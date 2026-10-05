"""Reusable preprocessing utilities for the Figshare brain tumor dataset."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder

EXPECTED_FIELDS = {"label", "PID", "image", "tumorBorder", "tumorMask"}
RAW_LABEL_TO_CLASS_INDEX = {1: 1, 2: 0, 3: 2}
CLASS_INDEX_TO_NAME = {0: "glioma", 1: "meningioma", 2: "pituitary tumor"}
REQUIRED_INVENTORY_COLUMNS = {"class_index", "patient_id"}


def _scalar(value: Any) -> Any:
    """Remove MATLAB singleton dimensions and return a Python scalar when possible."""
    array = np.asarray(value).squeeze()
    return array.item() if array.size == 1 else array


def decode_patient_id(value: Any) -> str:
    """Convert MATLAB's numeric character vector for ``PID`` into readable text."""
    array = np.asarray(value).squeeze()
    if array.dtype.kind in {"i", "u"}:
        return "".join(chr(int(item)) for item in array)
    return str(_scalar(array))


def canonicalize_patient_id(value: Any) -> str:
    """Return one stable patient identifier for MATLAB and CSV representations."""
    if isinstance(value, str):
        numbers = re.findall(r"\d+", value)
        if value.strip().startswith("[") and numbers:
            return decode_patient_id(np.asarray([int(item) for item in numbers], dtype=np.int64))
        return value.strip()
    return decode_patient_id(value)


def prepare_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an EDA inventory before splitting it."""
    missing = REQUIRED_INVENTORY_COLUMNS - set(inventory.columns)
    if missing:
        raise ValueError(f"inventory is missing columns: {sorted(missing)}")

    result = inventory.copy()
    result["patient_id"] = result["patient_id"].map(canonicalize_patient_id)
    result["class_index"] = pd.to_numeric(result["class_index"], errors="raise").astype(int)
    if not result["class_index"].isin(CLASS_INDEX_TO_NAME).all():
        raise ValueError("inventory contains an unknown class index")
    if result["patient_id"].eq("").any():
        raise ValueError("inventory contains an empty patient id")
    patient_classes = result.groupby("patient_id")["class_index"].nunique()
    if (patient_classes > 1).any():
        raise ValueError("a patient is assigned to more than one class")
    if "file" in result and result["file"].duplicated().any():
        raise ValueError("inventory contains duplicate file paths")
    if "class_name" in result:
        result["class_name"] = result["class_index"].map(CLASS_INDEX_TO_NAME)
    return result


def load_case(path: Path) -> dict[str, np.ndarray]:
    """Load one MATLAB v5 or v7.3 case.

    SciPy reads classic MATLAB files, while MATLAB v7.3 files are HDF5 files
    and therefore require ``h5py``. Both readers are normalized to the same
    dictionary of ``cjdata`` fields so the rest of the pipeline is format
    independent.
    """
    try:
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        if "cjdata" not in mat:
            raise KeyError("missing cjdata")
        cjdata = mat["cjdata"]
        fields = {
            name: np.asarray(getattr(cjdata, name))
            for name in EXPECTED_FIELDS
            if hasattr(cjdata, name)
        }
    except NotImplementedError:
        with h5py.File(path, "r") as mat:
            if "cjdata" not in mat:
                raise KeyError("missing cjdata")
            group = mat["cjdata"]
            fields = {
                name: np.asarray(group[name])
                for name in EXPECTED_FIELDS
                if name in group
            }
    missing = EXPECTED_FIELDS - fields.keys()
    if missing:
        raise KeyError(f"missing fields: {sorted(missing)}")
    return fields


def crop_brain(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Crop background around the brain and keep the mask spatially aligned.

    A threshold creates a rough foreground image, morphological closing fills
    small gaps, and the largest external contour supplies the crop box. The
    identical box is applied to both image and mask so their coordinates stay
    aligned.
    """
    image_array = np.asarray(image)
    mask_array = np.asarray(mask)
    if image_array.ndim != 2 or mask_array.shape != image_array.shape:
        raise ValueError("image and mask must be 2D arrays with identical shapes")

    image_float = image_array.astype(np.float32)
    maximum = float(np.max(image_float))
    if maximum <= 0:
        height, width = image_array.shape
        return image_array, mask_array, (0, 0, width, height)

    # Use a relative threshold because raw MRI intensity ranges differ by image.
    threshold = np.uint8(image_float > maximum * 0.05) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    # Close small holes and gaps before searching for the brain contour.
    foreground = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        height, width = image_array.shape
        return image_array, mask_array, (0, 0, width, height)

    x, y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return image_array[y : y + height, x : x + width], mask_array[y : y + height, x : x + width], (x, y, width, height)


def prepare_image(
    image: np.ndarray,
    mask: np.ndarray,
    image_size: int = 224,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Prepare one sample for CNN input.

    The image is cropped, resized, converted to ``float32``, normalized
    per image to ``[0, 1]``, and replicated across three channels. The mask
    uses nearest-neighbor resize so it remains binary.
    """
    cropped_image, cropped_mask, bbox = crop_brain(image, mask)
    # Area interpolation is suitable for shrinking continuous-valued images.
    resized_image = cv2.resize(cropped_image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    # Nearest-neighbor interpolation prevents fractional mask labels.
    resized_mask = cv2.resize(cropped_mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    image_float = resized_image.astype(np.float32)
    minimum, maximum = float(image_float.min()), float(image_float.max())
    # The constant-image branch avoids division by zero.
    normalized = (image_float - minimum) / (maximum - minimum) if maximum > minimum else np.zeros_like(image_float)
    # EfficientNet and the CNN models expect three input channels.
    rgb_image = np.repeat(normalized[..., np.newaxis], 3, axis=-1)
    return rgb_image, (resized_mask > 0).astype(np.uint8), bbox


def build_patient_level_split(
    inventory: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> pd.DataFrame:
    """Create deterministic train/validation/test assignments at patient level.

    Patients are shuffled independently inside each class, then all images
    belonging to a patient receive the same split. This prevents slices from
    one patient leaking across train, validation, and test.
    """
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio <= 0 for ratio in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must sum to 1")
    inventory = prepare_inventory(inventory)
    rng = np.random.default_rng(seed)
    assignments: list[dict[str, str]] = []
    for class_index, group in inventory.groupby("class_index", sort=True):
        # Stratify by class while assigning whole patients to one partition.
        patients = np.array(sorted(group["patient_id"].unique()), dtype=object)
        rng.shuffle(patients)
        train_end = int(round(len(patients) * train_ratio))
        validation_end = train_end + int(round(len(patients) * validation_ratio))
        for patient in patients[:train_end]:
            assignments.append({"class_index": int(class_index), "patient_id": str(patient), "split": "train"})
        for patient in patients[train_end:validation_end]:
            assignments.append({"class_index": int(class_index), "patient_id": str(patient), "split": "validation"})
        for patient in patients[validation_end:]:
            assignments.append({"class_index": int(class_index), "patient_id": str(patient), "split": "test"})

    split_by_patient = pd.DataFrame(assignments)
    # Attach the patient-level assignment back to every image row.
    result = inventory.merge(split_by_patient, on=["class_index", "patient_id"], how="left", validate="many_to_one")
    if result["split"].isna().any():
        raise RuntimeError("Some inventory rows were not assigned to a split")
    overlap = result.groupby("split")["patient_id"].unique().reindex(
        ("train", "validation", "test"), fill_value=[]
    )
    # Fail loudly if the split would cause patient leakage.
    for left in ("train", "validation", "test"):
        for right in ("train", "validation", "test"):
            if left < right and set(overlap[left]).intersection(overlap[right]):
                raise RuntimeError("Patient leakage detected between splits")
    return result


def load_split_manifest(path: Path) -> pd.DataFrame:
    """Load and validate a previously generated split manifest."""
    manifest = prepare_inventory(pd.read_csv(path))
    if "file" not in manifest:
        raise ValueError("split manifest is missing the file column")
    if "split" not in manifest:
        raise ValueError("split manifest is missing the split column")
    allowed_splits = {"train", "validation", "test"}
    if not manifest["split"].isin(allowed_splits).all():
        raise ValueError("split manifest contains an unknown split")
    return manifest


def save_split_manifest(
    inventory: pd.DataFrame,
    path: Path,
    *,
    seed: int = 42,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> pd.DataFrame:
    """Create, validate and persist a deterministic patient-level manifest."""
    manifest = build_patient_level_split(
        inventory,
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    return manifest


def one_hot_labels(class_indices: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """Convert class indices into fixed-width categorical vectors.

    For three classes, indices ``0``, ``1``, and ``2`` become
    ``[1, 0, 0]``, ``[0, 1, 0]``, and ``[0, 0, 1]`` respectively.
    """
    indices = np.asarray(class_indices, dtype=np.int64).reshape(-1, 1)
    if np.any((indices < 0) | (indices >= num_classes)):
        raise ValueError("class index outside the configured range")
    encoder = OneHotEncoder(categories=[np.arange(num_classes)], sparse_output=False, dtype=np.float32)
    return encoder.fit_transform(indices)
