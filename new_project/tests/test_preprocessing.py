import numpy as np
import pandas as pd
import h5py
from scipy.io import savemat

from src.preprocessing import (
    build_patient_level_split,
    canonicalize_patient_id,
    load_case,
    one_hot_labels,
    prepare_image,
    prepare_inventory,
)


def test_load_case_restores_same_image_and_mask_orientation_across_mat_formats(tmp_path):
    # A rectangular, asymmetric sample catches axis swaps hidden by square MRIs.
    image = np.arange(15, dtype=np.int16).reshape(3, 5)
    mask = np.zeros((3, 5), dtype=np.uint8)
    mask[1, 3] = 1
    border = np.asarray([[4, 2, 5, 2]], dtype=np.float64)
    fields = {
        "image": image,
        "tumorMask": mask,
        "tumorBorder": border,
        "label": np.asarray([[2]], dtype=np.int16),
        "PID": np.asarray([[ord(c) for c in "patient"]], dtype=np.uint16),
    }
    classic_path = tmp_path / "classic.mat"
    hdf5_path = tmp_path / "v73.mat"
    savemat(classic_path, {"cjdata": fields})
    with h5py.File(hdf5_path, "w", userblock_size=512) as mat:
        group = mat.create_group("cjdata")
        for name, value in fields.items():
            group.create_dataset(name, data=value.T)
    # MATLAB v7.3 header makes SciPy route the file to the HDF5 fallback.
    header = b"MATLAB 7.3 MAT-file, synthetic orientation fixture".ljust(116, b" ")
    header += b"\x00" * 8 + b"\x00\x02IM"
    with hdf5_path.open("r+b") as mat:
        mat.write(header)

    classic = load_case(classic_path)
    hdf5 = load_case(hdf5_path)
    for case in (classic, hdf5):
        np.testing.assert_array_equal(case["image"], image)
        np.testing.assert_array_equal(case["tumorMask"], mask)
        # Restoring array orientation does not exchange x and y coordinates.
        np.testing.assert_array_equal(case["tumorBorder"].ravel(), border.ravel())
        assert int(np.asarray(case["label"]).squeeze()) == 2
        np.testing.assert_array_equal(case["PID"].ravel(), fields["PID"].ravel())
    for size in (224, 240):
        classic_image, classic_mask, classic_bbox = prepare_image(
            classic["image"], classic["tumorMask"], image_size=size,
        )
        hdf5_image, hdf5_mask, hdf5_bbox = prepare_image(
            hdf5["image"], hdf5["tumorMask"], image_size=size,
        )
        np.testing.assert_array_equal(classic_image, hdf5_image)
        np.testing.assert_array_equal(classic_mask, hdf5_mask)
        assert classic_bbox == hdf5_bbox


def test_prepare_image_returns_normalized_rgb_and_binary_mask():
    image = np.zeros((20, 20), dtype=np.int16)
    image[4:16, 5:15] = np.arange(120).reshape(12, 10)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[8:12, 8:12] = 1

    prepared, prepared_mask, bbox = prepare_image(image, mask, image_size=16)

    assert prepared.shape == (16, 16, 3)
    assert prepared.dtype == np.float32
    assert 0.0 <= prepared.min() <= prepared.max() <= 1.0
    assert prepared_mask.shape == (16, 16)
    assert set(np.unique(prepared_mask)).issubset({0, 1})
    assert bbox[2] > 0 and bbox[3] > 0


def test_patient_level_split_has_no_overlap():
    inventory = pd.DataFrame(
        {
            "class_index": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "patient_id": ["a", "a", "b", "c", "c", "d", "e", "e", "f"],
        }
    )
    split = build_patient_level_split(inventory, seed=42)
    sets = {name: set(split.loc[split["split"] == name, "patient_id"]) for name in ("train", "validation", "test")}
    assert not sets["train"] & sets["validation"]
    assert not sets["train"] & sets["test"]
    assert not sets["validation"] & sets["test"]


def test_inventory_normalizes_matlab_patient_id_encoding():
    inventory = pd.DataFrame(
        {
            "file": ["1.mat"],
            "class_index": [0],
            "patient_id": ["[49 48 48 51 54 48]"],
        }
    )

    normalized = prepare_inventory(inventory)

    assert normalized.loc[0, "patient_id"] == "100360"
    assert canonicalize_patient_id("[49 48 48 51 54 48]") == "100360"


def test_split_is_reproducible():
    inventory = pd.DataFrame(
        {
            "file": [f"{index}.mat" for index in range(12)],
            "class_index": [0] * 4 + [1] * 4 + [2] * 4,
            "patient_id": [f"p{index}" for index in range(12)],
        }
    )

    first = build_patient_level_split(inventory, seed=7)
    second = build_patient_level_split(inventory, seed=7)

    pd.testing.assert_frame_equal(first, second)


def test_one_hot_labels():
    encoded = one_hot_labels(np.array([0, 2, 1]))
    np.testing.assert_array_equal(encoded, np.eye(3, dtype=np.float32)[[0, 2, 1]])
