import numpy as np
import pandas as pd

from src.preprocessing import build_patient_level_split, one_hot_labels, prepare_image


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


def test_one_hot_labels():
    encoded = one_hot_labels(np.array([0, 2, 1]))
    np.testing.assert_array_equal(encoded, np.eye(3, dtype=np.float32)[[0, 2, 1]])
