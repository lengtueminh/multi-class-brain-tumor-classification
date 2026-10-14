import numpy as np
import pandas as pd
import pytest

from src import data_loader


@pytest.fixture
def samples(monkeypatch):
    import tensorflow as tf

    frame = pd.DataFrame({"split": ["train"] * 6 + ["validation", "test"]})

    def generate(frame, raw_dir, image_size):
        for i in range(len(frame)):
            yield (np.full((image_size, image_size, 3), i / 10, np.float32),
                   np.eye(3, dtype=np.float32)[i % 3])

    monkeypatch.setattr(data_loader, "_sample_generator", generate)
    monkeypatch.setattr(data_loader, "_augment_batch", lambda x, y: (x + 0.25, y))
    return frame, tf


def test_clean_train_and_validation_keep_order_and_cardinality(samples, tmp_path):
    frame, tf = samples
    for split, count in (("train", 6), ("validation", 1), ("test", 1)):
        dataset = data_loader.build_tf_dataset(
            frame, tmp_path, split=split, image_size=4, batch_size=2,
        )
        assert int(tf.data.experimental.cardinality(dataset)) == (count + 1) // 2
        images = np.concatenate([x.numpy() for x, _ in dataset])
        np.testing.assert_allclose(images[:, 0, 0, 0], np.arange(count) / 10)


def test_training_augmentation_can_be_disabled_without_disabling_shuffle(samples, tmp_path):
    frame, tf = samples

    def build(augment):
        tf.keras.utils.set_random_seed(42)
        return data_loader.build_tf_dataset(
            frame, tmp_path, split="train", training=True, augment=augment,
            image_size=4, batch_size=2, seed=42,
        )

    clean = list(build(False).as_numpy_iterator())
    augmented = list(build(True).as_numpy_iterator())
    for (x, y), (aug_x, aug_y) in zip(clean, augmented, strict=True):
        np.testing.assert_allclose(aug_x, x + 0.25)
        np.testing.assert_array_equal(aug_y, y)
    order = np.concatenate([x[:, 0, 0, 0] for x, _ in clean])
    assert not np.array_equal(order, np.arange(6, dtype=np.float32) / 10)


def test_validation_rejects_training_transforms(samples, tmp_path):
    frame, _ = samples
    for option in (dict(training=True), dict(augment=True), dict(shuffle=True)):
        with pytest.raises(ValueError):
            data_loader.build_tf_dataset(frame, tmp_path, split="validation", **option)
