"""Shared manifest-backed data loader for the three model experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from src.preprocessing import CLASS_INDEX_TO_NAME, load_case, load_split_manifest, one_hot_labels, prepare_image


def _sample_generator(
    manifest: pd.DataFrame,
    raw_dir: Path,
    image_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for row in manifest.itertuples(index=False):
        case = load_case(raw_dir / row.file)
        image, _, _ = prepare_image(case["image"], case["tumorMask"], image_size=image_size)
        label = one_hot_labels(np.asarray([row.class_index], dtype=np.int64))[0]
        yield image, label


def build_tf_dataset(
    manifest: pd.DataFrame | Path,
    raw_dir: Path,
    *,
    split: str,
    image_size: int = 224,
    batch_size: int = 32,
    training: bool = False,
    seed: int = 42,
):
    """Build a lazy ``tf.data.Dataset`` with identical preprocessing for all models."""
    import tensorflow as tf

    frame = load_split_manifest(manifest) if isinstance(manifest, Path) else manifest.copy()
    frame = frame.loc[frame["split"].eq(split)].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"split {split!r} contains no samples")
    if split == "train" and training is False:
        raise ValueError("the training split must be requested with training=True")
    if training and split != "train":
        raise ValueError("augmentation is only allowed for the train split")

    output_signature = (
        tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(len(CLASS_INDEX_TO_NAME),), dtype=tf.float32),
    )
    dataset = tf.data.Dataset.from_generator(
        lambda: _sample_generator(frame, raw_dir, image_size),
        output_signature=output_signature,
    )
    if training:
        dataset = dataset.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)
        dataset = dataset.map(_augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _augment_batch(image, label):
    """Apply train-only, label-preserving image augmentation."""
    import tensorflow as tf

    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.random_flip_left_right(image)
    return tf.clip_by_value(image, 0.0, 1.0), label
