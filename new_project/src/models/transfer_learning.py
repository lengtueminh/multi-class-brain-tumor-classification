"""EfficientNetB2 extracted from ``notebooks/03_transfer_efficientnetb2.ipynb``."""

import tensorflow as tf


def build_transfer_efficientnetb2(image_size=240, num_classes=3, weights="imagenet"):
    """Build and compile the notebook model with its entire backbone trainable.

    Inputs are normalized to [0, 1] by the data loader, then scaled to [0, 255]
    before EfficientNet's built-in preprocessing. ``weights=None`` allows
    building the same architecture without downloading pretrained weights.
    """
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = tf.keras.layers.Lambda(lambda image: image * 255.0, name="to_255")(inputs)
    backbone = tf.keras.applications.EfficientNetB2(
        weights=weights, include_top=False, input_tensor=x
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="model4_original_efficientnetb2")

    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
    return model
