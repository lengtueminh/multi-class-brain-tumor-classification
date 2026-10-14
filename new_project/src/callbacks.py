"""Reusable training callbacks extracted from the model notebooks."""

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report


class ValidationMetrics(tf.keras.callbacks.Callback):
    """Log the simple CNN notebook's validation macro-F1 and class recalls."""

    def __init__(self, dataset, class_names):
        super().__init__()
        self.dataset = dataset
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []

        for images, labels in self.dataset:
            probabilities = self.model(images, training=False)
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            y_pred.extend(np.argmax(probabilities.numpy(), axis=1))

        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        metrics = {
            "val_macro_f1": report["macro avg"]["f1-score"],
            **{
                f"val_recall_{name.replace(' ', '_')}": report[name]["recall"]
                for name in self.class_names
            },
        }

        if logs is not None:
            logs.update(metrics)

        print("\n" + " | ".join(
            f"{name}: {value:.4f}" for name, value in metrics.items()
        ))


def simple_callbacks(validation_dataset, class_names, checkpoint_path):
    """The simple notebook's original callback order and settings."""
    return [
        ValidationMetrics(validation_dataset, class_names),
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor="val_loss", mode="min",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=4,
            min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=10,
            restore_best_weights=True, verbose=1,
        ),
    ]


def complex_callbacks(checkpoint_path, reports_dir, *, patience=20,
                      min_delta=0.0, decay_epochs=60):
    """The complex notebook's original callback order and settings."""
    import keras
    from pathlib import Path

    return [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.CSVLogger(str(Path(reports_dir) / "training_log.csv")),
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor="val_loss", mode="min", save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=patience, min_delta=min_delta,
            start_from_epoch=decay_epochs, restore_best_weights=False,
        ),
    ]


def transfer_callbacks(checkpoint_path):
    """The transfer notebook's original callback order and settings."""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", save_best_only=True,
            mode="max", verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.3, patience=5, min_delta=0.001, verbose=1,
        ),
    ]
