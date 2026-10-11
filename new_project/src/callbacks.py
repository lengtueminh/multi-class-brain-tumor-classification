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
