"""Classification metrics and plots for three classes."""

import json
from pathlib import Path
from time import perf_counter

from matplotlib.figure import Figure
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.preprocessing import CLASS_INDEX_TO_NAME

CLASS_NAMES = [CLASS_INDEX_TO_NAME[index] for index in range(3)]


def labels_to_indices(labels):
    """Convert one-hot labels to class indices; accept indices directly."""
    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[1] == 3:
        if not np.isin(labels, [0, 1]).all() or not np.all(labels.sum(axis=1) == 1):
            raise ValueError("Labels must be valid one-hot vectors")
        return labels.argmax(axis=1)
    if labels.ndim != 1 or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Labels must be class indices 0, 1, 2 or one-hot vectors")
    return labels.astype(np.int64)


def compute_metrics(y_true, y_prob):
    """Compute accuracy, per-class metrics and macro averages."""
    y_true = labels_to_indices(y_true)
    y_prob = np.asarray(y_prob)
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    if y_prob.shape != (len(y_true), 3):
        raise ValueError("Predictions must have shape (number of labels, 3)")
    if (not np.isfinite(y_prob).all() or np.any(y_prob < 0)
            or np.any(y_prob > 1) or not np.allclose(y_prob.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Predictions must be finite class probabilities summing to 1")

    y_pred = y_prob.argmax(axis=1)
    # Include all three classes, even if one is never predicted.
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0,
    )
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_metrics": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def collect_predictions(model, dataset):
    """Return paired labels, probabilities and inference time in ms/image."""
    all_labels, all_probabilities = [], []
    elapsed = 0.0
    image_count = 0
    for images, labels in dataset:
        # Warm up once before timing predictions.
        if image_count == 0:
            np.asarray(model.predict_on_batch(images))
        start = perf_counter()
        probabilities = np.asarray(model.predict_on_batch(images))
        elapsed += perf_counter() - start

        # Keep labels and predictions from the same batch.
        indices = labels_to_indices(labels)
        if probabilities.shape != (len(indices), 3):
            raise ValueError("Each prediction batch must match its label batch")
        all_labels.append(indices)
        all_probabilities.append(probabilities)
        image_count += len(indices)

    if image_count == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    # Timing excludes data loading and preprocessing.
    return (np.concatenate(all_labels), np.concatenate(all_probabilities),
            1000 * elapsed / image_count)


def save_json(data, path):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def plot_confusion_matrix(matrix, path, model_name):
    """Rows are true classes; columns are predicted classes."""
    fig = Figure(figsize=(7, 6))
    ax = fig.subplots()
    chart = ConfusionMatrixDisplay(np.asarray(matrix), display_labels=CLASS_NAMES)
    chart.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=20)
    ax.set_title(f"Confusion matrix: {model_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_learning_curves(history, path, model_name):
    """Plot training and validation loss and accuracy."""
    fig = Figure(figsize=(11, 4))
    axes = fig.subplots(1, 2)
    for ax, key in zip(axes, ["loss", "accuracy"]):
        epochs = range(1, len(history[key]) + 1)
        ax.plot(epochs, history[key], label="Train")
        ax.plot(epochs, history[f"val_{key}"], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key.capitalize())
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"Learning curves: {model_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def evaluate_model(model, test_dataset, model_name, reports_dir="reports"):
    """Evaluate the test set and save metrics and a confusion matrix."""
    y_true, y_prob, inference_time = collect_predictions(model, test_dataset)
    metrics = compute_metrics(y_true, y_prob)
    metrics.update({
        "model_name": model_name,
        "class_names": CLASS_NAMES,
        "num_test_images": len(y_true),
        "parameter_count": int(model.count_params()),
        "inference_time": float(inference_time),
        "inference_time_unit": "ms/image",
        "inference_time_method": "Warmed predict_on_batch calls; excludes data loading",
    })

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, reports_dir / f"{model_name}_metrics.json")
    plot_confusion_matrix(metrics["confusion_matrix"],
                        reports_dir / f"{model_name}_confusion_matrix.png", model_name)
    return metrics
