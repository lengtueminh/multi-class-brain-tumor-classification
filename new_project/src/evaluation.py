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


def evaluate_validation(model, dataset, model_name, reports_dir):
    """Preserve the complex notebook's validation report."""
    metrics = {
        key: float(value) for key, value in
        model.evaluate(dataset, return_dict=True, verbose=0).items()
    }
    y_true, y_prob, _ = collect_predictions(model, dataset)
    metrics.update(compute_metrics(y_true, y_prob))
    metrics.update(split="validation", num_validation_images=len(y_true))
    reports_dir = Path(reports_dir)
    save_json(metrics, reports_dir / f"{model_name}_validation_metrics.json")
    plot_confusion_matrix(metrics["confusion_matrix"],
                          reports_dir / f"{model_name}_validation_confusion_matrix.png", model_name)
    return metrics


def evaluate_simple_cnn(model, test_dataset, history, reports_dir, *, show=False):
    """Preserve the simple notebook's reports and end-to-end seconds/image timing."""
    import time
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    best_model = model
    # Cell 9 - Curves, test metrics and confusion matrix
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="train")
    plt.plot(history["val_accuracy"], label="validation")
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="train")
    plt.plot(history["val_loss"], label="validation")
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "simple_cnn_learning_curve.png", dpi=150)
    if show:
        plt.show()
    plt.close()

    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_dataset])
    start = time.perf_counter()
    probabilities = best_model.predict(test_dataset, verbose=1)
    inference_time = (time.perf_counter() - start) / len(y_true)
    y_pred = probabilities.argmax(axis=1)
    report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class_metrics": {name: report[name] for name in CLASS_NAMES},
        "parameter_count": int(best_model.count_params()),
        "inference_time": float(inference_time),
    }
    with open(reports_dir / "simple_cnn_metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))

    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]), annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(reports_dir / "simple_cnn_confusion_matrix.png", dpi=150)
    if show:
        plt.show()
    plt.close()
    return metrics


def evaluate_transfer(model, val_ds, test_ds, history, reports_dir, *, show=False):
    """Preserve EfficientNetB2's notebook evaluation and CSV reports."""
    import pandas as pd

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    best_model = model
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix

    val_metrics = best_model.evaluate(val_ds, return_dict=True)
    test_metrics = best_model.evaluate(test_ds, return_dict=True)
    print(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")

    history_df = pd.DataFrame(history)
    history_df["epoch"] = np.arange(1, len(history_df) + 1)
    history_df.to_csv(reports_dir / "history.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["epoch"], history_df["loss"], label="train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history_df["epoch"], history_df["accuracy"], label="train")
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "learning_curves.png", dpi=160)
    if show:
        plt.show()
    plt.close()

    y_true, y_pred = [], []
    for images, labels in test_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(best_model.predict(images, verbose=0), axis=1))

    report_df = pd.DataFrame(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, output_dict=True)).transpose()
    report_df.to_csv(reports_dir / "classification_report.csv")

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(reports_dir / "confusion_matrix.csv")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Original EfficientNetB2 - Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png", dpi=160)
    if show:
        plt.show()
    plt.close()
    return val_metrics, test_metrics, report_df, cm_df
