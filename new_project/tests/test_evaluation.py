import csv
import json

import numpy as np
import pytest

from src.evaluation import CLASS_NAMES, evaluate_model


@pytest.mark.parametrize("model_name", ["simple_cnn", "complex_cnn", "efficientnetb2"])
@pytest.mark.parametrize("split", ["test", "validation"])
def test_shared_evaluation_preserves_paired_predictions_and_metrics(tmp_path, model_name, split):
    # Reordering between iterations must not separate predictions from labels.
    class Dataset:
        iterations = 0

        def __iter__(self):
            self.iterations += 1
            batches = [
                (np.array([0, 1]), np.eye(3)[[0, 1]]),
                (np.array([2, 3]), np.eye(3)[[2, 0]]),
            ]
            return iter(batches if self.iterations % 2 else batches[::-1])

    class Model:
        def predict_on_batch(self, images):
            return np.eye(3)[np.array([0, 2, 2, 1])[images]]

        def evaluate(self, dataset, **kwargs):
            list(dataset)
            return {"loss": 0.75, "accuracy": 0.5}

        def count_params(self):
            return 123

    result = evaluate_model(Model(), Dataset(), model_name, tmp_path, split=split)
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 0, 1], [0, 0, 1]]
    assert result["accuracy"] == 0.5
    assert result["macro_precision"] == pytest.approx(0.5)
    assert result["macro_recall"] == pytest.approx(0.5)
    assert result["macro_f1"] == pytest.approx(4 / 9)
    assert result["per_class_metrics"][CLASS_NAMES[1]]["f1"] == 0
    assert result["loss"] == 0.75
    assert result["inference_time_unit"] == "ms/image"
    assert result["inference_time"] >= 0
    prefix = model_name if split == "test" else f"{model_name}_validation"
    assert json.loads((tmp_path / f"{prefix}_metrics.json").read_text()) == result
    with (tmp_path / f"{prefix}_classification_report.csv").open() as handle:
        report = {row[""]: row for row in csv.DictReader(handle)}
    assert float(report["macro avg"]["f1-score"]) == pytest.approx(result["macro_f1"])
    with (tmp_path / f"{prefix}_confusion_matrix.csv").open() as handle:
        matrix = [[int(row[name]) for name in CLASS_NAMES] for row in csv.DictReader(handle)]
    assert matrix == result["confusion_matrix"]
    assert (tmp_path / f"{prefix}_confusion_matrix.png").exists()


def test_shared_evaluation_includes_unpredicted_classes(tmp_path):
    class Model:
        def predict_on_batch(self, images):
            return np.array([[1.0, 0.0, 0.0]] * len(images))

        def evaluate(self, dataset, **kwargs):
            return {"loss": 1.0}

        def count_params(self):
            return 0

    dataset = [(np.zeros((3, 1)), np.eye(3))]
    result = evaluate_model(Model(), dataset, "simple_cnn", tmp_path)
    assert result["macro_precision"] == pytest.approx(1 / 9)
    assert result["macro_recall"] == pytest.approx(1 / 3)
    assert result["macro_f1"] == pytest.approx(1 / 6)
    assert result["per_class_metrics"][CLASS_NAMES[2]]["precision"] == 0
