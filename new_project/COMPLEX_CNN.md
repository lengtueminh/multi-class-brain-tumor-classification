# Selected Complex CNN

The final notebook is `notebooks/model_complex_cnn.ipynb`.
Its model definition is also available as `src.complex_cnn.build_complex_cnn`
for comparison with the other team members' models.

## Run the notebook

Restart the kernel and run all cells in order.
The notebook uses Keras with the PyTorch backend;
TensorFlow supplies the shared data pipeline. Environment dependencies are
recorded in `tools/requirements-gpu-benchmark.txt` and its lock file.
The shared split must already exist at `data/processed/split_manifest.csv`.

Internal explanations: [notebook and architecture](docs/internal/NOTEBOOK_COMPLEX_GIAI_THICH.md)
and [evaluation module](docs/internal/EVALUATION_GIAI_THICH.md).

The model has 456,755 parameters: sequential and parallel convolution branches,
fusion, two residual MBConv blocks, and a three-class classifier. BatchNorm uses
momentum 0.9 and epsilon 1e-3. Training retains seed 42, image size 224, batch
size 8, gradient accumulation 4, dropout 0.2, Adam with learning rate 2e-4,
and cosine decay over `train_batches * 60` steps to 1e-6. The maximum budget
is 90 epochs. Early stopping monitors validation loss with patience 20,
min_delta 0, and start_from_epoch 60. The global minimum validation-loss
checkpoint is loaded for evaluation.

Each new execution saves results under `runs/complex_cnn_seed42_<timestamp>/`,
with `models/complex_cnn_best.keras` and `reports/complex_cnn_*` inside that run.
It does not overwrite the selected results at the project root.

## Preserved selected results

The original completed BatchNorm run trained for 90 epochs and selected epoch 84
by validation loss. Its checkpoint is preserved byte for byte at
`models/complex_cnn_best.keras`. The main results are under `reports/`:

- `complex_cnn_metrics.json` and `complex_cnn_validation_metrics.json`
- `complex_cnn_history.json` and `complex_cnn_learning_curve.png`
- `complex_cnn_confusion_matrix.png` and `complex_cnn_validation_confusion_matrix.png`
- `complex_cnn_summary.txt` and `complex_cnn_training_config.json`
- `complex_cnn_split_manifest_used.csv` and `complex_cnn_training_log.csv`
- `complex_cnn_provenance.json`, including the source run and checkpoint SHA-256

Per-epoch diagnostic inference, per-epoch weights, diagnostic caches, and
overnight experiment summaries have been removed from the project. The retired
notebooks, tools, and original run folders are archived outside the repository
at `../complex_cnn_retired_20261010` (relative to the repository root), because
the execution policy blocked permanent bulk deletion. History and the training
log retain loss/accuracy for train and validation. Test results remain
exploratory because the test cohort was inspected in previous experiments.

## Use the model from Python

Run from `new_project/`. Set the backend before importing Keras; when inspecting
the selected CUDA run, also set the original mixed-precision policy.

```python
import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from src.complex_cnn import build_complex_cnn
from src.preprocessing import load_split_manifest
from pathlib import Path

# Build the same architecture and optimizer for a new training run.
manifest = load_split_manifest(Path("data/processed/split_manifest.csv"))
train_batches = (int(manifest["split"].eq("train").sum()) + 7) // 8
model = build_complex_cnn(decay_steps=train_batches * 60)

# Load the selected trained checkpoint for a model comparison.
best_model = keras.models.load_model("models/complex_cnn_best.keras", compile=False)
```

Use `src.evaluation.evaluate_model` with the shared test dataset to export
accuracy, macro precision/recall/F1, per-class metrics, parameter count,
inference time, and the confusion matrix in the common report format.
