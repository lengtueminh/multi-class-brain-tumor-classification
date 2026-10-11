# Project kế thừa

Đây là khu vực phát triển phiên bản mới dựa trên project gốc trong thư mục `../original/`.

- Đọc yêu cầu và phạm vi tại `PROJECT_SPEC.md`.
- Đặt notebook thử nghiệm trong `notebooks/`.
- Đặt mã nguồn tái sử dụng trong `src/`.
- Giữ dữ liệu đầu vào trong `data/raw/` và dữ liệu đã xử lý trong `data/processed/`.
- Lưu model, metrics và biểu đồ lần lượt trong `models/` và `reports/`.
- Đọc [HUONG_DAN_SU_DUNG.md](HUONG_DAN_SU_DUNG.md) trước khi chạy hoặc đóng góp code.

## Pipeline dùng chung

Ba model phải dùng cùng split và cùng cách chuẩn bị ảnh:

```python
from pathlib import Path

from src.data_loader import build_tf_dataset
from src.preprocessing import load_split_manifest
from src.reproducibility import set_global_seed

set_global_seed(42)
manifest = load_split_manifest(Path("data/processed/split_manifest.csv"))
train = build_tf_dataset(
    manifest,
    Path("data/raw"),
    split="train",
    training=True,
    image_size=224,
    batch_size=32,
    seed=42,
)
```

`training=True` chỉ được dùng với split `train`; validation và test không
augmentation. Không tạo lại split riêng trong notebook của từng model.

`src.preprocessing.load_case` restores the intended MATLAB image orientation:
HDF5/v7.3 images and masks are transposed together; classic MATLAB images are
already decoded in that orientation by SciPy. Tumor-border `(x, y)` values,
labels, and patient identifiers are unchanged. All three models now use this
orientation, with 224px inputs for simple/complex CNN and 240px for EfficientNetB2.

This changes the shared reader's previous HDF5 orientation for complex CNN and
EfficientNetB2. Historical checkpoints need their original preprocessing to
reproduce their results; new comparisons require retraining with this reader.
Saved notebook outputs and EDA image hashes still describe the run that produced
them until those notebooks are rerun. The existing patient split remains valid
and does not need regeneration.

## Training CLI

From the repository root, using the project virtual environment:

```powershell
.\.venv\Scripts\python.exe new_project/src/train.py --model simple_cnn
.\.venv\Scripts\python.exe new_project/src/train.py --model complex_cnn
.\.venv\Scripts\python.exe new_project/src/train.py --model efficientnetb2
```

Only `--model` is required. Default data/output paths use the script location,
so it also works from other working directories. From `new_project/`, module
execution also works: `python -m src.train --model simple_cnn`.

| Model | Backend | Device default | Precision | Size | Batch | Epochs |
|---|---|---|---|---|---|---|
| `simple_cnn` | TensorFlow | GPU if available, otherwise CPU | float32 | 224 | 32 | 50 |
| `complex_cnn` | PyTorch/Keras | CUDA required | mixed_float16 | 224 | 8 | 90 |
| `efficientnetb2` | TensorFlow | GPU if available, otherwise CPU | float32 | 240 | 32 | 50 |

The complex model retains gradient accumulation over four batches, the 60-epoch
cosine-decay schedule, and its original delayed early stopping. Simple CNN keeps
its validation-F1 callback and `val_loss` checkpoint; EfficientNetB2 keeps its
fully trainable ImageNet backbone and `val_accuracy` checkpoint. No frozen-head
phase or differential learning rates are introduced.

Unavailable requested GPUs fail clearly. Native Windows TensorFlow normally
runs on CPU. For a complex-model CPU run, explicitly use:

```powershell
.\.venv\Scripts\python.exe new_project/src/train.py --model complex_cnn --device cpu --precision float32
```

Optional overrides: `--epochs`, `--batch-size`, `--seed`, `--device`,
`--precision`, `--raw-dir`, `--manifest`, `--output-dir`, and
`--no-augmentation`. EfficientNetB2 additionally supports `--weights none` for
offline smoke tests; this trains from random initialization instead of ImageNet
and is not a transfer-learning result. Its default `--weights imagenet` may
download pretrained weights on first use.

Each run creates a new `runs/<model>_seed<seed>_<timestamp>/` directory, or the
new directory supplied with `--output-dir`. Existing output directories are
rejected. Checkpoints go in `models/`; `reports/` holds the resolved runtime,
manifest snapshot, model summary, history, curves, and evaluation. Generated
runs are ignored by Git.

Notebooks and CLI use the same modules: `src/models/`, `preprocessing.py`,
`data_loader.py`, `reproducibility.py`, `callbacks.py`, `training.py`, and
`evaluation.py`. Complex CNN keeps its original backend-specific seed setup.
The old `train_complex_cnn.py` import remains available for existing scripts.
Simple/transfer reports preserve their notebook formats and timing; compare
timing numbers only when measurement method, backend, precision, and hardware
match. Saved notebook outputs remain historical until rerun.

`configs/default.yaml` continues to configure the preprocessing and transfer
notebook; CLI defaults are listed above and overrides are recorded per run.
