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
