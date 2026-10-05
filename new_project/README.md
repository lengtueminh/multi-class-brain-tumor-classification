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