# Hướng dẫn sử dụng chung

Tài liệu này quy định cách chạy pipeline và lưu kết quả thống nhất cho cả
ba model trong project phân loại đa lớp u não.

## 1. Cấu trúc thư mục

```text
new_project/
├── configs/default.yaml       # Cấu hình dùng chung
├── data/raw/                  # Dữ liệu gốc, không chỉnh sửa
├── data/processed/            # Split manifest và dữ liệu sinh ra
├── notebooks/                 # Notebook thử nghiệm
├── src/                       # Code dùng chung và code model
├── tests/                     # Unit test
├── models/                    # Checkpoint, không commit file lớn
└── reports/                   # Metrics, biểu đồ và model summary
```

Không sửa trực tiếp thư mục `original/`. Mọi code mới đặt trong
`new_project/`.

## 2. Môi trường Python

Project sử dụng Python 3.13 và virtual environment `.venv` ở thư mục gốc.
Trên Windows, chạy lệnh từ thư mục repository:

```powershell
& ".\.venv\Scripts\Activate.ps1"
Set-Location ".\new_project"
```

Nếu môi trường chưa có các thư viện cần thiết, cài theo manifest của project
hoặc cài tối thiểu:

```powershell
pip install numpy pandas scipy scikit-learn opencv-python h5py tensorflow pytest
```

## 3. Dữ liệu và quy ước bắt buộc

Ba model phải sử dụng cùng:

- Dataset trong `data/raw/`.
- Split manifest tại `data/processed/split_manifest.csv`.
- Random seed `42`.
- Input `224 x 224 x 3`.
- Normalize pixel về `[0, 1]`.
- Class mapping:
  - `0 = glioma`
  - `1 = meningioma`
  - `2 = pituitary tumor`

Split được thực hiện theo patient, không theo từng ảnh. Một patient không được
xuất hiện ở nhiều split.

Chỉ tập `train` được augmentation. Validation và test phải giữ nguyên để kết
quả đánh giá công bằng.

## 4. Chạy và kiểm tra preprocessing

Các hàm dùng chung nằm trong:

- `src/preprocessing.py`
- `src/data_loader.py`
- `src/reproducibility.py`

Ví dụ tạo dataset cho model:

```python
from pathlib import Path

from src.data_loader import build_tf_dataset
from src.preprocessing import load_split_manifest
from src.reproducibility import set_global_seed

set_global_seed(42)

manifest = load_split_manifest(Path("data/processed/split_manifest.csv"))
raw_dir = Path("data/raw")

train_dataset = build_tf_dataset(
    manifest,
    raw_dir,
    split="train",
    image_size=224,
    batch_size=32,
    training=True,
    seed=42,
)

validation_dataset = build_tf_dataset(
    manifest,
    raw_dir,
    split="validation",
    image_size=224,
    batch_size=32,
)

test_dataset = build_tf_dataset(
    manifest,
    raw_dir,
    split="test",
    image_size=224,
    batch_size=32,
)
```

Không tạo split mới trong notebook của từng model. Nếu cần thay đổi split,
phải thống nhất với cả nhóm và cập nhật manifest dùng chung.

## 5. Chạy unit test

Từ thư mục `new_project/`:

```powershell
& "..\.venv\Scripts\python.exe" -m pytest tests -q
```

Trước khi push code, cần bảo đảm test pass và kiểm tra không có lỗi syntax hoặc
type trong các file đã thay đổi.

## 6. Quy ước huấn luyện model

Mỗi model cần có:

1. Hàm tạo model.
2. Hàm hoặc notebook huấn luyện.
3. Checkpoint tốt nhất.
4. Learning curve train/validation.
5. Metrics trên test set.
6. Confusion matrix.
7. Model summary và số lượng parameter.

Không dùng test set cho early stopping, chọn hyperparameter hoặc chọn
checkpoint tốt nhất.

### Model 1 — Simple CNN

- Huấn luyện từ đầu.
- Không sử dụng pretrained weights.
- Phụ trách thêm kiểm tra data pipeline và khả năng tái lập.

### Model 2 — Complex Sequential–Parallel CNN

- Dùng Functional API.
- Có nhánh tuần tự và nhánh song song.
- Nhánh song song sử dụng kernel `1x1`, `3x3`, `5x5`.
- Phụ trách thêm module evaluation và bảng so sánh.

### Model 3 — EfficientNetB0

- Giai đoạn 1: đóng băng backbone, chỉ train classifier head.
- Giai đoạn 2: mở một phần layer cuối để fine-tuning với learning rate nhỏ.
- Báo cáo riêng kết quả của hai giai đoạn.
- Phụ trách thêm Grad-CAM, ablation augmentation, Gaussian noise robustness
  và inference.

## 7. Quy ước lưu kết quả

Đặt tên file theo model, dùng chữ thường và dấu gạch dưới:

```text
models/
├── simple_cnn_best.keras
├── complex_cnn_best.keras
├── efficientnet_transfer_learning_best.keras
└── efficientnet_fine_tuning_best.keras

reports/
├── simple_cnn_metrics.json
├── simple_cnn_history.json
├── simple_cnn_confusion_matrix.png
├── complex_cnn_metrics.json
├── efficientnet_transfer_learning_metrics.json
└── efficientnet_fine_tuning_metrics.json
```

Metrics của các model phải có cùng các trường:

```text
accuracy
macro_precision
macro_recall
macro_f1
per_class_metrics
parameter_count
inference_time
```

Các file model, report và dữ liệu sinh ra được Git ignore theo mặc định. Chỉ
commit file kết quả nhỏ hoặc file mẫu khi cả nhóm thống nhất.

## 8. Quy trình Git

Mỗi người làm việc trên branch riêng theo model. Trước khi bắt đầu:

```powershell
git fetch origin
git merge main
```

Sau mỗi phần việc hoàn chỉnh:

```powershell
git status
git add <files>
git commit -m "Implement <task>"
git push -u origin <branch-name>
```

Không commit dữ liệu raw, checkpoint lớn hoặc file sinh tự động không cần
thiết. Nếu `main` có thay đổi mới, cập nhật branch trước khi mở pull request.

## 9. Checklist trước khi merge

- [ ] Code chạy được từ thư mục `new_project/`.
- [ ] Dùng đúng split manifest chung.
- [ ] Không có patient leakage.
- [ ] Augmentation chỉ áp dụng cho train.
- [ ] Seed và class mapping thống nhất.
- [ ] Unit test pass.
- [ ] Có checkpoint hoặc kết quả tương ứng.
- [ ] Có metrics, learning curve và confusion matrix.
- [ ] Không commit dữ liệu hoặc file nhị phân lớn.
- [ ] README hoặc notebook có lệnh chạy lại.
