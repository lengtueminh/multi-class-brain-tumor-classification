# Multi-class Brain Tumor Classification

Repository gồm project gốc và project kế thừa được phát triển từ project đó.

## Cấu trúc

```text
original/
  Code/                 Notebook và pipeline của project gốc
  dataset/              Dữ liệu gốc hoặc nơi đặt dữ liệu gốc
  saved model/           Model đã lưu của project gốc
  README.md             Tài liệu gốc

new_project/
  PROJECT_SPEC.md       Đặc tả và kế hoạch thí nghiệm mới
  notebooks/             Notebook cho thí nghiệm mới
  src/                   Mã nguồn Python dùng lại/tái cấu trúc
  configs/               Cấu hình dữ liệu và huấn luyện
  data/raw/              Dữ liệu đầu vào, không chỉnh sửa
  data/processed/        Dữ liệu sau tiền xử lý
  models/                Checkpoint và model sinh ra
  reports/               Metrics, biểu đồ và kết quả
  tests/                 Kiểm thử
```

## Quy ước

- `original/` là bản lưu project gốc; không sửa trực tiếp khi phát triển tính năng mới.
- Code kế thừa đặt trong `new_project/` và có thể tham chiếu ý tưởng/notebook từ `original/`.
- Dữ liệu và model lớn không commit vào Git; chỉ giữ `.gitkeep` để đánh dấu thư mục.