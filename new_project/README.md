# Project kế thừa

Phiên bản mới dựa trên project gốc trong `../original/`. Không sửa trực tiếp thư mục gốc.

- Phạm vi: `PROJECT_SPEC.md`; hướng dẫn chi tiết: [HUONG_DAN_SU_DUNG.md](HUONG_DAN_SU_DUNG.md).
- Mã nguồn dùng chung: `src/`; kiến trúc model: `src/models/`; notebook: `notebooks/`.
- Dữ liệu: `data/raw/`; manifest chia tập theo bệnh nhân: `data/processed/split_manifest.csv`.

## Xử lý dữ liệu

Cả ba model dùng chung manifest, chuẩn hóa pixel về `[0, 1]` và chỉ tăng cường
dữ liệu tập train. Không tự chia lại dữ liệu trong từng notebook.
Ảnh và mask HDF5/v7.3 được chuyển vị cùng nhau để khôi phục hướng ảnh MATLAB;
tọa độ biên u, nhãn và mã bệnh nhân được giữ nguyên. Manifest hiện tại vẫn hợp lệ.

Complex CNN và EfficientNetB2 cần huấn luyện lại để so sánh theo hướng ảnh mới.
Checkpoint cũ cần cách xử lý dữ liệu cũ để tái lập kết quả; kết quả đã lưu trong
notebook cũng chỉ được cập nhật khi chạy lại.

## Huấn luyện bằng CLI

Cài thư viện bằng `pip install -r requirements.txt` từ thư mục repository,
sau đó chạy trong môi trường Python của project:

```sh
python new_project/src/train.py --model simple_cnn
python new_project/src/train.py --model complex_cnn
python new_project/src/train.py --model efficientnetb2
```

Chỉ bắt buộc `--model`. Đường dẫn mặc định được xác định theo vị trí mã nguồn.

| Model | Thư viện thực thi | Thiết bị mặc định | Kiểu số | Ảnh | Batch | Epoch |
|---|---|---|---|---|---|---|
| `simple_cnn` | TensorFlow | GPU nếu có, còn lại CPU | float32 | 224 | 32 | 50 |
| `complex_cnn` | PyTorch/Keras | CUDA | mixed_float16 | 224 | 8 | 90 |
| `efficientnetb2` | TensorFlow | GPU nếu có, còn lại CPU | float32 | 240 | 32 | 50 |

Giữ nguyên optimizer, lịch learning rate và callback của từng model.
EfficientNetB2 dùng trọng số ImageNet, toàn bộ backbone được huấn luyện;
chưa áp dụng đóng băng backbone hay learning rate riêng cho từng phần.
Thiết bị GPU được yêu cầu nhưng không có sẽ báo lỗi rõ ràng. Để chạy Complex CNN
trên CPU, thêm `--device cpu --precision float32`.

Tùy chọn: `--epochs`, `--batch-size`, `--seed`, `--device`, `--precision`,
`--raw-dir`, `--manifest`, `--output-dir`, `--no-augmentation`.
Riêng EfficientNetB2 có `--weights none` để kiểm tra không cần tải trọng số;
đây là khởi tạo ngẫu nhiên, không phải học chuyển giao.

Mỗi lần chạy tạo thư mục mới `runs/<model>_seed<seed>_<timestamp>/` gồm
`models/` và `reports/` (cấu hình, manifest, lịch sử, biểu đồ, kết quả đánh giá).
`--output-dir` phải trỏ tới thư mục chưa tồn tại. Kết quả chạy không được đưa vào Git.
CLI dùng mặc định ở bảng trên; sửa `configs/default.yaml` không thay đổi mặc định CLI.

## Google Colab

Chọn môi trường GPU, đưa mã nguồn cùng dữ liệu gốc và manifest lên Colab,
cài `requirements.txt`, rồi chạy các lệnh trên với tiền tố `!`.
Dữ liệu và manifest không nằm trong Git, cần cung cấp riêng.
Nên đọc dữ liệu từ ổ đĩa cục bộ Colab và lưu kết quả vào Google Drive qua
`--output-dir`; mỗi model dùng một thư mục mới. CLI chưa hỗ trợ tiếp tục lần chạy bị ngắt.
