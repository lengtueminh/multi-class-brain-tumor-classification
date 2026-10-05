# Phân công công việc

## Mục tiêu

Chia phần việc còn lại cho 3 thành viên theo nguyên tắc mỗi người phụ trách một model. Các phần việc dùng chung được phân bổ thêm dựa trên độ phức tạp của từng model để khối lượng tương đối cân bằng.

## Phân công chính

| Thành viên | Model phụ trách | Phần việc bổ sung | Sản phẩm bắt buộc |
|---|---|---|---|
| **TM** | **Model 1: Simple CNN** | Data loader, patient-level split và kiểm tra khả năng tái lập | Code model, notebook huấn luyện, checkpoint, model summary, learning curve, metrics và confusion matrix |
| **CA** | **Model 2: Complex Sequential–Parallel CNN** | Module đánh giá và bảng so sánh kết quả | Code model, notebook huấn luyện, checkpoint, model summary, learning curve, metrics và confusion matrix |
| **TH** | **Model 3: EfficientNetB0** gồm transfer learning và fine-tuning | Grad-CAM, ablation augmentation, robustness với Gaussian noise và inference | Code model, hai checkpoint, kết quả riêng cho hai giai đoạn, learning curves, metrics và confusion matrix |

## Người 1 — Simple CNN và data pipeline

### Model

- Xây dựng Simple CNN từ đầu, không dùng pretrained weights.
- Sử dụng convolutional layer, pooling layer, global average pooling, dense layer và dropout.
- Huấn luyện, lưu checkpoint tốt nhất và đánh giá trên test set.

### Phần việc bổ sung

- Hoàn thiện data loader dùng chung cho train, validation và test.
- Kiểm tra patient-level split, bảo đảm không có patient trùng giữa các tập.
- Lưu split manifest và class mapping để có thể tái lập.
- Kiểm tra seed, kích thước ảnh `224x224x3` và quy tắc normalize.
- Bổ sung test cho data loader và split nếu cần.

### Đầu ra

- `models/simple_cnn_best.keras`
- Notebook hoặc script huấn luyện Simple CNN.
- Model summary.
- Learning curve train/validation.
- Metrics và confusion matrix.
- Tài liệu ngắn mô tả preprocessing và baseline.

## Người 2 — Complex Sequential–Parallel CNN và evaluation

### Model

- Xây dựng model bằng Functional API.
- Có nhánh tuần tự gồm các convolution nối tiếp.
- Có nhánh song song với các kernel `1x1`, `3x3` và `5x5`.
- Kiểm tra kích thước tensor trước khi `Concatenate` hoặc `Add`.
- Huấn luyện, lưu checkpoint tốt nhất và đánh giá trên test set.

### Phần việc bổ sung

- Viết module đánh giá dùng chung cho cả ba model.
- Tính các metric:
  - Accuracy.
  - Macro precision.
  - Macro recall.
  - Macro F1-score.
  - Precision, recall và F1-score theo từng class.
  - Số lượng parameter.
  - Inference time.
- Xuất metrics thành JSON hoặc CSV trong `reports/`.
- Tạo bảng so sánh kết quả của ba model.
- Tổng hợp các cặp class thường bị nhầm từ confusion matrix.

### Đầu ra

- `models/complex_cnn_best.keras`
- Notebook hoặc script huấn luyện Complex CNN.
- Module evaluation dùng chung.
- Model summary.
- Learning curve train/validation.
- Metrics và confusion matrix.
- Bảng so sánh ba model.

## Người 3 — EfficientNetB0, explainability và robustness

### Model

- Sử dụng EfficientNetB0 với ImageNet weights.
- Giai đoạn 1: đóng băng backbone và chỉ huấn luyện classifier head.
- Giai đoạn 2: mở một phần các layer cuối, ví dụ 30 layer, để fine-tuning với learning rate nhỏ.
- Báo cáo riêng kết quả của transfer learning và fine-tuning.

### Phần việc bổ sung

- Viết Grad-CAM dùng chung cho cả ba model.
- Tạo heatmap cho một số mẫu đại diện của từng class.
- Kiểm tra heatmap tập trung vào vùng u hay vùng nền/artefact.
- Thực hiện ablation augmentation trên Model 3:
  - Không augmentation.
  - Có augmentation.
- Kiểm tra robustness bằng Gaussian noise trên bản sao của test set ở một hoặc hai mức cường độ.
- Viết script inference nhận ảnh mới và trả về class cùng xác suất.

### Đầu ra

- `models/efficientnet_transfer_learning_best.keras`
- `models/efficientnet_fine_tuning_best.keras`
- Notebook hoặc script huấn luyện hai giai đoạn.
- Learning curve cho transfer learning và fine-tuning.
- Metrics, confusion matrix và model summary.
- Hình Grad-CAM cho cả ba model.
- Báo cáo ablation augmentation.
- Báo cáo robustness.
- Script inference.

## Quy ước chung

Ba model phải dùng cùng:

- Patient-level train/validation/test split.
- Class mapping:
  - `0 = glioma`
  - `1 = meningioma`
  - `2 = pituitary tumor`
- Input kích thước `224x224x3`.
- Quy tắc normalize.
- Random seed.
- Tiêu chí đánh giá.

Không được sử dụng test set để chọn hyperparameter, early stopping hoặc checkpoint tốt nhất.

Mỗi model cần lưu tối thiểu:

```text
models/<model_name>_best.keras
reports/<model_name>_metrics.json
reports/<model_name>_history.json
reports/<model_name>_confusion_matrix.png
reports/<model_name>_summary.txt
```

## Thứ tự thực hiện

1. Người 1 hoàn thiện data loader, split manifest và class mapping.
2. Người 2 hoàn thiện evaluation module và format kết quả chung.
3. Cả ba xây dựng, huấn luyện và lưu checkpoint model của mình.
4. Người 2 chạy evaluation thống nhất trên output của cả ba model.
5. Người 3 chạy Grad-CAM, ablation và robustness.
6. Người 1 kiểm tra khả năng tái lập toàn bộ pipeline.
7. Cả nhóm tổng hợp bảng so sánh, phân tích lỗi và viết kết luận.

## Checklist hoàn thành

- [ ] Có code tạo model.
- [ ] Có code hoặc notebook huấn luyện.
- [ ] Có checkpoint tốt nhất.
- [ ] Có kết quả validation và test.
- [ ] Có learning curve.
- [ ] Có confusion matrix.
- [ ] Có model summary và số parameter.
- [ ] Kết quả được lưu theo cùng format.
- [ ] Có test cho phần code mới.
- [ ] Có hướng dẫn chạy ngắn gọn.
