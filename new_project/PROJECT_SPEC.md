# Phân loại đa lớp u não từ ảnh MRI bằng ba kiến trúc Deep Learning trong 2 tuần

## 1. Bối cảnh nghiên cứu

U não là bệnh lý nguy hiểm, trong đó việc xác định đúng loại u có ý nghĩa quan trọng đối với chẩn đoán và điều trị. MRI là phương pháp hình ảnh phổ biến để quan sát não, nhưng các lớp `glioma`, `meningioma` và `pituitary tumor` có thể có đặc điểm hình thái tương tự nhau. Dự án xây dựng các mô hình Computer Vision để tự động phân loại ảnh MRI thành ba lớp trên.

CNN có khả năng tự học các đặc trưng từ ảnh, từ cạnh và texture ở các lớp đầu đến hình dạng và vùng tổn thương ở các lớp sâu. Trong phạm vi hai tuần, dự án tập trung vào ba model cốt lõi: CNN cơ bản, CNN có kiến trúc tuần tự kết hợp song song, và EfficientNet sử dụng transfer learning/fine-tuning.

## 2. Bài toán

Cho tập dữ liệu ảnh MRI và nhãn tương ứng:

$$
\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{N}
$$

Trong đó $x_i$ là ảnh MRI và $y_i$ thuộc một trong ba lớp:

```text
0 -> glioma
1 -> meningioma
2 -> pituitary tumor
```

Mục tiêu là học hàm phân loại:

$$
f_\theta: \mathbb{R}^{H \times W \times 3} \rightarrow \{0,1,2\}
$$

sao cho nhãn dự đoán gần với nhãn thật và mô hình có khả năng tổng quát trên ảnh chưa từng xuất hiện trong training.

## 3. Dữ liệu và tiền xử lý

### 3.1. Dataset

Project sử dụng Figshare Brain Tumor Dataset:

| Thuộc tính | Thông tin |
|---|---|
| Số ảnh | 3.064 ảnh MRI |
| Số bệnh nhân | 233 |
| Số lớp | 3 |
| Lớp | Glioma, meningioma, pituitary tumor |
| Ảnh đầu vào | MRI T1-weighted contrast-enhanced |
| Dataset | [Figshare Brain Tumor Dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) |

Phân chia dữ liệu cần được thực hiện trước augmentation. Chỉ tập training được augmentation; validation và test giữ nguyên để đánh giá công bằng.

### 3.2. Pipeline tiền xử lý

Ảnh MRI gốc --> Crop brain contour--> Chia train / validation / test --> Augmentation chỉ trên train --> Resize 224x224x3 --> Label encoding và one-hot encoding --> Đưa vào ba model  

Các bước:

1. Crop vùng não bằng threshold, erosion, dilation và contour lớn nhất.
2. Chia dữ liệu theo class với random seed cố định.
3. Augmentation trên tập train bằng rotation, translation và flip.
4. Resize toàn bộ ảnh về `224 x 224 x 3`.
5. Mã hóa nhãn thành vector one-hot có kích thước 3.

Dữ liệu sau tiền xử lý có dạng:

```text
X: (batch_size, 224, 224, 3)
y: (batch_size, 3)
```

## 4. Ba model cần xây dựng

### 4.1. Model 1: CNN đơn giản

Model 1 là baseline được huấn luyện từ đầu, không dùng pretrained weights.

```mermaid
flowchart LR
    A["Input<br/>(224,224,3)"] --> B["Conv2D 32, 3x3<br/>(224,224,32)"]
    B --> C["ReLU + MaxPooling<br/>(112,112,32)"]
    C --> D["Conv2D 64, 3x3<br/>(112,112,64)"]
    D --> E["ReLU + MaxPooling<br/>(56,56,64)"]
    E --> F["GlobalAveragePooling2D<br/>(64)"]
    F --> G["Dense 128 + ReLU<br/>(128)"]
    G --> H["Dropout 0.5<br/>(128)"]
    H --> I["Dense 3 + Softmax<br/>(3)"]
```

| Layer | Input | Output | Ý nghĩa |
|---|---|---|---|
| Conv2D 32 | `(224,224,3)` | `(224,224,32)` | Học đặc trưng cấp thấp |
| MaxPooling | `(224,224,32)` | `(112,112,32)` | Giảm kích thước không gian |
| Conv2D 64 | `(112,112,32)` | `(112,112,64)` | Học đặc trưng phức tạp hơn |
| MaxPooling | `(112,112,64)` | `(56,56,64)` | Giảm chi phí tính toán |
| GlobalAveragePooling | `(56,56,64)` | `(64)` | Tóm tắt mỗi feature map |
| Dense | `(64)` | `(128)` | Kết hợp các đặc trưng |
| Dense Softmax | `(128)` | `(3)` | Xác suất của ba class |

### 4.2. Model 2: CNN phức tạp có tuần tự và song song

Model 2 sử dụng Functional API và gồm hai kiểu xử lý:

- Nhánh tuần tự: các convolution nối tiếp để học đặc trưng theo nhiều cấp.
- Nhánh song song: nhiều convolution với kernel khác nhau xử lý cùng input để học đặc trưng ở nhiều scale.

```mermaid
flowchart TD
    A["Input<br/>(224,224,3)"]
    A --> S1["Sequential branch<br/>Conv3x3 -> BN -> ReLU"]
    S1 --> S2["Conv3x3 -> Pooling"]
    S2 --> S3["Conv3x3 -> Pooling"]
    A --> P["Parallel block"]
    P --> P1["Branch 1<br/>Conv1x1"]
    P --> P2["Branch 2<br/>Conv3x3"]
    P --> P3["Branch 3<br/>Conv5x5"]
    P1 --> C["Concatenate 3 branches"]
    P2 --> C
    P3 --> C
    S3 --> M["Merge sequential + parallel"]
    C --> M
    M --> G["GlobalAveragePooling2D"]
    G --> D["Dense + Dropout"]
    D --> O["Dense 3 + Softmax"]
```

Model 2 chỉ dùng ba nhánh `1x1`, `3x3` và `5x5` để giảm thời gian thiết kế và debug.

Ý nghĩa các nhánh:

- `1x1`: kết hợp channel với chi phí thấp.
- `3x3`: học chi tiết cục bộ.
- `5x5`: quan sát vùng không gian lớn hơn.
- Trước khi `Concatenate` hoặc `Add`, các nhánh phải có cùng kích thước không gian. Model 2 kiểm tra liệu việc kết hợp nhiều receptive field có tốt hơn CNN tuần tự đơn giản hay không; đổi lại số parameter và nguy cơ overfitting tăng.

### 4.3. Model 3: Transfer learning và fine-tuning

Model 3 kế thừa backbone đã học từ ImageNet. Trong phạm vi hai tuần, sử dụng EfficientNetB0 để giảm chi phí tính toán, sau đó thực hiện transfer learning và fine-tuning trên cùng bộ dữ liệu.

```mermaid
flowchart LR
    A["Input<br/>(224,224,3)"] --> B["EfficientNetB0 backbone<br/>ImageNet weights"]
    B --> C["GlobalAveragePooling2D"]
    C --> D["Dropout 0.5"]
    D --> E["Dense 3 + Softmax"]
    E --> F["Glioma / Meningioma / Pituitary"]
```

#### Giai đoạn transfer learning

Đóng băng backbone và chỉ huấn luyện classifier head:

```python
backbone.trainable = False
```

#### Giai đoạn fine-tuning

Mở một phần các layer cuối của backbone và huấn luyện với learning rate nhỏ:

```python
backbone.trainable = True

for layer in backbone.layers[:-30]:
    layer.trainable = False
```

Fine-tuning giúp các feature cuối thích nghi với ảnh MRI. Cần báo cáo riêng kết quả của transfer learning và fine-tuning.

## 5. Hàm mất mát và tối ưu

Với bài toán ba lớp, sử dụng categorical cross-entropy:

$$
L=-\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{3} y_{ic}\log(\hat{y}_{ic})
$$

Trong đó $y_{ic}$ là nhãn thật dạng one-hot và $\hat{y}_{ic}$ là xác suất model dự đoán cho class $c$.

Optimizer đề xuất là Adam. Learning rate của fine-tuning phải nhỏ hơn giai đoạn train classifier:

```text
Classifier training: 1e-3
Fine-tuning:         1e-5 hoặc 1e-6
```

Có thể dùng thêm `EarlyStopping`, `ReduceLROnPlateau` và `ModelCheckpoint`.

## 6. Thiết kế thí nghiệm và đánh giá

Ba model phải dùng cùng train/validation/test split, thứ tự class, image size, random seed và tiêu chí đánh giá.

| Metric | Ý nghĩa |
|---|---|
| Accuracy | Tỉ lệ dự đoán đúng trên toàn bộ ảnh |
| Precision | Trong các ảnh được dự đoán là một class, bao nhiêu ảnh đúng |
| Recall/Sensitivity | Trong các ảnh thật thuộc class, bao nhiêu ảnh được tìm đúng |
| F1-score | Trung bình điều hòa giữa precision và recall |
| Confusion matrix | Các cặp class thường bị nhầm |

Nên báo cáo `macro average` bên cạnh accuracy để các class có trọng số ngang nhau.

| Model | Parameters | Accuracy | Macro Precision | Macro Recall | Macro F1 | Inference time |
|---|---:|---:|---:|---:|---:|---:|
| Simple CNN | Đo từ model | Đo thực nghiệm | Đo thực nghiệm | Đo thực nghiệm | Đo thực nghiệm | Đo thực nghiệm |
| Complex sequential-parallel CNN | Đo từ model | Đo thực nghiệm | Đo thực nghiệm | Đo thực nghiệm | Đo thực nghiệm | Đo thực nghiệm |
| EfficientNet transfer learning | Đo từ model | Đo thực nghiệm | Đo thực nghiệm | Đo thực nghiệm | Đo thực nghiệm | Đo thực nghiệm |
| EfficientNet fine-tuning | Đo từ model | Đo thực nghiệm | Đo thực nghiệm | Đo thực nghiệm | Đo thực nghiệm | Đo thực nghiệm |

Các giá trị trong bài báo gốc chỉ là reference, không phải kết quả đảm bảo của project mới.

## 7. Explainability với Grad-CAM

Grad-CAM dùng gradient của class dự đoán đối với feature map ở convolution layer cuối:

$$
\alpha_k^c=\frac{1}{Z}\sum_i\sum_j\frac{\partial y^c}{\partial A_{ij}^k}
$$

$$
L_{Grad-CAM}^c=ReLU\left(\sum_k \alpha_k^c A^k\right)
$$

Trong đó $A^k$ là feature map thứ $k$, $y^c$ là score của class $c$, và $\alpha_k^c$ là mức quan trọng của feature map $k$.

Grad-CAM cần được áp dụng cho cả ba model. Heatmap nên được kiểm tra xem có tập trung vào vùng u hay vào vùng nền/artefact. Đây là công cụ giải thích, không thay thế chẩn đoán y khoa.

## 8. Phạm vi thí nghiệm trong 2 tuần

Để bảo đảm tính khả thi, thí nghiệm được giới hạn vào các kết quả cốt lõi:

1. Huấn luyện và đánh giá ba model trên cùng một train/validation/test split.
2. Áp dụng Grad-CAM cho một số mẫu đại diện của cả ba model.
3. Thực hiện ablation augmentation chỉ trên Model 3 với hai cấu hình: có và không có augmentation.
4. Kiểm tra robustness chỉ bằng Gaussian noise trên test set.
5. Không thực hiện cross-dataset validation trong phiên bản hai tuần.

## 9. Phân tích lỗi và robustness

Cần phân tích confusion matrix, các cặp class thường bị nhầm và khoảng cách giữa training accuracy và test accuracy. Có thể xem thêm một số ảnh dự đoán sai nếu còn thời gian.

Robustness chỉ được kiểm tra bằng Gaussian noise ở một hoặc hai mức cường độ. Noise phải được áp dụng trên bản sao của test set, không thay đổi test set gốc.

## 10. Nguyên tắc triển khai

1. Xây dựng Model 1 và Model 2 từ đầu; Model 3 dùng EfficientNetB0.
2. Tách transfer learning và fine-tuning thành hai bước ngắn trong cùng quy trình.
3. Dùng một pipeline dữ liệu duy nhất cho cả ba model.
4. Không dùng test set làm validation trong quá trình chọn model.
5. Dùng mapping class cố định: `0=glioma`, `1=meningioma`, `2=pituitary tumor`.
6. Dùng input thống nhất `224 x 224 x 3` cho cả ba model.
7. Lưu metric, learning curve, confusion matrix và model summary cho từng thí nghiệm.

## 11. Sản phẩm đầu ra

- Ba kiến trúc model và code huấn luyện tương ứng.
- Checkpoint của model tốt nhất.
- Bảng so sánh accuracy, precision, recall, F1-score và số parameter.
- Learning curve của train/validation.
- Confusion matrix cho từng model.
- Phân tích các ảnh dự đoán sai.
- Grad-CAM cho từng class.
- Báo cáo ablation augmentation trên Model 3.
- Báo cáo robustness với Gaussian noise.
- Script inference nhận ảnh mới và trả về class cùng xác suất.

## 12. Tài liệu tham khảo

1. Cheng et al. (2017), Figshare Brain Tumor Dataset.
2. Tan and Le (2019), *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*.
3. Selvaraju et al. (2017), *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*.
4. Zulfiqar, Bajwa and Mehmood (2023), *Multi-class classification of brain tumor types from MR images using EfficientNets*, Biomedical Signal Processing and Control, 84, 104777.
5. [Repository tham khảo](https://github.com/FatimaZulfiqar/multi-class-brain-tumor-classification).

## 13. Kết luận

Dự án trong hai tuần đánh giá ba hướng xây dựng model cho phân loại ảnh MRI: CNN đơn giản làm baseline, CNN phức tạp kết hợp xử lý tuần tự và song song với ba nhánh song song, và EfficientNetB0 sử dụng transfer learning/fine-tuning.

Thiết kế này cho phép trả lời ba câu hỏi: CNN cơ bản hoạt động đến đâu, việc thêm nhánh song song có cải thiện biểu diễn ảnh hay không, và pretrained backbone có đem lại lợi ích so với model huấn luyện từ đầu hay không. Kết luận cuối cùng phải dựa trên cùng một protocol đánh giá và kết quả thực nghiệm của ba model.
