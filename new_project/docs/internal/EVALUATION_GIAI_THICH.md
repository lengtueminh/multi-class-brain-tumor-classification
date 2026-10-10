# Tài liệu nội bộ: hiểu sâu `src/evaluation.py`

Module này chuyển dự đoán của model thành metric, confusion matrix và các file báo cáo. Nó không train model, không chọn checkpoint, không chia dữ liệu và không tự so sánh cả ba model. Notebook phải chọn checkpoint bằng validation trước khi gọi đánh giá test. Tài liệu giải thích code hiện tại, không thay đổi hành vi của module.

## 1. Luồng dữ liệu và hợp đồng đầu vào

```text
dataset → collect_predictions(model, dataset)
           ├─ y_true: (N,), chỉ số lớp
           ├─ y_prob: (N,3), xác suất dự đoán
           └─ inference_time: ms/ảnh
                 ↓
           compute_metrics(y_true, y_prob)
                 ↓
           evaluate_model thêm metadata
                 ↓
           save_json + plot_confusion_matrix
```

Dataset phải yield `(images, labels)`. Labels được phép là indices `(B,)` hoặc one-hot `(B,3)`. Model phải trả xác suất `(B,3)` theo đúng thứ tự lớp. Nếu output là logits chưa softmax, shape đúng chưa đủ: kiểm tra xác suất sẽ từ chối hoặc giá trị sẽ không mang đúng ý nghĩa mong muốn.

| Index | Tên lớp trong project |
|---|---|
| 0 | glioma |
| 1 | meningioma |
| 2 | pituitary tumor |

`CLASS_NAMES` được lấy từ preprocessing, không suy ra từ tên folder. Dùng đúng thứ tự output của model là điều kiện bắt buộc khi so sánh. Module cố định ba lớp qua `range(3)`, shape `(N,3)` và `labels=[0,1,2]`; không phải evaluation tổng quát cho số lớp tùy ý.

## 2. `labels_to_indices`: thống nhất cách biểu diễn nhãn

`np.asarray` chuyển list, array hoặc đối tượng tương thích sang NumPy. Nhãn one-hot hợp lệ có đúng ba cột, mỗi phần tử thuộc {0,1}, tổng mỗi hàng bằng 1. Ví dụ `[0,1,0]` → index 1. `argmax(axis=1)` lấy vị trí của số 1 trên từng hàng.

Nhãn dạng index phải có một chiều và chỉ gồm 0,1,2; hàm ép về int64. Các float 0.0/1.0/2.0 cũng vượt qua kiểm tra giá trị rồi được ép kiểu. Vector `(N,1)`, nhãn -1/3, soft labels như `[0.2,0.8,0]` và multi-hot như `[1,1,0]` bị từ chối. Vì vậy module không dùng trực tiếp để đánh giá label smoothing hoặc multi-label classification.

Hàm không kiểm tra empty array ở đây; kiểm tra dataset rỗng được thực hiện trong `collect_predictions` hoặc `compute_metrics`. Các `ValueError` thuộc module evaluation vẫn cần thiết để tránh báo cáo sai dữ liệu; việc bỏ Python `assert` trong notebook không xóa các kiểm tra này.

## 3. `compute_metrics`: từ xác suất sang phân loại

### 3.1. Kiểm tra shape và xác suất

Hàm chuyển y_true về indices, y_prob về array; yêu cầu N>0 và shape `(N,3)`. Xác suất phải hữu hạn (không NaN/Inf), nằm trong [0,1] và tổng mỗi hàng gần 1. `np.allclose(..., atol=1e-5)` có cả tolerance tương đối mặc định của NumPy, không chỉ tolerance tuyệt đối; đây là kiểm tra gần bằng, không đòi hỏi tổng bitwise bằng 1.

Sau đó `y_pred = y_prob.argmax(axis=1)` lấy lớp có xác suất cao nhất. Nếu hai lớp có xác suất bằng nhau, NumPy chọn vị trí đầu tiên đạt cực đại. Metric được tính trên nhãn argmax, không đánh giá mức độ calibration hay sự tự tin của toàn bộ phân phối. Model đoán đúng với p=0.51 và p=0.99 có thể có cùng accuracy nhưng crossentropy khác nhau.

### 3.2. TP, FP, FN và support

Xét riêng lớp k theo cách one-vs-rest:

| Đại lượng | Ý nghĩa |
|---|---|
| TP_k | Nhãn thật k và dự đoán k |
| FP_k | Nhãn thật khác k nhưng dự đoán k |
| FN_k | Nhãn thật k nhưng dự đoán khác k |
| support_k | Số ảnh có nhãn thật k = TP_k + FN_k |

`precision_k = TP_k/(TP_k+FP_k)`: trong những ảnh được model gọi là k, bao nhiêu ảnh đúng? `recall_k = TP_k/(TP_k+FN_k)`: trong những ảnh thật sự là k, tìm được bao nhiêu? `F1_k = 2*precision_k*recall_k/(precision_k+recall_k)`, tương đương `2TP_k/(2TP_k+FP_k+FN_k)` khi mẫu số khác 0.

`precision_recall_fscore_support` trả bốn arrays, mỗi array có một phần tử cho mỗi lớp theo `labels=[0,1,2]`. Không truyền `average`, nên hàm trả metric từng lớp; module tự lấy mean sau đó. `zero_division=0` gán 0 khi phép tính tương ứng không xác định, chẳng hạn không dự đoán lớp nào đó. Việc khai báo đủ ba labels giữ vị trí và số lớp nhất quán ngay cả khi một lớp vắng mặt. [Tài liệu scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)

### 3.3. Accuracy và macro averages

`accuracy = số ảnh đúng / N`. Với bài toán ba lớp single-label, accuracy có thể cao dù một lớp ít ảnh được nhận diện kém. Macro precision, recall và F1 lần lượt là trung bình số học của ba metric tương ứng theo lớp, mỗi lớp có trọng số 1/3.

**Macro F1 = mean(F1_glioma, F1_meningioma, F1_pituitary)**. Không lấy macro precision/recall rồi tính harmonic mean. Weighted average sẽ đặt trọng số theo support; module không dùng weighted average. Micro average cộng TP/FP/FN trước; module cũng không xuất nó.

Khi một lớp không có nhãn thật trong tập đánh giá, `zero_division=0` cùng danh sách cố định vẫn đưa giá trị 0 tương ứng vào các macro metrics khi phép chia không xác định. Do đó cần đọc support, không chỉ nhìn macro score. Trong bài toán single-label đủ các lớp, macro recall tương ứng cách lấy trung bình recall theo lớp thường được gọi là balanced accuracy; module chỉ xuất tên `macro_recall`.

### 3.4. Ví dụ tính tay

Confusion matrix, **hàng = lớp thật, cột = lớp dự đoán**:

| Thật / Dự đoán | Glioma | Meningioma | Pituitary | Support |
|---|---:|---:|---:|---:|
| Glioma | 8 | 1 | 1 | 10 |
| Meningioma | 2 | 3 | 0 | 5 |
| Pituitary | 1 | 0 | 4 | 5 |

Tổng 20 ảnh, đúng 8+3+4=15 → accuracy 0.75. Với glioma: TP=8, FP=2+1=3, FN=1+1=2 → precision=8/11, recall=8/10, F1=16/21.

| Lớp | Precision | Recall | F1 |
|---|---:|---:|---:|
| Glioma | 0.727273 | 0.8 | 0.761905 |
| Meningioma | 0.75 | 0.6 | 0.666667 |
| Pituitary | 0.8 | 0.8 | 0.8 |
| Macro | **0.759091** | **0.733333** | **0.742857** |

Ma trận không đối xứng: ô thật glioma/dự đoán meningioma có ý nghĩa khác ô thật meningioma/dự đoán glioma. Tổng hàng giúp tính recall; tổng cột giúp tính precision; đường chéo là các dự đoán đúng. Đây là counts chưa normalize, không phải phần trăm.

### 3.5. Kiểu dữ liệu trả về

Các scalar được ép thành `float`/`int` Python, confusion matrix thành list lồng nhau bằng `.tolist()`. Nhờ vậy JSON encoder không gặp kiểu NumPy khó serialize. `per_class_metrics` là dictionary theo tên lớp; mỗi lớp có precision, recall, f1, support. `compute_metrics` không tính loss, parameter count hay thời gian, vì nó chỉ nhận labels và probabilities, không nhận model.

## 4. `collect_predictions`: ghép nhãn đúng với dự đoán và đo thời gian

Hàm duyệt dataset **một lần**, thu labels và predictions từ cùng batch. Không đọc labels ở một lượt rồi predictions ở lượt khác: với dataset shuffle/augmentation, hai lượt có thể khác thứ tự và làm metric sai dù cả hai arrays cùng chiều dài.

Hai list ban đầu chứa arrays của từng batch. Cuối hàm `np.concatenate` ghép theo chiều đầu thành `(N,)` và `(N,3)`. Batch cuối không cần đủ B ảnh; chỉ cần prediction shape khớp số labels của chính batch. Bộ nhớ giữ tất cả labels/probabilities đến cuối, cỡ O(N×3), không giữ toàn bộ ảnh.

Batch đầu được predict một lần trước khi bắt đầu timer để warm-up. Dự đoán warm-up bị bỏ; cùng batch đầu được predict lần nữa và kết quả lần có đo thời gian mới được lưu. Model ở inference mode; BN dùng moving statistics, Dropout không hoạt động. Hàm không dùng test labels để cập nhật weights.

### 4.1. Thời gian đo chính xác đoạn nào?

```python
start = perf_counter()
probabilities = np.asarray(model.predict_on_batch(images))
elapsed += perf_counter() - start
```

Timer bao gồm `predict_on_batch` và chuyển output sang NumPy. Batch `images,labels` đã được iterator yield trước timer, nên thời gian trực tiếp chờ đọc MATLAB/crop/resize bị loại khỏi phép đo. Tuy nhiên input transfer, backend overhead và output transfer/conversion có thể nằm trong timer; prefetch hoặc công việc khác vẫn có thể tranh tài nguyên.

`1000 * elapsed / image_count` là tổng giây predict chia tổng số ảnh, đổi thành ms/ảnh. Đây là throughput quy về từng ảnh với batching đang dùng; không phải latency request một ảnh riêng lẻ, cũng không phải thời gian toàn pipeline từ file đến quyết định.

Không có một lời gọi synchronize GPU tường minh. Chuyển prediction sang NumPy thường buộc kết quả hoàn thành/di chuyển về host, nhưng việc đo vẫn phụ thuộc backend. Không nên gọi số đo này là benchmark kernel GPU thuần hoặc so sánh các model dùng batch/device/precision khác nhau như thể điều kiện giống nhau.

### 4.2. Khi nào so sánh inference time có ý nghĩa?

Giữ cùng dataset, batch size, image size, phần cứng, precision, backend, mức warm-up và điều kiện tải máy. Một lượt có thể nhiễu; nếu muốn kết luận tốc độ nên chạy nhiều lượt độc lập và báo cáo phân phối, chẳng hạn median. Module hiện tại xuất một scalar của một lượt, không kèm độ lệch chuẩn hay confidence interval.

Validation trong notebook gọi `evaluate()` để lấy loss rồi gọi `collect_predictions()` để lấy macro metrics: đó là hai lượt inference trên cùng validation dataset không shuffle/augmentation. Test dùng `evaluate_model()` nên không lấy test loss qua `evaluate()`; không đọc nhầm JSON test như thể có trường loss.

## 5. `save_json`: lưu kết quả có thể dùng lại

`json.dumps(..., indent=2, allow_nan=False)` tạo JSON thụt lề và từ chối NaN/Infinity. `Path(path).write_text(..., encoding="utf-8")` ghi file UTF-8, có thể ghi đè file cùng tên. Helper không tạo parent directory; caller phải bảo đảm thư mục tồn tại. Không có cơ chế ghi file atomic: nếu quá trình ghi bị gián đoạn thì file có thể chưa hoàn chỉnh.

Các giá trị NumPy tùy ý không tự động được chuyển bởi helper. `compute_metrics` và `evaluate_model` đã ép kiểu trước khi gọi; code khác dùng `save_json` cũng phải tạo cấu trúc serialize được.

## 6. `plot_confusion_matrix`: vẽ đúng cách đọc

`Figure` tạo đối tượng hình trực tiếp thay vì thông qua trạng thái toàn cục của `pyplot`. `fig.subplots()` lấy một axes. `ConfusionMatrixDisplay` nhận counts và tên ba lớp theo đúng thứ tự. `cmap="Blues"` dùng màu đậm hơn cho count lớn; `values_format="d"` in số nguyên; xoay nhãn x 20 độ để dễ đọc.

Title chứa tên model. `tight_layout()` tự điều chỉnh khoảng cách để giảm cắt nhãn, `savefig(..., dpi=150)` lưu hình. Hàm không show hình và không return Figure; notebook dùng `Image(filename=...)` để hiển thị file đã lưu. Cũng như `save_json`, caller chịu trách nhiệm thư mục cha tồn tại.

Nếu tự đưa ma trận đã normalize thành float cho helper này, format `d` không phù hợp; helper hiện được thiết kế cho counts integer. Không nên dùng màu một mình để kết luận mức độ nhầm lẫn giữa các lớp có support khác nhau.

## 7. `plot_learning_curves`: history nào được mong đợi?

Hàm nhận dictionary có `loss`, `accuracy`, `val_loss`, `val_accuracy`. Hai subplot lần lượt là loss và accuracy. `zip(axes,["loss","accuracy"])` ghép mỗi axes với một metric. `range(1,len(history[key])+1)` tạo trục epoch 1-based; train/validation được vẽ với cùng trục.

History phải có các keys và độ dài tương ứng hợp lệ; hàm không có bộ kiểm tra riêng để chữa missing keys hoặc độ dài lệch. Legend và grid giúp đọc hai đường, title chứa model_name. Hàm không tự tìm epoch tốt nhất, không smoothing, không vẽ macro F1 hay learning rate.

Learning curves hữu ích để xem tối ưu hóa và khả năng tổng quát hóa, nhưng train loss trong training mode có augmentation/Dropout, validation loss trong inference mode không có chúng. Khoảng cách hai đường không chỉ phản ánh overfitting. Người đọc không nên tự lấy test score từng epoch để chọn checkpoint.

## 8. `evaluate_model`: hàm điều phối báo cáo test

Hàm gọi thu predictions/timing rồi tính metrics, thêm model_name, class_names, số ảnh, count_params, inference_time, đơn vị và mô tả phương pháp đo. `count_params` bao gồm trainable weights và trạng thái non-trainable như moving BN statistics, không chỉ số weights cập nhật bằng gradient.

Sau đó nó tạo reports_dir nếu cần, ghi `<model_name>_metrics.json`, vẽ `<model_name>_confusion_matrix.png`, trả dictionary metrics cho notebook. Gọi lại cùng tên và cùng folder sẽ ghi đè report. Model name được caller truyền vào; module không tự xác minh rằng nó khớp tên trong model hoặc rằng đây thật sự là checkpoint tốt nhất.

Không lưu raw probabilities, không có ROC/AUC, sensitivity/specificity tên riêng, calibration, metric theo patient, các khoảng tin cậy hoặc danh sách cặp lớp nhầm nhiều nhất. Một số đại lượng có thể tính tiếp từ confusion matrix, nhưng chúng chưa được module xuất. Nó cũng chưa tạo bảng tổng hợp ba model: caller cần thu các JSON vào bảng riêng.

Ví dụ gọi từ `new_project/`, sau khi đã tạo test_dataset theo pipeline chung:

```python
from src.evaluation import evaluate_model

metrics = evaluate_model(best_model, test_dataset, "complex_cnn", reports_dir="reports")
print(metrics["macro_f1"])
```

## 9. Giới hạn khi diễn giải kết quả

Ảnh của cùng patient có thể tương quan; metric từng slice không phải metric từng patient. Patient có nhiều slices đóng góp nhiều vào accuracy. Patient split là điều kiện về cách chia dữ liệu; patient-level evaluation là bước tổng hợp dự đoán khác mà module chưa thực hiện.

Support giữa các lớp không bằng nhau nên cần xem cùng accuracy, macro scores và từng lớp. Recall thấp nghĩa nhiều ảnh của lớp đó bị bỏ sót; precision thấp nghĩa nhiều ảnh thuộc lớp khác bị gọi nhầm thành lớp đó. Không biến kết quả thăm dò trên dataset thành khẳng định chất lượng chẩn đoán lâm sàng.

## 10. Giải thích từng dòng module

Số dòng dưới đây là số dòng thật trong `src/evaluation.py` ở phiên bản được đọc. Dòng trống được bỏ vì chỉ phân cách khối. Phụ lục bao gồm imports, kiểm tra, công thức gọi thư viện, vẽ biểu đồ và ghi file.

<!-- LINE_BY_LINE_APPENDIX -->

| Dòng | Code | Giải thích |
|---:|---|---|
| 1 | `"""Classification metrics and plots for three classes."""` | Docstring mô tả nhiệm vụ hoặc cấu trúc của hàm/module; không thực hiện thuật toán. |
| 3 | `import json` | Dùng JSON encoder để xuất dictionary metrics thành văn bản. |
| 4 | `from pathlib import Path` | Dùng Path để ghép, kiểm tra, tạo và ghi đường dẫn thay vì ghép chuỗi thủ công. |
| 5 | `from time import perf_counter` | Dùng đồng hồ đơn điệu độ phân giải cao để đo khoảng thời gian, không phải giờ lịch. |
| 7 | `from matplotlib.figure import Figure` | Tạo Figure trực tiếp, tránh phụ thuộc vào trạng thái toàn cục pyplot. |
| 8 | `import numpy as np` | Dùng NumPy cho arrays, ceil, argmin và xử lý dự đoán. |
| 9 | `from sklearn.metrics import (` | Bắt đầu import các hàm thống kê phân loại và tiện ích vẽ của scikit-learn. |
| 10 | `ConfusionMatrixDisplay,` | Import đối tượng vẽ confusion matrix với tên lớp trên trục. |
| 11 | `accuracy_score,` | Import hàm tính tỉ lệ nhãn dự đoán bằng nhãn thật. |
| 12 | `confusion_matrix,` | Import hàm đếm cặp lớp thật và lớp dự đoán. |
| 13 | `precision_recall_fscore_support,` | Import hàm tính precision/recall/F1/support theo từng lớp. |
| 14 | `)` | Đóng cấu trúc/lời gọi đã mở ở các dòng trên; không tạo một bước xử lý mới. |
| 16 | `from src.preprocessing import CLASS_INDEX_TO_NAME` | Dùng mapping lớp chung, tránh evaluation có thứ tự nhãn khác pipeline. |
| 18 | `CLASS_NAMES = [CLASS_INDEX_TO_NAME[index] for index in range(3)]` | Tạo danh sách tên lớp theo index 0,1,2; toàn module cố định ba lớp. |
| 21 | `def labels_to_indices(labels):` | Định nghĩa hàm `labels_to_indices` và các tham số; thân hàm chỉ chạy khi được gọi. Xem phần cơ chế của hàm tương ứng ở trên. |
| 22 | `"""Convert one-hot labels to class indices; accept indices directly."""` | Docstring mô tả nhiệm vụ hoặc cấu trúc của hàm/module; không thực hiện thuật toán. |
| 23 | `labels = np.asarray(labels)` | Chuyển labels về NumPy để kiểm tra shape, giá trị và argmax. |
| 24 | `if labels.ndim == 2 and labels.shape[1] == 3:` | Chọn nhánh xử lý one-hot khi labels là ma trận có đúng ba cột. |
| 25 | `if not np.isin(labels, [0, 1]).all() or not np.all(labels.sum(axis=1) == 1):` | Từ chối phần tử khác 0/1 hoặc hàng không có đúng một giá trị 1. |
| 26 | `raise ValueError("Labels must be valid one-hot vectors")` | Dừng hàm và báo ValueError với thông điệp này khi đầu vào vi phạm điều kiện ngay phía trên; không tiếp tục tính metric sai. |
| 27 | `return labels.argmax(axis=1)` | Lấy index của lớp trên từng hàng one-hot; output shape (N,). |
| 28 | `if labels.ndim != 1 or not np.isin(labels, [0, 1, 2]).all():` | Ở nhánh index, yêu cầu vector một chiều và mọi giá trị thuộc 0,1,2. |
| 29 | `raise ValueError("Labels must be class indices 0, 1, 2 or one-hot vectors")` | Dừng hàm và báo ValueError với thông điệp này khi đầu vào vi phạm điều kiện ngay phía trên; không tiếp tục tính metric sai. |
| 30 | `return labels.astype(np.int64)` | Trả indices kiểu int64 để thống nhất kiểu nhãn giữa các nguồn dữ liệu. |
| 33 | `def compute_metrics(y_true, y_prob):` | Định nghĩa hàm `compute_metrics` và các tham số; thân hàm chỉ chạy khi được gọi. Xem phần cơ chế của hàm tương ứng ở trên. |
| 34 | `"""Compute accuracy, per-class metrics and macro averages."""` | Docstring mô tả nhiệm vụ hoặc cấu trúc của hàm/module; không thực hiện thuật toán. |
| 35 | `y_true = labels_to_indices(y_true)` | Chuẩn hóa nhãn thật thành vector index trước khi tính metrics. |
| 36 | `y_prob = np.asarray(y_prob)` | Chuẩn hóa probabilities về NumPy; chưa biến chúng thành nhãn dự đoán. |
| 37 | `if len(y_true) == 0:` | Kiểm tra có ít nhất một nhãn; metric trên tập rỗng không có ý nghĩa. |
| 38 | `raise ValueError("Cannot evaluate an empty dataset")` | Dừng hàm và báo ValueError với thông điệp này khi đầu vào vi phạm điều kiện ngay phía trên; không tiếp tục tính metric sai. |
| 39 | `if y_prob.shape != (len(y_true), 3):` | Yêu cầu đúng một hàng prediction cho mỗi nhãn và đúng ba xác suất mỗi hàng. |
| 40 | `raise ValueError("Predictions must have shape (number of labels, 3)")` | Dừng hàm và báo ValueError với thông điệp này khi đầu vào vi phạm điều kiện ngay phía trên; không tiếp tục tính metric sai. |
| 41 | `if (not np.isfinite(y_prob).all() or np.any(y_prob < 0)` | Bắt đầu kiểm tra xác suất không NaN/Inf và không âm. |
| 42 | `or np.any(y_prob > 1) or not np.allclose(y_prob.sum(axis=1), 1, atol=1e-5)):` | Kiểm tra không lớn hơn 1 và tổng từng hàng gần 1; allclose còn có rtol mặc định. |
| 43 | `raise ValueError("Predictions must be finite class probabilities summing to 1")` | Dừng hàm và báo ValueError với thông điệp này khi đầu vào vi phạm điều kiện ngay phía trên; không tiếp tục tính metric sai. |
| 45 | `y_pred = y_prob.argmax(axis=1)` | Chuyển phân phối xác suất thành nhãn có xác suất cao nhất; tie lấy index đầu. |
| 46 | `# Include all three classes, even if one is never predicted.` | Comment giải thích ý đồ đoạn code; không chạy và không tạo tensors/weights. |
| 47 | `precision, recall, f1, support = precision_recall_fscore_support(` | Nhận bốn arrays metrics, mỗi vị trí tương ứng một lớp được khai báo. |
| 48 | `y_true, y_pred, labels=[0, 1, 2], zero_division=0,` | Dùng nhãn thật/dự đoán, cố định đủ ba lớp và trả 0 cho phép chia không xác định. |
| 49 | `)` | Đóng cấu trúc/lời gọi đã mở ở các dòng trên; không tạo một bước xử lý mới. |
| 50 | `per_class = {}` | Tạo dictionary để ánh xạ tên lớp sang metrics tương ứng. |
| 51 | `for index, name in enumerate(CLASS_NAMES):` | Duyệt cả index và tên lớp theo thứ tự mapping chung. |
| 52 | `per_class[name] = {` | Bắt đầu dictionary precision/recall/F1/support của một lớp. |
| 53 | `"precision": float(precision[index]),` | Lấy precision của đúng lớp và chuyển scalar NumPy thành float Python. |
| 54 | `"recall": float(recall[index]),` | Lấy recall của đúng lớp, ép float Python. |
| 55 | `"f1": float(f1[index]),` | Lấy F1 đã tính riêng cho lớp, không tính từ macro scores. |
| 56 | `"support": int(support[index]),` | Lấy số nhãn thật thuộc lớp và ép int Python. |
| 57 | `}` | Đóng cấu trúc/lời gọi đã mở ở các dòng trên; không tạo một bước xử lý mới. |
| 58 | `return {` | Bắt đầu dictionary kết quả trả cho caller. |
| 59 | `"accuracy": float(accuracy_score(y_true, y_pred)),` | Tính tỉ lệ ảnh đúng trên toàn bộ N ảnh. |
| 60 | `"macro_precision": float(precision.mean()),` | Trung bình precision của ba lớp với trọng số bằng nhau. |
| 61 | `"macro_recall": float(recall.mean()),` | Trung bình recall của ba lớp, không dùng trọng số số lượng ảnh. |
| 62 | `"macro_f1": float(f1.mean()),` | Trung bình ba F1 từng lớp; không lấy harmonic mean của macro precision/recall. |
| 63 | `"per_class_metrics": per_class,` | Đưa dictionary từng lớp vào kết quả cuối. |
| 64 | `"confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),` | Đếm hàng thật/cột dự đoán theo thứ tự 0,1,2 và chuyển matrix sang list để serialize. |
| 65 | `}` | Đóng cấu trúc/lời gọi đã mở ở các dòng trên; không tạo một bước xử lý mới. |
| 68 | `def collect_predictions(model, dataset):` | Định nghĩa hàm `collect_predictions` và các tham số; thân hàm chỉ chạy khi được gọi. Xem phần cơ chế của hàm tương ứng ở trên. |
| 69 | `"""Return paired labels, probabilities and inference time in ms/image."""` | Docstring mô tả nhiệm vụ hoặc cấu trúc của hàm/module; không thực hiện thuật toán. |
| 70 | `all_labels, all_probabilities = [], []` | Tạo hai list chứa arrays nhãn và predictions của từng batch. |
| 71 | `elapsed = 0.0` | Khởi tạo tổng thời gian inference theo đơn vị giây. |
| 72 | `image_count = 0` | Khởi tạo tổng số ảnh thực sự đã thu predictions. |
| 73 | `for images, labels in dataset:` | Lấy images và labels từ cùng batch; dữ liệu đã yield trước khi timer bắt đầu. |
| 74 | `# Warm up once before timing predictions.` | Comment giải thích ý đồ đoạn code; không chạy và không tạo tensors/weights. |
| 75 | `if image_count == 0:` | Từ chối dataset không yield ảnh nào. |
| 76 | `np.asarray(model.predict_on_batch(images))` | Predict batch đầu để warm-up và chuyển output sang NumPy; bỏ kết quả, không tính thời gian. |
| 77 | `start = perf_counter()` | Bắt đầu đo ngay trước predict_on_batch. |
| 78 | `probabilities = np.asarray(model.predict_on_batch(images))` | Chạy inference và lấy output NumPy; conversion/transfer output nằm trong đoạn đo. |
| 79 | `elapsed += perf_counter() - start` | Cộng thời gian vừa đo của batch vào tổng giây. |
| 81 | `# Keep labels and predictions from the same batch.` | Comment giải thích ý đồ đoạn code; không chạy và không tạo tensors/weights. |
| 82 | `indices = labels_to_indices(labels)` | Chuẩn hóa labels của chính batch đã predict, không đọc lại dataset ở lượt khác. |
| 83 | `if probabilities.shape != (len(indices), 3):` | Bảo đảm mỗi ảnh trong batch có đúng ba probabilities, kể cả batch cuối ngắn. |
| 84 | `raise ValueError("Each prediction batch must match its label batch")` | Dừng hàm và báo ValueError với thông điệp này khi đầu vào vi phạm điều kiện ngay phía trên; không tiếp tục tính metric sai. |
| 85 | `all_labels.append(indices)` | Lưu vector indices của batch; chưa ghép với các batch trước. |
| 86 | `all_probabilities.append(probabilities)` | Lưu matrix probabilities của cùng batch theo cùng thứ tự. |
| 87 | `image_count += len(indices)` | Cộng số ảnh thực tế của batch, không giả định batch cuối đầy đủ. |
| 89 | `if image_count == 0:` | Từ chối dataset không yield ảnh nào. |
| 90 | `raise ValueError("Cannot evaluate an empty dataset")` | Dừng hàm và báo ValueError với thông điệp này khi đầu vào vi phạm điều kiện ngay phía trên; không tiếp tục tính metric sai. |
| 91 | `# Timing excludes data loading and preprocessing.` | Comment giải thích ý đồ đoạn code; không chạy và không tạo tensors/weights. |
| 92 | `return (np.concatenate(all_labels), np.concatenate(all_probabilities),` | Ghép batch labels/probabilities theo chiều 0 thành (N,) và (N,3). |
| 93 | `1000 * elapsed / image_count)` | Đổi tổng thời gian giây sang ms rồi chia số ảnh; trả throughput quy về ms/ảnh. |
| 96 | `def save_json(data, path):` | Định nghĩa hàm `save_json` và các tham số; thân hàm chỉ chạy khi được gọi. Xem phần cơ chế của hàm tương ứng ở trên. |
| 97 | `Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")` | Encode JSON thụt lề, từ chối NaN/Inf, ghi UTF-8; không tự tạo parent và có thể ghi đè. |
| 100 | `def plot_confusion_matrix(matrix, path, model_name):` | Định nghĩa hàm `plot_confusion_matrix` và các tham số; thân hàm chỉ chạy khi được gọi. Xem phần cơ chế của hàm tương ứng ở trên. |
| 101 | `"""Rows are true classes; columns are predicted classes."""` | Docstring mô tả nhiệm vụ hoặc cấu trúc của hàm/module; không thực hiện thuật toán. |
| 102 | `fig = Figure(figsize=(7, 6))` | Tạo hình confusion matrix kích thước 7×6 inch. |
| 103 | `ax = fig.subplots()` | Tạo một axes để vẽ confusion matrix. |
| 104 | `chart = ConfusionMatrixDisplay(np.asarray(matrix), display_labels=CLASS_NAMES)` | Gắn ma trận counts và tên lớp theo cùng thứ tự. |
| 105 | `chart.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=20)` | Vẽ màu xanh, ghi counts integer trong ô và xoay nhãn cột 20 độ. |
| 106 | `ax.set_title(f"Confusion matrix: {model_name}")` | Đặt title phân biệt model, không thay nội dung ma trận. |
| 107 | `fig.tight_layout()` | Tự điều chỉnh khoảng cách để nhãn/title hạn chế bị cắt. |
| 108 | `fig.savefig(path, dpi=150)` | Ghi hình ra đường dẫn với DPI 150; caller chuẩn bị thư mục cha. |
| 111 | `def plot_learning_curves(history, path, model_name):` | Định nghĩa hàm `plot_learning_curves` và các tham số; thân hàm chỉ chạy khi được gọi. Xem phần cơ chế của hàm tương ứng ở trên. |
| 112 | `"""Plot training and validation loss and accuracy."""` | Docstring mô tả nhiệm vụ hoặc cấu trúc của hàm/module; không thực hiện thuật toán. |
| 113 | `fig = Figure(figsize=(11, 4))` | Tạo hình learning curves kích thước 11×4 inch. |
| 114 | `axes = fig.subplots(1, 2)` | Tạo hai axes trên một hàng: loss và accuracy. |
| 115 | `for ax, key in zip(axes, ["loss", "accuracy"]):` | Ghép từng axes với tên metric để dùng lại logic vẽ. |
| 116 | `epochs = range(1, len(history[key]) + 1)` | Tạo các epoch từ 1 đến số phần tử history của metric. |
| 117 | `ax.plot(epochs, history[key], label="Train")` | Vẽ metric train đã ghi trong fit, không phải inference trên train clean. |
| 118 | `ax.plot(epochs, history[f"val_{key}"], label="Validation")` | Vẽ metric validation tương ứng trên cùng trục epoch. |
| 119 | `ax.set_xlabel("Epoch")` | Đặt nhãn trục x là epoch. |
| 120 | `ax.set_ylabel(key.capitalize())` | Đặt nhãn trục y theo metric, viết hoa chữ đầu. |
| 121 | `ax.legend()` | Hiển thị chú giải Train/Validation. |
| 122 | `ax.grid(alpha=0.3)` | Thêm lưới mờ giúp đọc giá trị. |
| 123 | `fig.suptitle(f"Learning curves: {model_name}")` | Đặt title chung cho cả hai subplot. |
| 124 | `fig.tight_layout()` | Tự điều chỉnh khoảng cách để nhãn/title hạn chế bị cắt. |
| 125 | `fig.savefig(path, dpi=150)` | Ghi hình ra đường dẫn với DPI 150; caller chuẩn bị thư mục cha. |
| 128 | `def evaluate_model(model, test_dataset, model_name, reports_dir="reports"):` | Định nghĩa hàm `evaluate_model` và các tham số; thân hàm chỉ chạy khi được gọi. Xem phần cơ chế của hàm tương ứng ở trên. |
| 129 | `"""Evaluate the test set and save metrics and a confusion matrix."""` | Docstring mô tả nhiệm vụ hoặc cấu trúc của hàm/module; không thực hiện thuật toán. |
| 130 | `y_true, y_prob, inference_time = collect_predictions(model, test_dataset)` | Thu toàn bộ labels/probabilities và thời gian; không train hoặc chọn checkpoint. |
| 131 | `metrics = compute_metrics(y_true, y_prob)` | Tính metrics phân loại từ các arrays đã thu. |
| 132 | `metrics.update({` | Bắt đầu bổ sung metadata và thống kê model vào dictionary metrics. |
| 133 | `"model_name": model_name,` | Ghi tên do caller truyền; không tự nhận diện file checkpoint. |
| 134 | `"class_names": CLASS_NAMES,` | Lưu thứ tự lớp để đọc confusion matrix và output đúng. |
| 135 | `"num_test_images": len(y_true),` | Lưu tổng số ảnh, không phải tổng số patient. |
| 136 | `"parameter_count": int(model.count_params()),` | Đếm cả trainable và non-trainable parameters, không đo FLOPs. |
| 137 | `"inference_time": float(inference_time),` | Lưu thời gian scalar đã tính bằng ms/ảnh. |
| 138 | `"inference_time_unit": "ms/image",` | Ghi đơn vị để tránh đọc nhầm thành giây/batch. |
| 139 | `"inference_time_method": "Warmed predict_on_batch calls; excludes data loading",` | Ghi mô tả phương pháp; giới hạn đo lường chi tiết nằm ở phần timing của tài liệu. |
| 140 | `})` | Đóng cấu trúc/lời gọi đã mở ở các dòng trên; không tạo một bước xử lý mới. |
| 142 | `reports_dir = Path(reports_dir)` | Chuẩn hóa đường dẫn reports thành Path. |
| 143 | `reports_dir.mkdir(parents=True, exist_ok=True)` | Tạo reports và thư mục cha nếu cần; thư mục đã có không gây lỗi. |
| 144 | `save_json(metrics, reports_dir / f"{model_name}_metrics.json")` | Xuất JSON test theo tên model; có thể ghi đè report cùng tên. |
| 145 | `plot_confusion_matrix(metrics["confusion_matrix"],` | Truyền counts từ kết quả metrics cho helper vẽ ma trận. |
| 146 | `reports_dir / f"{model_name}_confusion_matrix.png", model_name)` | Chọn file PNG và title model của report test. |
| 147 | `return metrics` | Trả dictionary để notebook trình bày hoặc caller tổng hợp bảng so sánh. |
