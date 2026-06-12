# CSC4005 Lab 7 Report – Compression: KD + Quantization Trade-offs

## 1. Thông tin
- Họ tên: Nguyễn ĐỨc Hoàng
- Mã sinh viên: 1771040015
- Lớp: KHMT 17-01
- Link GitHub repo: https://github.com/FIT-DNU-CS-16-01/csc4005-lab7-khmt_1701_nhom5
- Kỹ thuật chọn: Quantization
- Link W&B nếu dùng KD: Không sử dụng
- Link model nếu không commit trực tiếp:
- Baseline ONNX: https://github.com/FIT-DNU-CS-16-01/csc4005-lab7-khmt_1701_nhom5/blob/main/models/vit_smartcampus.onnx
- Compressed ONNX INT8: https://github.com/FIT-DNU-CS-16-01/csc4005-lab7-khmt_1701_nhom5/blob/main/models/      vit_smartcampus_dynamic_int8.onnx
- Quantization report: https://github.com/FIT-DNU-CS-16-01/csc4005-lab7-khmt_1701_nhom5/blob/main/models/quantization_report.json
- Benchmark result: https://github.com/FIT-DNU-CS-16-01/csc4005-lab7-khmt_1701_nhom5/blob/main/outputs/benchmark_quantization.csv
- Trade-off table: https://github.com/FIT-DNU-CS-16-01/csc4005-lab7-khmt_1701_nhom5/blob/main/outputs/tradeoff_table.md

## 2. Mô tả baseline model

| Nội dung | Giá trị |
|---|---|
| Bài toán | Smart Campus Scene Classification |
| Dataset | MIT Indoor Scenes 67 subset |
| Số lớp | 5 |
| Baseline model | classroom, computerroom, library, corridor, office |
| Baseline format | PyTorch / ONNX | ResNet18 fine-tuned |
| Baseline checkpoint/ONNX | models/vit_smartcampus.onnx |
| Baseline checkpoint PyTorch | checkpoints/resnet18_smartcampus_best.pt |
| Baseline model size | 42.636 MB |

Baseline model được huấn luyện trên subset 5 lớp của bộ dữ liệu MIT Indoor Scenes 67. Sau khi huấn luyện, model được export sang định dạng ONNX để phục vụ quá trình benchmark và nén mô hình bằng quantization.

## 3. Kỹ thuật nén đã chọn

### Nếu chọn Quantization

| Thông tin | Giá trị |
|---|---|
| Loại quantization | Dynamic / Static |
| Input model | models/vit_smartcampus.onnx |
| Output model | models/vit_smartcampus_dynamic_int8.onnx |
| Dạng dữ liệu sau nén | INT8 / khác |
| Công cụ | onnxruntime.quantization |

Mô tả ngắn:

Trong bài lab này, nhóm sử dụng ONNX Dynamic Quantization để nén mô hình baseline đã được export sang định dạng ONNX. Ban đầu, khi quantization toàn bộ model, ONNX Runtime CPU gặp lỗi ConvInteger ở các tầng convolution. Vì vậy, nhóm chọn cách quantization an toàn hơn bằng cách chỉ lượng tử hóa các toán tử MatMul và Gemm. Cách làm này giúp model sau nén vẫn chạy ổn định trên ONNX Runtime CPU, đồng thời vẫn đáp ứng yêu cầu so sánh accuracy, macro-F1, latency, throughput và model size trước/sau nén.



### Nếu chọn Knowledge Distillation

| Thông tin | Giá trị |
|---|---|
| Teacher model | Không sử dụng |
| Student model | Không sử dụng |
| alpha | Không áp dụng |
| temperature | Không áp dụng |
| epochs | Không áp dụng |
| batch size | Không áp dụng |
| optimizer | Không áp dụng |

Công thức loss sử dụng:

Không áp dụng vì nhóm không chọn Knowledge Distillation.

## 4. Kết quả đánh giá

| Model | Accuracy | Macro-F1 | Model size (MB) |
|---|---:|---:|---:|
| Baseline | 0.9412 | 0.9257 | 42.636 |
| Compressed | 0.9412 | 0.9257 | 42.633 |

Nhận xét:

- Accuracy giảm: 0.0000, tương đương 0%.
- Macro-F1 giảm: 0.0000, tương đương 0%
- Mức giảm này hoàn toàn chấp nhận được vì chất lượng dự đoán của model sau nén không thay đổi so với baseline.
- Tuy nhiên, model size giảm rất ít do nhóm chỉ quantize các toán tử MatMul/Gemm để tránh lỗi ConvInteger khi chạy trên ONNX Runtime CPU.

## 5. Kết quả benchmark

| Model | Batch size | Mean latency (ms) | P95 latency (ms) | Throughput (img/s) | Size (MB) |
|---|---:|---:|---:|---:|---:|
| Baseline ONNX FP32 | 1 | 12.477 | 15.618 | 80.149 | 42.636 |
| Compressed ONNX INT8 | 1 | 12.674 | 15.685 | 78.904 | 42.633 |
| Baseline ONNX FP32 | 4 | 48.934 | 54.009 | 81.743 | 42.636 |
| Compressed ONNX INT8 | 4 | 50.085 | 58.025 | 79.865 | 42.633 |
| Baseline ONNX FP32 | 8 | 98.523 | 108.140 | 81.200 | 42.636 |
| Compressed ONNX INT8 | 8 | 99.089 | 108.468 | 80.736 | 42.633 |

## 6. Bảng trade-off

| Model | Accuracy | Macro-F1 | Mean latency @bs=1 | Throughput @bs=1 | Size | Nhận xét |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 0.9412 | 0.9257 | 12.477 ms | 80.149 img/s | 42.636 MB | Model gốc, độ chính xác tốt, chạy ổn định trên CPU |
| Compressed | 0.9412 | 0.9257 | 12.674 ms | 78.904 img/s | 42.633 MB | Model sau quantization giữ nguyên accuracy/F1 nhưng chưa cải thiện rõ về tốc độ và dung lượng |

## 7. Phân tích

Trả lời:

1. Mô hình sau nén nhỏ hơn bao nhiêu phần trăm?
- Model baseline có kích thước khoảng 42.636 MB, model sau quantization có kích thước khoảng 42.633 MB. Như vậy, kích thước model giảm khoảng 0.0074%. Mức giảm này rất nhỏ.
Nguyên nhân là do nhóm chỉ quantize các toán tử MatMul và Gemm để đảm bảo model chạy ổn định trên ONNX Runtime CPU. Nếu quantize toàn bộ các tầng convolution, model có thể giảm mạnh hơn nhưng bị lỗi ConvInteger khi benchmark trên máy hiện tại.
2. Latency giảm hay tăng?
- Latency tăng nhẹ. Với batch size = 1:
Baseline latency: 12.477 ms
Compressed latency: 12.674 ms
Như vậy model sau quantization chậm hơn khoảng 0.197 ms. Mức tăng này không lớn nhưng cho thấy quantization trong trường hợp này chưa đem lại lợi ích rõ rệt về tốc độ.
3. Throughput thay đổi thế nào?
- Throughput giảm nhẹ. Với batch size = 1:
Baseline throughput: 80.149 img/s
Compressed throughput: 78.904 img/s
Như vậy throughput giảm khoảng 1.245 img/s. Sự thay đổi này nhỏ, nhưng compressed model chưa nhanh hơn baseline.
4. Accuracy/F1 giảm nhiều không?
- Accuracy và Macro-F1 không giảm.
Baseline Accuracy: 0.9412
Compressed Accuracy: 0.9412
Baseline Macro-F1: 0.9257
Compressed Macro-F1: 0.9257
Điều này cho thấy model sau quantization vẫn giữ được chất lượng dự đoán tương đương model gốc.
5. Nếu triển khai trên CPU hoặc edge device, bạn có chọn compressed model không?
- Trong trường hợp kết quả hiện tại, nhóm chưa ưu tiên chọn compressed model để triển khai vì model sau nén không giảm đáng kể về dung lượng và cũng không cải thiện latency/throughput.
Tuy nhiên, compressed model vẫn có ưu điểm là giữ nguyên accuracy và macro-F1. Nếu hệ thống yêu cầu thử nghiệm quy trình nén nhanh, hoặc cần chuẩn bị cho các bước tối ưu sâu hơn như static quantization, compressed model vẫn là một lựa chọn có thể cân nhắc.
6. Nếu không chọn, lý do là gì?
- Lý do chưa chọn compressed model là:

- Model size giảm rất ít.
- Latency tăng nhẹ.
- Throughput giảm nhẹ.
- Chưa có lợi ích rõ rệt so với baseline.

Nếu cần deploy thật trên CPU hoặc edge device, nhóm nên thử thêm static quantization với calibration data hoặc Knowledge Distillation để tạo student model nhỏ hơn.

## 8. Khi nào chọn KD, khi nào chọn Quantization?

1.  Khi nào quantization phù hợp?

Quantization phù hợp khi:

- Đã có model baseline tốt và muốn nén nhanh.
- Muốn triển khai model trên CPU hoặc thiết bị có tài nguyên hạn chế.
- Muốn giảm kích thước model hoặc tăng tốc inference mà không train lại từ đầu.
- Có thể chấp nhận accuracy giảm nhẹ sau nén.

Trong bài lab này, quantization phù hợp vì nhóm đã có model ONNX baseline và có thể thực hiện nén trực tiếp bằng onnxruntime.quantization.

2. Khi nào KD phù hợp?

Knowledge Distillation phù hợp khi:

- Muốn tạo một student model nhỏ hơn đáng kể so với teacher model.
- Có thời gian và tài nguyên để train lại model.
- Muốn giảm dung lượng và tăng tốc inference mạnh hơn quantization.
- Teacher model có accuracy tốt và có thể hướng dẫn student học tốt hơn.

KD thường phù hợp hơn khi hệ thống cần model rất nhẹ để chạy trên edge device, mobile hoặc môi trường giới hạn bộ nhớ.

3. Nếu được làm lại, bạn sẽ chọn kỹ thuật nào cho hệ thống Smart Campus?

Nếu được làm lại, nhóm sẽ chọn kết hợp cả hai kỹ thuật:

- Dùng Knowledge Distillation để huấn luyện student model nhỏ hơn, ví dụ MobileNetV2 hoặc ResNet18 nhỏ.
- Sau đó dùng Quantization để nén tiếp student model.

Cách kết hợp này có thể giúp giảm model size tốt hơn, cải thiện latency rõ hơn và vẫn giữ accuracy ở mức chấp nhận được cho hệ thống Smart Campus Scene Classification.

## 9. Kết luận

rong Lab 7, nhóm đã thực hiện nén mô hình cho bài toán Smart Campus Scene Classification bằng kỹ thuật ONNX Dynamic Quantization. Baseline model được huấn luyện bằng ResNet18 trên subset 5 lớp của MIT Indoor Scenes 67 và được export sang định dạng ONNX. Sau đó, nhóm tạo model INT8 bằng onnxruntime.quantization.

Kết quả cho thấy accuracy và macro-F1 của model sau nén không thay đổi so với baseline, đều đạt Accuracy = 0.9412 và Macro-F1 = 0.9257. Tuy nhiên, do chỉ quantize an toàn các toán tử MatMul/Gemm, kích thước model giảm rất ít, từ 42.636 MB xuống 42.633 MB. Latency của model sau nén tăng nhẹ từ 12.477 ms lên 12.674 ms ở batch size = 1.

Trade-off quan trọng nhất là quantization giúp giữ nguyên chất lượng dự đoán nhưng trong cấu hình hiện tại chưa cải thiện đáng kể về tốc độ và dung lượng. Qua bài lab này, nhóm hiểu rõ hơn rằng hiệu quả nén phụ thuộc vào loại model, loại toán tử được quantize và khả năng hỗ trợ của runtime trên phần cứng triển khai. Nếu muốn tối ưu tốt hơn, nhóm nên thử Knowledge Distillation hoặc Static Quantization với calibration data.
