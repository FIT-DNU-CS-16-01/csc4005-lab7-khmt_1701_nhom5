# Trade-off Table - Lab 7

| Model                     |   Size MB |   Accuracy |   Macro-F1 |   Latency ms |   Throughput img/s | Nhận xét                                                 |
|:--------------------------|----------:|-----------:|-----------:|-------------:|-------------------:|:---------------------------------------------------------|
| Baseline ONNX FP32        |    42.636 |     0.9412 |     0.9257 |       12.477 |             80.149 | Model gốc, độ chính xác tốt hơn nhưng chưa nén           |
| Dynamic Quantization INT8 |    42.633 |     0.9412 |     0.9257 |       12.674 |             78.904 | Nén an toàn MatMul/Gemm, chạy được trên ONNX Runtime CPU |

## Nhận xét

Mô hình sau Dynamic Quantization vẫn giữ được khả năng suy luận trên ONNX Runtime CPU. Trong thí nghiệm này, chỉ lượng tử hóa các toán tử MatMul/Gemm để tránh lỗi ConvInteger trên CPU, vì vậy kích thước model giảm không nhiều. Tuy nhiên, mô hình nén vẫn có thể dùng để so sánh latency, throughput, accuracy và macro-F1 với baseline.

## Kết luận

Với bài toán Smart Campus Scene Classification gồm 5 lớp classroom, computerroom, library, corridor và office, ONNX Dynamic Quantization phù hợp khi cần thử nghiệm nén mô hình nhanh và triển khai trên CPU. Nếu muốn giảm kích thước mạnh hơn, cần dùng static quantization hoặc mô hình student nhỏ hơn bằng Knowledge Distillation.
