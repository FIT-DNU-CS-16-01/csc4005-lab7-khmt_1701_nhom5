import json
import os
import pandas as pd
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)

with open("outputs/eval_baseline_onnx.json", "r", encoding="utf-8") as f:
    base_eval = json.load(f)

with open("outputs/eval_quantized_onnx.json", "r", encoding="utf-8") as f:
    quant_eval = json.load(f)

bench = pd.read_csv("outputs/benchmark_quantization.csv")

# lấy kết quả batch_size = 1 để ghi báo cáo
base_bench = bench[(bench["model"] == "baseline_onnx") & (bench["batch_size"] == 1)].iloc[0]
quant_bench = bench[(bench["model"] == "quantized_int8") & (bench["batch_size"] == 1)].iloc[0]

base_size = os.path.getsize("models/vit_smartcampus.onnx") / (1024 * 1024)
quant_size = os.path.getsize("models/vit_smartcampus_dynamic_int8.onnx") / (1024 * 1024)

rows = [
    {
        "Model": "Baseline ONNX FP32",
        "Size MB": round(base_size, 3),
        "Accuracy": round(base_eval["accuracy"], 4),
        "Macro-F1": round(base_eval["macro_f1"], 4),
        "Latency ms": round(float(base_bench["mean_latency_ms"]), 3),
        "Throughput img/s": round(float(base_bench["throughput_img_per_sec"]), 3),
        "Nhận xét": "Model gốc, độ chính xác tốt hơn nhưng chưa nén"
    },
    {
        "Model": "Dynamic Quantization INT8",
        "Size MB": round(quant_size, 3),
        "Accuracy": round(quant_eval["accuracy"], 4),
        "Macro-F1": round(quant_eval["macro_f1"], 4),
        "Latency ms": round(float(quant_bench["mean_latency_ms"]), 3),
        "Throughput img/s": round(float(quant_bench["throughput_img_per_sec"]), 3),
        "Nhận xét": "Nén an toàn MatMul/Gemm, chạy được trên ONNX Runtime CPU"
    }
]

df = pd.DataFrame(rows)
df.to_csv("outputs/tradeoff_table.csv", index=False, encoding="utf-8-sig")

md = "# Trade-off Table - Lab 7\n\n"
md += df.to_markdown(index=False)
md += """

## Nhận xét

Mô hình sau Dynamic Quantization vẫn giữ được khả năng suy luận trên ONNX Runtime CPU. Trong thí nghiệm này, chỉ lượng tử hóa các toán tử MatMul/Gemm để tránh lỗi ConvInteger trên CPU, vì vậy kích thước model giảm không nhiều. Tuy nhiên, mô hình nén vẫn có thể dùng để so sánh latency, throughput, accuracy và macro-F1 với baseline.

## Kết luận

Với bài toán Smart Campus Scene Classification gồm 5 lớp classroom, computerroom, library, corridor và office, ONNX Dynamic Quantization phù hợp khi cần thử nghiệm nén mô hình nhanh và triển khai trên CPU. Nếu muốn giảm kích thước mạnh hơn, cần dùng static quantization hoặc mô hình student nhỏ hơn bằng Knowledge Distillation.
"""

with open("outputs/tradeoff_table.md", "w", encoding="utf-8") as f:
    f.write(md)

print(df)
print("Saved outputs/tradeoff_table.csv")
print("Saved outputs/tradeoff_table.md")
