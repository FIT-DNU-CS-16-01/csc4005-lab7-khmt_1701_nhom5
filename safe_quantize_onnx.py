from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path
import os
import json

input_onnx = "models/vit_smartcampus.onnx"
output_onnx = "models/vit_smartcampus_dynamic_int8.onnx"

Path("models").mkdir(exist_ok=True)

baseline_size = os.path.getsize(input_onnx) / (1024 * 1024)

quantize_dynamic(
    model_input=input_onnx,
    model_output=output_onnx,
    op_types_to_quantize=["MatMul", "Gemm"],
    weight_type=QuantType.QInt8
)

compressed_size = os.path.getsize(output_onnx) / (1024 * 1024)

report = {
    "method": "onnx_dynamic_quantization_safe",
    "input_onnx": input_onnx,
    "output_onnx": output_onnx,
    "op_types_to_quantize": ["MatMul", "Gemm"],
    "weight_type": "QInt8",
    "baseline_size_mb": baseline_size,
    "compressed_size_mb": compressed_size,
    "size_reduction_percent": (baseline_size - compressed_size) / baseline_size * 100
}

with open("models/quantization_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(report)
