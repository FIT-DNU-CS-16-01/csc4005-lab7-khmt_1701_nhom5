import torch
import torch.nn as nn
from torchvision.models import resnet18
from pathlib import Path

Path("models").mkdir(exist_ok=True)

ckpt_path = "checkpoints/resnet18_smartcampus_best.pt"
onnx_path = "models/vit_smartcampus.onnx"

ckpt = torch.load(ckpt_path, map_location="cpu")
classes = ckpt.get("classes", ["classroom", "computerroom", "library", "corridor", "office"])

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    onnx_path,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "logits": {0: "batch_size"}
    },
    opset_version=17,
    export_params=True,
    do_constant_folding=True,
    external_data=False,
    dynamo=False
)

print("Exported ONNX:", onnx_path)
print("Classes:", classes)
