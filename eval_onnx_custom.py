import argparse
import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def make_splits(n, seed=42):
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--data_dir", default="data/mit_indoor_smartcampus_5")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds_full = datasets.ImageFolder(args.data_dir, transform=tf)
    _, _, test_idx = make_splits(len(ds_full))
    test_ds = Subset(ds_full, test_idx)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    sess = ort.InferenceSession(args.onnx_path)
    input_name = sess.get_inputs()[0].name

    y_true = []
    y_pred = []

    for x, y in loader:
        logits = sess.run(None, {input_name: x.numpy().astype(np.float32)})[0]
        pred = logits.argmax(axis=1)

        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred.tolist())

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")

    result = {
        "model": Path(args.onnx_path).name,
        "accuracy": acc,
        "macro_f1": mf1,
        "num_test": len(test_ds),
        "classes": ds_full.classes
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(result)


if __name__ == "__main__":
    main()
