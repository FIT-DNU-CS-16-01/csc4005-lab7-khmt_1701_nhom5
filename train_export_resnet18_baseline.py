from pathlib import Path
import json
import random
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def make_splits(n, seed=42):
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(y.tolist())
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    return acc, mf1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/mit_indoor_smartcampus_5")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds_full = datasets.ImageFolder(data_dir, transform=train_tf)
    eval_ds_full = datasets.ImageFolder(data_dir, transform=eval_tf)

    train_idx, val_idx, test_idx = make_splits(len(train_ds_full))

    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(eval_ds_full, val_idx)
    test_ds = Subset(eval_ds_full, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_classes = len(train_ds_full.classes)
    print("Classes:", train_ds_full.classes)
    print("Train/Val/Test:", len(train_ds), len(val_ds), len(test_ds))

    try:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        print("Using pretrained ResNet18")
    except Exception as e:
        print("Cannot load pretrained weights, using random init:", e)
        model = resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_acc, val_f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: loss={total_loss/len(train_loader):.4f}, val_acc={val_acc:.4f}, val_macro_f1={val_f1:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": train_ds_full.classes,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
            }, "checkpoints/resnet18_smartcampus_best.pt")

    ckpt = torch.load("checkpoints/resnet18_smartcampus_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_acc, test_f1 = evaluate(model, test_loader, device)

    metrics = {
        "model": "resnet18_baseline",
        "accuracy": test_acc,
        "macro_f1": test_f1,
        "classes": train_ds_full.classes,
        "num_test": len(test_ds)
    }

    with open("outputs/eval_baseline_onnx.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)

    torch.onnx.export(
        model,
        dummy,
        "models/vit_smartcampus.onnx",
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17
    )

    print("Saved checkpoint: checkpoints/resnet18_smartcampus_best.pt")
    print("Saved ONNX: models/vit_smartcampus.onnx")
    print("Saved metrics: outputs/eval_baseline_onnx.json")
    print("Test accuracy:", test_acc)
    print("Test macro-F1:", test_f1)


if __name__ == "__main__":
    main()
