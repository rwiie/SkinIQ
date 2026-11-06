"""Fine-tune EfficientNet-V2 on a prepared dataset (proposal 3.6.1) and evaluate
it on the held-out test split (proposal 3.8).

Run training/prepare_data.py first. On Google Colab use training/train_colab.ipynb.

    python training/train.py --task skin_type
    python training/train.py --task skin_condition

    # skin type v2 experiment (anti-overfitting settings), saved to models/skin_type_v2/
    python training/train.py --task skin_type --run-name skin_type_v2 \
        --model tf_efficientnetv2_b0.in1k --img-size 224 --lr 1e-4 --freeze-epochs 3 \
        --aug strong --drop-rate 0.3 --drop-path-rate 0.2 --epochs 25 --patience 6

Writes to models/<run-name>/ (default: models/<task>/):
    model.pt              best checkpoint (by validation macro-F1) + labels and settings
    history.csv           loss / accuracy / macro-F1 per epoch
    metrics.json          test accuracy, precision, recall, F1 (macro, weighted, per class)
    confusion_matrix.csv  and confusion_matrix.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skiniq.config import DATA_DIR, IMAGE_SIZE, MODELS_DIR  # noqa: E402
from skiniq.vision.model import DEFAULT_MODEL, build_model, eval_transform, train_transform  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=["skin_type", "skin_condition"])
    p.add_argument("--data-dir", type=Path, default=DATA_DIR / "processed")
    p.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    p.add_argument("--run-name", default=None, help="output subfolder (default: the task name)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--no-pretrained", action="store_true", help="random init (offline smoke tests only)")
    p.add_argument("--img-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--aug", choices=["mild", "strong"], default="mild")
    p.add_argument("--drop-rate", type=float, default=0.0, help="dropout before the classifier")
    p.add_argument("--drop-path-rate", type=float, default=0.0, help="stochastic depth")
    p.add_argument("--freeze-epochs", type=int, default=0,
                   help="train only the classifier head for the first N epochs")
    p.add_argument("--patience", type=int, default=5, help="early stop after N epochs without val macro-F1 gain")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=None, help="cap images per split (smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(root: Path, split: str, transform, max_samples: int | None, seed: int):
    ds = ImageFolder(root / split, transform=transform)
    if max_samples and len(ds) > max_samples:
        idx = random.Random(seed).sample(range(len(ds)), max_samples)
        sub = Subset(ds, sorted(idx))
        sub.classes, sub.targets = ds.classes, [ds.targets[i] for i in sub.indices]
        return sub
    return ds


def class_weights(targets: list[int], n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1 (handles class imbalance)."""
    counts = Counter(targets)
    w = torch.tensor([len(targets) / (n_classes * max(counts[c], 1)) for c in range(n_classes)])
    return w / w.mean()


def make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze everything except the classifier head.

    Training only the new head first lets it fit before the pretrained features
    are updated, which reduces early overfitting on small, noisy datasets.
    """
    for p in model.parameters():
        p.requires_grad = trainable
    for p in model.get_classifier().parameters():
        p.requires_grad = True


@torch.no_grad()
def evaluate(model, loader, device, criterion=None) -> dict:
    model.eval()
    preds, targets, losses = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x)
        if criterion is not None:
            losses.append(criterion(logits.float(), y).item() * len(y))
        preds += logits.argmax(1).tolist()
        targets += y.tolist()
    out = {
        "accuracy": accuracy_score(targets, preds),
        "macro_f1": f1_score(targets, preds, average="macro", zero_division=0),
        "preds": preds,
        "targets": targets,
    }
    if losses:
        out["loss"] = sum(losses) / len(targets)
    return out


def save_confusion_png(cm: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 3, 1.1 * len(labels) + 2))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> dict:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = args.data_dir / args.task
    out = args.out_dir / (args.run_name or args.task)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = load_split(root, "train", train_transform(args.img_size, args.aug), args.max_samples, args.seed)
    val_ds = load_split(root, "val", eval_transform(args.img_size), args.max_samples, args.seed)
    test_ds = load_split(root, "test", eval_transform(args.img_size), args.max_samples, args.seed)
    labels = train_ds.classes
    assert val_ds.classes == labels == test_ds.classes, "class folders differ between splits"
    print(f"[{args.task}] device={device} classes={labels}")
    print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) > args.batch_size, **loader_kw)
    val_dl = DataLoader(val_ds, **loader_kw)
    test_dl = DataLoader(test_ds, **loader_kw)

    model = build_model(
        len(labels), args.model, pretrained=not args.no_pretrained,
        drop_rate=args.drop_rate, drop_path_rate=args.drop_path_rate,
    ).to(device)
    weights = None if args.no_class_weights else class_weights(train_ds.targets, len(labels))
    if weights is not None:
        print("  class weights:", {c: round(w, 2) for c, w in zip(labels, weights.tolist())})
    criterion = nn.CrossEntropyLoss(
        weight=None if weights is None else weights.to(device), label_smoothing=args.label_smoothing
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_dl)
    scheduler = make_scheduler(optimizer, args.epochs * steps_per_epoch, int(args.warmup_epochs * steps_per_epoch))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history, best_f1, best_epoch, stale = [], -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        set_backbone_trainable(model, trainable=epoch > args.freeze_epochs)
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss += loss.item() * len(y)
            seen += len(y)

        val = evaluate(model, val_dl, device, criterion)
        row = {
            "epoch": epoch,
            "train_loss": round(run_loss / max(seen, 1), 4),
            "val_loss": round(val["loss"], 4),
            "val_accuracy": round(val["accuracy"], 4),
            "val_macro_f1": round(val["macro_f1"], 4),
            "lr": scheduler.get_last_lr()[0],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        print("  " + "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, stale = val["macro_f1"], epoch, 0
            torch.save({
                "state_dict": model.state_dict(),
                "labels": labels,
                "model_name": args.model,
                "image_size": args.img_size,
                "task": args.task,
                "epoch": epoch,
                "val_macro_f1": best_f1,
            }, out / "model.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"  early stop: no val macro-F1 gain for {args.patience} epochs")
                break

    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    # Test-set evaluation with the best checkpoint (proposal 3.8)
    ckpt = torch.load(out / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test = evaluate(model, test_dl, device)
    report = classification_report(
        test["targets"], test["preds"], labels=list(range(len(labels))),
        target_names=labels, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(test["targets"], test["preds"], labels=list(range(len(labels))))
    metrics = {
        "task": args.task,
        "run_name": args.run_name or args.task,
        "model": args.model,
        "image_size": args.img_size,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "val_macro_f1": round(best_f1, 4),
        "test_size": len(test["targets"]),
        "test_accuracy": round(report["accuracy"], 4),
        "test_macro": {k: round(v, 4) for k, v in report["macro avg"].items() if k != "support"},
        "test_weighted": {k: round(v, 4) for k, v in report["weighted avg"].items() if k != "support"},
        "per_class": {c: {k: round(v, 4) for k, v in report[c].items()} for c in labels},
        "settings": {k: str(v) for k, v in vars(args).items()},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out / "confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["actual \\ predicted"] + labels)
        for label, r in zip(labels, cm.tolist()):
            w.writerow([label] + r)
    save_confusion_png(cm, labels, out / "confusion_matrix.png",
                       f"{args.run_name or args.task} — test accuracy {metrics['test_accuracy']:.1%}")

    print(f"  best epoch {best_epoch}: val macro-F1 {best_f1:.4f}")
    print(f"  TEST accuracy {metrics['test_accuracy']:.4f}  macro-F1 {metrics['test_macro']['f1-score']:.4f}")
    print(f"  saved -> {out}")
    return metrics


if __name__ == "__main__":
    main()
