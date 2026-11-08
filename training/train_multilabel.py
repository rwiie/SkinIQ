"""Multi-label skin concern training: EfficientNet-V2 with six independent sigmoid outputs
and binary cross-entropy, evaluated per concern.

Labels come from two sources:
- Annotated photos (data/labels/multilabel_labels.csv): all six concerns known.
- Unannotated close-ups: only their dataset folder concern is known (positive).
  The other five are unknown. They are treated as *weak* negatives: target 0 with a
  small loss weight (--unknown-negative-weight, default 0.1), because most close-ups
  show one main concern. Without any negatives the model would learn to answer "yes"
  to everything. Set the weight to 0 to ignore unknown labels completely.
  Unannotated full faces have no known concern and are skipped.

Copies of an original photo share its labels. Validation (threshold tuning, model
selection) and test (metrics) use annotated photos only.

    python training/train_multilabel.py --labels data/labels/multilabel_labels.csv

Writes models/<run-name>/ (default models/skin_concerns/): model.pt (with per-concern
thresholds), history.csv, metrics.json, per_concern.csv, thresholds.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from skiniq.config import DATA_DIR, IMAGE_SIZE, MODELS_DIR, SKIN_CONDITIONS  # noqa: E402
from skiniq.vision.model import DEFAULT_MODEL, build_model, eval_transform, train_transform  # noqa: E402
from train import make_scheduler, seed_everything, set_backbone_trainable  # noqa: E402

C = list(SKIN_CONDITIONS)


# ---- data ---------------------------------------------------------------------------

def build_table(processed: Path, labels_csv: Path | None, unknown_negative_weight: float = 0.1) -> pd.DataFrame:
    """One row per image file: split, targets y_<c>, loss weights m_<c> (1 = known,
    `unknown_negative_weight` = assumed negative, 0 = ignored), annotated/unsure flags."""
    labels = pd.DataFrame()
    if labels_csv is not None and Path(labels_csv).exists():
        labels = pd.read_csv(labels_csv).set_index("key")

    rows = []
    for dataset in ("skin_type", "skin_condition"):
        manifest = pd.read_csv(processed / dataset / "manifest.csv")
        folder_labels = manifest.groupby("group")["label"].agg(set)
        for r in manifest.itertuples():
            key = f"{dataset}:{r.group}"
            row = {"path": str(processed / r.path), "split": r.split, "key": key, "dataset": dataset,
                   "annotated": False, "unsure": False}
            if key in labels.index:
                lab = labels.loc[key]
                if int(lab["unusable"]):
                    continue
                row.update({f"y_{c}": float(lab[c]) for c in C})
                row.update({f"m_{c}": 1.0 for c in C})
                row.update(annotated=True, unsure=bool(int(lab["unsure"])))
            elif dataset == "skin_condition":
                known = folder_labels[r.group]
                row.update({f"y_{c}": float(c in known) for c in C})
                row.update({f"m_{c}": 1.0 if c in known else unknown_negative_weight for c in C})
            else:
                continue  # unannotated full face: nothing known
            rows.append(row)
    return pd.DataFrame(rows)


class MultiLabelImages(Dataset):
    def __init__(self, table: pd.DataFrame, transform) -> None:
        self.paths = table["path"].tolist()
        self.y = torch.tensor(table[[f"y_{c}" for c in C]].values, dtype=torch.float32)
        self.m = torch.tensor(table[[f"m_{c}" for c in C]].values, dtype=torch.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.y[i], self.m[i]


# ---- loss, thresholds and metrics --------------------------------------------------

def masked_bce(logits: torch.Tensor, y: torch.Tensor, m: torch.Tensor, pos_weight: torch.Tensor | None = None):
    """Binary cross-entropy, weighted mean over labels by their weight (0 = ignored)."""
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")
    return (loss * m).sum() / m.sum().clamp(min=1e-8)  # weighted mean over known labels


def positive_weights(table: pd.DataFrame, cap: float = 10.0) -> torch.Tensor:
    """negatives / positives per concern over known labels, capped; 1.0 when unknown."""
    w = []
    for c in C:
        known = table[table[f"m_{c}"] == 1]
        pos = (known[f"y_{c}"] == 1).sum()
        neg = (known[f"y_{c}"] == 0).sum()
        w.append(min(cap, neg / pos) if pos and neg else 1.0)
    return torch.tensor(w, dtype=torch.float32)


def tune_thresholds(probs: np.ndarray, y: np.ndarray, default: float = 0.5) -> dict[str, float]:
    """Per concern, the threshold (0.05-0.95) with the best F1 on validation."""
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    out = {}
    for j, c in enumerate(C):
        if y[:, j].min() == y[:, j].max():  # only one class present: can't tune
            out[c] = default
            continue
        scores = [f1_score(y[:, j], probs[:, j] >= t, zero_division=0) for t in grid]
        out[c] = float(grid[int(np.argmax(scores))])
    return out


def per_concern_metrics(probs: np.ndarray, y: np.ndarray, thresholds: dict[str, float]) -> pd.DataFrame:
    rows = []
    for j, c in enumerate(C):
        pred = probs[:, j] >= thresholds[c]
        both = 0 < y[:, j].sum() < len(y)
        rows.append({
            "concern": c,
            "threshold": thresholds[c],
            "precision": precision_score(y[:, j], pred, zero_division=0),
            "recall": recall_score(y[:, j], pred, zero_division=0),
            "f1": f1_score(y[:, j], pred, zero_division=0),
            "roc_auc": roc_auc_score(y[:, j], probs[:, j]) if both else float("nan"),
            "pr_auc": average_precision_score(y[:, j], probs[:, j]) if both else float("nan"),
            "positives": int(y[:, j].sum()),
            "negatives": int(len(y) - y[:, j].sum()),
        })
    return pd.DataFrame(rows).set_index("concern").round(4)


def overall_metrics(probs: np.ndarray, y: np.ndarray, thresholds: dict[str, float]) -> dict:
    pred = np.stack([probs[:, j] >= thresholds[c] for j, c in enumerate(C)], axis=1).astype(int)
    clear = y.sum(axis=1) == 0
    return {
        "photos": int(len(y)),
        "macro_f1": round(f1_score(y, pred, average="macro", zero_division=0), 4),
        "micro_f1": round(f1_score(y, pred, average="micro", zero_division=0), 4),
        "exact_match": round(float((pred == y).all(axis=1).mean()), 4),
        "hamming_loss": round(float((pred != y).mean()), 4),
        "clear_photos": int(clear.sum()),
        # Share of clear-skin photos wrongly given at least one concern
        "clear_false_alarm_rate": round(float(pred[clear].any(axis=1).mean()), 4) if clear.any() else None,
        "multi_concern_photos": int((y.sum(axis=1) >= 2).sum()),
    }


@torch.no_grad()
def predict_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs, ys, ms = [], [], []
    for x, y, m in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x.to(device))
        probs.append(torch.sigmoid(logits.float()).cpu())
        ys.append(y)
        ms.append(m)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy(), torch.cat(ms).numpy()


def photo_level(table: pd.DataFrame, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average predictions over copies of the same photo, so each photo counts once."""
    df = table[["key"]].copy()
    df[[f"p_{c}" for c in C]] = probs
    df[[f"y_{c}" for c in C]] = table[[f"y_{c}" for c in C]].values
    g = df.groupby("key").mean()
    return g[[f"p_{c}" for c in C]].values, g[[f"y_{c}" for c in C]].values.round().astype(int)


# ---- training -----------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, default=DATA_DIR / "labels" / "multilabel_labels.csv")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR / "processed")
    p.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    p.add_argument("--run-name", default="skin_concerns")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--img-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--freeze-epochs", type=int, default=0)
    p.add_argument("--aug", choices=["mild", "strong"], default="mild")
    p.add_argument("--drop-rate", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--unknown-negative-weight", type=float, default=0.1,
                   help="loss weight for the assumed-negative unknown labels of unannotated close-ups")
    p.add_argument("--min-val-photos", type=int, default=50,
                   help="annotated validation photos needed to select by ROC-AUC; else validation loss")
    p.add_argument("--include-unsure-in-test", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out_dir / args.run_name
    out.mkdir(parents=True, exist_ok=True)

    table = build_table(args.data_dir, args.labels, args.unknown_negative_weight)
    train_t = table[table.split == "train"]
    val_t = table[(table.split == "val") & table.annotated]
    test_t = table[(table.split == "test") & table.annotated]
    if not args.include_unsure_in_test:
        test_t = test_t[~test_t.unsure]
    if test_t.empty:
        raise SystemExit("No annotated test photos yet: label the test sample first (annotate_colab.ipynb).")
    select_by_auc = val_t.key.nunique() >= args.min_val_photos
    if not select_by_auc:  # too few annotated val photos: use partial labels and val loss
        val_t = table[table.split == "val"]
    print(f"[multi-label] device={device} train={len(train_t)} files "
          f"({train_t.annotated.sum()} from annotated photos) val={len(val_t)} test={len(test_t)} "
          f"({test_t.key.nunique()} photos); select by {'val ROC-AUC' if select_by_auc else 'val loss'}")

    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_dl = DataLoader(MultiLabelImages(train_t, train_transform(args.img_size, args.aug)), shuffle=True,
                          drop_last=len(train_t) > args.batch_size, **kw)
    val_dl = DataLoader(MultiLabelImages(val_t, eval_transform(args.img_size)), **kw)
    test_dl = DataLoader(MultiLabelImages(test_t, eval_transform(args.img_size)), **kw)

    model = build_model(len(C), args.model, pretrained=not args.no_pretrained, drop_rate=args.drop_rate).to(device)
    pos_weight = positive_weights(train_t).to(device)
    print("  positive weights:", dict(zip(C, [round(w, 2) for w in pos_weight.tolist()])))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = len(train_dl)
    scheduler = make_scheduler(optimizer, args.epochs * steps, int(args.warmup_epochs * steps))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history, best, best_epoch, stale = [], -math.inf, 0, 0
    for epoch in range(1, args.epochs + 1):
        set_backbone_trainable(model, trainable=epoch > args.freeze_epochs)
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y, m in train_dl:
            x, y, m = x.to(device), y.to(device), m.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(x)
            loss = masked_bce(logits.float(), y, m, pos_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item() * len(x)
            n += len(x)

        probs, ys, ms = predict_probs(model, val_dl, device)
        val_loss = masked_bce(torch.tensor(probs).clamp(1e-6, 1 - 1e-6).logit(), torch.tensor(ys), torch.tensor(ms)).item()
        row = {"epoch": epoch, "train_loss": round(total / max(n, 1), 4), "val_loss": round(val_loss, 4)}
        if select_by_auc:
            p_photo, y_photo = photo_level(val_t, probs)
            aucs = [roc_auc_score(y_photo[:, j], p_photo[:, j]) for j in range(len(C)) if 0 < y_photo[:, j].sum() < len(y_photo)]
            row["val_mean_roc_auc"] = round(float(np.mean(aucs)), 4) if aucs else float("nan")
            score = row["val_mean_roc_auc"]
        else:
            score = -val_loss
        row["seconds"] = round(time.time() - t0, 1)
        history.append(row)
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()))

        if score > best:
            best, best_epoch, stale = score, epoch, 0
            torch.save({"state_dict": model.state_dict(), "labels": C, "model_name": args.model,
                        "image_size": args.img_size, "multilabel": True, "epoch": epoch}, out / "model.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"  early stop: no improvement for {args.patience} epochs")
                break
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    # Best checkpoint: tune per-concern thresholds on validation, evaluate on test
    ckpt = torch.load(out / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    if select_by_auc:
        p_val, y_val = photo_level(val_t, predict_probs(model, val_dl, device)[0])
        thresholds = tune_thresholds(p_val, y_val)
    else:
        thresholds = {c: 0.5 for c in C}
    ckpt["thresholds"] = thresholds
    torch.save(ckpt, out / "model.pt")

    p_test, y_test = photo_level(test_t, predict_probs(model, test_dl, device)[0])
    per = per_concern_metrics(p_test, y_test, thresholds)
    overall = overall_metrics(p_test, y_test, thresholds)
    per.to_csv(out / "per_concern.csv")
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    metrics = {
        "run_name": args.run_name, "model": args.model, "best_epoch": best_epoch, "epochs_run": len(history),
        "selected_by": "val mean ROC-AUC" if select_by_auc else "val loss (too few annotated val photos)",
        "train_files": int(len(train_t)), "train_files_fully_annotated": int(train_t.annotated.sum()),
        "test": overall, "per_concern": per.to_dict(orient="index"), "thresholds": thresholds,
        "settings": {k: str(v) for k, v in vars(args).items()},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(per.to_string())
    print("  overall:", overall)
    print(f"  saved -> {out}")
    return metrics


if __name__ == "__main__":
    main()
