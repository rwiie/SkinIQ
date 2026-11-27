"""Score saved multi-label models against a label file, without retraining.

Use it to compare models trained on different labels against the same answer key: for example v2
(trained and first scored on the earlier labels) against the author's relabelled test photos, where
v4 was scored.

Each model is scored on the annotated test photos (unsure ones left out, as in training), two ways:
- "stored": the per-concern thresholds saved in the checkpoint (tuned on the labels it was trained with);
- "retuned": thresholds re-tuned on the validation photos of --labels (max F1 per concern), as
  training does. This is the fair comparison when the label files differ.
With --tta each photo is also scored averaged with its mirror image, as the app runs v2.

    python training/evaluate_multilabel.py --labels my_labels.csv \\
        --model v2=models/skin_concerns_v2/model.pt --model v4=models/skin_concerns_v4/model.pt --tta

Writes <out-dir>/comparison.csv (one row per model, set-up and threshold choice) and
<out-dir>/per_concern.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from skiniq.config import DATA_DIR  # noqa: E402
from skiniq.vision.model import build_model, eval_transform  # noqa: E402
from train_multilabel import (C, MultiLabelImages, build_table, overall_metrics,  # noqa: E402
                              per_concern_metrics, photo_level, tune_thresholds)


@torch.no_grad()
def predict(model, loader, device, tta: bool = False) -> np.ndarray:
    """Sigmoid probabilities per file; with tta, averaged with the mirror image."""
    model.eval()
    out = []
    for x, _, _ in loader:
        x = x.to(device)
        p = torch.sigmoid(model(x).float())
        if tta:
            p = (p + torch.sigmoid(model(torch.flip(x, dims=[3])).float())) / 2
        out.append(p.cpu())
    return torch.cat(out).numpy()


def summarise(p_test: np.ndarray, y_test: np.ndarray, thresholds: dict[str, float]) -> tuple[dict, pd.DataFrame]:
    """Overall row (macro precision, recall and mean ROC-AUC added) and the per-concern table."""
    per = per_concern_metrics(p_test, y_test, thresholds)
    row = overall_metrics(p_test, y_test, thresholds)
    row.update(macro_precision=round(per.precision.mean(), 4), macro_recall=round(per.recall.mean(), 4),
               mean_roc_auc=round(per.roc_auc.mean(), 4), mean_pr_auc=round(per.pr_auc.mean(), 4))
    return row, per


def mean_auc(p: np.ndarray, y: np.ndarray) -> float:
    aucs = [roc_auc_score(y[:, j], p[:, j]) for j in range(len(C)) if 0 < y[:, j].sum() < len(y)]
    return round(float(np.mean(aucs)), 4) if aucs else float("nan")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, required=True, help="label file to score against")
    p.add_argument("--model", action="append", required=True, metavar="NAME=PATH",
                   help="checkpoint to score (repeat for several)")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR / "processed")
    p.add_argument("--out-dir", type=Path, default=Path("training/results/label_comparison"))
    p.add_argument("--tta", action="store_true", help="also score with mirror-image averaging")
    p.add_argument("--include-unsure-in-test", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args(argv)


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = build_table(args.data_dir, args.labels)
    val_t = table[(table.split == "val") & table.annotated]
    test_t = table[(table.split == "test") & table.annotated]
    if not args.include_unsure_in_test:
        test_t = test_t[~test_t.unsure]
    print(f"[evaluate] labels={args.labels} val={val_t.key.nunique()} photos test={test_t.key.nunique()} photos "
          f"device={device}")

    rows, pers = [], []
    for spec in args.model:
        name, _, path = spec.partition("=")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = build_model(len(ckpt["labels"]), ckpt["model_name"], pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        size = ckpt["image_size"]
        kw = dict(batch_size=args.batch_size, num_workers=args.num_workers)
        val_dl = DataLoader(MultiLabelImages(val_t, eval_transform(size)), **kw)
        test_dl = DataLoader(MultiLabelImages(test_t, eval_transform(size)), **kw)
        for tta in ([False, True] if args.tta else [False]):
            setup = "mirror averaging" if tta else "single image"
            p_val, y_val = photo_level(val_t, predict(model, val_dl, device, tta))
            p_test, y_test = photo_level(test_t, predict(model, test_dl, device, tta))
            choices = {"retuned on these val labels": tune_thresholds(p_val, y_val)}
            if ckpt.get("thresholds") and not tta:  # stored thresholds were tuned without mirror averaging
                choices = {"stored in checkpoint": ckpt["thresholds"], **choices}
            for choice, thr in choices.items():
                row, per = summarise(p_test, y_test, thr)
                rows.append({"model": name, "setup": setup, "thresholds": choice,
                             "val_mean_roc_auc": mean_auc(p_val, y_val), **row,
                             "threshold_values": json.dumps(thr)})
                per = per.reset_index()
                per.insert(0, "thresholds", choice)
                per.insert(0, "setup", setup)
                per.insert(0, "model", name)
                pers.append(per)
                print(f"  {name:>4} | {setup:16} | {choice:28} | macro F1 {row['macro_f1']:.3f} "
                      f"| mean ROC-AUC {row['mean_roc_auc']:.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.out_dir / "comparison.csv", index=False)
    pd.concat(pers).to_csv(args.out_dir / "per_concern.csv", index=False)
    print(f"  saved -> {args.out_dir}")
    return result


if __name__ == "__main__":
    main()
