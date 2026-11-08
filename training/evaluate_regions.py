"""Compare whole-image and region-based (YuNet) analysis on annotated test photos.

Uses a multi-label checkpoint from train_multilabel.py and its per-concern thresholds.
Each annotated test photo is scored twice: once as a whole image, once by region
analysis (highest score per concern across face regions). Results are reported per
concern and overall, separately for full faces and close-ups.

    python training/evaluate_regions.py --model models/skin_concerns/model.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from skiniq.config import DATA_DIR, MODELS_DIR, SKIN_CONDITIONS  # noqa: E402
from skiniq.vision.model import SkinClassifier  # noqa: E402
from skiniq.vision.regions import analyse_regions  # noqa: E402
from train_multilabel import overall_metrics, per_concern_metrics  # noqa: E402

C = list(SKIN_CONDITIONS)


def evaluate(classifier, labels: pd.DataFrame, image_root: Path, include_unsure: bool = False) -> dict:
    test = labels[(labels.split == "test") & (labels.unusable == 0)]
    if not include_unsure:
        test = test[test.unsure == 0]
    if test.empty:
        raise SystemExit("No annotated test photos to evaluate.")
    th = classifier.thresholds or {c: 0.5 for c in C}

    rows = []
    for r in test.itertuples():
        img = Image.open(image_root / r.image).convert("RGB")
        whole = classifier.predict(img)
        ra = analyse_regions(img, classifier)
        rows.append({"dataset": r.dataset, "face_found": ra.face_found,
                     **{f"y_{c}": getattr(r, c) for c in C},
                     **{f"whole_{c}": whole[c] for c in C},
                     **{f"region_{c}": ra.condition_scores[c] for c in C}})
    df = pd.DataFrame(rows)

    report = {}
    for subset, part in [("all", df), ("full faces", df[df.dataset == "skin_type"]),
                         ("close-ups", df[df.dataset == "skin_condition"]),
                         ("roboflow", df[df.dataset == "skin_problem"])]:
        if part.empty:
            continue
        y = part[[f"y_{c}" for c in C]].values.astype(int)
        report[subset] = {}
        for method in ("whole", "region"):
            p = part[[f"{method}_{c}" for c in C]].values
            report[subset][method] = {
                "overall": overall_metrics(p, y, th),
                "per_concern": per_concern_metrics(p, y, th).to_dict(orient="index"),
            }
    report["faces_found"] = {name: int(df[df.dataset == ds].face_found.sum())
                             for name, ds in (("full faces", "skin_type"), ("close-ups", "skin_condition"),
                                              ("roboflow", "skin_problem"))}
    return report


def summary_table(report: dict) -> pd.DataFrame:
    rows = []
    for subset, methods in report.items():
        if subset == "faces_found":
            continue
        for method, res in methods.items():
            o = res["overall"]
            rows.append({"photos": subset, "method": method, "n": o["photos"], "macro F1": o["macro_f1"],
                         "exact match": o["exact_match"], "clear false alarms": o["clear_false_alarm_rate"],
                         **{f"F1 {c}": res["per_concern"][c]["f1"] for c in C}})
    return pd.DataFrame(rows)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=MODELS_DIR / "skin_concerns" / "model.pt")
    p.add_argument("--labels", type=Path, default=DATA_DIR / "labels" / "multilabel_labels.csv")
    p.add_argument("--images", type=Path, default=DATA_DIR / "processed")
    p.add_argument("--out", type=Path, default=None, help="default: next to the model")
    p.add_argument("--include-unsure", action="store_true")
    args = p.parse_args(argv)

    clf = SkinClassifier.load(args.model)
    if not clf.multilabel:
        raise SystemExit("Needs a multi-label checkpoint from train_multilabel.py")
    report = evaluate(clf, pd.read_csv(args.labels), args.images, args.include_unsure)
    out = args.out or args.model.parent
    (out / "region_evaluation.json").write_text(json.dumps(report, indent=2, default=float))
    table = summary_table(report)
    table.to_csv(out / "region_evaluation.csv", index=False)
    print(table.to_string(index=False))
    print("faces found:", report["faces_found"])
    return report


if __name__ == "__main__":
    main()
