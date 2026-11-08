"""Build the multi-label annotation queue: one row per original photo, in priority order.

Copies of the same original photo (Roboflow augmentations) share labels, so only one
representative file per photo group is annotated. The queue starts with a stratified
test sample, then a validation sample, then training photos, so annotation can stop
at any point and still leave a fully labelled test set.

Close-ups from the skin condition dataset start with their folder label(s) ticked,
Roboflow images with their boxed concerns ticked; full faces from the skin type
dataset start with nothing ticked.

    python training/annotation_queue.py      # -> data/labels/annotation_queue.csv
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skiniq.config import DATA_DIR, SKIN_CONDITIONS  # noqa: E402

DATASETS = ("skin_type", "skin_condition", "skin_problem")
PRIORITY_SAMPLE = {  # (split, dataset) -> number of photos annotated first
    ("test", "skin_type"): 100,
    ("test", "skin_condition"): 100,
    ("test", "skin_problem"): 100,
    ("val", "skin_type"): 67,
    ("val", "skin_condition"): 67,
    ("val", "skin_problem"): 66,
}
TIERS = ["test sample", "val sample", "train", "test rest", "val rest"]


def load_groups(processed: Path) -> pd.DataFrame:
    """One row per original-photo group with a representative image and prefilled labels."""
    rows = []
    for dataset in ("skin_type", "skin_condition"):
        m = pd.read_csv(processed / dataset / "manifest.csv")
        m["is_copy"] = m["original"].str.contains("-Copy", case=False)
        for group, g in m.groupby("group"):
            rep = g.sort_values(["is_copy", "low_res", "path"]).iloc[0]  # prefer a non-copy, full-res file
            prefill = sorted(set(g["label"])) if dataset == "skin_condition" else []
            rows.append({
                "key": f"{dataset}:{group}",
                "dataset": dataset,
                "group": group,
                "split": rep["split"],
                "folder_labels": "|".join(sorted(set(g["label"]))),
                "prefill": "|".join(prefill),
                "image": rep["path"],
                "md5": rep["md5"],
                "copies": len(g),
            })
    roboflow = processed / "skin_problem" / "manifest.csv"
    if roboflow.exists():
        m = pd.read_csv(roboflow)
        for group, g in m.groupby("group"):
            rep = g.sort_values("path").iloc[0]
            concerns = sorted(set("|".join(g["concerns"]).split("|")))
            rows.append({
                "key": f"skin_problem:{group}", "dataset": "skin_problem", "group": group, "split": rep["split"],
                "folder_labels": "|".join(concerns), "prefill": "|".join(concerns),
                "image": rep["path"], "md5": rep["md5"], "copies": len(g),
            })
    return pd.DataFrame(rows)


def _stratified(df: pd.DataFrame, n: int, rng: random.Random) -> list[str]:
    """Pick n keys spread evenly over folder labels (round-robin)."""
    pools = {lab: rng.sample(list(g["key"]), len(g)) for lab, g in df.groupby("folder_labels")}
    picked: list[str] = []
    while len(picked) < min(n, len(df)):
        for lab in sorted(pools):
            if pools[lab] and len(picked) < n:
                picked.append(pools[lab].pop())
    return picked


def _interleave(*lists: list[str]) -> list[str]:
    out = []
    for i in range(max(map(len, lists), default=0)):
        out += [lst[i] for lst in lists if i < len(lst)]
    return out


def build_queue(groups: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    order: list[str] = []
    tier_of: dict[str, str] = {}
    for split in ("test", "val"):
        sampled = [
            _stratified(groups[(groups.split == split) & (groups.dataset == ds)], PRIORITY_SAMPLE.get((split, ds), 0), rng)
            for ds in DATASETS
        ]
        for k in _interleave(*sampled):
            order.append(k)
            tier_of[k] = f"{split} sample"
    chosen = set(order)
    for split, tier in (("train", "train"), ("test", "test rest"), ("val", "val rest")):
        rest = [
            _stratified(groups[(groups.split == split) & (groups.dataset == ds) & ~groups.key.isin(chosen)],
                        len(groups), rng)
            for ds in DATASETS
        ]
        for k in _interleave(*rest):
            order.append(k)
            tier_of[k] = tier
    q = groups.set_index("key").loc[order].reset_index()
    q.insert(0, "order", range(1, len(q) + 1))
    q.insert(1, "tier", q["key"].map(tier_of))
    return q


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed", type=Path, default=DATA_DIR / "processed")
    p.add_argument("--out", type=Path, default=DATA_DIR / "labels" / "annotation_queue.csv")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    queue = build_queue(load_groups(args.processed), args.seed)
    assert set("|".join(queue.prefill).split("|")) - {""} <= set(SKIN_CONDITIONS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(args.out, index=False)
    print(f"{len(queue)} photos -> {args.out}")
    print(queue.groupby(["tier", "dataset"], sort=False).size().to_string())


if __name__ == "__main__":
    main()
