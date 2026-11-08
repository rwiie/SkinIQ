"""Prepare the Roboflow "Skin Problem Detection" dataset (COCO object detection) as
multi-label skin concern data.

Source: Skin-Problem-Detection-Relabel-Clean3 v1, Roboflow Universe
(parin-kittipongdaja), CC BY 4.0. 3,240 images (640x640), 10 box classes.

Steps
1. Convert bounding boxes to six image-level labels: a concern is 1 if the image has
   at least one box of a class mapped to it (Acne and Whiteheads -> acne, Dark-Spots ->
   hyperpigmentation, Skin-Redness -> redness, Englarged-Pores -> enlarged_pores).
   Dry-Skin, Oily-Skin and Eyebags are not skin concerns in SkinIQ and are ignored.
2. Drop images with no box for any of the six concerns. These come from sub-datasets
   that only labelled dry/oily skin or eye bags, so a missing box doesn't mean the
   concern is absent (some aren't faces at all).
3. Re-split 70/15/15 by original photo (Roboflow file name or identical bytes), so no
   photo has copies in two splits, stratified by each image's rarest concern.

Missing boxes are not trustworthy negatives (the dataset merges sub-datasets that each
labelled their own concerns); train_multilabel.py treats them as weak negatives.

    python training/prepare_roboflow.py   # data/raw/skin_problem -> data/processed/skin_problem
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skiniq.config import DATA_DIR, DATA_SPLIT, SKIN_CONDITIONS  # noqa: E402
from skiniq.data_prep import ImageRecord, assign_groups, assign_splits, file_md5  # noqa: E402

CLASS_TO_CONCERN = {
    "Acne": "acne",
    "Whiteheads": "acne",
    "Blackheads": "blackheads",
    "Dark-Spots": "hyperpigmentation",
    "Skin-Redness": "redness",
    "Englarged-Pores": "enlarged_pores",
    "Wrinkles": "wrinkles",
}
IGNORED_CLASSES = {"Dry-Skin", "Oily-Skin", "Eyebags", "Acne-Blackhead-Wrinkles-f6HR-vJCz-Vyox"}
SPLITS = {"train": DATA_SPLIT["train"], "val": DATA_SPLIT["val"], "test": DATA_SPLIT["test"]}


def read_coco(raw: Path) -> pd.DataFrame:
    """One row per image across the original train/valid/test folders."""
    rows = []
    for folder in ("train", "valid", "test"):
        ann_file = raw / folder / "_annotations.coco.json"
        if not ann_file.exists():
            continue
        coco = json.loads(ann_file.read_text())
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        unknown = set(cats.values()) - set(CLASS_TO_CONCERN) - IGNORED_CLASSES
        if unknown:
            raise ValueError(f"Unmapped Roboflow classes: {sorted(unknown)}")
        boxes: dict[int, list] = {}
        for a in coco["annotations"]:
            boxes.setdefault(a["image_id"], []).append((cats[a["category_id"]], [round(v, 1) for v in a["bbox"]]))
        for img in coco["images"]:
            b = boxes.get(img["id"], [])
            concerns = sorted({CLASS_TO_CONCERN[c] for c, _ in b if c in CLASS_TO_CONCERN})
            rows.append({
                "original_split": folder,
                "file": img["file_name"],
                "path": raw / folder / img["file_name"],
                "concerns": concerns,
                "classes": sorted({c for c, _ in b}),
                "boxes": [{"class": c, "concern": CLASS_TO_CONCERN.get(c), "bbox": bb} for c, bb in b],
            })
    if not rows:
        raise SystemExit(f"No COCO annotations found under {raw}")
    return pd.DataFrame(rows)


def stratify_label(concerns: list[str], counts: Counter) -> str:
    """The image's rarest concern, so rare concerns are spread over all splits."""
    return min(concerns, key=lambda c: (counts[c], c))


def prepare(raw: Path, out: Path, seed: int = 42) -> tuple[pd.DataFrame, Counter]:
    images = read_coco(raw)
    dropped = Counter()
    keep = images[images.concerns.str.len() > 0].copy()
    dropped["no SkinIQ concern boxed (only dry/oily skin, eye bags or none)"] = len(images) - len(keep)

    counts = Counter(c for cs in keep.concerns for c in cs)
    records = [ImageRecord(path=p, label=stratify_label(cs, counts), md5=file_md5(p))
               for p, cs in zip(keep.path, keep.concerns)]
    assign_groups(records)

    # Byte-identical copies: keep one (labels are merged, since a box in either copy is real)
    by_md5: dict[str, int] = {}
    keep = keep.reset_index(drop=True)
    merged_concerns: dict[int, set] = {}
    survivors = []
    for i, r in enumerate(records):
        if r.md5 in by_md5:
            merged_concerns[by_md5[r.md5]] |= set(keep.at[i, "concerns"])
            dropped["exact duplicate"] += 1
            continue
        by_md5[r.md5] = i
        merged_concerns[i] = set(keep.at[i, "concerns"])
        survivors.append(i)
    keep = keep.loc[survivors].copy()
    keep["concerns"] = [sorted(merged_concerns[i]) for i in survivors]
    records = [records[i] for i in survivors]
    assign_splits(records, SPLITS, seed=seed)

    keep["split"] = [r.split for r in records]
    keep["group"] = [r.group for r in records]
    keep["source"] = [r.source for r in records]
    keep["md5"] = [r.md5 for r in records]

    if out.exists():
        shutil.rmtree(out)
    rel_paths = []
    for row in keep.itertuples():
        dst = out / row.split / row.file
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise FileExistsError(dst)
        try:
            os.link(row.path, dst)
        except OSError:
            shutil.copy2(row.path, dst)
        rel_paths.append(dst.relative_to(out.parent).as_posix())
    keep["path"] = rel_paths

    manifest = pd.DataFrame({
        "path": keep.path, "split": keep.split, "group": keep.group, "source": keep.source, "md5": keep.md5,
        "concerns": keep.concerns.str.join("|"), "classes": keep.classes.str.join("|"),
        "original_split": keep.original_split, "boxes": keep.boxes.apply(json.dumps),
    })
    manifest.to_csv(out / "manifest.csv", index=False)
    return manifest, dropped


def report(manifest: pd.DataFrame, dropped: Counter, scanned: int) -> str:
    lines = [
        "## Skin problem detection (Roboflow, multi-label)",
        "",
        f"- Images scanned: {scanned}",
        f"- Images kept: {len(manifest)} from {manifest.group.nunique()} original-photo groups",
    ]
    lines += [f"- Removed ({reason}): {n}" for reason, n in dropped.items()]
    lines += ["", "| Concern | Train | Val | Test | Total |", "|---|---:|---:|---:|---:|"]
    for c in SKIN_CONDITIONS:
        has = manifest[manifest.concerns.str.split("|").apply(lambda cs: c in cs)]
        n = has.split.value_counts()
        lines.append(f"| {c} | {n.get('train', 0)} | {n.get('val', 0)} | {n.get('test', 0)} | {len(has)} |")
    sizes = manifest.split.value_counts()
    multi = (manifest.concerns.str.count(r"\|") >= 1).sum()
    lines += ["", f"Images: train {sizes.get('train', 0)}, val {sizes.get('val', 0)}, test {sizes.get('test', 0)}; "
                  f"{multi} images have two or more concerns.", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=DATA_DIR / "raw" / "skin_problem")
    p.add_argument("--out", type=Path, default=DATA_DIR / "processed" / "skin_problem")
    p.add_argument("--report", type=Path, default=Path(__file__).resolve().parent / "data_report_roboflow.md")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    manifest, dropped = prepare(args.raw, args.out, args.seed)
    scanned = len(manifest) + sum(dropped.values())
    # leakage check: each photo group in exactly one split
    assert manifest.groupby("group").split.nunique().max() == 1
    args.report.write_text("# Roboflow dataset preparation report\n\n" + report(manifest, dropped, scanned))
    print(f"skin_problem: kept {len(manifest)}/{scanned} -> {args.out}")
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
