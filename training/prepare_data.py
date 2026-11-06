"""Prepare the skin type and skin condition image datasets (proposal 3.5).

Removes duplicates, re-splits 70/15/15 by original photo (no leakage), and writes
an image-folder layout that Hugging Face `datasets` loads with `image` and `label`
columns:

    data/processed/<task>/{train,val,test}/<label>/<file>.jpg
    data/processed/<task>/manifest.csv      one row per kept image
    data/processed/<task>/removed.csv       one row per dropped image, with reason
    training/data_report.md                 class counts per split (for the report)

Usage (from the repo root):
    python training/prepare_data.py
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skiniq.config import DATA_DIR, DATA_SPLIT, ROOT_DIR, SKIN_CONDITIONS, SKIN_TYPES  # noqa: E402
from skiniq.data_prep import PrepResult, prepare  # noqa: E402

SPLITS = {"train": DATA_SPLIT["train"], "val": DATA_SPLIT["val"], "test": DATA_SPLIT["test"]}
LOW_RES = 256  # images smaller than this on either side are flagged

# Dataset folder name -> label in skiniq.config
CONDITION_FOLDERS = {
    "acne": "acne",
    "blackheades": "blackheads",  # misspelt in the source dataset
    "pigmentation": "hyperpigmentation",
    "redness": "redness",
    "pores": "enlarged_pores",
    "wrinkles": "wrinkles",
}


def skin_type_dirs(raw: Path) -> dict[str, list[Path]]:
    # The dataset's own train/valid/test split leaks, so all three are pooled and re-split.
    return {t: [raw / "skin_type" / s / t for s in ("train", "valid", "test")] for t in SKIN_TYPES}


def skin_condition_dirs(raw: Path) -> dict[str, list[Path]]:
    assert set(CONDITION_FOLDERS.values()) == set(SKIN_CONDITIONS)
    return {label: [raw / "skin_condition" / folder] for folder, label in CONDITION_FOLDERS.items()}


def _place(src: Path, dst: Path) -> None:
    """Hard-link when possible (no extra disk space), otherwise copy."""
    if dst.exists():
        raise FileExistsError(f"Two images map to {dst}")
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_task(task: str, result: PrepResult, out_root: Path, raw: Path) -> dict:
    """CSV paths are relative to the parent of ``raw`` (e.g. ``raw/skin_type/...``)."""
    base = raw.resolve().parent
    out = out_root / task
    if out.exists():
        shutil.rmtree(out)

    rows = []
    for r in result.kept:
        dst = out / r.split / r.label / r.path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        _place(r.path, dst)
        with Image.open(r.path) as im:
            w, h = im.size
        rows.append({
            "path": dst.relative_to(out_root).as_posix(),
            "label": r.label,
            "split": r.split,
            "group": r.group,
            "source": r.source,
            "width": w,
            "height": h,
            "low_res": int(min(w, h) < LOW_RES),
            "original": r.path.resolve().relative_to(base).as_posix(),
            "md5": r.md5,
        })

    with open(out / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with open(out / "removed.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["original", "label", "reason"])
        for r, reason in result.removed:
            writer.writerow([r.path.resolve().relative_to(base).as_posix(), r.label, reason])

    return {
        "rows": rows,
        "removed": Counter(reason for _, reason in result.removed),
        "groups": len({r.group for r in result.kept}),
    }


def check_no_leakage(rows: list[dict]) -> None:
    split_by_group: dict[int, set[str]] = {}
    for row in rows:
        split_by_group.setdefault(row["group"], set()).add(row["split"])
    leaking = [g for g, s in split_by_group.items() if len(s) > 1]
    if leaking:
        raise AssertionError(f"{len(leaking)} photo groups appear in more than one split")


def report_section(title: str, labels: tuple[str, ...], stats: dict, scanned: int) -> str:
    rows = stats["rows"]
    counts = Counter((r["label"], r["split"]) for r in rows)
    low = Counter(r["label"] for r in rows if r["low_res"])
    lines = [
        f"## {title}",
        "",
        f"- Images scanned: {scanned}",
        f"- Images kept: {len(rows)} from {stats['groups']} original-photo groups",
    ]
    for reason, n in sorted(stats["removed"].items()):
        lines.append(f"- Removed ({reason}): {n}")
    lines += [
        "",
        "| Class | Train | Val | Test | Total | Share | Low-res (<256px) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in labels:
        tr, va, te = (counts[(c, s)] for s in ("train", "val", "test"))
        tot = tr + va + te
        lines.append(
            f"| {c} | {tr} | {va} | {te} | {tot} | {tot / len(rows):.1%} | {low[c]} |"
        )
    totals = [sum(counts[(c, s)] for c in labels) for s in ("train", "val", "test")]
    lines.append(
        f"| **Total** | {totals[0]} | {totals[1]} | {totals[2]} | {len(rows)} | 100% | {sum(low.values())} |"
    )
    lines.append("")
    lines.append(
        "Split share: "
        + ", ".join(f"{s} {t / len(rows):.1%}" for s, t in zip(("train", "val", "test"), totals))
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=DATA_DIR / "raw")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "processed")
    parser.add_argument("--report", type=Path, default=ROOT_DIR / "training" / "data_report.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = [
        ("skin_type", "Skin type (olimov)", SKIN_TYPES, skin_type_dirs(args.raw), True),
        ("skin_condition", "Skin condition (drishya)", SKIN_CONDITIONS, skin_condition_dirs(args.raw), False),
    ]

    sections = [
        "# Image dataset preparation report",
        "",
        "Generated by `training/prepare_data.py` (seed "
        f"{args.seed}). Split 70/15/15 by original photo: every augmented copy of a photo "
        "is in the same split, so no test image has a near-copy in training.",
        "",
    ]
    for task, title, labels, dirs, single_label in tasks:
        missing = [str(d) for ds in dirs.values() for d in ds if not d.is_dir()]
        if missing:
            sys.exit(f"Missing dataset folders (see data/README.md):\n  " + "\n  ".join(missing))
        result = prepare(dirs, SPLITS, single_label=single_label, seed=args.seed)
        stats = write_task(task, result, args.out, args.raw)
        check_no_leakage(stats["rows"])
        scanned = len(result.kept) + len(result.removed)
        sections.append(report_section(title, labels, stats, scanned))
        print(f"{task}: kept {len(result.kept)}/{scanned} -> {args.out / task}")

    args.report.write_text("\n".join(sections))
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
