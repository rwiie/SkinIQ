"""Keep one image per photo: no augmented copies and no duplicates (skin type + skin condition).

The prepared datasets (training/prepare_data.py) keep every augmented copy of a photo
(``<photo>_jpg.rf.<hash>.jpg``) together in one split. This step goes further and keeps a single
image per photo:

1. Files sharing an original-photo name stay together only if they look alike (CLIP image
   embedding, cosine >= --same-photo): generic names such as "3.jpg" link unrelated photos.
2. Near-identical files are merged across names and across the two datasets
   (cosine >= --duplicate): the same photo saved twice, or reused by another dataset.
3. Per photo, one image is kept: a labelled one if any (so data/labels/*.csv stay valid, test
   and validation photos first), otherwise a skin condition copy (it carries a concern label),
   then the copy with the least black padding (augmented rotations and shifts add black
   borders), then the largest.

A skin condition photo keeps every concern folder it appeared in (``labels`` column). The
Roboflow dataset is not included.

    python training/make_unique.py   # data/processed -> data/processed_unique

Writes data/processed_unique/<task>/{manifest,removed}.csv, the image folders (hard links) and
training/data_report_unique.md.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skiniq.config import DATA_DIR, ROOT_DIR, SKIN_CONDITIONS, SKIN_TYPES  # noqa: E402
from skiniq.data_prep import border_darkness, photo_clusters  # noqa: E402

TASKS = ("skin_type", "skin_condition")
SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}
NEW_GROUP_OFFSET = 100_000  # ids for photos split off a name group


def load(data_dir: Path) -> pd.DataFrame:
    frames = []
    for task in TASKS:
        m = pd.read_csv(data_dir / task / "manifest.csv")
        m["task"] = task
        frames.append(m)
    m = pd.concat(frames, ignore_index=True)
    m["key"] = m.task + ":" + m.group.astype(str)
    return m


def embed(m: pd.DataFrame, data_dir: Path, cache: Path | None) -> np.ndarray:
    if cache and cache.exists():
        e = np.load(cache)
        if len(e) == len(m):
            return e
    from skiniq.vision.face_models import clip_embedding

    e = np.zeros((len(m), 512), np.float32)
    for i, p in enumerate(m.path):
        e[i] = clip_embedding(np.asarray(Image.open(data_dir / p).convert("RGB")))
        if i % 1000 == 0:
            print(f"  embedded {i}/{len(m)}", flush=True)
    if cache:
        np.save(cache, e)
    return e


def labelled(label_files: list[Path]) -> tuple[set[str], set[str]]:
    """Keys and image paths that carry the author's labels."""
    keys, images = set(), set()
    for f in label_files:
        if f.exists():
            lab = pd.read_csv(f)
            keys |= set(lab.key)
            images |= set(lab.image.dropna())
    return keys, images


def choose(m: pd.DataFrame, clusters: list[int], keys: set[str], images: set[str], data_dir: Path) -> pd.DataFrame:
    """One row per photo, with the kept file, its group id and all its labels."""
    m = m.assign(cluster=clusters)
    m["is_labelled_image"] = m.path.isin(images)
    m["is_labelled_key"] = m.key.isin(keys)
    m["split_rank"] = m.split.map(SPLIT_PRIORITY)
    m["border"] = [border_darkness(np.asarray(Image.open(data_dir / p).convert("L"))) for p in m.path]
    m["pixels"] = m.width * m.height
    m["has_concern_label"] = m.task == "skin_condition"  # keeps a concern label for training
    order = m.sort_values(["is_labelled_image", "is_labelled_key", "split_rank", "has_concern_label", "border",
                           "pixels", "path"], ascending=[False, False, True, False, True, False, True])
    kept = order.drop_duplicates("cluster").copy()

    # A name group split into several photos: the labelled (or largest) part keeps the group id,
    # so label files still match; the others get new ids.
    size = m.groupby("cluster").size()
    kept["cluster_size"] = kept.cluster.map(size)
    kept = kept.sort_values(["is_labelled_image", "is_labelled_key", "cluster_size"], ascending=False)
    seen, new_id, groups = set(), NEW_GROUP_OFFSET, []
    for r in kept.itertuples():
        if r.key in seen:
            groups.append(new_id)
            new_id += 1
        else:
            seen.add(r.key)
            groups.append(r.group)
    kept["group"] = groups
    kept["key"] = kept.task + ":" + kept.group.astype(str)

    # all labels of the photo within the kept file's dataset (a close-up can show several concerns)
    labels = m.groupby(["cluster", "task"]).label.agg(lambda s: "|".join(sorted(set(s))))
    kept["labels"] = [labels[(c, t)] for c, t in zip(kept.cluster, kept.task)]
    return kept


def write(kept: pd.DataFrame, m: pd.DataFrame, data_dir: Path, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    cols = ["path", "label", "labels", "split", "group", "source", "width", "height", "low_res",
            "original", "md5", "cluster_size"]
    for task in TASKS:
        rows = []
        for r in kept[kept.task == task].itertuples():
            rel = Path(task) / r.split / r.label / Path(r.path).name
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(data_dir / r.path, dst)
            except OSError:
                shutil.copy2(data_dir / r.path, dst)
            rows.append({**{c: getattr(r, c) for c in cols if c != "path"}, "path": rel.as_posix()})
        pd.DataFrame(rows)[cols].to_csv(out / task / "manifest.csv", index=False)

        keep_paths = set(kept.path)
        kept_by_cluster = kept.set_index("cluster").path
        dropped = m[(m.task == task) & ~m.path.isin(keep_paths)]
        reasons = []
        for r in dropped.itertuples():
            twin = kept_by_cluster[r.cluster]
            same_group = kept.loc[kept.cluster == r.cluster, "source"].iloc[0] == r.source
            reasons.append("augmented copy" if same_group else f"duplicate of {twin}")
        dropped.assign(reason=reasons)[["path", "label", "split", "reason"]].to_csv(out / task / "removed.csv",
                                                                                     index=False)


def report(kept: pd.DataFrame, m: pd.DataFrame, path: Path, same_photo: float, duplicate: float) -> None:
    lines = ["# One image per photo (no augmented copies or duplicates)", "",
             f"Generated by `training/make_unique.py` (CLIP cosine: same photo >= {same_photo}, "
             f"duplicate >= {duplicate}). Roboflow is not included.", "",
             "| Dataset | Files before | Photos kept | Removed |", "|---|---:|---:|---:|"]
    for task in TASKS:
        a, b = (m.task == task).sum(), (kept.task == task).sum()
        lines.append(f"| {task} | {a} | {b} | {a - b} |")
    lines.append(f"| **Total** | {len(m)} | {len(kept)} | {len(m) - len(kept)} |")
    cross = kept[kept.cluster.map(m.groupby("cluster").task.nunique()) > 1]
    lines += ["", f"Photos found in both datasets (one copy kept): {len(cross)}", ""]
    for task, labels in (("skin_type", SKIN_TYPES), ("skin_condition", SKIN_CONDITIONS)):
        k = kept[kept.task == task]
        lines += [f"## {task}", "", "| Class | Train | Val | Test | Total |", "|---|---:|---:|---:|---:|"]
        for c in labels:
            has = k[k.labels.str.split("|").map(lambda s, c=c: c in s)]
            n = Counter(has.split)
            lines.append(f"| {c} | {n['train']} | {n['val']} | {n['test']} | {len(has)} |")
        n = Counter(k.split)
        lines += [f"| **Photos** | {n['train']} | {n['val']} | {n['test']} | {len(k)} |", ""]
        if task == "skin_condition":
            lines += [f"Photos in more than one concern folder: {(k.labels.str.count('[|]') > 0).sum()}", ""]
    path.write_text("\n".join(lines))


def main(argv=None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR / "processed")
    p.add_argument("--out", type=Path, default=DATA_DIR / "processed_unique")
    p.add_argument("--labels", type=Path, nargs="*",
                   default=sorted((DATA_DIR / "labels").glob("*labels*.csv")))
    p.add_argument("--embeddings", type=Path, default=None, help="cache file for CLIP embeddings (.npy)")
    p.add_argument("--same-photo", type=float, default=0.80)
    p.add_argument("--duplicate", type=float, default=0.95)
    p.add_argument("--report", type=Path, default=ROOT_DIR / "training" / "data_report_unique.md")
    args = p.parse_args(argv)

    m = load(args.data_dir)
    e = embed(m, args.data_dir, args.embeddings)
    clusters = photo_clusters(list(zip(m.task, m.group)), e, args.same_photo, args.duplicate)
    keys, images = labelled(args.labels)
    kept = choose(m, clusters, keys, images, args.data_dir)
    m = m.assign(cluster=clusters)
    write(kept, m, args.data_dir, args.out)
    report(kept, m, args.report, args.same_photo, args.duplicate)
    lost = sorted((keys & set(m.key)) - set(kept.key))
    if lost:
        print(f"  {len(lost)} labelled photos were duplicates of another labelled photo and were dropped: {lost[:10]}")
    print(f"kept {len(kept)} of {len(m)} images -> {args.out}; report -> {args.report}")
    return kept


if __name__ == "__main__":
    main()
