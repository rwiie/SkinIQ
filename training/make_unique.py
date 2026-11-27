"""Keep one image per photo: no augmented copies and no duplicates (skin type + skin condition).

The prepared datasets (training/prepare_data.py) keep every augmented copy of a photo together in
one split. This step keeps a single image per photo, using the photo list in data/labels:

- ``unique_photos.csv``: one row per photo (5,813): the image kept, its dataset, class, concern
  folders, split and whether it shows a full face.
- ``photo_files.csv``: every prepared image file and the photo it belongs to. Copies were found by
  exact bytes, by shared original names confirmed with keypoint matching, and by keypoint matches
  across names and datasets (robust to rotation, flipping and cropping). Roboflow is not included.

Every photo gets its own key (``<dataset>:<group>``). A name group that turned out to hold several
different photos keeps its group id for the photo with a labelled image, the others get new ids
from NEW_GROUP_OFFSET. Label files are remapped to these keys through their ``image`` column, so a
label always follows the picture that was labelled.

    python training/make_unique.py   # data/processed -> data/processed_unique + unique_labels.csv

Writes data/processed_unique/<dataset>/manifest.csv with the image folders (hard links), and
data/labels/unique_labels.csv (all label files merged, later files win, keys remapped).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skiniq.config import DATA_DIR  # noqa: E402

TASKS = ("skin_type", "skin_condition")
NEW_GROUP_OFFSET = 100_000  # ids for photos split off a name group
LABEL_FILES = ("multilabel_labels.csv", "fullface_claude_labels.csv")  # later files win


def assign_keys(photos: pd.DataFrame, labelled_photo_ids: set[int]) -> pd.DataFrame:
    """One unique (dataset, group) per photo; a labelled photo keeps its original group id."""
    photos = photos.copy()
    photos["labelled"] = photos.photo_id.isin(labelled_photo_ids)
    order = photos.sort_values(["labelled", "images", "photo_id"], ascending=[False, False, True])
    seen, new_id, groups = set(), NEW_GROUP_OFFSET, {}
    for r in order.itertuples():
        k = (r.dataset, r.group)
        if k in seen:
            groups[r.photo_id] = new_id
            new_id += 1
        else:
            seen.add(k)
            groups[r.photo_id] = r.group
    photos["group"] = photos.photo_id.map(groups)
    photos["key"] = photos.dataset + ":" + photos.group.astype(str)
    return photos.drop(columns="labelled")


def merge_labels(label_files: list[Path], files: pd.DataFrame, photos: pd.DataFrame) -> pd.DataFrame:
    """All label files in order (later wins), each row moved to the key of the photo it labels."""
    image_to_photo = dict(zip(files.path, files.photo_id))
    rows = []
    for f in label_files:
        if f.exists():
            lab = pd.read_csv(f)
            lab["photo_id"] = lab.image.map(image_to_photo)
            rows.append(lab[lab.photo_id.notna()])  # Roboflow rows have no photo here
    if not rows:
        return pd.DataFrame()
    lab = pd.concat(rows, ignore_index=True).drop_duplicates("photo_id", keep="last")
    p = photos.set_index("photo_id")
    lab["photo_id"] = lab.photo_id.astype(int)
    lab["key"] = lab.photo_id.map(p.key)
    lab["dataset"] = lab.photo_id.map(p.dataset)
    lab["split"] = lab.photo_id.map(p.split)  # the photo's split decides
    lab["image"] = lab.photo_id.map(p.image)
    return lab


def write(photos: pd.DataFrame, data_dir: Path, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    for task in TASKS:
        rows = []
        for r in photos[photos.dataset == task].itertuples():
            rel = Path(task) / r.split / r.label / Path(r.image).name
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(data_dir / r.image, dst)
            except OSError:
                shutil.copy2(data_dir / r.image, dst)
            labels = r.folder_labels if isinstance(r.folder_labels, str) and r.folder_labels else r.label
            rows.append({"path": rel.as_posix(), "label": r.label, "labels": labels, "split": r.split,
                         "group": r.group, "photo_id": r.photo_id, "full_face": r.full_face, "md5": r.md5,
                         "copies": r.images})
        (out / task).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / task / "manifest.csv", index=False)


def main(argv=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR / "processed")
    p.add_argument("--out", type=Path, default=DATA_DIR / "processed_unique")
    p.add_argument("--photos", type=Path, default=DATA_DIR / "labels" / "unique_photos.csv")
    p.add_argument("--files", type=Path, default=DATA_DIR / "labels" / "photo_files.csv")
    p.add_argument("--labels", type=Path, nargs="*", default=[DATA_DIR / "labels" / f for f in LABEL_FILES])
    p.add_argument("--labels-out", type=Path, default=DATA_DIR / "labels" / "unique_labels.csv")
    args = p.parse_args(argv)

    photos = pd.read_csv(args.photos)
    files = pd.read_csv(args.files)
    image_to_photo = dict(zip(files.path, files.photo_id))
    labelled = set()
    for f in args.labels:
        if f.exists():
            labelled |= {int(x) for x in pd.read_csv(f).image.map(image_to_photo).dropna()}
    photos = assign_keys(photos, labelled)
    labels = merge_labels(args.labels, files, photos)
    write(photos, args.data_dir, args.out)
    labels.drop(columns="photo_id").to_csv(args.labels_out, index=False)
    print(f"{len(photos)} photos -> {args.out}; {len(labels)} labelled photos -> {args.labels_out}")
    return photos, labels


if __name__ == "__main__":
    main()
