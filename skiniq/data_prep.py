"""Image dataset preparation (proposal 3.5): duplicate removal and a leakage-free
70/15/15 split.

Both Kaggle datasets were exported from Roboflow, which saves several augmented
copies of each original photo (``<original>_jpg.rf.<hash>.jpg``). Splitting files
at random puts copies of the same photo in train and test and inflates test
accuracy, so files are grouped by original photo and whole groups are assigned
to a split.

Grouping links two files when they share an original-photo name (across every
class of a dataset) or have byte-identical content. A false link only forces
two files into the same split, so linking generously is safe.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

_ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+\.(jpe?g|png)$", re.IGNORECASE)
_COPY_MARKER = re.compile(r"(-Copy)+", re.IGNORECASE)


def source_name(filename: str) -> str:
    """Return the original-photo name a Roboflow export file was made from.

    >>> source_name("oily_157e4d-Copy_jpg.rf.14a6b4.jpg")
    'oily_157e4d_jpg'
    """
    name = _ROBOFLOW_SUFFIX.sub("", filename)
    if name == filename:  # not a Roboflow export: drop the extension instead
        name = Path(filename).stem
    return _COPY_MARKER.sub("", name)


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


@dataclass
class ImageRecord:
    path: Path
    label: str
    md5: str
    source: str = ""
    group: int = -1
    split: str = ""

    def __post_init__(self) -> None:
        if not self.source:
            self.source = source_name(self.path.name)


@dataclass
class PrepResult:
    kept: list[ImageRecord]
    removed: list[tuple[ImageRecord, str]] = field(default_factory=list)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def assign_groups(records: list[ImageRecord]) -> None:
    """Set ``record.group`` so linked files (same source name or same bytes) share an id."""
    uf = _UnionFind(len(records))
    first_by_key: dict[tuple[str, str], int] = {}
    for i, r in enumerate(records):
        for key in (("source", r.source), ("md5", r.md5)):
            if key in first_by_key:
                uf.union(i, first_by_key[key])
            else:
                first_by_key[key] = i
    roots: dict[int, int] = {}
    for i, r in enumerate(records):
        r.group = roots.setdefault(uf.find(i), len(roots))


def remove_duplicates(records: list[ImageRecord], single_label: bool) -> PrepResult:
    """Drop byte-identical copies and, for single-label tasks, ambiguous photos.

    - Same bytes, same label: keep the first copy.
    - Same bytes, different labels: drop every copy (the label is unknown).
    - ``single_label`` (skin type): a group whose files carry more than one label
      is the same photo labelled two ways, so the whole group is dropped.
      Skin conditions are not single-label (a face can show acne and blackheads),
      so mixed-label groups are kept there; they still share one split.
    """
    removed: list[tuple[ImageRecord, str]] = []

    labels_by_md5: dict[str, set[str]] = defaultdict(set)
    for r in records:
        labels_by_md5[r.md5].add(r.label)

    seen: set[str] = set()
    survivors: list[ImageRecord] = []
    for r in records:
        if len(labels_by_md5[r.md5]) > 1:
            removed.append((r, "identical image under different labels"))
        elif r.md5 in seen:
            removed.append((r, "exact duplicate"))
        else:
            seen.add(r.md5)
            survivors.append(r)

    if single_label:
        labels_by_group: dict[int, set[str]] = defaultdict(set)
        for r in survivors:
            labels_by_group[r.group].add(r.label)
        kept = []
        for r in survivors:
            if len(labels_by_group[r.group]) > 1:
                removed.append((r, "same photo under different labels"))
            else:
                kept.append(r)
        survivors = kept

    return PrepResult(kept=survivors, removed=removed)


def assign_splits(
    records: list[ImageRecord],
    ratios: dict[str, float],
    seed: int = 42,
) -> None:
    """Assign each group to one split, keeping every class close to ``ratios``.

    Groups are placed largest first (after a seeded shuffle) into the split
    that is furthest below its target image count for the group's classes.
    """
    groups: dict[int, list[ImageRecord]] = defaultdict(list)
    for r in records:
        groups[r.group].append(r)

    class_totals = Counter(r.label for r in records)
    target = {s: {c: n * frac for c, n in class_totals.items()} for s, frac in ratios.items()}
    current = {s: Counter() for s in ratios}

    order = list(groups)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda g: len(groups[g]), reverse=True)  # stable: ties stay shuffled

    for g in order:
        counts = Counter(r.label for r in groups[g])

        def deficit(split: str) -> float:
            return sum(
                (target[split][c] - current[split][c]) / class_totals[c] * n
                for c, n in counts.items()
            )

        best = max(ratios, key=deficit)
        current[best].update(counts)
        for r in groups[g]:
            r.split = best


def scan_folder(class_dirs: dict[str, list[Path]]) -> list[ImageRecord]:
    """Build records from ``{label: [folders...]}``; every image in the folders gets that label."""
    records = []
    for label, folders in class_dirs.items():
        for folder in folders:
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in IMAGE_SUFFIXES:
                    records.append(ImageRecord(path=p, label=label, md5=file_md5(p)))
    return records


def prepare(
    class_dirs: dict[str, list[Path]],
    ratios: dict[str, float],
    single_label: bool,
    seed: int = 42,
) -> PrepResult:
    records = scan_folder(class_dirs)
    assign_groups(records)
    result = remove_duplicates(records, single_label=single_label)
    assign_splits(result.kept, ratios, seed=seed)
    return result


# ---- one image per photo (no augmented copies, no duplicates) ----------------------------------

def border_darkness(gray) -> float:
    """Share of near-black pixels in the outer 5% frame of a greyscale array (0-255).

    Rotated or shifted augmented copies are padded with black, so the copy with the least
    dark border is the one closest to the original photo.
    """
    import numpy as np

    a = np.asarray(gray, dtype=float)
    h, w = a.shape[:2]
    bh, bw = max(1, h // 20), max(1, w // 20)
    frame = np.concatenate([a[:bh].ravel(), a[-bh:].ravel(), a[:, :bw].ravel(), a[:, -bw:].ravel()])
    return float((frame < 12).mean())


def photo_clusters(groups: list, embeddings, same_photo: float, duplicate: float) -> list[int]:
    """Cluster image files into photos.

    ``groups[i]`` is the name-based group of file i (files sharing an original-photo name) and
    ``embeddings[i]`` its unit-length image embedding. Within a name group, files are only kept
    together when they look alike (cosine >= ``same_photo``): generic names such as "3.jpg" can
    link unrelated photos. Across groups, files that are near-identical (cosine >= ``duplicate``)
    are merged: the same photo saved under two names or in two datasets.
    Returns a cluster id per file.
    """
    import numpy as np

    e = np.asarray(embeddings, dtype=np.float32)
    uf = _UnionFind(len(groups))
    members: dict = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    for idx in members.values():
        if len(idx) > 1:
            sim = e[idx] @ e[idx].T
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    if sim[a, b] >= same_photo:
                        uf.union(idx[a], idx[b])
    for start in range(0, len(e), 2048):
        sim = e[start:start + 2048] @ e.T
        for a, b in zip(*np.nonzero(sim >= duplicate)):
            i = start + int(a)
            if i < b:
                uf.union(i, int(b))
    roots: dict[int, int] = {}
    return [roots.setdefault(uf.find(i), len(roots)) for i in range(len(groups))]
