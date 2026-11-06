from collections import Counter
from pathlib import Path

import pytest

from skiniq.data_prep import (
    ImageRecord,
    assign_groups,
    assign_splits,
    prepare,
    remove_duplicates,
    source_name,
)

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def rec(name: str, label: str, md5: str) -> ImageRecord:
    return ImageRecord(path=Path(name), label=label, md5=md5)


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("oily_157e4d_jpg.rf.14a6b48788ba75f6.jpg", "oily_157e4d_jpg"),
        ("oily_157e4d-Copy_jpg.rf.ad8b6fa89c7bd024.jpg", "oily_157e4d_jpg"),
        ("x-Copy-Copy_jpg.rf.0a.jpg", "x_jpg"),
        ("23_PNG_jpg.rf.4fbd18b4.jpg", "23_PNG_jpg"),
        ("plain_photo.jpg", "plain_photo"),
    ],
)
def test_source_name(filename, expected):
    assert source_name(filename) == expected


def test_augmented_copies_share_a_group():
    records = [
        rec("a_jpg.rf.111.jpg", "oily", "m1"),
        rec("a-Copy_jpg.rf.222.jpg", "oily", "m2"),
        rec("b_jpg.rf.333.jpg", "oily", "m3"),
    ]
    assign_groups(records)
    assert records[0].group == records[1].group != records[2].group


def test_identical_bytes_link_differently_named_files():
    records = [rec("a_jpg.rf.1.jpg", "dry", "same"), rec("zzz_jpg.rf.2.jpg", "dry", "same")]
    assign_groups(records)
    assert records[0].group == records[1].group


def test_exact_duplicate_same_label_keeps_one():
    records = [rec("a_jpg.rf.1.jpg", "dry", "same"), rec("b_jpg.rf.2.jpg", "dry", "same")]
    assign_groups(records)
    result = remove_duplicates(records, single_label=True)
    assert len(result.kept) == 1
    assert [reason for _, reason in result.removed] == ["exact duplicate"]


def test_identical_image_with_two_labels_is_dropped_entirely():
    records = [rec("a_jpg.rf.1.jpg", "dry", "same"), rec("b_jpg.rf.2.jpg", "oily", "same")]
    assign_groups(records)
    result = remove_duplicates(records, single_label=False)
    assert result.kept == []


def test_single_label_task_drops_photo_labelled_two_ways():
    # Same original photo (shared source name) filed under two skin types.
    records = [
        rec("p_jpg.rf.1.jpg", "dry", "m1"),
        rec("p_jpg.rf.2.jpg", "oily", "m2"),
        rec("q_jpg.rf.3.jpg", "oily", "m3"),
    ]
    assign_groups(records)
    result = remove_duplicates(records, single_label=True)
    assert [r.path.name for r in result.kept] == ["q_jpg.rf.3.jpg"]


def test_multi_label_task_keeps_photo_with_two_conditions():
    records = [rec("p_jpg.rf.1.jpg", "acne", "m1"), rec("p_jpg.rf.2.jpg", "blackheads", "m2")]
    assign_groups(records)
    result = remove_duplicates(records, single_label=False)
    assert len(result.kept) == 2


def make_dataset(n_groups_per_class: dict[str, int], copies: int = 3) -> list[ImageRecord]:
    records = []
    for label, n in n_groups_per_class.items():
        for g in range(n):
            for c in range(copies):
                records.append(rec(f"{label}{g}_jpg.rf.{c}.jpg", label, f"{label}-{g}-{c}"))
    assign_groups(records)
    return records


def test_groups_never_span_splits():
    records = make_dataset({"oily": 50, "dry": 30})
    assign_splits(records, RATIOS)
    splits_per_group = {}
    for r in records:
        splits_per_group.setdefault(r.group, set()).add(r.split)
    assert all(len(s) == 1 for s in splits_per_group.values())


def test_each_class_close_to_70_15_15():
    records = make_dataset({"oily": 200, "dry": 100, "combination": 40})
    assign_splits(records, RATIOS)
    for label in ("oily", "dry", "combination"):
        counts = Counter(r.split for r in records if r.label == label)
        total = sum(counts.values())
        for split, frac in RATIOS.items():
            assert counts[split] / total == pytest.approx(frac, abs=0.05), (label, split)


def test_split_is_reproducible_with_seed():
    a = make_dataset({"oily": 40, "dry": 40})
    b = make_dataset({"oily": 40, "dry": 40})
    assign_splits(a, RATIOS, seed=7)
    assign_splits(b, RATIOS, seed=7)
    assert [r.split for r in a] == [r.split for r in b]


def test_prepare_end_to_end(tmp_path):
    from PIL import Image

    for label in ("oily", "dry"):
        d = tmp_path / label
        d.mkdir()
        for g in range(20):
            for c in range(2):
                color = (g * 10, c * 100, 50 if label == "oily" else 150)
                Image.new("RGB", (8, 8), color).save(d / f"{label}{g}_jpg.rf.{c}.jpg")

    result = prepare({"oily": [tmp_path / "oily"], "dry": [tmp_path / "dry"]}, RATIOS, single_label=True)

    assert len(result.kept) == 80
    assert {r.split for r in result.kept} == {"train", "val", "test"}
