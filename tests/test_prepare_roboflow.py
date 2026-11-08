import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "training" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pr = load("prepare_roboflow")
CATS = ["Acne-Blackhead-Wrinkles-f6HR-vJCz-Vyox", "Acne", "Blackheads", "Dark-Spots", "Dry-Skin",
        "Englarged-Pores", "Eyebags", "Oily-Skin", "Skin-Redness", "Whiteheads", "Wrinkles"]


def write_coco(raw: Path, folder: str, images: list[tuple[str, list[str], int]]):
    """images: (file name, box class names, colour seed)."""
    d = raw / folder
    d.mkdir(parents=True, exist_ok=True)
    coco = {"categories": [{"id": i, "name": n} for i, n in enumerate(CATS)], "images": [], "annotations": []}
    for img_id, (name, classes, seed) in enumerate(images):
        Image.new("RGB", (32, 32), (seed % 255, (seed * 7) % 255, 90)).save(d / name)
        coco["images"].append({"id": img_id, "file_name": name, "width": 32, "height": 32})
        for c in classes:
            coco["annotations"].append({"id": len(coco["annotations"]), "image_id": img_id,
                                        "category_id": CATS.index(c), "bbox": [1, 2, 10, 10]})
    (d / "_annotations.coco.json").write_text(json.dumps(coco))


@pytest.fixture
def raw(tmp_path):
    raw = tmp_path / "raw"
    train = [(f"{i}_jpg.rf.a{i}.jpg", ["Acne"], i) for i in range(20)]
    train += [("w_jpg.rf.b1.jpg", ["Wrinkles", "Dark-Spots"], 101),
              ("wh_jpg.rf.b2.jpg", ["Whiteheads", "Blackheads"], 102),
              ("dry_jpg.rf.b3.jpg", ["Dry-Skin", "Eyebags"], 103),       # no SkinIQ concern -> dropped
              ("none_jpg.rf.b4.jpg", [], 104),                           # no boxes -> dropped
              ("red_jpg.rf.b5.jpg", ["Skin-Redness", "Oily-Skin"], 105)]
    write_coco(raw, "train", train)
    # Same original photo ("dup_jpg") in train and valid -> must end up in one split
    write_coco(raw, "valid", [("dup_jpg.rf.c1.jpg", ["Englarged-Pores"], 200)])
    write_coco(raw, "test", [("dup_jpg.rf.c2.jpg", ["Acne"], 201)])
    return raw


def test_boxes_become_six_image_labels(raw):
    df = pr.read_coco(raw).set_index("file")
    assert df.loc["w_jpg.rf.b1.jpg", "concerns"] == ["hyperpigmentation", "wrinkles"]
    assert df.loc["wh_jpg.rf.b2.jpg", "concerns"] == ["acne", "blackheads"]   # whiteheads -> acne
    assert df.loc["red_jpg.rf.b5.jpg", "concerns"] == ["redness"]             # oily skin ignored
    assert df.loc["dry_jpg.rf.b3.jpg", "concerns"] == []


def test_unknown_class_is_an_error(tmp_path):
    raw = tmp_path / "raw"
    write_coco(raw, "train", [("a.jpg", ["Acne"], 1)])
    coco = json.loads((raw / "train" / "_annotations.coco.json").read_text())
    coco["categories"].append({"id": 99, "name": "Freckles"})
    (raw / "train" / "_annotations.coco.json").write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="Freckles"):
        pr.read_coco(raw)


def test_prepare_drops_unlabelled_and_keeps_copies_together(raw, tmp_path):
    out = tmp_path / "processed" / "skin_problem"
    manifest, dropped = pr.prepare(raw, out)
    assert dropped["no SkinIQ concern boxed (only dry/oily skin, eye bags or none)"] == 2
    assert len(manifest) == 25
    dup = manifest[manifest.source == "dup_jpg"]
    assert len(dup) == 2 and dup.split.nunique() == 1 and dup.group.nunique() == 1
    assert all((out.parent / p).exists() for p in manifest.path)
    assert set(manifest.split) == {"train", "val", "test"}
    boxes = json.loads(manifest.set_index("source").loc["w_jpg", "boxes"])
    assert {b["concern"] for b in boxes} == {"wrinkles", "hyperpigmentation"}


def test_exact_duplicate_kept_once_with_merged_labels(tmp_path):
    raw = tmp_path / "raw"
    write_coco(raw, "train", [("x_jpg.rf.1.jpg", ["Acne"], 5)] + [(f"{i}_jpg.rf.{i}.jpg", ["Acne"], 40 + i) for i in range(6)])
    write_coco(raw, "valid", [("y_jpg.rf.2.jpg", ["Wrinkles"], 5)])  # same pixels as x, different name
    manifest, dropped = pr.prepare(raw, tmp_path / "out" / "skin_problem")
    assert dropped["exact duplicate"] == 1
    assert manifest[manifest.md5.duplicated()].empty
    merged = manifest[manifest.concerns.str.contains("wrinkles")]
    assert merged.concerns.tolist() == ["acne|wrinkles"]


def test_training_table_reads_roboflow_boxes_as_positives_and_weak_negatives(raw, tmp_path):
    tm = load("train_multilabel")
    processed = tmp_path / "processed"
    pr.prepare(raw, processed / "skin_problem")
    for ds in ("skin_type", "skin_condition"):
        (processed / ds).mkdir(parents=True)
        pd.DataFrame(columns=["path", "label", "split", "group"]).to_csv(processed / ds / "manifest.csv", index=False)
    table = tm.build_table(processed, None)
    row = table[table.path.str.contains("w_jpg")].iloc[0]
    assert row["y_wrinkles"] == 1 and row["m_wrinkles"] == 1 and row["y_hyperpigmentation"] == 1
    assert row["y_acne"] == 0 and row["m_acne"] == pytest.approx(0.1)
    assert set(table.dataset) == {"skin_problem"}
