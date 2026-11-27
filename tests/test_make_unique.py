import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from skiniq.data_prep import border_darkness, photo_clusters

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
spec = importlib.util.spec_from_file_location("make_unique", ROOT / "training" / "make_unique.py")
mu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mu)


def unit(*v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_border_darkness_prefers_unpadded_copy():
    photo = np.full((100, 100), 150.0)
    rotated = photo.copy()
    rotated[:, :15] = 0  # black padding from a rotation
    assert border_darkness(photo) == 0
    assert border_darkness(rotated) > 0.2


def test_unrelated_photo_with_the_same_name_is_split_off():
    # group 0: two copies of one photo plus an unrelated photo that shares the generic name
    e = [unit(1, 0, 0), unit(0.98, 0.2, 0), unit(0, 1, 0)]
    c = photo_clusters([0, 0, 0], e, same_photo=0.8, duplicate=0.95)
    assert c[0] == c[1] != c[2]


def test_same_photo_under_two_names_or_datasets_is_merged():
    e = [unit(1, 0, 0), unit(1, 0.01, 0), unit(0, 0, 1)]
    c = photo_clusters([("skin_type", 5), ("skin_condition", 9), ("skin_condition", 10)], e,
                       same_photo=0.8, duplicate=0.95)
    assert c[0] == c[1] != c[2]


def test_similar_but_different_photos_with_different_names_stay_apart():
    e = [unit(1, 0, 0), unit(0.9, 0.44, 0)]  # cosine ~0.9: alike, not identical
    c = photo_clusters([1, 2], e, same_photo=0.8, duplicate=0.95)
    assert c[0] != c[1]


def write_task(root, task, rows):
    out = []
    for path, label, split, group, source, pad in rows:
        p = root / task / split / label / path
        p.parent.mkdir(parents=True, exist_ok=True)
        a = np.full((60, 60, 3), 160, np.uint8)
        if pad:
            a[:, :10] = 0
        Image.fromarray(a).save(p)
        out.append({"path": f"{task}/{split}/{label}/{path}", "label": label, "split": split, "group": group,
                    "source": source, "width": 60, "height": 60, "low_res": 1, "original": "", "md5": path})
    pd.DataFrame(out).to_csv(root / task / "manifest.csv", index=False)


def test_one_image_per_photo_end_to_end(tmp_path):
    data = tmp_path / "processed"
    write_task(data, "skin_type", [
        ("a1.jpg", "oily", "test", 0, "a", True),    # labelled copy: must be kept even though padded
        ("a2.jpg", "oily", "test", 0, "a", False),
        ("b1.jpg", "dry", "train", 1, "b", False),
    ])
    write_task(data, "skin_condition", [
        ("c1.jpg", "acne", "train", 0, "c", True),   # augmented copy with black padding
        ("c2.jpg", "acne", "train", 0, "c", False),
        ("c3.jpg", "redness", "train", 0, "c", False),  # same photo in a second concern folder
        ("d1.jpg", "redness", "train", 1, "d", False),  # same photo as skin_type b1 (reused dataset)
        ("x1.jpg", "wrinkles", "val", 0, "c", False),   # unrelated photo sharing the name "c"
    ])
    # embeddings in manifest order: skin_type a1 a2 b1, then skin_condition c1 c2 c3 d1 x1
    e = np.stack([unit(1, 0, 0, 0), unit(1, 0.05, 0, 0), unit(0, 1, 0, 0),
                  unit(0, 0, 1, 0), unit(0, 0, 1, 0.05), unit(0, 0.05, 1, 0), unit(0, 1, 0.01, 0),
                  unit(0, 0, 0, 1)])
    np.save(tmp_path / "e.npy", e)
    pd.DataFrame([{"key": "skin_type:0", "image": "skin_type/test/oily/a1.jpg"}]).to_csv(tmp_path / "lab.csv",
                                                                                     index=False)
    out = tmp_path / "unique"
    mu.main(["--data-dir", str(data), "--out", str(out), "--labels", str(tmp_path / "lab.csv"),
             "--embeddings", str(tmp_path / "e.npy"), "--report", str(tmp_path / "r.md")])

    st = pd.read_csv(out / "skin_type" / "manifest.csv")
    sc = pd.read_csv(out / "skin_condition" / "manifest.csv")
    assert list(st.path.map(lambda p: Path(p).name)) == ["a1.jpg"]  # labelled copy kept; b1 is d1's twin
    assert sorted(Path(p).name for p in sc.path) == ["c2.jpg", "d1.jpg", "x1.jpg"]  # unpadded copy kept
    c = sc.set_index(sc.path.map(lambda p: Path(p).name))
    assert set(c.loc["c2.jpg", "labels"].split("|")) == {"acne", "redness"}
    assert c.loc["x1.jpg", "group"] >= mu.NEW_GROUP_OFFSET   # split off the "c" name group
    assert c.loc["c2.jpg", "group"] == 0
    assert (out / "skin_condition" / "train" / "acne" / "c2.jpg").exists()
    assert set(pd.read_csv(out / "skin_condition" / "removed.csv").reason) == {"augmented copy"}
    assert set(pd.read_csv(out / "skin_type" / "removed.csv").reason) == {
        "augmented copy", "duplicate of skin_condition/train/redness/d1.jpg"}


def test_training_reads_all_concern_folders_of_a_photo(tmp_path):
    from tests.test_train_multilabel import tm, write_dataset

    write_dataset(tmp_path, "skin_condition", [(0, "train", "acne")])
    write_dataset(tmp_path, "skin_type", [(0, "train", "oily")])
    m = pd.read_csv(tmp_path / "skin_condition" / "manifest.csv")
    m["labels"] = "acne|redness"
    m.to_csv(tmp_path / "skin_condition" / "manifest.csv", index=False)
    r = tm.build_table(tmp_path, None).set_index("key").loc["skin_condition:0"]
    assert r["y_acne"] == 1 and r["y_redness"] == 1 and r["m_redness"] == 1
    assert r["y_wrinkles"] == 0 and r["m_wrinkles"] == 0.1
