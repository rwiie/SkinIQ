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


def make_photo(root, path):
    p = root / path
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((40, 40, 3), 160, np.uint8)).save(p)


def test_one_image_per_photo_end_to_end(tmp_path):
    data = tmp_path / "processed"
    paths = ["skin_type/test/oily/a1.jpg", "skin_type/test/oily/a2.jpg", "skin_condition/train/acne/c1.jpg",
             "skin_condition/train/redness/c2.jpg", "skin_condition/val/wrinkles/x1.jpg"]
    for p in paths:
        make_photo(data, p)
    photos = pd.DataFrame([
        # photo 0: skin type face (a1 kept, a2 its copy)
        {"photo_id": 0, "key": "skin_type:0", "dataset": "skin_type", "group": 0, "image": paths[0], "md5": "a",
         "label": "oily", "folder_labels": None, "split": "test", "full_face": 1, "images": 2},
        # photo 1: close-up filed under acne and redness
        {"photo_id": 1, "key": "skin_condition:0", "dataset": "skin_condition", "group": 0, "image": paths[2],
         "md5": "c", "label": "acne", "folder_labels": "acne|redness", "split": "train", "full_face": 0,
         "images": 2},
        # photo 2: a different picture that shares the name group 0
        {"photo_id": 2, "key": "skin_condition:0", "dataset": "skin_condition", "group": 0, "image": paths[4],
         "md5": "x", "label": "wrinkles", "folder_labels": "wrinkles", "split": "val", "full_face": 1,
         "images": 1},
    ])
    files = pd.DataFrame({"path": paths, "photo_id": [0, 0, 1, 1, 2]})
    photos.to_csv(tmp_path / "photos.csv", index=False)
    files.to_csv(tmp_path / "files.csv", index=False)
    c = ["acne", "blackheads", "hyperpigmentation", "redness", "enlarged_pores", "wrinkles"]
    row = lambda key, image, ann, pos: {"key": key, "dataset": key.split(":")[0], "split": "x", "image": image,
                                         **{k: int(k in pos) for k in c}, "unsure": 0, "unusable": 0,
                                         "annotator": ann}
    pd.DataFrame([row("skin_type:0", paths[1], "author", ["wrinkles"]),          # labelled on a copy
                  row("skin_condition:0", paths[4], "author", ["wrinkles"])]).to_csv(tmp_path / "a.csv",
                                                                                      index=False)
    pd.DataFrame([row("skin_type:0", paths[0], "second", ["acne"])]).to_csv(tmp_path / "b.csv", index=False)

    out = tmp_path / "unique"
    photos_out, labels = mu.main(["--data-dir", str(data), "--out", str(out), "--photos", str(tmp_path / "photos.csv"),
                                  "--files", str(tmp_path / "files.csv"), "--labels", str(tmp_path / "a.csv"),
                                  str(tmp_path / "b.csv"), "--labels-out", str(tmp_path / "labels.csv")])

    sc = pd.read_csv(out / "skin_condition" / "manifest.csv").set_index("photo_id")
    assert sc.loc[2, "group"] == 0                        # labelled photo keeps the name group's id
    assert sc.loc[1, "group"] >= mu.NEW_GROUP_OFFSET      # the other picture gets a new id
    assert sc.loc[1, "labels"] == "acne|redness"
    assert (out / "skin_condition" / "train" / "acne" / "c1.jpg").exists()
    st = pd.read_csv(out / "skin_type" / "manifest.csv")
    assert list(st.path) == ["skin_type/test/oily/a1.jpg"]   # one image per photo
    lab = labels.set_index("key")
    assert lab.loc["skin_type:0", "annotator"] == "second"   # later file wins for the same photo
    assert lab.loc["skin_type:0", "split"] == "test"         # the photo's split, not the label file's
    assert lab.loc["skin_condition:0", "image"] == paths[4]  # label followed the picture it was made on


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
