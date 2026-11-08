import importlib.util
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


ann = load("annotation")


@pytest.fixture
def queue(tmp_path):
    rows = []
    for i, (ds, prefill) in enumerate([("skin_condition", "acne"), ("skin_type", ""), ("skin_condition", "blackheads|wrinkles")]):
        img = tmp_path / "img" / f"{i}.jpg"
        img.parent.mkdir(exist_ok=True)
        Image.new("RGB", (800, 600), (200, 150, 120)).save(img)
        rows.append({"order": i + 1, "tier": "test sample", "key": f"{ds}:{i}", "dataset": ds, "group": i,
                     "split": "test", "folder_labels": prefill or "oily", "prefill": prefill,
                     "image": f"{i}.jpg", "md5": f"m{i}", "copies": 1})
    return pd.DataFrame(rows)


def test_session_prefills_folder_labels_and_saves(queue, tmp_path):
    store = ann.LabelStore(tmp_path / "labels.csv")
    s = ann.AnnotationSession(queue, store, annotator="raneem")
    assert s.initial_labels() == ({"acne"}, False)
    s.submit({"acne", "redness"}, unsure=True)

    saved = pd.read_csv(tmp_path / "labels.csv")
    row = saved.iloc[0]
    assert row["acne"] == 1 and row["redness"] == 1 and row["wrinkles"] == 0
    assert row["unsure"] == 1 and row["annotator"] == "raneem"
    assert s.initial_labels() == (set(), False)  # full face: nothing prefilled


def test_resume_starts_at_first_unlabelled(queue, tmp_path):
    path = tmp_path / "labels.csv"
    s = ann.AnnotationSession(queue, ann.LabelStore(path))
    s.submit({"acne"})
    resumed = ann.AnnotationSession(queue, ann.LabelStore(path))
    assert resumed.current()["key"] == "skin_type:1"


def test_back_shows_saved_labels_and_resubmit_overwrites(queue, tmp_path):
    s = ann.AnnotationSession(queue, ann.LabelStore(tmp_path / "labels.csv"))
    s.submit({"acne"})
    s.back()
    assert s.initial_labels() == ({"acne"}, False)
    s.submit({"acne", "hyperpigmentation"})
    labels = pd.read_csv(tmp_path / "labels.csv")
    assert len(labels) == 1 and labels.iloc[0]["hyperpigmentation"] == 1
    assert s.current()["key"] == "skin_type:1"


def test_unusable_clears_labels(queue, tmp_path):
    s = ann.AnnotationSession(queue, ann.LabelStore(tmp_path / "labels.csv"))
    s.submit({"acne"}, unusable=True)
    row = pd.read_csv(tmp_path / "labels.csv").iloc[0]
    assert row["unusable"] == 1 and row["acne"] == 0


def test_unknown_concern_rejected(queue, tmp_path):
    s = ann.AnnotationSession(queue, ann.LabelStore(tmp_path / "labels.csv"))
    with pytest.raises(ValueError):
        s.submit({"freckles"})


def test_finishing_queue_and_progress(queue, tmp_path):
    s = ann.AnnotationSession(queue, ann.LabelStore(tmp_path / "labels.csv"))
    for _ in range(3):
        s.submit(set())
    assert s.finished
    assert s.progress().loc["test sample"].tolist() == [3, 3]


def test_widget_ui_clicks_save_and_navigate(queue, tmp_path):
    pytest.importorskip("ipywidgets")
    s = ann.AnnotationSession(queue, ann.LabelStore(tmp_path / "labels.csv"), annotator="raneem")
    ui = ann.build_ui(s, tmp_path / "img")
    assert ui.toggles["acne"].value is True  # prefilled
    ui.toggles["redness"].value = True
    ui.save.click()
    assert ui.info.value.startswith("#2 ") and all(not t.value for t in ui.toggles.values())
    ui.unusable.click()
    ui.back.click()
    assert "already labelled" in ui.info.value
    labels = pd.read_csv(tmp_path / "labels.csv").set_index("key")
    assert labels.loc["skin_condition:0", "redness"] == 1
    assert labels.loc["skin_type:1", "unusable"] == 1


def test_queue_puts_stratified_test_sample_first(tmp_path):
    q = load("annotation_queue")
    for ds, labels in (("skin_type", ["dry", "oily"]), ("skin_condition", ["acne", "redness"])):
        d = tmp_path / ds
        d.mkdir()
        rows = []
        for i in range(40):
            split = "test" if i < 10 else "val" if i < 20 else "train"
            lab = labels[i % 2]
            rows.append({"path": f"{ds}/{split}/{lab}/{i}.jpg", "label": lab, "split": split, "group": i,
                         "low_res": 0, "original": f"raw/{ds}/{i}.jpg", "md5": f"{ds}{i}"})
        pd.DataFrame(rows).to_csv(d / "manifest.csv", index=False)
    queue = q.build_queue(q.load_groups(tmp_path))
    assert len(queue) == 80 and queue.key.is_unique
    first = queue.head(20)
    assert (first.split == "test").all()
    assert set(first.dataset) == {"skin_type", "skin_condition"}
    assert (queue[queue.dataset == "skin_type"].prefill.fillna("") == "").all()
    assert set(queue[queue.dataset == "skin_condition"].prefill) == {"acne", "redness"}
