import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from skiniq.config import SKIN_CONDITIONS

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("train_multilabel", ROOT / "training" / "train_multilabel.py")
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)
C = list(SKIN_CONDITIONS)


def write_dataset(root: Path, dataset: str, rows: list[tuple[int, str, str]]):
    """rows: (group, split, folder label) -> one tiny image per group."""
    manifest = []
    for group, split, label in rows:
        rel = f"{dataset}/{split}/{label}/{group}.jpg"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shade = 60 + (group * 37) % 180
        Image.new("RGB", (40, 40), (shade, 120, 200 - shade // 2)).save(root / rel)
        manifest.append({"path": rel, "label": label, "split": split, "group": group})
    pd.DataFrame(manifest).to_csv(root / dataset / "manifest.csv", index=False)


def label_row(key, split, positives=(), unsure=0, unusable=0):
    return {"key": key, "dataset": key.split(":")[0], "split": split, "image": "", "md5": "",
            **{c: int(c in positives) for c in C}, "unsure": unsure, "unusable": unusable,
            "annotator": "t", "labelled_at": ""}


@pytest.fixture
def small(tmp_path):
    write_dataset(tmp_path, "skin_condition", [(0, "train", "acne"), (1, "train", "redness"), (2, "test", "acne")])
    write_dataset(tmp_path, "skin_type", [(0, "train", "oily"), (1, "train", "dry"), (2, "test", "normal")])
    labels = pd.DataFrame([
        label_row("skin_condition:0", "train", ["acne", "blackheads"]),
        label_row("skin_type:0", "train", []),                    # clear face
        label_row("skin_type:2", "test", ["wrinkles"], unsure=1),
        label_row("skin_condition:2", "test", unusable=1),
    ])
    labels.to_csv(tmp_path / "labels.csv", index=False)
    return tm.build_table(tmp_path, tmp_path / "labels.csv").set_index("key")


def test_annotated_photo_has_all_labels_known(small):
    r = small.loc["skin_condition:0"]
    assert r.annotated and all(r[f"m_{c}"] == 1 for c in C)
    assert r["y_acne"] == 1 and r["y_blackheads"] == 1 and r["y_redness"] == 0


def test_unannotated_closeup_knows_its_folder_concern_and_weak_negatives(small):
    r = small.loc["skin_condition:1"]
    assert not r.annotated
    assert r["y_redness"] == 1 and r["m_redness"] == 1
    others = [c for c in C if c != "redness"]
    assert all(r[f"y_{c}"] == 0 and r[f"m_{c}"] == pytest.approx(0.1) for c in others)


def test_unknown_negative_weight_zero_ignores_unknown_labels(tmp_path):
    write_dataset(tmp_path, "skin_condition", [(0, "train", "acne")])
    write_dataset(tmp_path, "skin_type", [(0, "train", "oily")])
    r = tm.build_table(tmp_path, None, unknown_negative_weight=0.0).iloc[0]
    assert sum(r[f"m_{c}"] for c in C) == 1


def test_clear_face_is_all_known_negatives(small):
    r = small.loc["skin_type:0"]
    assert all(r[f"y_{c}"] == 0 and r[f"m_{c}"] == 1 for c in C)


def test_unannotated_faces_and_unusable_photos_are_skipped(small):
    assert "skin_type:1" not in small.index
    assert "skin_condition:2" not in small.index
    assert small.loc["skin_type:2"].unsure


def test_masked_bce_weights_weak_negatives():
    y = torch.zeros(1, 6)
    full = tm.masked_bce(torch.full((1, 6), 3.0), y, torch.ones(1, 6))
    weak = tm.masked_bce(torch.full((1, 6), 3.0), y, torch.full((1, 6), 0.1))
    assert weak.item() == pytest.approx(full.item())  # same per-label loss, normalised by total weight
    mixed = tm.masked_bce(torch.tensor([[3.0, 3, 3, 3, 3, 3]]), torch.tensor([[1.0, 0, 0, 0, 0, 0]]),
                          torch.tensor([[1.0, 0.1, 0.1, 0.1, 0.1, 0.1]]))
    pos = torch.nn.functional.softplus(torch.tensor(-3.0)).item()
    neg = torch.nn.functional.softplus(torch.tensor(3.0)).item()
    assert mixed.item() == pytest.approx((pos + 0.5 * neg) / 1.5, rel=1e-4)


def test_masked_bce_ignores_unknown_labels():
    y = torch.tensor([[1.0, 0, 0, 0, 0, 0]])
    m = torch.tensor([[1.0, 0, 0, 0, 0, 0]])
    a = tm.masked_bce(torch.tensor([[2.0, 0, 0, 0, 0, 0]]), y, m)
    b = tm.masked_bce(torch.tensor([[2.0, 9, -9, 5, 3, -4]]), y, m)
    assert a.item() == pytest.approx(b.item())
    assert a.item() == pytest.approx(torch.nn.functional.softplus(torch.tensor(-2.0)).item())


def test_positive_weights_balance_rare_concerns():
    t = pd.DataFrame({**{f"y_{c}": [1, 0, 0, 0] for c in C}, **{f"m_{c}": [1, 1, 1, 1] for c in C}})
    t["y_redness"] = [1, 1, 0, 0]
    t["m_wrinkles"] = 0
    w = dict(zip(C, tm.positive_weights(t).tolist()))
    assert w["acne"] == pytest.approx(3.0) and w["redness"] == pytest.approx(1.0) and w["wrinkles"] == 1.0


def test_threshold_tuning_finds_separating_value():
    probs = np.tile(np.array([[0.1], [0.2], [0.7], [0.8]]), (1, 6))
    y = np.tile(np.array([[0], [0], [1], [1]]), (1, 6))
    th = tm.tune_thresholds(probs, y)
    assert all(0.2 < t <= 0.7 for t in th.values())


def test_threshold_defaults_when_concern_absent():
    th = tm.tune_thresholds(np.full((4, 6), 0.3), np.zeros((4, 6), int))
    assert set(th.values()) == {0.5}


def test_overall_metrics_counts_clear_false_alarms():
    th = {c: 0.5 for c in C}
    probs = np.array([[0.9] + [0.1] * 5, [0.1] * 6, [0.6] + [0.1] * 5])
    y = np.array([[1, 0, 0, 0, 0, 0], [0] * 6, [0] * 6])
    m = tm.overall_metrics(probs, y, th)
    assert m["clear_photos"] == 2 and m["clear_false_alarm_rate"] == 0.5
    assert m["exact_match"] == pytest.approx(2 / 3, abs=1e-4)


def test_photo_level_averages_copies():
    table = pd.DataFrame({"key": ["a", "a", "b"], **{f"y_{c}": [1, 1, 0] for c in C}})
    probs = np.array([[0.2] * 6, [0.6] * 6, [0.9] * 6])
    p, y = tm.photo_level(table, probs)
    assert p[0, 0] == pytest.approx(0.4) and y.tolist()[0][0] == 1 and len(p) == 2


def test_training_end_to_end(tmp_path):
    rows_c, rows_t, labels = [], [], []
    for g in range(24):
        split = "train" if g < 12 else "val" if g < 18 else "test"
        label = C[g % len(C)]
        rows_c.append((g, split, label))
        rows_t.append((g, split, "oily"))
        labels.append(label_row(f"skin_condition:{g}", split, [label]))
        labels.append(label_row(f"skin_type:{g}", split, [] if g % 2 else ["wrinkles"]))
    write_dataset(tmp_path, "skin_condition", rows_c)
    write_dataset(tmp_path, "skin_type", rows_t)
    pd.DataFrame(labels).to_csv(tmp_path / "labels.csv", index=False)

    metrics = tm.main([
        "--labels", str(tmp_path / "labels.csv"), "--data-dir", str(tmp_path), "--out-dir", str(tmp_path / "m"),
        "--model", "tf_efficientnetv2_b0", "--no-pretrained", "--img-size", "32", "--epochs", "1",
        "--batch-size", "8", "--num-workers", "0", "--min-val-photos", "5",
    ])
    out = tmp_path / "m" / "skin_concerns"
    ckpt = torch.load(out / "model.pt", weights_only=False)
    assert ckpt["multilabel"] and set(ckpt["thresholds"]) == set(C)
    assert metrics["selected_by"] == "val mean ROC-AUC"
    assert metrics["test"]["photos"] == 12
    assert set(json.loads((out / "metrics.json").read_text())["per_concern"]) == set(C)


def test_no_annotated_test_photos_stops_with_message(tmp_path):
    write_dataset(tmp_path, "skin_condition", [(0, "train", "acne"), (1, "test", "acne")])
    write_dataset(tmp_path, "skin_type", [(0, "train", "oily")])
    with pytest.raises(SystemExit, match="label the test sample"):
        tm.main(["--labels", str(tmp_path / "none.csv"), "--data-dir", str(tmp_path), "--out-dir", str(tmp_path),
                 "--no-pretrained", "--model", "tf_efficientnetv2_b0", "--num-workers", "0"])


def test_region_evaluation_compares_whole_and_region(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("evaluate_regions", ROOT / "training" / "evaluate_regions.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    from skiniq.vision import regions

    monkeypatch.setattr(regions, "detect_face", lambda img: None)  # whole-image fallback

    class Fake:
        multilabel, thresholds = True, {c: 0.5 for c in C}

        def predict(self, img):
            return {c: 0.9 if c == "acne" else 0.1 for c in C}

    rows = []
    for i, (ds, pos) in enumerate([("skin_type", []), ("skin_type", ["acne"]), ("skin_condition", ["acne", "redness"])]):
        Image.new("RGB", (60, 60)).save(tmp_path / f"{i}.jpg")
        r = label_row(f"{ds}:{i}", "test", pos)
        r["image"] = f"{i}.jpg"
        rows.append(r)
    report = ev.evaluate(Fake(), pd.DataFrame(rows), tmp_path)
    assert report["all"]["whole"]["overall"]["photos"] == 3
    assert report["full faces"]["region"]["overall"]["clear_false_alarm_rate"] == 1.0  # acne on the clear face
    assert report["close-ups"]["whole"]["per_concern"]["redness"]["recall"] == 0.0
    assert len(ev.summary_table(report)) == 6


def test_unsure_training_photos_are_skipped_but_unsure_test_photos_kept(tmp_path):
    write_dataset(tmp_path, "skin_condition", [(0, "train", "acne")])
    write_dataset(tmp_path, "skin_type", [(0, "train", "oily"), (1, "train", "dry"), (2, "test", "normal")])
    pd.DataFrame([
        label_row("skin_type:0", "train", [], unsure=1),
        label_row("skin_type:1", "train", []),
        label_row("skin_type:2", "test", [], unsure=1),
    ]).to_csv(tmp_path / "labels.csv", index=False)
    keys = set(tm.build_table(tmp_path, tmp_path / "labels.csv").key)
    assert "skin_type:0" not in keys and {"skin_type:1", "skin_type:2"} <= keys
