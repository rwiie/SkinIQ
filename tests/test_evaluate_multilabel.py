import importlib.util
import sys
from pathlib import Path

import pandas as pd
import torch

from tests.test_train_multilabel import C, label_row, tm, write_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
spec = importlib.util.spec_from_file_location("evaluate_multilabel", ROOT / "training" / "evaluate_multilabel.py")
em = importlib.util.module_from_spec(spec)
spec.loader.exec_module(em)


class Asymmetric(torch.nn.Module):
    """Scores the left half of the image only, so the mirror image gives a different answer."""

    def forward(self, x):
        left = x[..., : x.shape[-1] // 2].mean(dim=(1, 2, 3))
        return left[:, None].repeat(1, len(C))


def test_mirror_averaging_averages_photo_and_mirror():
    x = torch.zeros(1, 3, 4, 4)
    x[..., :2] = 2.0  # bright left half: mirror image has a dark left half
    loader = [(x, torch.zeros(1, len(C)), torch.ones(1, len(C)))]
    plain = em.predict(Asymmetric(), loader, torch.device("cpu"))
    mirrored = em.predict(Asymmetric(), loader, torch.device("cpu"), tta=True)
    expected = (torch.sigmoid(torch.tensor(2.0)) + torch.sigmoid(torch.tensor(0.0))) / 2
    assert abs(plain[0, 0] - torch.sigmoid(torch.tensor(2.0)).item()) < 1e-6
    assert abs(mirrored[0, 0] - expected.item()) < 1e-6


def test_scores_a_checkpoint_against_another_label_file(tmp_path):
    rows_c, rows_t, labels = [], [], []
    for g in range(24):
        split = "train" if g < 12 else "val" if g < 18 else "test"
        rows_c.append((g, split, C[g % len(C)]))
        rows_t.append((g, split, "oily"))
        labels.append(label_row(f"skin_condition:{g}", split, [C[g % len(C)]]))
        labels.append(label_row(f"skin_type:{g}", split, [] if g % 2 else ["wrinkles"]))
    write_dataset(tmp_path, "skin_condition", rows_c)
    write_dataset(tmp_path, "skin_type", rows_t)
    pd.DataFrame(labels).to_csv(tmp_path / "labels.csv", index=False)
    tm.main(["--labels", str(tmp_path / "labels.csv"), "--data-dir", str(tmp_path), "--out-dir", str(tmp_path / "m"),
             "--model", "tf_efficientnetv2_b0", "--no-pretrained", "--img-size", "32", "--epochs", "1",
             "--batch-size", "8", "--num-workers", "0", "--min-val-photos", "5"])

    # A second answer key: every test and validation face now also has redness
    relabelled = pd.DataFrame(labels)
    relabelled.loc[relabelled.split != "train", "redness"] = 1
    relabelled.to_csv(tmp_path / "relabelled.csv", index=False)
    result = em.main(["--labels", str(tmp_path / "relabelled.csv"), "--data-dir", str(tmp_path),
                      "--model", f"tiny={tmp_path / 'm' / 'skin_concerns' / 'model.pt'}", "--tta",
                      "--out-dir", str(tmp_path / "out"), "--batch-size", "8", "--num-workers", "0"])

    # single image: stored + retuned thresholds; mirror averaging: retuned only
    assert list(zip(result.setup, result.thresholds)) == [
        ("single image", "stored in checkpoint"), ("single image", "retuned on these val labels"),
        ("mirror averaging", "retuned on these val labels")]
    assert (result.photos == 12).all()
    per = pd.read_csv(tmp_path / "out" / "per_concern.csv")
    assert (per[per.concern == "redness"].positives == 12).all()  # scored against the new labels
    assert (tmp_path / "out" / "comparison.csv").exists()
