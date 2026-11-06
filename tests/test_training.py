import importlib.util
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from skiniq.vision.model import SkinClassifier, eval_transform, train_transform

ROOT = Path(__file__).resolve().parent.parent


def load_train_module():
    spec = importlib.util.spec_from_file_location("train", ROOT / "training" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transforms_output_model_input_shape():
    img = Image.new("RGB", (640, 480), (200, 150, 120))
    assert train_transform(64)(img).shape == (3, 64, 64)
    assert eval_transform(64)(img).shape == (3, 64, 64)


def test_class_weights_favour_rare_classes():
    train = load_train_module()
    w = train.class_weights([0] * 90 + [1] * 10, n_classes=2)
    assert w[1] > w[0]
    assert w.mean().item() == pytest.approx(1.0)


@pytest.fixture
def tiny_dataset(tmp_path):
    for split in ("train", "val", "test"):
        for label, color in (("dry", (220, 200, 180)), ("oily", (120, 80, 60))):
            d = tmp_path / "skin_type" / split / label
            d.mkdir(parents=True)
            for i in range(4):
                Image.new("RGB", (40, 40), tuple(c + i for c in color)).save(d / f"{i}.jpg")
    return tmp_path


def test_training_writes_checkpoint_and_metrics(tiny_dataset, tmp_path):
    train = load_train_module()
    out = tmp_path / "models"
    metrics = train.main([
        "--task", "skin_type", "--data-dir", str(tiny_dataset), "--out-dir", str(out),
        "--model", "tf_efficientnetv2_b0", "--no-pretrained", "--img-size", "32",
        "--epochs", "1", "--batch-size", "4", "--num-workers", "0",
    ])

    task_dir = out / "skin_type"
    for name in ("model.pt", "history.csv", "metrics.json", "confusion_matrix.csv", "confusion_matrix.png"):
        assert (task_dir / name).exists(), name
    assert metrics["test_size"] == 8
    assert set(json.loads((task_dir / "metrics.json").read_text())["per_class"]) == {"dry", "oily"}

    clf = SkinClassifier.load(task_dir / "model.pt", device="cpu")
    probs = clf.predict(Image.new("RGB", (100, 100)))
    assert set(probs) == {"dry", "oily"}
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)
    assert list(probs.values()) == sorted(probs.values(), reverse=True)
    assert isinstance(clf.model, torch.nn.Module)


def test_strong_augmentation_keeps_shape():
    img = Image.new("RGB", (640, 480), (200, 150, 120))
    assert train_transform(64, "strong")(img).shape == (3, 64, 64)
    with pytest.raises(ValueError):
        train_transform(64, "extreme")


def test_freeze_leaves_only_classifier_trainable():
    from skiniq.vision.model import build_model

    train = load_train_module()
    model = build_model(4, "tf_efficientnetv2_b0", pretrained=False)
    head = {id(p) for p in model.get_classifier().parameters()}

    train.set_backbone_trainable(model, trainable=False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and all(id(p) in head for p in trainable)

    train.set_backbone_trainable(model, trainable=True)
    assert all(p.requires_grad for p in model.parameters())


def test_frozen_epoch_does_not_change_backbone(tiny_dataset, tmp_path):
    train = load_train_module()
    out = tmp_path / "models"
    common = [
        "--task", "skin_type", "--data-dir", str(tiny_dataset), "--out-dir", str(out),
        "--model", "tf_efficientnetv2_b0", "--no-pretrained", "--img-size", "32",
        "--epochs", "1", "--batch-size", "4", "--num-workers", "0", "--patience", "1",
    ]
    train.main(common + ["--run-name", "frozen", "--freeze-epochs", "1", "--aug", "strong",
                         "--drop-rate", "0.3", "--drop-path-rate", "0.2"])
    torch.manual_seed(42)  # same random init as the run above (seed 42)
    from skiniq.vision.model import build_model

    init = build_model(2, "tf_efficientnetv2_b0", pretrained=False).state_dict()
    trained = torch.load(out / "frozen" / "model.pt", weights_only=False)

    assert trained["state_dict"]["conv_stem.weight"].equal(init["conv_stem.weight"])
    assert not trained["state_dict"]["classifier.weight"].equal(init["classifier.weight"])
    metrics = json.loads((out / "frozen" / "metrics.json").read_text())
    assert metrics["run_name"] == "frozen" and metrics["settings"]["aug"] == "strong"
