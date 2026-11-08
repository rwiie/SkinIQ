"""EfficientNet-V2 model and image transforms, shared by training and the app
so both preprocess images the same way (proposal 3.6.1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import transforms as T

from skiniq.config import IMAGE_SIZE

DEFAULT_MODEL = "tf_efficientnetv2_s.in21k_ft_in1k"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(
    num_classes: int,
    model_name: str = DEFAULT_MODEL,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
) -> torch.nn.Module:
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )


class AddGaussianNoise:
    """Light sensor-style noise on a [0, 1] tensor (proposal 3.6.1: noise injection)."""

    def __init__(self, std: float = 0.02, p: float = 0.3) -> None:
        self.std, self.p = std, p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            x = (x + torch.randn_like(x) * self.std).clamp(0, 1)
        return x


def train_transform(size: int = IMAGE_SIZE, strength: str = "mild") -> T.Compose:
    """Training augmentation. Never shifts hue, because colour carries redness and pigmentation.

    - ``mild``: the default; the source images already contain Roboflow rotations and mosaics.
    - ``strong``: heavier cropping, lighting changes and random erasing, to fight overfitting
      on small or noisy datasets (used for the skin type v2 experiment).
    """
    if strength not in ("mild", "strong"):
        raise ValueError(f"unknown augmentation strength: {strength}")
    strong = strength == "strong"
    steps = [
        T.RandomResizedCrop(size, scale=(0.5, 1.0) if strong else (0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15 if strong else 10),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1) if strong
        else T.ColorJitter(brightness=0.15, contrast=0.15),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
        T.ToTensor(),
        AddGaussianNoise(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if strong:
        steps.append(T.RandomErasing(p=0.25, scale=(0.02, 0.15)))
    return T.Compose(steps)


def eval_transform(size: int = IMAGE_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@dataclass
class SkinClassifier:
    """A trained classifier loaded from a checkpoint written by training/train.py."""

    model: torch.nn.Module
    labels: list[str]
    image_size: int
    device: torch.device
    multilabel: bool = False                  # sigmoid outputs (independent concerns)
    thresholds: dict[str, float] | None = None  # per-concern decision thresholds (multi-label)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "SkinClassifier":
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(path, map_location=dev, weights_only=False)
        model = build_model(len(ckpt["labels"]), ckpt["model_name"], pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.to(dev).eval()
        return cls(model=model, labels=ckpt["labels"], image_size=ckpt["image_size"], device=dev,
                   multilabel=bool(ckpt.get("multilabel", False)), thresholds=ckpt.get("thresholds"))

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict[str, float]:
        """Return {label: probability}, highest first.

        Single-label models: probabilities sum to 1 (softmax). Multi-label models: each
        concern has its own independent probability (sigmoid).
        """
        x = eval_transform(self.image_size)(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x)[0]
        probs = (logits.sigmoid() if self.multilabel else logits.softmax(dim=0)).tolist()
        return dict(sorted(zip(self.labels, probs), key=lambda kv: kv[1], reverse=True))
