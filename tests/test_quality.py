"""Photo quality check: one clear, well-lit face of reasonable size."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from skiniq.vision.quality import check_photo

ROOT = Path(__file__).resolve().parent.parent
# A front-facing selfie from the (uncommitted) skin type test split; see data/README.md
GOOD = ROOT / "data/processed/skin_type/test/combination/combination_-_15e129e6a2bedb36bde8_jpg.rf.3923751f22b92088ce49a26f7cedfcba.jpg"


@pytest.fixture(scope="module")
def face():
    if not GOOD.exists():
        pytest.skip("dataset not available (run training/prepare_data.py)")
    img = Image.open(GOOD).convert("RGB").resize((640, 640)).crop((0, 80, 640, 560))
    return np.asarray(img)


def test_good_selfie_is_accepted(face):
    r = check_photo(Image.fromarray(face))
    assert r.ok and r.faces == 1 and r.problems == []


@pytest.mark.parametrize("make, message", [
    (lambda a: cv2.GaussianBlur(a, (0, 0), 6), "blurry"),
    (lambda a: (a * 0.18).astype("uint8"), "too dark"),
    (lambda a: np.clip(a * 0.5 + 170, 0, 255).astype("uint8"), "overexposed"),
    (lambda a: np.full((480, 640, 3), 128, "uint8"), "No face"),
    (lambda a: np.hstack([a, a]), "More than one face"),
    (lambda a: cv2.resize(a, (160, 160)), "too small"),
])
def test_bad_photos_are_rejected_with_a_reason(face, make, message):
    r = check_photo(Image.fromarray(make(face)))
    assert not r.ok and message in r.message


def test_small_face_in_large_photo_is_rejected(face):
    canvas = np.full((1600, 1600, 3), 200, "uint8")
    small = cv2.resize(face, (face.shape[1] // 3, face.shape[0] // 3))
    canvas[600:600 + small.shape[0], 600:600 + small.shape[1]] = small
    r = check_photo(Image.fromarray(canvas))
    assert not r.ok and ("too small" in r.message or "No face" in r.message)


def test_large_photo_is_accepted(face):
    big = cv2.resize(face, (face.shape[1] * 4, face.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
    assert check_photo(Image.fromarray(big)).ok
