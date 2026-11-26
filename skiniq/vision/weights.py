"""Download the trained skin concern model on first use.

Model weights (82 MB) are not stored in git. If models/skin_concerns/model.pt is
missing, it is downloaded from the project's shared Google Drive folder and checked
against its SHA-256 before use, so a partial or wrong file is never loaded.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import requests

from skiniq.config import MODELS_DIR

SKIN_CONCERN_MODEL = MODELS_DIR / "skin_concerns" / "model.pt"
# Multi-label skin concern model v2 (training/results/skin_concerns_v2), shared on Google Drive
MODEL_URL = os.environ.get(
    "SKINIQ_MODEL_URL",
    "https://drive.usercontent.google.com/download?id=1uGmAZtnQp62_kSLDpwrxu0Rh5LB52CJn&export=download&confirm=t",
)
MODEL_SHA256 = "41eb371460564dff080c1e1dd5062798c55746a48b7ba5141fd717a34540581a"
# v2 is run with test-time augmentation (the photo and its mirror image, averaged). Thresholds for that
# set-up were re-tuned on the validation photos (max F1 per concern), as in training; on the test photos
# this raised macro F1 from 0.661 to 0.688 (training/results/skin_concerns_v2_tta/).
TTA_THRESHOLDS = {"acne": 0.55, "blackheads": 0.35, "hyperpigmentation": 0.30, "redness": 0.55,
                  "enlarged_pores": 0.45, "wrinkles": 0.30}
TIMEOUT_S = 60


class ModelUnavailable(RuntimeError):
    """The model file is missing and could not be downloaded."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(path: Path = SKIN_CONCERN_MODEL, url: str = MODEL_URL,
                 expected_sha256: str | None = MODEL_SHA256) -> Path:
    """Return the model path, downloading it first if it is missing."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=TIMEOUT_S) as resp:
            if resp.status_code != 200:
                raise ModelUnavailable(f"model download failed (HTTP {resp.status_code})")
            if "text/html" in resp.headers.get("Content-Type", ""):
                raise ModelUnavailable("model download returned a web page, not the model file")
            with open(part, "wb") as f:
                for chunk in resp.iter_content(1 << 20):
                    f.write(chunk)
    except requests.RequestException as e:
        part.unlink(missing_ok=True)
        raise ModelUnavailable(f"model download failed ({type(e).__name__}); check your internet connection") from None
    except ModelUnavailable:
        part.unlink(missing_ok=True)
        raise
    if expected_sha256 and sha256(part) != expected_sha256:
        part.unlink(missing_ok=True)
        raise ModelUnavailable("downloaded model file is corrupted (checksum mismatch)")
    os.replace(part, path)
    return path
