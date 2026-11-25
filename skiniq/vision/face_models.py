"""Trained models behind the photo check: face parts, hands, and drawings.

- Face parsing (BiSeNet ResNet-18, yakhyo/face-parsing via jbrownkramer/face-parsing, MIT code; trained
  on CelebAMask-HQ, non-commercial research and education): labels every pixel of the face as skin,
  eyebrow, eye, glasses, nose, lips, hair, hat... so we can check that each eye is open and visible,
  that no glasses or sunglasses cover the eyes, that the nose and lips are not covered (face mask), and
  that hair or a hat does not cover the forehead.
- Hands (Google MediaPipe palm detector + hand landmark model, Apache-2.0, ONNX from the OpenCV Model
  Zoo): a hand touching or covering the face. Hands are labelled as skin by face parsing, so they need
  their own detector.
- CLIP (openai/clip-vit-base-patch32, MIT; quantized ONNX by Xenova) image embedding with two small
  logistic-regression classifiers trained for this project (assets/clip_probes.json): drawing or
  cartoon vs real photo, and hand on face.

Models are downloaded on first use (about 150 MB) into models/photo_check/ and checked against their
SHA-256. Calibration and evaluation: docs/photo_check.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from skiniq.config import MODELS_DIR
from skiniq.vision.weights import ensure_model

MODEL_DIR = MODELS_DIR / "photo_check"
_HF = "https://huggingface.co"
MODELS = {
    "face_parsing": ("face_parsing_bisenet_resnet18.onnx",
                     f"{_HF}/jbrownkramer/face-parsing/resolve/main/resnet18.onnx",
                     "0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f"),
    "palm": ("palm_detection_mediapipe_2023feb.onnx",
             f"{_HF}/opencv/palm_detection_mediapipe/resolve/main/palm_detection_mediapipe_2023feb.onnx",
             "78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c"),
    "hand_landmarks": ("handpose_estimation_mediapipe_2023feb.onnx",
                       f"{_HF}/opencv/handpose_estimation_mediapipe/resolve/main/handpose_estimation_mediapipe_2023feb.onnx",
                       "db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c"),
    "clip_vision": ("clip_vit_b32_vision_quantized.onnx",
                    f"{_HF}/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_quantized.onnx",
                    "583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299"),
}
PROBES_PATH = Path(__file__).resolve().parent / "assets" / "clip_probes.json"

# face-parsing labels (CelebAMask-HQ order used by face-parsing.PyTorch / yakhyo)
BG, SKIN, LBROW, RBROW, LEYE, REYE, GLASSES, LEAR, REAR, EARRING, NOSE, MOUTH, ULIP, LLIP, NECK, NECKL, CLOTH, HAIR, HAT = \
    range(19)

PARSE_SIZE = 512
CROP_SCALE = 1.7          # face-parsing crop: square of 1.7 x the face box, centred on it

# Calibrated on 3,923 dataset selfies (contact-sheet review) - see docs/photo_check.md
PARTS = {
    "min_eye_area": 0.0015,        # visible eye pixels / (eye distance)^2 below this = closed or covered
    "min_glasses_share": 0.15,     # share of BOTH eye areas labelled glasses = glasses / sunglasses
    "max_lens_lightness": 0.55,    # lens lightness / skin lightness below this = sunglasses (dark lenses)
    "min_nose_share": 0.10,        # share of the nose area labelled nose
    "min_lips_share": 0.05,        # share of the mouth area labelled lips or mouth
    "max_forehead_covered": 0.55,  # share of the band above the eyebrows labelled hair or hat
    "min_skin_redness": 3.0,       # median a* of the skin; below = sheet/clay mask or a colour filter
    "min_highlights": 65.0,        # L* of the brightest 5% of the photo; below = photo too dark
    "max_greyscale_saturation": 25,  # 90th percentile HSV saturation of the photo; below = black and white
}
HANDS = {"palm_score": 0.8, "landmark_conf": 0.5, "palm_candidate": 0.3, "min_overlap": 0.15}


class ModelsUnavailable(RuntimeError):
    """A photo-check model could not be downloaded."""


def model_path(name: str, download: bool = True) -> Path:
    filename, url, sha = MODELS[name]
    path = MODEL_DIR / filename
    if download:
        try:
            ensure_model(path, url, sha)
        except Exception as e:  # ModelUnavailable or a bad checksum
            raise ModelsUnavailable(f"{name}: {e}") from None
    return path


@lru_cache(maxsize=None)
def _onnx(name: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(model_path(name)), opts, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=None)
def _palm_detector(score: float):
    from skiniq.vision.vendor.mp_palmdet import MPPalmDet

    return MPPalmDet(str(model_path("palm")), scoreThreshold=score)


@lru_cache(maxsize=None)
def _hand_landmarks():
    from skiniq.vision.vendor.mp_handpose import MPHandPose

    return MPHandPose(str(model_path("hand_landmarks")), confThreshold=0.0)


@lru_cache(maxsize=1)
def _probes() -> dict:
    return json.loads(PROBES_PATH.read_text())


def available() -> bool:
    """True if every model is present or can be downloaded."""
    try:
        for name in MODELS:
            model_path(name)
        return True
    except ModelsUnavailable:
        return False


# ---- face parsing ----------------------------------------------------------------------------------

@dataclass
class FaceParse:
    labels: np.ndarray     # PARSE_SIZE x PARSE_SIZE label ids
    valid: np.ndarray      # True where the crop is inside the photo
    points: np.ndarray     # 5 landmarks (eye, eye, nose, mouth, mouth) in crop pixels
    crop: np.ndarray       # the RGB crop


def parse_face(rgb: np.ndarray, face: np.ndarray) -> FaceParse:
    x, y, w, h = (float(v) for v in face[:4])
    side = max(w, h) * CROP_SCALE
    s = PARSE_SIZE / side
    m = np.float32([[s, 0, -(x + w / 2 - side / 2) * s], [0, s, -(y + h / 2 - side / 2) * s]])
    crop = cv2.warpAffine(rgb, m, (PARSE_SIZE, PARSE_SIZE), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    valid = cv2.warpAffine(np.full(rgb.shape[:2], 255, np.uint8), m, (PARSE_SIZE, PARSE_SIZE),
                           flags=cv2.INTER_NEAREST) > 0
    valid &= crop.max(axis=2) > 6   # pure black (letterbox bars, cut-outs) is not hair: ignore it
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    blob = ((crop / 255.0 - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
    logits = _onnx("face_parsing").run(None, {"input": blob})[0][0]
    points = np.c_[np.asarray(face[4:14], np.float64).reshape(5, 2), np.ones(5)] @ m.T
    return FaceParse(logits.argmax(0).astype(np.uint8), valid, points, crop)


def _box(arr: np.ndarray, cx: float, cy: float, half_w: float, half_h: float) -> np.ndarray:
    return arr[max(0, int(cy - half_h)):max(0, int(cy + half_h)), max(0, int(cx - half_w)):max(0, int(cx + half_w))]


def parts_measures(p: FaceParse) -> dict[str, float]:
    """How visible each part of the face is, from the parsed labels (areas outside the photo are ignored)."""
    lab = np.where(p.valid, p.labels, 255)     # 255 = outside the photo
    e1, e2 = p.points[0], p.points[1]
    left, right = (e1, e2) if e1[0] <= e2[0] else (e2, e1)
    iod = max(float(np.hypot(*(right - left))), 1.0)

    def share(region: np.ndarray, ids) -> float:
        inside = region != 255
        return float(np.isin(region[inside], ids).mean()) if inside.any() else 0.0

    m: dict[str, float] = {}
    lum = cv2.cvtColor(p.crop, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) * (100 / 255)
    lum_box, lens = [], []
    for name, e in (("left", left), ("right", right)):
        region = _box(lab, e[0], e[1], 0.45 * iod, 0.3 * iod)
        m[f"eye_{name}"] = float(np.isin(region, (LEYE, REYE)).sum()) / (iod * iod)
        m[f"glasses_{name}"] = share(region, (GLASSES,))
        lens.append(_box(lum, e[0], e[1], 0.45 * iod, 0.3 * iod)[region == GLASSES])
    skin_px = (p.labels == SKIN) & p.valid
    skin_l = float(np.median(lum[skin_px])) if skin_px.sum() > 50 else 50.0
    lens_px = np.concatenate(lens)
    # through a clear lens the eye and skin show (light); sunglasses are dark
    m["lens_lightness"] = float(np.median(lens_px)) / max(skin_l, 1.0) if len(lens_px) > 30 else 1.0
    m["nose"] = share(_box(lab, p.points[2][0], p.points[2][1] - 0.15 * iod, 0.35 * iod, 0.4 * iod), (NOSE,))
    mouth = (p.points[3] + p.points[4]) / 2
    m["lips"] = share(_box(lab, mouth[0], mouth[1], 0.55 * iod, 0.35 * iod), (ULIP, LLIP, MOUTH))
    cx, ey = (left[0] + right[0]) / 2, (left[1] + right[1]) / 2
    m["forehead_covered"] = share(_box(lab, cx, ey - 0.75 * iod, 0.6 * iod, 0.25 * iod), (HAIR, HAT))
    skin = cv2.cvtColor(p.crop, cv2.COLOR_RGB2LAB)[(p.labels == SKIN) & p.valid].astype(np.float32)
    m["skin_redness"] = float(np.median(skin[:, 1]) - 128) if len(skin) > 50 else 0.0
    return m


def parts_problems(m: dict[str, float], colour_photo: bool = True) -> list[str]:
    """Problem codes from parts_measures (most specific first)."""
    t = PARTS
    codes = []
    if min(m["glasses_left"], m["glasses_right"]) >= t["min_glasses_share"]:
        codes.append("sunglasses" if m["lens_lightness"] < t["max_lens_lightness"] else "glasses")
    elif min(m["eye_left"], m["eye_right"]) < t["min_eye_area"]:
        codes.append("eye_hidden")
    if m["nose"] < t["min_nose_share"] or m["lips"] < t["min_lips_share"]:
        codes.append("mask")
    if m["forehead_covered"] > t["max_forehead_covered"]:
        codes.append("forehead_covered")
    if colour_photo and m["skin_redness"] < t["min_skin_redness"] and "mask" not in codes:
        codes.append("unnatural_skin")
    return codes


# ---- colour, exposure --------------------------------------------------------------------------------

def is_greyscale(rgb: np.ndarray) -> bool:
    small = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
    sat = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[..., 1]
    return float(np.percentile(sat, 90)) < PARTS["max_greyscale_saturation"]


def highlights(rgb: np.ndarray) -> float:
    """Lightness (L*, 0-100) of the brightest 5% of the photo. In a well-lit photo something is bright
    (background, clothes, eyes, skin highlights) whatever the person's skin tone; in a dark photo nothing is."""
    small = cv2.resize(rgb, (256, max(1, round(256 * rgb.shape[0] / rgb.shape[1]))), interpolation=cv2.INTER_AREA)
    lum = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) * (100 / 255)
    return float(np.percentile(lum, 95))


# ---- hands ---------------------------------------------------------------------------------------------

def hand_on_face(rgb: np.ndarray, face: np.ndarray) -> tuple[bool, float, float]:
    """(hand found, best palm score, best landmark confidence) for hands overlapping the face.

    The palm detector runs on the whole photo and on a close-up around the face (a hand on the face is
    then larger); each palm overlapping the face is confirmed by the hand landmark model.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    x, y, w, h = (float(v) for v in face[:4])
    fx0, fy0, fx1, fy1 = x - 0.1 * w, y - 0.1 * h, x + 1.1 * w, y + 1.1 * h
    side = 2.2 * max(w, h)
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    close = bgr[y0:int(cy + side / 2), x0:int(cx + side / 2)]
    det, landmarks = _palm_detector(HANDS["palm_candidate"]), _hand_landmarks()
    best_palm = best_conf = 0.0
    for img, (ox, oy) in ((bgr, (0, 0)), (close, (x0, y0))):
        if img.size == 0:
            continue
        for palm in det.infer(img):
            px0, py0, px1, py1 = palm[0] + ox, palm[1] + oy, palm[2] + ox, palm[3] + oy
            overlap = max(0, min(px1, fx1) - max(px0, fx0)) * max(0, min(py1, fy1) - max(py0, fy0))
            if overlap < HANDS["min_overlap"] * (px1 - px0) * (py1 - py0):
                continue
            best_palm = max(best_palm, float(palm[-1]))
            res = landmarks.infer(img, palm)
            if res is not None:
                best_conf = max(best_conf, float(res[-1]))
    return best_palm >= HANDS["palm_score"] or best_conf >= HANDS["landmark_conf"], best_palm, best_conf


# ---- CLIP classifiers ----------------------------------------------------------------------------------

def clip_embedding(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = 224 / min(h, w)
    im = cv2.resize(rgb, (max(224, round(w * s)), max(224, round(h * s))), interpolation=cv2.INTER_CUBIC)
    y0, x0 = (im.shape[0] - 224) // 2, (im.shape[1] - 224) // 2
    im = im[y0:y0 + 224, x0:x0 + 224]
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    blob = ((im / 255.0 - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
    e = _onnx("clip_vision").run(None, {"pixel_values": blob})[0][0]
    return e / np.linalg.norm(e)


def clip_probability(embedding: np.ndarray, probe: str) -> float:
    p = _probes()[probe]
    z = float(np.dot(embedding, np.asarray(p["weights"], np.float32)) + p["bias"])
    return 1.0 / (1.0 + np.exp(-z))


def clip_threshold(probe: str) -> float:
    return float(_probes()[probe]["threshold"])
