"""Photo quality check before skin analysis (upload and camera).

A photo is accepted only if it shows exactly one face that is fully visible, framed head
and shoulders like the camera outline, looking straight at the camera, sharp and evenly lit.

How it works: YuNet finds the face and five landmarks (eyes, nose tip, mouth corners). The
face is warped so the eyes and mouth land on fixed points, then each part of the face is
measured against the person's *own* skin, so the checks work the same for every skin tone:

- eyes: a visible eye has contrast (white, iris, lashes); a covered eye is flat and
  skin-coloured; sunglasses are dark and flat
- lower face: a mask makes the nose, mouth and chin differ in colour from the forehead and
  hides the lips (smooth, not redder than the forehead)
- left vs right: a hand, hair or object on one side changes its colour or removes its detail
- glasses: a frame gives strong edges across the nose bridge and under the eyes
- lighting: clipped highlights, crushed shadows and one-sided shadow on the skin itself
  (not average brightness, which would penalise darker skin)

Thresholds were calibrated on 900 selfies from the project's skin-type dataset and on copies
of them with sunglasses, glasses, masks and covered halves drawn on (see docs/photo_check.md).
They are in VISIBILITY, LIGHTING and FRAMING; the browser camera guide uses the same values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps

from skiniq.vision import face_models as fm

from skiniq.vision.regions import FACE_SCORE_THRESHOLD, _get_detector

MIN_SIDE_PX = 240          # shortest image side
MIN_FACE_PX = 100          # face box width in the original photo
MIN_SHARPNESS = 8.0
MAX_ANALYSIS_SIDE = 1280   # large photos are downscaled for detection (faster, same result)

# Head-and-shoulders framing, as fractions of the photo (the camera outline uses the same values)
FRAMING = {
    "min_face_height": 0.30,   # face box height / photo height: smaller = too far away
    "max_face_height": 0.60,   # larger = too close (shoulders not in the photo)
    "min_centre_x": 0.35, "max_centre_x": 0.65,
    "min_centre_y": 0.25, "max_centre_y": 0.55,
    "max_chin": 0.80,          # bottom of the face box, leaving room for neck and shoulders
    "edge_margin": 0.02,       # the face box must not touch the photo's edges
    "max_roll_deg": 15.0,      # head tilted sideways
    "max_yaw": 0.25,           # nose off the eye midline, as a share of the eye distance (~15°): head turned
}

# Uploaded photos: a close-up is fine (no need to show the neck or shoulders), but the whole face must be in
# the photo with a little room around it, centred and facing the camera. Captured photos keep FRAMING.
UPLOAD_FRAMING = {
    **FRAMING,
    "max_face_height": 1.0,    # no "too close" limit
    "max_chin": 1.0,           # the chin may be near the bottom (neck not needed)
    "max_centre_y": 0.65,      # a close-up face sits around the middle of the photo
    "face_margin": 0.08,       # room around the face box, as a share of the face size (forehead and chin in frame)
}

# Visibility of the face (measured on the aligned face; see face_visibility)
VISIBILITY = {
    "eye_min_contrast": 0.06,        # eye contrast / skin lightness below this = eye covered...
    "eye_covered_min_rel": 0.55,     # ...if the eye is not dark (else it is sunglasses)
    "sunglasses_max_rel": 0.35,      # both eyes this dark relative to the skin...
    "sunglasses_min_dark": 0.80,     # ...and mostly dark pixels
    "mask_min_colour_diff": 18.0,    # lower face vs forehead colour difference (a*, b*)...
    "mask_max_lip_red": 4.0,         # ...with no lips redder than the forehead
    "mask_max_mouth_texture": 0.035, # or a smooth mouth area...
    "mask_max_lip_red_flat": 3.0,    # ...with no red lips
    "half_max_colour_diff": 20.0,    # left vs right cheek colour difference
    "half_min_texture_ratio": 0.50,  # detail of the plainer half / the other half
    "glasses_min_bridge_edges": 2.8, # edge strength across the nose bridge (vs cheeks)...
    "glasses_min_under_edges": 2.2,  # ...and along the lower rims
}

LIGHTING = {
    "max_clipped": 0.10,       # share of skin pixels blown out to white
    "max_crushed": 0.20,       # share of skin pixels crushed to black (0.10 flagged well-lit darker skin)
    "max_side_diff": 30.0,     # left vs right cheek lightness difference (L*, 0-100); even across skin tones
}

MESSAGES = {
    "too_far": "Move closer: your head and shoulders should fill the photo, like the outline in the camera.",
    "too_close": "Move back: show your head and shoulders, not just your face.",
    "off_centre": "Centre your face in the photo, with your shoulders below it.",
    "cut_off": "Your whole face must be in the photo. Move back or centre your face.",
    "tilted": "Keep your head straight, not tilted to one side.",
    "turned": "Face the camera straight on, not turned to one side.",
    "sunglasses": "Sunglasses detected. Remove them so your eyes and the skin around them can be seen.",
    "glasses": "Glasses detected. Remove them so the skin around your eyes can be analysed.",
    "mask": "Your nose or mouth is covered (for example by a face mask). Remove anything covering your face.",
    "eye_hidden": "One or both of your eyes are closed or covered. Keep both eyes open, look at the camera, "
                  "and move hair or hands away from them.",
    "forehead_covered": "Your forehead is covered (by hair, a hat or a headband). Pull your hair back so your "
                        "whole forehead shows.",
    "hand": "A hand is touching or covering your face. Keep your hands away from your face.",
    "unnatural_skin": "Your skin colour looks unnatural, as if covered by a sheet or clay mask or changed by a "
                      "filter. Remove any mask and use a photo without filters.",
    "greyscale": "This is a black-and-white photo. Use a colour photo so your skin can be analysed.",
    "drawing": "This looks like a drawing, cartoon or illustration. Use a real photo of your face.",
    "eye_covered": "One of your eyes is covered. Move hair, hands or objects away and look at the camera.",
    "half_covered": "Part of your face is covered on one side. Move hair, hands or objects away from your face.",
    "too_dark": "The photo is too dark. Face a window or a light so your whole face is lit.",
    "too_bright": "The photo is overexposed. Avoid direct flash or strong light on your face.",
    "uneven_light": "One side of your face is in shadow. Face the light so both sides are evenly lit.",
    "blurry": "The photo is blurry. Hold the camera still and make sure it is in focus.",
}


@dataclass
class QualityReport:
    ok: bool
    problems: list[str] = field(default_factory=list)
    faces: int = 0
    face_box: tuple[int, int, int, int] | None = None
    sharpness: float | None = None
    brightness: float | None = None
    codes: list[str] = field(default_factory=list)       # machine-readable problem codes
    measures: dict[str, float] = field(default_factory=dict)

    @property
    def message(self) -> str:
        return " ".join(self.problems)


def settings() -> dict:
    """Thresholds shared with the browser camera guide."""
    return {"framing": FRAMING, "visibility": VISIBILITY, "lighting": LIGHTING, "messages": MESSAGES,
            "parts": fm.PARTS, "hands": fm.HANDS}


# ---- face alignment and region measurements ----------------------------------------------------

CANON_W, CANON_H = 192, 224
CANON_EYE_L, CANON_EYE_R, CANON_MOUTH = (64, 80), (128, 80), (96, 152)
# (centre x, centre y, width, height) in the aligned face
PATCHES = {
    "eye_l": (64, 80, 34, 18), "eye_r": (128, 80, 34, 18),
    "brow_mid": (96, 62, 22, 14), "forehead": (96, 34, 64, 18),
    "cheek_l": (58, 122, 26, 24), "cheek_r": (134, 122, 26, 24),
    "nose": (96, 118, 18, 16), "mouth": (96, 152, 40, 14), "chin": (96, 188, 36, 16),
    "bridge": (96, 82, 14, 14), "under_l": (64, 101, 34, 10), "under_r": (128, 101, 34, 10),
}
HALF_L, HALF_R = (slice(50, 200), slice(30, 88)), (slice(50, 200), slice(104, 162))
SKIN_PATCHES = ("cheek_l", "cheek_r", "nose", "forehead")


def _landmarks(face: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(image-left eye, image-right eye, nose tip, mouth centre) from a YuNet row."""
    e1, e2 = face[4:6].astype(np.float32), face[6:8].astype(np.float32)
    left, right = (e1, e2) if e1[0] <= e2[0] else (e2, e1)
    mouth = ((face[10:12] + face[12:14]) / 2).astype(np.float32)
    return left, right, face[8:10].astype(np.float32), mouth


def align_face(rgb: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Warp the face so the eyes and the mouth centre land on fixed points (CANON_W x CANON_H)."""
    left, right, _, mouth = _landmarks(face)
    m = cv2.getAffineTransform(np.float32([left, right, mouth]),
                               np.float32([CANON_EYE_L, CANON_EYE_R, CANON_MOUTH]))
    return cv2.warpAffine(rgb, m, (CANON_W, CANON_H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def _patch(arr: np.ndarray, name: str) -> np.ndarray:
    x, y, w, h = PATCHES[name]
    return arr[y - h // 2:y + h // 2, x - w // 2:x + w // 2]


def face_visibility(rgb: np.ndarray, face: np.ndarray) -> dict[str, float]:
    """Measurements of how visible each part of the face is, and how it is lit."""
    canon = align_face(rgb, face)
    lab = cv2.cvtColor(canon, cv2.COLOR_RGB2LAB).astype(np.float32)
    lum, a_ch, b_ch = lab[..., 0] * (100 / 255), lab[..., 1] - 128, lab[..., 2] - 128

    def median_lab(name: str) -> np.ndarray:
        return np.array([np.median(_patch(ch, name)) for ch in (lum, a_ch, b_ch)])

    med = {n: median_lab(n) for n in PATCHES}
    skin = np.median(np.stack([med["cheek_l"], med["cheek_r"], med["nose"]]), axis=0)
    upper = np.median(np.stack([med["brow_mid"], med["forehead"]]), axis=0)
    lower = np.median(np.stack([med["nose"], med["mouth"], med["chin"]]), axis=0)
    skin_l = max(float(skin[0]), 1.0)

    m: dict[str, float] = {}
    for side in "lr":
        eye = _patch(lum, f"eye_{side}")
        m[f"eye_{side}_rel"] = float(np.mean(eye)) / skin_l
        m[f"eye_{side}_dark"] = float(np.mean(eye < 0.6 * skin_l))
        m[f"eye_{side}_contrast"] = float(np.std(eye)) / skin_l
    m["lower_colour_diff"] = float(np.linalg.norm(lower[1:] - upper[1:]))
    m["lip_red"] = float(med["mouth"][1] - upper[1])
    m["mouth_texture"] = float(np.std(_patch(lum, "mouth"))) / max(float(upper[0]), 1.0)
    m["cheek_colour_diff"] = float(np.linalg.norm(med["cheek_l"][1:] - med["cheek_r"][1:]))
    m["cheek_light_diff"] = float(abs(med["cheek_l"][0] - med["cheek_r"][0]))

    grey = cv2.cvtColor(canon, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = np.abs(cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3))
    grad = np.hypot(gx, gy)
    cheek_detail = (float(np.mean(_patch(grad, "cheek_l"))) + float(np.mean(_patch(grad, "cheek_r")))) / 2 + 4
    m["bridge_edges"] = float(np.mean(_patch(grad, "bridge"))) / cheek_detail
    m["under_edges"] = max(float(np.mean(_patch(gy, "under_l"))), float(np.mean(_patch(gy, "under_r")))) / cheek_detail
    left, right = float(np.mean(grad[HALF_L])), float(np.mean(grad[HALF_R]))
    m["half_texture_ratio"] = min(left, right) / max(left, right, 1e-6)

    skin_px = np.concatenate([_patch(canon, n).reshape(-1, 3) for n in SKIN_PATCHES])
    m["clipped"] = float(np.mean(skin_px.min(axis=1) >= 250))
    m["crushed"] = float(np.mean(skin_px.max(axis=1) <= 25))
    m["skin_lightness"] = skin_l
    return m


def visibility_problems(m: dict[str, float]) -> list[str]:
    """Problem codes from face_visibility measurements (order: most specific first)."""
    v, li = VISIBILITY, LIGHTING
    codes = []
    eyes_rel = (m["eye_l_rel"], m["eye_r_rel"])
    if max(eyes_rel) < v["sunglasses_max_rel"] and min(m["eye_l_dark"], m["eye_r_dark"]) > v["sunglasses_min_dark"]:
        codes.append("sunglasses")
    elif m["bridge_edges"] > v["glasses_min_bridge_edges"] and m["under_edges"] > v["glasses_min_under_edges"]:
        codes.append("glasses")
    if any(m[f"eye_{s}_contrast"] < v["eye_min_contrast"] and m[f"eye_{s}_rel"] > v["eye_covered_min_rel"]
           for s in "lr"):
        codes.append("eye_covered")
    if (m["lower_colour_diff"] > v["mask_min_colour_diff"] and m["lip_red"] < v["mask_max_lip_red"]) or \
            (m["mouth_texture"] < v["mask_max_mouth_texture"] and m["lip_red"] < v["mask_max_lip_red_flat"]):
        codes.append("mask")
        if "glasses" in codes:  # a mask's top edge runs under the eyes like a glasses rim
            codes.remove("glasses")
    if m["cheek_colour_diff"] > v["half_max_colour_diff"] or m["half_texture_ratio"] < v["half_min_texture_ratio"]:
        if "eye_covered" not in codes:
            codes.append("half_covered")
    if m["crushed"] > li["max_crushed"]:
        codes.append("too_dark")
    elif m["clipped"] > li["max_clipped"]:
        codes.append("too_bright")
    if m["cheek_light_diff"] > li["max_side_diff"]:
        codes.append("uneven_light")
    return codes


def framing_problems(face: np.ndarray, width: int, height: int, framing: dict | None = None) -> list[str]:
    """Problem codes for framing and head pose (head and shoulders by default; UPLOAD_FRAMING for uploads)."""
    f = FRAMING if framing is None else framing
    x, y, w, h = (float(v) for v in face[:4])
    codes = []
    mx = max(f["edge_margin"] * width, f.get("face_margin", 0.0) * w)
    my = max(f["edge_margin"] * height, f.get("face_margin", 0.0) * h)
    if x < mx or y < my or x + w > width - mx or y + h > height - my:
        codes.append("cut_off")
    face_h = h / height
    if face_h < f["min_face_height"]:
        codes.append("too_far")
    elif (f["max_face_height"] < 1.0 or f["max_chin"] < 1.0) and (
            face_h > f["max_face_height"] or (y + h) / height > f["max_chin"]):
        codes.append("too_close")
        if "cut_off" in codes:  # an oversized face runs off the edges: moving back fixes both
            codes.remove("cut_off")
    cx, cy = (x + w / 2) / width, (y + h / 2) / height
    if "cut_off" not in codes and not (f["min_centre_x"] <= cx <= f["max_centre_x"]
                                       and f["min_centre_y"] <= cy <= f["max_centre_y"]):
        codes.append("off_centre")
    left, right, nose, _ = _landmarks(face)
    eye_dist = float(np.hypot(*(right - left)))
    if eye_dist > 0:
        if abs(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0]))) > f["max_roll_deg"]:
            codes.append("tilted")
        if abs(float(nose[0]) - float(left[0] + right[0]) / 2) / eye_dist > f["max_yaw"]:
            codes.append("turned")
    return codes


# ---- the check ----------------------------------------------------------------------------------

def model_problems(rgb: np.ndarray, face: np.ndarray, measures: dict[str, float]) -> list[str]:
    """Problem codes from the trained models plus the lighting measures (raises ModelsUnavailable)."""
    embedding = fm.clip_embedding(rgb)
    if fm.clip_probability(embedding, "drawing") >= fm.clip_threshold("drawing"):
        return ["drawing"]
    grey = fm.is_greyscale(rgb)
    parse = fm.parse_face(rgb, face)
    parts = fm.parts_measures(parse)
    measures.update({f"parts_{k}": v for k, v in parts.items()})
    codes = ["greyscale"] if grey else []
    codes += fm.parts_problems(parts, colour_photo=not grey)
    hand, palm, conf = fm.hand_on_face(rgb, face)
    measures.update(hand_palm=palm, hand_landmarks=conf)
    if hand:                                      # pre-trained MediaPipe palm + hand-landmark models only
        codes.append("hand")
    measures["highlights"] = fm.highlights(rgb)
    li = LIGHTING
    if measures["crushed"] > li["max_crushed"] or measures["highlights"] < fm.PARTS["min_highlights"]:
        codes.append("too_dark")
    elif measures["clipped"] > li["max_clipped"]:
        codes.append("too_bright")
    if measures["cheek_light_diff"] > li["max_side_diff"] and "hand" not in codes:
        codes.append("uneven_light")
    return codes


def _faces(rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    det = _get_detector()
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    if faces is None:
        return np.zeros((0, 15))
    return faces[faces[:, -1] >= FACE_SCORE_THRESHOLD]


def check_photo(image: Image.Image, source: str = "camera") -> QualityReport:
    """Return whether the photo is good enough, with user-facing reasons if not.

    ``source="upload"`` relaxes the framing (UPLOAD_FRAMING: a close-up of the whole face is fine);
    ``"camera"`` (captured photos) keeps head-and-shoulders framing, as the camera guide shows.

    Uses the trained models in skiniq.vision.face_models (face parts, hands, drawings); if they cannot
    be downloaded, falls back to the rule-based visibility checks (face_visibility / visibility_problems).
    """
    image = ImageOps.exif_transpose(image).convert("RGB")   # phone photos store their rotation separately
    if min(image.size) < MIN_SIDE_PX:
        return QualityReport(False, [f"The photo is too small ({image.width}×{image.height} px). "
                                     f"Use a photo at least {MIN_SIDE_PX} px on its shortest side."],
                             codes=["photo_too_small"])
    scale = min(1.0, MAX_ANALYSIS_SIDE / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)))
    rgb = np.asarray(image)
    faces = _faces(rgb)
    if len(faces) == 0:
        return QualityReport(False, ["No face was detected. Use a front-facing photo of your head and shoulders, "
                                     "with nothing covering your face, in good light."], codes=["no_face"])
    if len(faces) > 1:
        # a much smaller second face (e.g. someone in the background) is tolerated
        areas = sorted((f[2] * f[3] for f in faces), reverse=True)
        if areas[1] > 0.25 * areas[0]:
            return QualityReport(False, ["More than one face was detected. Use a photo with only your face."],
                                 faces=len(faces), codes=["several_faces"])
    face = max(faces, key=lambda f: f[2] * f[3])
    x, y, fw, fh = (float(v) for v in face[:4])
    w, h = image.size
    box = (max(0, int(x)), max(0, int(y)), min(w, int(x + fw)), min(h, int(y + fh)))

    if source not in ("camera", "upload"):
        raise ValueError(f"unknown photo source: {source}")
    framing = framing_problems(face, w, h, UPLOAD_FRAMING if source == "upload" else FRAMING)
    if fw / scale < MIN_FACE_PX and "too_far" not in framing:
        framing.append("too_far")
    measures = face_visibility(rgb, face)
    try:
        visibility = model_problems(rgb, face, measures)
    except fm.ModelsUnavailable:
        visibility = visibility_problems(measures)      # rule-based fallback (offline)
    if "drawing" in visibility:
        return QualityReport(False, [MESSAGES["drawing"]], faces=len(faces), face_box=box, codes=["drawing"],
                             measures={k: round(v, 4) for k, v in measures.items()})
    # something covering the face can shift the landmarks and look like a turned head: report it first
    pose = [c for c in framing if c in ("tilted", "turned")]
    codes = [c for c in framing if c not in pose] + visibility + pose

    grey = cv2.cvtColor(rgb[box[1]:box[3], box[0]:box[2]], cv2.COLOR_RGB2GRAY)
    grey = cv2.resize(grey, (256, max(1, round(256 * grey.shape[0] / max(1, grey.shape[1])))))
    sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    if sharpness < MIN_SHARPNESS:
        codes.append("blurry")
    problems = [MESSAGES[c] for c in dict.fromkeys(codes)]
    return QualityReport(not problems, problems, faces=len(faces), face_box=box, sharpness=round(sharpness, 1),
                         brightness=round(float(grey.mean()), 1), codes=list(dict.fromkeys(codes)),
                         measures={k: round(v, 4) for k, v in measures.items()})
