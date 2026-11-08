"""Region-based skin condition analysis.

The condition model gives one main condition per image (its six probabilities
sum to 1). To find several conditions on one face, and where they are, the face
is located with YuNet, aligned so the eyes are level, and cut into five regions
(forehead, nose, left cheek, right cheek, chin). Each region and the whole face
are classified separately. The training images are mostly close-ups, so regions
are also closer to what the model has seen.

When no face is found (for example a close-up of one cheek), the whole image is
analysed instead, as in training.

Status: experimental, not used by the pipeline yet. The condition model has no
"clear skin" class, so on ordinary full-face photos it labels many healthy regions
as blackheads or acne (a condition was reported on 43 of 56 test faces). Region
analysis becomes reliable once the model is retrained with a clear-skin class.

YuNet: https://github.com/opencv/opencv_zoo (MIT licence, see assets/LICENSE-yunet).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from skiniq.config import CONDITION_DETECTED_THRESHOLD, SKIN_CONDITIONS

YUNET_PATH = Path(__file__).resolve().parent / "assets" / "face_detection_yunet_2023mar.onnx"
FACE_SCORE_THRESHOLD = 0.7
MIN_REGION_PX = 48           # smaller crops carry too little detail to classify
MIN_SKIN_FRACTION = 0.5      # skip regions that are mostly background, hair or clothing
REGIONS = ("forehead", "nose", "left_cheek", "right_cheek", "chin")
WHOLE_FACE = "whole_face"
WHOLE_IMAGE = "whole_image"

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 in pixels


@dataclass
class Face:
    box: Box
    right_eye: tuple[float, float]  # the person's right eye (image left)
    left_eye: tuple[float, float]
    nose: tuple[float, float]
    mouth_right: tuple[float, float]
    mouth_left: tuple[float, float]
    score: float


@dataclass
class RegionResult:
    name: str
    box: Box                      # in the aligned image
    probabilities: dict[str, float]

    @property
    def top(self) -> tuple[str, float]:
        c = max(self.probabilities, key=self.probabilities.get)
        return c, self.probabilities[c]


@dataclass
class RegionAnalysis:
    face_found: bool
    regions: list[RegionResult]
    aligned_image: Image.Image | None = field(default=None, repr=False)
    thresholds: dict[str, float] | None = None  # set for multi-label models

    def _found_in(self, r: RegionResult) -> list[str]:
        """Concerns a region shows: every concern over its threshold (multi-label),
        or the confident top prediction (single-label)."""
        if self.thresholds is not None:
            return [c for c in SKIN_CONDITIONS if r.probabilities[c] >= self.thresholds[c]]
        c, p = r.top
        return [c] if p >= CONDITION_DETECTED_THRESHOLD else []

    @property
    def condition_scores(self) -> dict[str, float]:
        """Per condition, the highest probability any region gives it (independent 0-1 scores)."""
        return {c: round(max(r.probabilities[c] for r in self.regions), 4) for c in SKIN_CONDITIONS}

    @property
    def locations(self) -> dict[str, list[str]]:
        """Regions where each condition is the confident top prediction."""
        found: dict[str, list[str]] = {}
        for r in self.regions:
            if r.name in (WHOLE_FACE, WHOLE_IMAGE):
                continue
            for c in self._found_in(r):
                found.setdefault(c, []).append(r.name)
        return found

    def to_image_analysis(self):
        """Region scores as fusion input (highest score per concern across regions)."""
        from skiniq.fusion import ImageAnalysis

        return ImageAnalysis(conditions=self.condition_scores, independent=True,
                             thresholds=self.thresholds, locations=self.locations)


# ---- face detection and alignment ------------------------------------------------------

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), FACE_SCORE_THRESHOLD)
    return _detector


def detect_face(image: Image.Image) -> Face | None:
    """Return the most confident face, or None."""
    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    det = _get_detector()
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    if faces is None or len(faces) == 0:
        return None
    f = max(faces, key=lambda row: row[-1])
    x, y, fw, fh = f[:4]
    pts = [(float(f[i]), float(f[i + 1])) for i in range(4, 14, 2)]
    return Face(
        box=(int(x), int(y), int(x + fw), int(y + fh)),
        right_eye=pts[0], left_eye=pts[1], nose=pts[2], mouth_right=pts[3], mouth_left=pts[4],
        score=float(f[-1]),
    )


def align_face(image: Image.Image, face: Face) -> tuple[Image.Image, Face]:
    """Rotate the image so the eyes are level; return the image and the moved landmarks."""
    (rx, ry), (lx, ly) = face.right_eye, face.left_eye
    angle = float(np.degrees(np.arctan2(ly - ry, lx - rx)))
    if abs(angle) < 2:
        return image, face
    center = ((rx + lx) / 2, (ry + ly) / 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rgb = np.asarray(image.convert("RGB"))
    rotated = cv2.warpAffine(rgb, m, (rgb.shape[1], rgb.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    def move(p):
        return tuple((m @ np.array([p[0], p[1], 1.0])).tolist())

    corners = np.array([[face.box[0], face.box[1], 1], [face.box[2], face.box[1], 1],
                        [face.box[0], face.box[3], 1], [face.box[2], face.box[3], 1]], float) @ m.T
    box = (int(corners[:, 0].min()), int(corners[:, 1].min()), int(corners[:, 0].max()), int(corners[:, 1].max()))
    moved = Face(box, move(face.right_eye), move(face.left_eye), move(face.nose),
                 move(face.mouth_right), move(face.mouth_left), face.score)
    return Image.fromarray(rotated), moved


def region_boxes(face: Face, image_size: tuple[int, int]) -> dict[str, Box]:
    """Region boxes from an eye-levelled face, scaled by the distance between the eyes."""
    w, h = image_size
    (rx, ry), (lx, ly) = face.right_eye, face.left_eye
    d = max(abs(lx - rx), 1.0)
    eye_y = (ry + ly) / 2
    cx = (rx + lx) / 2
    nx, ny = face.nose
    mouth_y = (face.mouth_right[1] + face.mouth_left[1]) / 2
    mouth_x = (face.mouth_right[0] + face.mouth_left[0]) / 2

    boxes = {
        "forehead": (cx - 0.8 * d, eye_y - 1.1 * d, cx + 0.8 * d, eye_y - 0.35 * d),
        "nose": (nx - 0.35 * d, eye_y + 0.15 * d, nx + 0.35 * d, ny + 0.25 * d),
        # The person's left cheek is on the image right
        "left_cheek": (lx - 0.35 * d, ny - 0.15 * d, lx + 0.4 * d, mouth_y + 0.1 * d),
        "right_cheek": (rx - 0.4 * d, ny - 0.15 * d, rx + 0.35 * d, mouth_y + 0.1 * d),
        "chin": (mouth_x - 0.5 * d, mouth_y + 0.2 * d, mouth_x + 0.5 * d, mouth_y + 0.85 * d),
    }
    clipped = {}
    for name, (x0, y0, x1, y1) in boxes.items():
        box = (max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1)))
        if box[2] - box[0] >= MIN_REGION_PX and box[3] - box[1] >= MIN_REGION_PX:
            clipped[name] = box
    return clipped


def skin_fraction(image: Image.Image) -> float:
    """Share of pixels in a common YCrCb skin-colour range (Cr 133-173, Cb 77-127;
    Chai & Ngan, 1999). A coarse filter for background and hair, not a skin detector."""
    ycrcb = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    return float((mask > 0).mean())


# ---- analysis ------------------------------------------------------------------------

def analyse_regions(image: Image.Image, classifier) -> RegionAnalysis:
    """Classify each face region and the whole face (or the whole image if no face).

    `classifier` needs a `predict(PIL image) -> {condition: probability}` method,
    e.g. `skiniq.vision.model.SkinClassifier`.
    """
    image = image.convert("RGB")
    thresholds = getattr(classifier, "thresholds", None) if getattr(classifier, "multilabel", False) else None
    face = detect_face(image)
    if face is None:
        full = (0, 0, image.width, image.height)
        return RegionAnalysis(False, [RegionResult(WHOLE_IMAGE, full, classifier.predict(image))], image,
                              thresholds)

    aligned, face = align_face(image, face)
    crops = {WHOLE_FACE: _pad_box(face.box, aligned.size), **region_boxes(face, aligned.size)}
    results = []
    for name, box in crops.items():
        crop = aligned.crop(box)
        if name != WHOLE_FACE and skin_fraction(crop) < MIN_SKIN_FRACTION:
            continue
        results.append(RegionResult(name, box, classifier.predict(crop)))
    return RegionAnalysis(True, results, aligned, thresholds)


def _pad_box(box: Box, size: tuple[int, int], pad: float = 0.1) -> Box:
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return (max(0, int(x0 - px)), max(0, int(y0 - py)), min(size[0], int(x1 + px)), min(size[1], int(y1 + py)))


def draw_regions(analysis: RegionAnalysis) -> Image.Image:
    """Aligned image with each region outlined and labelled with its top condition."""
    img = np.asarray(analysis.aligned_image).copy()
    for r in analysis.regions:
        if r.name in (WHOLE_FACE, WHOLE_IMAGE):
            continue
        found = analysis._found_in(r)
        text = ", ".join(f"{c} {r.probabilities[c]:.0%}" for c in found) or "clear"
        color = (220, 60, 60) if found else (60, 170, 90)
        cv2.rectangle(img, r.box[:2], r.box[2:], color, 2)
        cv2.putText(img, f"{r.name}: {text}", (r.box[0] + 3, r.box[1] + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return Image.fromarray(img)
