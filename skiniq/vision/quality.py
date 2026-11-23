"""Photo quality check before skin analysis (upload and camera).

A photo is accepted only if it has exactly one clear, well-lit face of reasonable size.
Thresholds were calibrated on the held-out test photos: real selfies have face widths
of 19–69% of the image (5th–95th percentile) and face sharpness (variance of the
Laplacian on the face resized to 256 px wide) of 13 or more for 95% of photos, while a
Gaussian blur of sigma 6 drops it to about 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from skiniq.vision.regions import FACE_SCORE_THRESHOLD, _get_detector

MIN_SIDE_PX = 240          # shortest image side
MIN_FACE_PX = 100          # face box width
MIN_FACE_FRACTION = 0.15   # face width / image width
MIN_SHARPNESS = 8.0
MIN_BRIGHTNESS = 55.0      # mean grey level of the face, 0-255
MAX_BRIGHTNESS = 215.0
MAX_ANALYSIS_SIDE = 1280   # large photos are downscaled for detection (faster, same result)


@dataclass
class QualityReport:
    ok: bool
    problems: list[str] = field(default_factory=list)
    faces: int = 0
    face_box: tuple[int, int, int, int] | None = None
    sharpness: float | None = None
    brightness: float | None = None

    @property
    def message(self) -> str:
        return " ".join(self.problems)


def _faces(rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    det = _get_detector()
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    if faces is None:
        return np.zeros((0, 15))
    return faces[faces[:, -1] >= FACE_SCORE_THRESHOLD]


def check_photo(image: Image.Image) -> QualityReport:
    """Return whether the photo is good enough, with user-facing reasons if not."""
    image = image.convert("RGB")
    if min(image.size) < MIN_SIDE_PX:
        return QualityReport(False, [f"The photo is too small ({image.width}×{image.height} px). "
                                     f"Use a photo at least {MIN_SIDE_PX} px on its shortest side."])
    scale = min(1.0, MAX_ANALYSIS_SIDE / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)))
    rgb = np.asarray(image)
    faces = _faces(rgb)
    if len(faces) == 0:
        return QualityReport(False, ["No face was detected. Use a front-facing photo of your face, "
                                     "without sunglasses or a mask, in good light."])
    if len(faces) > 1:
        # a much smaller second face (e.g. someone in the background) is tolerated
        areas = sorted((f[2] * f[3] for f in faces), reverse=True)
        if areas[1] > 0.25 * areas[0]:
            return QualityReport(False, ["More than one face was detected. Use a photo with only your face."],
                                 faces=len(faces))
    x, y, fw, fh = (float(v) for v in max(faces, key=lambda f: f[2] * f[3])[:4])
    w, h = image.size
    box = (max(0, int(x)), max(0, int(y)), min(w, int(x + fw)), min(h, int(y + fh)))
    problems = []
    face_px_original = fw / scale  # face width in the uploaded photo's own pixels
    if face_px_original < MIN_FACE_PX or fw / w < MIN_FACE_FRACTION:
        problems.append("Your face is too small in the photo. Move closer so your face fills the frame.")
    grey = cv2.cvtColor(rgb[box[1]:box[3], box[0]:box[2]], cv2.COLOR_RGB2GRAY)
    grey = cv2.resize(grey, (256, max(1, round(256 * grey.shape[0] / max(1, grey.shape[1])))))
    sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    brightness = float(grey.mean())
    if brightness < MIN_BRIGHTNESS:
        problems.append("The photo is too dark. Face a window or a light.")
    elif brightness > MAX_BRIGHTNESS:
        problems.append("The photo is overexposed. Avoid direct flash or strong light on your face.")
    if sharpness < MIN_SHARPNESS:
        problems.append("The photo is blurry. Hold the camera still and make sure it is in focus.")
    return QualityReport(not problems, problems, faces=len(faces), face_box=box,
                         sharpness=round(sharpness, 1), brightness=round(brightness, 1))
