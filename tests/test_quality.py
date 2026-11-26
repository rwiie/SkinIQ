"""Photo quality check: one fully visible face, framed head and shoulders, straight on, well lit."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from skiniq.vision import quality as q
from skiniq.vision.quality import check_photo, framing_problems, visibility_problems

ROOT = Path(__file__).resolve().parent.parent
TEST_SPLIT = ROOT / "data/processed/skin_type/test"   # uncommitted dataset (data/README.md)


# ---- rules on measurements (no photos needed) ------------------------------------------------

CLEAN = {  # typical measurements of a fully visible, evenly lit face (medians of the calibration set)
    "eye_l_rel": 0.65, "eye_r_rel": 0.66, "eye_l_dark": 0.38, "eye_r_dark": 0.37,
    "eye_l_contrast": 0.23, "eye_r_contrast": 0.22, "lower_colour_diff": 5.0, "lip_red": 8.5,
    "mouth_texture": 0.17, "cheek_colour_diff": 4.0, "cheek_light_diff": 6.0, "bridge_edges": 1.2,
    "under_edges": 0.7, "half_texture_ratio": 0.85, "clipped": 0.0, "crushed": 0.0, "skin_lightness": 70.0,
}


def test_clean_face_has_no_visibility_problems():
    assert visibility_problems(CLEAN) == []


@pytest.mark.parametrize("change, code", [
    ({"eye_l_rel": 0.18, "eye_r_rel": 0.2, "eye_l_dark": 1.0, "eye_r_dark": 0.98}, "sunglasses"),
    ({"bridge_edges": 5.3, "under_edges": 5.0}, "glasses"),
    ({"eye_l_contrast": 0.01, "eye_l_rel": 0.9}, "eye_covered"),                       # a hand over one eye
    ({"lower_colour_diff": 31.0, "lip_red": -13.0}, "mask"),                            # blue surgical mask
    ({"mouth_texture": 0.01, "lip_red": 0.0}, "mask"),                                  # white or black mask
    ({"cheek_colour_diff": 24.0}, "half_covered"),
    ({"half_texture_ratio": 0.4}, "half_covered"),                                      # smooth hand on one side
    ({"crushed": 0.3}, "too_dark"),
    ({"clipped": 0.2}, "too_bright"),
    ({"cheek_light_diff": 35.0}, "uneven_light"),
])
def test_each_visibility_problem_is_detected(change, code):
    assert code in visibility_problems({**CLEAN, **change})


def test_dark_eyes_with_contrast_are_not_sunglasses():
    # dark eye makeup or deep-set eyes: dark but with contrast (not both eyes flat and black)
    assert visibility_problems({**CLEAN, "eye_l_rel": 0.4, "eye_r_rel": 0.42}) == []


def test_mask_edge_is_not_also_reported_as_glasses():
    codes = visibility_problems({**CLEAN, "lower_colour_diff": 31.0, "lip_red": -13.0,
                                 "bridge_edges": 3.0, "under_edges": 3.5})
    assert "mask" in codes and "glasses" not in codes


def _face(cx=0.5, cy=0.42, h=0.45, width=640, height=480, roll=0.0, yaw=0.0):
    """A YuNet-style row for a face of height h (share of the photo) centred at (cx, cy)."""
    fh = h * height
    fw = fh * 0.8
    x, y = cx * width - fw / 2, cy * height - fh / 2
    eye_d = fw * 0.4
    mx, my = cx * width, y + fh * 0.4
    dx, dy = eye_d / 2 * np.cos(np.radians(roll)), eye_d / 2 * np.sin(np.radians(roll))
    left, right = (mx - dx, my - dy), (mx + dx, my + dy)
    nose = (mx + yaw * eye_d, y + fh * 0.6)
    return np.array([x, y, fw, fh, *left, *right, *nose, mx - eye_d / 3, y + fh * 0.8,
                     mx + eye_d / 3, y + fh * 0.8, 0.95])


@pytest.mark.parametrize("face, codes", [
    (_face(), []),
    (_face(h=0.2), ["too_far"]),
    (_face(h=0.75), ["too_close"]),                     # also off the bottom edge: "move back" covers it
    (_face(cx=0.8), ["off_centre"]),
    (_face(cy=0.15, h=0.4), ["cut_off"]),
    (_face(roll=25), ["tilted"]),
    (_face(yaw=0.4), ["turned"]),
])
def test_head_and_shoulders_framing(face, codes):
    assert framing_problems(face, 640, 480) == codes


@pytest.mark.parametrize("face, codes", [
    (_face(h=0.75, cy=0.5), []),                        # a close-up of the whole face is fine for uploads
    (_face(h=0.6, cy=0.62), []),                        # chin near the bottom, no neck: fine
    (_face(h=0.9, cy=0.5), ["cut_off"]),                # forehead or chin outside the photo
    (_face(h=0.6, cx=0.72, cy=0.5), ["off_centre"]),
    (_face(h=0.2), ["too_far"]),
    (_face(h=0.6, cy=0.5, roll=25), ["tilted"]),
    (_face(h=0.6, cy=0.5, yaw=0.4), ["turned"]),
])
def test_upload_framing_allows_close_ups_but_needs_a_whole_centred_straight_face(face, codes):
    assert framing_problems(face, 640, 480, q.UPLOAD_FRAMING) == codes


def test_portrait_photos_use_the_same_framing():
    assert framing_problems(_face(width=900, height=1200), 900, 1200) == []
    assert framing_problems(_face(h=0.8, width=900, height=1200), 900, 1200) == ["too_close"]


def test_camera_guide_gets_the_same_thresholds():
    s = q.settings()
    assert s["framing"] is q.FRAMING and s["visibility"] is q.VISIBILITY and s["lighting"] is q.LIGHTING
    assert set(s["messages"]) >= {"sunglasses", "glasses", "mask", "eye_covered", "half_covered", "too_far",
                                  "too_close", "too_dark", "too_bright", "uneven_light"}


# ---- real photos (skipped without the dataset) ----------------------------------------------

def head_and_shoulders(img: np.ndarray, face: np.ndarray, face_height: float = 0.45) -> np.ndarray:
    """Place a face close-up on a 640x480 frame framed like the camera outline."""
    x, y, w, h = face[:4]
    s = face_height * 480 / h
    big = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    canvas = np.full((480, 640, 3), 190, np.uint8)
    ox, oy = int(320 - (x + w / 2) * s), int(0.42 * 480 - (y + h / 2) * s)
    sx, sy, dx, dy = max(0, -ox), max(0, -oy), max(0, ox), max(0, oy)
    cw, ch = min(big.shape[1] - sx, 640 - dx), min(big.shape[0] - sy, 480 - dy)
    canvas[dy:dy + ch, dx:dx + cw] = big[sy:sy + ch, sx:sx + cw]
    return canvas


def draw_on_face(img: np.ndarray, face: np.ndarray, kind: str) -> np.ndarray:
    """Copy of the photo with sunglasses, glasses, a mask, a hand over one eye or a dark half drawn on."""
    left, right, _, mouth = q._landmarks(face)
    to_img = cv2.getAffineTransform(np.float32([q.CANON_EYE_L, q.CANON_EYE_R, q.CANON_MOUTH]),
                                    np.float32([left, right, mouth]))
    layer = np.zeros((q.CANON_H, q.CANON_W, 3), np.uint8)
    mask = np.zeros((q.CANON_H, q.CANON_W), np.uint8)
    skin = tuple(int(v) for v in np.median(q.align_face(img, face)[110:135, 45:150].reshape(-1, 3), axis=0))
    if kind == "sunglasses":
        for cx in (64, 128):
            for target, colour in ((layer, (25, 25, 30)), (mask, 255)):
                cv2.ellipse(target, (cx, 82), (26, 18), 0, 0, 360, colour, -1)
    elif kind == "glasses":
        for target, colour in ((layer, (30, 30, 30)), (mask, 255)):
            for cx in (64, 128):
                cv2.rectangle(target, (cx - 27, 66), (cx + 27, 100), colour, 3)
            cv2.line(target, (91, 78), (101, 78), colour, 3)
    elif kind == "mask":
        pts = np.int32([[22, 108], [96, 96], [170, 108], [162, 200], [96, 222], [30, 200]])
        cv2.fillPoly(layer, [pts], (170, 200, 230))
        cv2.fillPoly(mask, [pts], 255)
    elif kind == "eye_covered":
        cv2.ellipse(layer, (64, 84), (34, 26), 0, 0, 360, skin, -1)
        cv2.ellipse(mask, (64, 84), (34, 26), 0, 0, 360, 255, -1)
    elif kind == "dark_half":
        cv2.rectangle(layer, (0, 0), (96, q.CANON_H), (40, 35, 30), -1)
        cv2.rectangle(mask, (0, 0), (96, q.CANON_H), 255, -1)
    h, w = img.shape[:2]
    over = cv2.warpAffine(layer, to_img, (w, h))
    alpha = cv2.warpAffine(mask, to_img, (w, h)).astype(np.float32)[..., None] / 255
    return (img * (1 - alpha) + over * alpha).astype(np.uint8)


@pytest.fixture(scope="module")
def framed():
    """(photo, face) for the first test-split selfie that passes when framed head and shoulders."""
    if not TEST_SPLIT.exists():
        pytest.skip("dataset not available (run training/prepare_data.py)")
    for path in sorted(TEST_SPLIT.glob("*/*.jpg"))[:200]:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        faces = q._faces(rgb)
        if len(faces) != 1 or faces[0][2] < 160:
            continue
        photo = head_and_shoulders(rgb, faces[0])
        if check_photo(Image.fromarray(photo)).ok:
            return photo, q._faces(photo)[0], rgb, faces[0]
    pytest.skip("no suitable selfie found in the test split")


def test_head_and_shoulders_selfie_is_accepted(framed):
    r = check_photo(Image.fromarray(framed[0]))
    assert r.ok and r.faces == 1 and r.problems == [] and r.codes == []


@pytest.mark.parametrize("kind, code", [
    ("sunglasses", "sunglasses"), ("mask", "mask"), ("eye_covered", "eye_hidden"),
])
def test_covered_faces_are_rejected(framed, kind, code):
    photo, face = framed[0], framed[1]
    r = check_photo(Image.fromarray(draw_on_face(photo, face, kind)))
    assert not r.ok and code in r.codes and q.MESSAGES[code] in r.message


# Real dataset photos with genuine problems (the models are trained on real faces, so drawn-on glasses or
# hands are not a fair test); paths under data/processed, skipped without the dataset.
REAL_CASES = {
    "hand": ("skin_type/train/normal/normal_f6e4461d9ab63dd8c1e6_jpg.rf.4f19202e50c3292d7312c3e4b8cfc3ff.jpg", "hand"),
    "glasses": ("skin_type/train/dry/dry_new_d9cac0f138c33b674999_jpg.rf.a8dd6292433dce9cedf5b04af73210b0.jpg", "glasses"),
    "sunglasses": ("skin_type/train/oily/oily_f3777e14df39fb4b57ab_jpg.rf.de90bb81ade9df0caf247c4d942dd4ca.jpg",
                   "sunglasses"),
    "eyes closed": ("skin_type/train/oily/oily_new_edd87f49dd25491258b8_jpg.rf.063ee5386f563a86ce27729a9813ac1c.jpg",
                    "eye_hidden"),
    "fringe": ("skin_type/train/dry/dry_058306bb9c7116981e12_jpg.rf.9919e7c263b9b14e2ae7496b3e1406aa.jpg",
               "forehead_covered"),
    "cap": ("skin_type/val/oily/oily_new_ba4897b430da14daf959_jpg.rf.fe0659da8a6673537742834c7cbc669d.jpg",
            "forehead_covered"),
}


@pytest.mark.parametrize("case", list(REAL_CASES))
def test_real_photos_with_problems_are_rejected_with_the_right_reason(case):
    rel, code = REAL_CASES[case]
    path = ROOT / "data/processed" / rel
    if not path.exists():
        pytest.skip("dataset not available")
    img = Image.open(path).convert("RGB")
    img.thumbnail((q.MAX_ANALYSIS_SIDE, q.MAX_ANALYSIS_SIDE))   # as check_photo does before detecting
    rgb = np.asarray(img)
    faces = q._faces(rgb)
    r = check_photo(Image.fromarray(head_and_shoulders(rgb, max(faces, key=lambda f: f[2] * f[3]))))
    assert not r.ok and code in r.codes and q.MESSAGES[code] in r.message


def test_half_covered_face_is_rejected(framed):
    r = check_photo(Image.fromarray(draw_on_face(framed[0], framed[1], "dark_half")))
    assert not r.ok and set(r.codes) & {"half_covered", "uneven_light", "no_face", "eye_hidden", "too_dark"}


@pytest.mark.parametrize("make, code", [
    (lambda a: cv2.GaussianBlur(a, (0, 0), 6), "blurry"),
    (lambda a: (a * 0.12).astype("uint8"), "too_dark"),
    (lambda a: np.clip(a.astype(int) * 1.8 + 60, 0, 255).astype("uint8"), "too_bright"),
    (lambda a: np.full((480, 640, 3), 128, "uint8"), "no_face"),
    (lambda a: np.hstack([a, a]), "several_faces"),
])
def test_bad_photos_are_rejected_with_a_reason(framed, make, code):
    r = check_photo(Image.fromarray(make(framed[0])))
    assert not r.ok and code in r.codes


def test_one_side_in_shadow_is_rejected(framed):
    photo = framed[0].astype(np.float32)
    x = int(framed[1][0] + framed[1][2] / 2)
    photo[:, :x] *= 0.35                                         # light from one side only
    r = check_photo(Image.fromarray(photo.astype(np.uint8)))
    assert not r.ok and set(r.codes) & {"uneven_light", "half_covered", "too_dark"}


def test_face_close_up_is_rejected_as_too_close(framed):
    _, _, rgb, face = framed
    r = check_photo(Image.fromarray(head_and_shoulders(rgb, face, face_height=0.75)))
    assert not r.ok and "too_close" in r.codes and q.MESSAGES["too_close"] in r.message


def test_uploaded_close_up_is_accepted_but_captured_close_up_is_not(framed):
    _, _, rgb, face = framed
    close_up = Image.fromarray(head_and_shoulders(rgb, face, face_height=0.65))   # whole face, no shoulders
    assert check_photo(close_up, source="upload").ok
    assert "too_close" in check_photo(close_up, source="camera").codes


def test_face_far_away_is_rejected_as_too_far(framed):
    _, _, rgb, face = framed
    r = check_photo(Image.fromarray(head_and_shoulders(rgb, face, face_height=0.2)))
    assert not r.ok and "too_far" in r.codes


def test_small_photo_is_rejected():
    r = check_photo(Image.new("RGB", (160, 160), (128, 128, 128)))
    assert not r.ok and r.codes == ["photo_too_small"]


def test_large_photo_is_accepted(framed):
    big = cv2.resize(framed[0], (2560, 1920), interpolation=cv2.INTER_CUBIC)
    assert check_photo(Image.fromarray(big)).ok
