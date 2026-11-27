"""Trained-model photo checks (skiniq/vision/face_models.py): face parts, colour, exposure, CLIP classifiers."""

import re

import numpy as np
import pytest

from skiniq.vision import face_models as fm

VISIBLE = {  # a clear, front-facing face: both eyes open, no glasses, nose and lips visible, forehead clear
    "eye_left": 0.035, "eye_right": 0.033, "glasses_left": 0.0, "glasses_right": 0.0, "lens_lightness": 1.0,
    "nose": 0.56, "lips": 0.24, "forehead_covered": 0.05, "skin_redness": 13.0,
}


def test_visible_face_has_no_problems():
    assert fm.parts_problems(VISIBLE) == []


@pytest.mark.parametrize("change, code", [
    ({"eye_left": 0.0}, "eye_hidden"),                                          # one eye closed or covered
    ({"eye_left": 0.0, "eye_right": 0.0}, "eye_hidden"),
    ({"glasses_left": 0.6, "glasses_right": 0.5, "lens_lightness": 0.9}, "glasses"),     # clear lenses
    ({"glasses_left": 0.6, "glasses_right": 0.5, "lens_lightness": 0.3, "eye_left": 0, "eye_right": 0},
     "sunglasses"),                                                             # dark lenses
    ({"nose": 0.02}, "mask"),
    ({"lips": 0.0}, "mask"),
    ({"forehead_covered": 0.8}, "forehead_covered"),                            # fringe, hat, headband
    ({"skin_redness": 1.0}, "unnatural_skin"),                                  # sheet / clay mask, filter
])
def test_each_face_part_problem_is_detected(change, code):
    assert code in fm.parts_problems({**VISIBLE, **change})


def test_glasses_are_reported_once_not_as_closed_eyes():
    codes = fm.parts_problems({**VISIBLE, "glasses_left": 0.6, "glasses_right": 0.6, "eye_left": 0, "eye_right": 0})
    assert "eye_hidden" not in codes and len([c for c in codes if c in ("glasses", "sunglasses")]) == 1


def test_glasses_on_one_eye_only_are_not_glasses():
    # e.g. a magnifying glass or a phone held over one eye: not a pair of glasses
    assert "glasses" not in fm.parts_problems({**VISIBLE, "glasses_left": 0.9, "glasses_right": 0.0})


def test_black_and_white_photo_skips_the_skin_colour_check():
    assert "unnatural_skin" not in fm.parts_problems({**VISIBLE, "skin_redness": 0.0}, colour_photo=False)


def test_greyscale_detection():
    rng = np.random.default_rng(0)
    grey = np.repeat(rng.integers(0, 255, (200, 200, 1), dtype=np.uint8), 3, axis=2)
    colour = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    assert fm.is_greyscale(grey) and not fm.is_greyscale(colour)


def test_highlights_measure_exposure():
    scene = np.full((300, 400, 3), 120, np.uint8)
    scene[:50] = 245                                      # a bright window or wall somewhere in the photo
    dark = (scene * 0.3).astype(np.uint8)
    assert fm.highlights(scene) > fm.PARTS["min_highlights"] > fm.highlights(dark)


def test_clip_classifiers_are_well_formed():
    for probe in ("drawing",):
        p = fm._probes()[probe]
        assert len(p["weights"]) == 512 and 0 < p["threshold"] < 1 and p["trained_on"] and p["validation"]
        e = np.random.default_rng(1).normal(size=512).astype(np.float32)
        assert 0.0 <= fm.clip_probability(e / np.linalg.norm(e), probe) <= 1.0


def test_model_list_has_https_urls_and_checksums():
    for filename, url, sha in fm.MODELS.values():
        assert filename.endswith(".onnx") and url.startswith("https://huggingface.co/")
        assert re.fullmatch(r"[0-9a-f]{64}", sha)


def test_settings_share_the_model_thresholds_with_the_camera():
    from skiniq.vision.quality import settings
    s = settings()
    assert s["parts"] is fm.PARTS and s["hands"] is fm.HANDS


# ---- with the models (downloaded on first use) -------------------------------------------------

@pytest.fixture(scope="module")
def models():
    if not fm.available():
        pytest.skip("photo-check models not available (no internet)")


def test_face_parsing_finds_the_parts_of_a_real_face(models):
    from pathlib import Path

    from PIL import Image

    from skiniq.vision.quality import _faces
    paths = sorted(Path(__file__).resolve().parent.parent.glob("data/processed/skin_type/test/normal/*.jpg"))
    if not paths:
        pytest.skip("dataset not available")
    for path in paths[:40]:
        img = Image.open(path).convert("RGB")
        img.thumbnail((1280, 1280))
        rgb = np.asarray(img)
        faces = _faces(rgb)
        if len(faces) == 1 and faces[0][2] > 150:
            m = fm.parts_measures(fm.parse_face(rgb, faces[0]))
            if fm.parts_problems(m) == []:
                t = fm.PARTS
                assert m["nose"] >= t["min_nose_share"] and m["lips"] >= t["min_lips_share"]
                assert min(m["eye_left"], m["eye_right"]) >= t["min_eye_area"] and m["skin_redness"] >= t["min_skin_redness"]
                return
    pytest.skip("no clear face in the sample")


def test_hand_detector_finds_nothing_on_an_empty_scene(models):
    scene = np.full((480, 640, 3), 200, np.uint8)
    found, palm, conf = fm.hand_on_face(scene, np.array([240, 120, 160, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9]))
    assert not found and palm < fm.HANDS["palm_score"]
