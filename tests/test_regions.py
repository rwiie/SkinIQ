"""Region analysis (experimental): geometry, fallback and aggregation, with a fake classifier."""

import numpy as np
import pytest
from PIL import Image

from skiniq.config import SKIN_CONDITIONS
from skiniq.vision import regions
from skiniq.vision.regions import (
    REGIONS,
    WHOLE_FACE,
    WHOLE_IMAGE,
    Face,
    RegionAnalysis,
    RegionResult,
    align_face,
    analyse_regions,
    region_boxes,
    skin_fraction,
)

SKIN = (224, 172, 140)  # inside the YCrCb skin range


def probs(**p):
    rest = [c for c in SKIN_CONDITIONS if c not in p]
    left = 1 - sum(p.values())
    return {**p, **{c: left / len(rest) for c in rest}}


class FakeClassifier:
    def __init__(self, result):
        self.result, self.calls = result, 0

    def predict(self, image):
        self.calls += 1
        return self.result


def face_at(scale=100.0, ox=200.0, oy=200.0, tilt=0.0):
    """Synthetic upright face; eyes `scale` px apart, optionally tilted (left eye lower by `tilt`)."""
    return Face(
        box=(int(ox - scale), int(oy - scale), int(ox + scale), int(oy + 1.6 * scale)),
        right_eye=(ox - scale / 2, oy), left_eye=(ox + scale / 2, oy + tilt),
        nose=(ox, oy + 0.6 * scale), mouth_right=(ox - 0.4 * scale, oy + scale),
        mouth_left=(ox + 0.4 * scale, oy + scale), score=0.95,
    )


def test_region_boxes_cover_all_regions_inside_image():
    boxes = region_boxes(face_at(), (400, 500))
    assert set(boxes) == set(REGIONS)
    for x0, y0, x1, y1 in boxes.values():
        assert 0 <= x0 < x1 <= 400 and 0 <= y0 < y1 <= 500


def test_region_boxes_are_anatomically_ordered():
    b = region_boxes(face_at(), (400, 500))
    centre_y = {k: (v[1] + v[3]) / 2 for k, v in b.items()}
    assert centre_y["forehead"] < centre_y["nose"] < centre_y["chin"]
    # Person's right cheek is on the image left
    assert b["right_cheek"][0] < b["nose"][0] < b["left_cheek"][0]


def test_small_face_drops_regions_too_small_to_classify():
    assert region_boxes(face_at(scale=30, ox=60, oy=60), (200, 200)) == {}


def test_regions_outside_image_are_clipped_or_dropped():
    boxes = region_boxes(face_at(oy=40), (400, 500))  # forehead above the image top
    assert "forehead" not in boxes
    assert all(y0 >= 0 for _, y0, _, _ in boxes.values())


def test_align_face_levels_the_eyes():
    img = Image.new("RGB", (400, 500), SKIN)
    _, aligned = align_face(img, face_at(tilt=30))
    assert aligned.right_eye[1] == pytest.approx(aligned.left_eye[1], abs=0.5)


def test_skin_fraction_separates_skin_from_background():
    assert skin_fraction(Image.new("RGB", (50, 50), SKIN)) > 0.9
    assert skin_fraction(Image.new("RGB", (50, 50), (10, 10, 10))) < 0.1


def test_no_face_falls_back_to_whole_image(monkeypatch):
    monkeypatch.setattr(regions, "detect_face", lambda img: None)
    clf = FakeClassifier(probs(acne=0.9))
    result = analyse_regions(Image.new("RGB", (300, 300), SKIN), clf)
    assert not result.face_found
    assert [r.name for r in result.regions] == [WHOLE_IMAGE]
    assert result.locations == {}  # whole image has no location


def test_face_classifies_whole_face_and_skin_regions(monkeypatch):
    monkeypatch.setattr(regions, "detect_face", lambda img: face_at())
    clf = FakeClassifier(probs(blackheads=0.8))
    result = analyse_regions(Image.new("RGB", (400, 500), SKIN), clf)
    assert result.face_found
    assert {r.name for r in result.regions} == {WHOLE_FACE, *REGIONS}
    assert result.locations == {"blackheads": list(REGIONS)}


def test_background_regions_are_skipped(monkeypatch):
    monkeypatch.setattr(regions, "detect_face", lambda img: face_at())
    arr = np.full((500, 400, 3), SKIN, np.uint8)
    arr[:180] = 0  # dark band over the forehead
    result = analyse_regions(Image.fromarray(arr), FakeClassifier(probs(acne=0.7)))
    assert "forehead" not in {r.name for r in result.regions}


def test_condition_scores_take_highest_region_and_locations_need_confidence():
    analysis = RegionAnalysis(True, [
        RegionResult(WHOLE_FACE, (0, 0, 9, 9), probs(acne=0.6)),
        RegionResult("nose", (0, 0, 9, 9), probs(blackheads=0.9)),
        RegionResult("left_cheek", (0, 0, 9, 9), probs(acne=0.7)),
        RegionResult("chin", (0, 0, 9, 9), probs(redness=0.4)),
    ])
    scores = analysis.condition_scores
    assert scores["blackheads"] == pytest.approx(0.9) and scores["acne"] == pytest.approx(0.7)
    assert analysis.locations == {"blackheads": ["nose"], "acne": ["left_cheek"]}  # chin 40%: not confident


def test_multilabel_region_reports_every_concern_over_its_threshold():
    th = {c: 0.5 for c in SKIN_CONDITIONS} | {"redness": 0.3}
    p = {c: 0.1 for c in SKIN_CONDITIONS} | {"acne": 0.7, "redness": 0.35, "wrinkles": 0.45}
    analysis = RegionAnalysis(True, [
        RegionResult(WHOLE_FACE, (0, 0, 9, 9), p),
        RegionResult("left_cheek", (0, 0, 9, 9), p),
        RegionResult("forehead", (0, 0, 9, 9), {c: 0.1 for c in SKIN_CONDITIONS} | {"wrinkles": 0.8}),
    ], thresholds=th)
    assert analysis.locations == {"acne": ["left_cheek"], "redness": ["left_cheek"], "wrinkles": ["forehead"]}
    img = analysis.to_image_analysis()
    assert img.independent and img.conditions["wrinkles"] == pytest.approx(0.8)
    assert img.locations["wrinkles"] == ["forehead"] and img.thresholds == th


def test_clear_region_gets_no_location():
    th = {c: 0.5 for c in SKIN_CONDITIONS}
    analysis = RegionAnalysis(True, [RegionResult("nose", (0, 0, 9, 9), {c: 0.1 for c in SKIN_CONDITIONS})],
                              thresholds=th)
    assert analysis.locations == {}


def test_analyse_regions_uses_multilabel_classifier_thresholds(monkeypatch):
    monkeypatch.setattr(regions, "detect_face", lambda img: face_at())
    clf = FakeClassifier({c: 0.1 for c in SKIN_CONDITIONS} | {"acne": 0.6, "blackheads": 0.6})
    clf.multilabel, clf.thresholds = True, {c: 0.5 for c in SKIN_CONDITIONS}
    result = analyse_regions(Image.new("RGB", (400, 500), SKIN), clf)
    assert set(result.locations) == {"acne", "blackheads"}
