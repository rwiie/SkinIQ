import json

import pytest

from skiniq.config import SKIN_CONDITIONS, SKIN_TYPES
from skiniq.environment import EnvironmentData
from skiniq.fusion import USCE, ImageAnalysis, build_usce


# ---- environment ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "uv, level",
    [(0, "low"), (2.9, "low"), (3, "moderate"), (6, "high"), (8, "very_high"), (11, "extreme")],
)
def test_uv_level_follows_who_categories(uv, level):
    assert EnvironmentData(25, 50, uv).uv_level == level


def test_environment_flags_for_hot_humid_high_uv(kl_afternoon):
    assert kl_afternoon.flags == {"high_uv": True, "humid": True, "dry_air": False, "hot": True, "cold": False}


def test_environment_flags_for_cold_dry_air():
    flags = EnvironmentData(temperature_c=5, humidity_pct=30, uv_index=1).flags
    assert flags["cold"] and flags["dry_air"] and not flags["humid"] and not flags["high_uv"]


@pytest.mark.parametrize("humidity, uv", [(-1, 3), (101, 3), (50, -0.5)])
def test_environment_rejects_impossible_values(humidity, uv):
    with pytest.raises(ValueError):
        EnvironmentData(25, humidity, uv)


# ---- image analysis -------------------------------------------------------------------

def test_image_analysis_rejects_missing_condition():
    probs = {c: 1 / 5 for c in SKIN_CONDITIONS[:5]}
    with pytest.raises(ValueError, match="exactly"):
        ImageAnalysis(conditions=probs)


def test_image_analysis_rejects_probabilities_not_summing_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        ImageAnalysis(conditions={c: 0.5 for c in SKIN_CONDITIONS})


# ---- skin type ------------------------------------------------------------------------

def test_skin_type_comes_from_questionnaire(answers, image):
    usce = build_usce(answers(oiliness="very_oily", oil_location="all_over", tightness="never"), image(acne=0.9))
    assert usce.skin_type == "oily"
    assert usce.skin_type_source == "questionnaire"
    assert usce.skin_type_reasons


def test_image_skin_type_ignored_at_default_weight(answers):
    photo_says_dry = ImageAnalysis(
        conditions={c: 1 / 6 for c in SKIN_CONDITIONS},
        skin_type={"normal": 0, "oily": 0, "dry": 1.0, "combination": 0},
    )
    usce = build_usce(answers(), photo_says_dry)
    assert usce.skin_type == "normal"
    assert usce.skin_type_source == "questionnaire"


def test_image_skin_type_blends_when_weight_set(answers):
    photo_says_dry = ImageAnalysis(
        conditions={c: 1 / 6 for c in SKIN_CONDITIONS},
        skin_type={"normal": 0, "oily": 0, "dry": 1.0, "combination": 0},
    )
    usce = build_usce(answers(), photo_says_dry, skin_type_image_weight=0.9)
    assert usce.skin_type == "dry"
    assert usce.skin_type_source == "questionnaire (10%) + photo (90%)"
    assert sum(usce.skin_type_scores.values()) == pytest.approx(1, abs=1e-3)


# ---- condition concerns ---------------------------------------------------------------

def test_confident_photo_condition_is_detected(answers, image):
    usce = build_usce(answers(), image(acne=0.85))
    assert usce.detected_conditions == ["acne"]
    assert usce.concerns["acne"].sources == ["photo"]


def test_unsure_photo_gives_possible_not_detected(answers, image):
    # No "clear skin" class: a spread-out prediction must not be reported as a detection
    usce = build_usce(answers(), image(redness=0.35, wrinkles=0.3))
    assert usce.detected_conditions == []
    assert usce.possible_conditions == ["redness", "wrinkles"]


def test_known_condition_raises_concern_without_photo(answers):
    usce = build_usce(answers(known_conditions=["melasma"]))
    assert not usce.has_photo
    assert usce.concerns["hyperpigmentation"].status == "detected"
    assert usce.concerns["hyperpigmentation"].sources == ["questionnaire"]
    assert usce.concerns["hyperpigmentation"].photo is None


def test_photo_and_questionnaire_agree_lists_both_sources(answers, image):
    usce = build_usce(answers(breakouts="frequently"), image(acne=0.6))
    assert usce.concerns["acne"].score == pytest.approx(0.7)
    assert usce.concerns["acne"].sources == ["photo", "questionnaire"]


@pytest.mark.parametrize("breakouts, status", [("frequently", "detected"), ("occasionally", "possible"), ("never", "none")])
def test_breakout_frequency_sets_acne_concern(answers, breakouts, status):
    assert build_usce(answers(breakouts=breakouts)).concerns["acne"].status == status


def test_rosacea_counts_as_redness_and_medical_condition(answers):
    usce = build_usce(answers(known_conditions=["rosacea"]))
    assert usce.concerns["redness"].status == "detected"
    assert usce.medical_conditions == ["rosacea"]
    assert usce.is_sensitive


# ---- sensitivity and safety -----------------------------------------------------------

@pytest.mark.parametrize(
    "sensitivity, reacts, expected",
    [("not", "no", False), ("slightly", "sometimes", False), ("moderately", "no", True), ("not", "yes", True)],
)
def test_sensitive_flag(answers, sensitivity, reacts, expected):
    assert build_usce(answers(sensitivity=sensitivity, reacts_to_products=reacts)).is_sensitive is expected


def test_pregnancy_flag(answers):
    assert build_usce(answers(pregnant="yes")).pregnant is True


# ---- answer validation ----------------------------------------------------------------

@pytest.mark.parametrize("missing", ["age_group", "pregnant", "sensitivity", "reacts_to_products", "breakouts"])
def test_missing_required_answer_raises(answers, missing):
    a = answers()
    del a[missing]
    with pytest.raises(ValueError, match=missing):
        build_usce(a)


def test_invalid_multi_select_option_raises(answers):
    with pytest.raises(ValueError, match="avoid_ingredients"):
        build_usce(answers(avoid_ingredients=["parabens"]))


def test_none_option_is_dropped_from_multi_select(answers):
    usce = build_usce(answers(avoid_ingredients=["none", "fragrance"], known_conditions=["none"]))
    assert usce.avoid_ingredients == ["fragrance"]
    assert usce.medical_conditions == []


def test_multi_select_accepts_single_string(answers):
    assert build_usce(answers(goals="brightening")).goals == ["brightening"]


# ---- embedding and serialisation ------------------------------------------------------

def test_vector_matches_feature_names_and_is_scaled(answers, image, kl_afternoon):
    usce = build_usce(answers(avoid_ingredients=["fragrance"]), image(acne=0.9), kl_afternoon)
    vec = usce.vector()
    assert len(vec) == len(USCE.feature_names()) == 36
    assert all(0 <= v <= 1 for v in vec)
    named = dict(zip(USCE.feature_names(), vec))
    assert named["concern:acne"] == pytest.approx(0.9)
    assert named["avoid:fragrance"] == 1.0 and named["avoid:alcohol"] == 0.0
    assert named["env:high_uv"] == 1.0 and named["env:known"] == 1.0
    assert sum(named[f"skin_type:{t}"] for t in SKIN_TYPES) == pytest.approx(1, abs=1e-3)


def test_missing_environment_gives_zero_env_features(answers):
    named = dict(zip(USCE.feature_names(), build_usce(answers()).vector()))
    assert all(v == 0 for k, v in named.items() if k.startswith("env:"))
    assert build_usce(answers()).env_flags["high_uv"] is False


def test_to_dict_is_json_serialisable(answers, image, kl_afternoon):
    d = build_usce(answers(known_conditions=["acne"]), image(acne=0.7), kl_afternoon).to_dict()
    restored = json.loads(json.dumps(d))
    assert restored["skin_type"] == "normal"
    assert restored["detected_conditions"] == ["acne"]
    assert restored["environment"]["uv_level"] == "very_high"
    assert list(restored["concerns"])[0] == "acne"  # highest concern first


# ---- multi-label / region input -------------------------------------------------------

def multi(**probs):
    return {c: probs.get(c, 0.05) for c in SKIN_CONDITIONS}


def test_independent_probabilities_need_not_sum_to_one(answers):
    img = ImageAnalysis(conditions=multi(acne=0.9, blackheads=0.8, hyperpigmentation=0.7), independent=True)
    usce = build_usce(answers(), img)
    assert usce.detected_conditions == ["acne", "blackheads", "hyperpigmentation"]


def test_single_label_input_still_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        ImageAnalysis(conditions=multi(acne=0.9, blackheads=0.8))


def test_per_concern_threshold_decides_photo_detection(answers):
    thresholds = {c: 0.5 for c in SKIN_CONDITIONS} | {"redness": 0.3, "acne": 0.8}
    img = ImageAnalysis(conditions=multi(redness=0.35, acne=0.6), independent=True, thresholds=thresholds)
    usce = build_usce(answers(), img)
    assert usce.concerns["redness"].status == "detected"   # 0.35 >= its tuned 0.3
    assert usce.concerns["acne"].status == "possible"      # 0.6 < its tuned 0.8


def test_self_report_still_detects_when_photo_threshold_is_high(answers):
    thresholds = {c: 0.9 for c in SKIN_CONDITIONS}
    img = ImageAnalysis(conditions=multi(acne=0.6), independent=True, thresholds=thresholds)
    assert build_usce(answers(breakouts="frequently"), img).concerns["acne"].status == "detected"


def test_thresholds_must_cover_every_concern():
    with pytest.raises(ValueError, match="thresholds"):
        ImageAnalysis(conditions=multi(), independent=True, thresholds={"acne": 0.5})


def test_locations_reach_the_concern(answers):
    img = ImageAnalysis(conditions=multi(blackheads=0.9), independent=True,
                        locations={"blackheads": ["nose", "chin"]})
    usce = build_usce(answers(), img)
    assert usce.concerns["blackheads"].locations == ["nose", "chin"]
    assert usce.to_dict()["concerns"]["blackheads"]["locations"] == ["nose", "chin"]
