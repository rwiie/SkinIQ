"""End-to-end pipeline and explanations (proposal 3.6.5, 3.8 XAI checks)."""

import json

import pytest
from PIL import Image

from skiniq import pipeline
from skiniq.config import SKIN_CONDITIONS
from skiniq.environment import EnvironmentData
from skiniq.explain import key_ingredients
from skiniq.fusion import build_usce
from skiniq.pipeline import analyse, build_routine
from skiniq.products import Product, ingredient_functions, ingredient_groups
from skiniq.recommender import Recommendation
from skiniq.rules.base import ProductAssessment


class FakeClassifier:
    multilabel = True
    thresholds = {c: 0.5 for c in SKIN_CONDITIONS}

    def __init__(self, **probs):
        self.probs = {c: probs.get(c, 0.05) for c in SKIN_CONDITIONS}

    def predict(self, image):
        return self.probs


@pytest.fixture
def selfie():
    return Image.new("RGB", (224, 224), (210, 160, 130))


def walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_strings(v)


# ---- full pipeline -----------------------------------------------------------------------

def test_full_result_has_the_four_sections_and_is_json(answers, selfie, kl_afternoon):
    r = analyse(answers(breakouts="frequently"), selfie, kl_afternoon, classifier=FakeClassifier(acne=0.9))
    json.dumps(r)
    assert {"skin_analysis", "advice", "routine", "disclaimer", "usce", "notes"} <= set(r)
    assert r["skin_analysis"]["concerns"][0]["concern"] == "acne"
    assert "photo analysis 90%" in r["skin_analysis"]["concerns"][0]["text"]
    assert [s["step"] for s in r["routine"]["morning"]] == ["cleanser", "toner", "treatment", "moisturizer",
                                                           "sunscreen"]
    assert [s["step"] for s in r["routine"]["evening"]] == ["cleanser", "toner", "treatment", "moisturizer"]
    assert any("UV peaks at 10" in a for a in r["advice"])
    assert r["notes"] == []


def test_every_recommended_product_is_explained(answers, selfie, kl_afternoon):
    r = analyse(answers(goals=["brightening"]), selfie, kl_afternoon, classifier=FakeClassifier())
    for step in r["routine"]["morning"] + r["routine"]["evening"]:
        p = step["product"]
        assert p["summary"] and (p["key_ingredients"] or p["why"])


def test_morning_routine_has_no_retinoid_or_aha(answers, kl_afternoon):
    r = analyse(answers(goals=["anti_aging"], age_group="36_49"), environment=kl_afternoon)
    from skiniq.products import load_products

    by_title = {p.title: p for p in load_products()}
    for step in r["routine"]["morning"]:
        p = by_title[step["product"]["product"]]
        assert not (p.has("retinoid") or p.has("aha"))


def test_without_photo_or_weather_still_works_with_notes(answers):
    r = analyse(answers(known_conditions=["melasma"]))
    assert any("Weather data" in n for n in r["notes"])
    assert r["skin_analysis"]["photo_used"] is False
    assert r["skin_analysis"]["concerns"][0]["concern"] == "hyperpigmentation"
    assert "your questionnaire answers" in r["skin_analysis"]["concerns"][0]["text"]


def test_missing_model_skips_photo_with_note(answers, selfie, monkeypatch):
    monkeypatch.setattr(pipeline, "load_classifier", lambda *a: None)
    r = analyse(answers(), selfie, EnvironmentData(25, 50, 3))
    assert any("could not be analysed" in n for n in r["notes"])
    assert r["skin_analysis"]["photo_used"] is False


def test_clear_skin_gets_no_concerns_text(answers, selfie):
    r = analyse(answers(), selfie, EnvironmentData(25, 50, 3), classifier=FakeClassifier())
    assert r["skin_analysis"]["concerns"] == []
    assert r["skin_analysis"]["no_concerns_text"] == "No skin concerns were detected."


def test_safety_rules_hold_end_to_end(answers):
    """Pregnant user avoiding fragrance: no retinoid, hydroquinone or fragrance anywhere."""
    from skiniq.products import load_products

    r = analyse(answers(pregnant="yes", avoid_ingredients=["fragrance"], goals=["anti_aging", "brightening"]))
    by_title = {p.title: p for p in load_products()}
    picked = [s["product"]["product"] for s in r["routine"]["morning"] + r["routine"]["evening"]]
    picked += [w["product"]["product"] for w in r["routine"]["weekly"]]
    for title in picked:
        p = by_title[title]
        assert not (p.has("retinoid") or p.has("hydroquinone") or p.has("fragrance")), title
    assert any("Pregnancy" in a for a in r["advice"])


def test_explanations_never_mention_the_wrong_skin_type(answers):
    r = analyse(answers(oiliness="dry", tightness="often", flakiness="yes"))
    text = " ".join(walk_strings(r["skin_analysis"]))
    assert "point to dry skin" in text and "oily" not in r["skin_analysis"]["skin_type_text"]


# ---- routine assembly --------------------------------------------------------------------

def product(pid, step, *ingredients):
    ings = tuple(ingredients)
    f, c = ingredient_functions(ings)
    return Product(id=pid, brand="B", name=f"P{pid}", category="X", step=step, ingredients=ings,
                   tags=frozenset(), groups=ingredient_groups(ings), functions=f, concerns=c)


def rec(p, score):
    return Recommendation(p, score, 0.0, score, [], ProductAssessment(p))


def test_routine_avoids_conflicts_by_taking_the_next_best():
    ranked = {
        "toner": [rec(product(1, "toner", "Glycolic Acid"), 0.9)],
        "treatment": [rec(product(2, "treatment", "Retinol"), 0.9), rec(product(3, "treatment", "Niacinamide"), 0.5)],
    }
    steps, conflicts = build_routine(ranked, ("toner", "treatment"), "evening")
    assert [s.pick.product.id for s in steps] == [1, 3]   # retinol + AHA avoided
    assert conflicts == []


def test_unavoidable_conflict_is_reported_with_advice():
    ranked = {
        "toner": [rec(product(1, "toner", "Salicylic Acid"), 0.9)],
        "treatment": [rec(product(2, "treatment", "Retinol"), 0.9)],
    }
    _, conflicts = build_routine(ranked, ("toner", "treatment"), "evening")
    assert conflicts and conflicts[0]["advice"] == "Use them on alternate nights"


def test_morning_drops_photosensitising_products():
    ranked = {"treatment": [rec(product(2, "treatment", "Retinol"), 0.9), rec(product(3, "treatment", "Niacinamide"), 0.4)]}
    steps, _ = build_routine(ranked, ("treatment",), "morning")
    assert steps[0].pick.product.id == 3 and steps[0].alternatives == []


# ---- explanations ------------------------------------------------------------------------

def test_key_ingredients_match_user_needs(answers):
    u = build_usce(answers(goals=["acne_control"]))
    p = product(1, "treatment", "Water", "Glycerin", "Salicylic Acid", "Xanthan Gum")
    keys = key_ingredients(p, u)
    assert keys[0]["ingredient"] == "Salicylic Acid" and "acne_control" in keys[0]["functions"]
    assert all(k["ingredient"] not in ("Water", "Xanthan Gum") for k in keys)


@pytest.mark.parametrize("age", ["18_25", "26_35", "36_49", "50_plus"])
def test_results_include_guidance_for_the_age_group(answers, age):
    g = analyse(answers(age_group=age))["skin_analysis"]["age_guidance"]
    assert g["age_group"] == age and g["care_focus"] and g["common_concerns"] and g["skin_changes"]


def test_typed_avoid_ingredient_never_reaches_the_routine(answers):
    r = analyse(answers(avoid_ingredients=["other"], avoid_other="niacinamide"))
    picks = [s["product"] for part in ("morning", "evening") for s in r["routine"][part]]
    from skiniq.products import load_products
    by_title = {p.title: p for p in load_products()}
    assert all(not by_title[p["product"]].containing("niacinamide") for p in picks)
    assert any("Niacinamide" in tip for tip in r["advice"])


def test_results_pdf_has_every_section(answers, kl_afternoon):
    import io

    from skiniq.report import build_pdf

    PdfReader = pytest.importorskip("pypdf").PdfReader

    r = analyse(answers(known_conditions=["other"], known_conditions_other="Perioral dermatitis"),
                environment=kl_afternoon)
    pdf = build_pdf(r)
    assert pdf.startswith(b"%PDF")
    text = " ".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)
    for heading in ["Skin analysis", "Skin type", "Skin care for your age", "today's weather", "Advice for you",
                    "Your routine", "Morning", "Evening", "not a medical diagnosis"]:
        assert heading in text, heading
    assert "Perioral dermatitis" in text


@pytest.mark.parametrize("extra", [{}, {"avoid_ingredients": ["other"], "avoid_other": "niacinamide"},
                                   {"known_conditions": ["other"], "known_conditions_other": "Perioral dermatitis",
                                    "avoid_ingredients": ["other"], "avoid_other": "niacinamide"}])
def test_acne_routines_do_not_stack_exfoliants(answers, extra):
    # Regression: with a short candidate list every acne pick was a BHA product (up to 12 conflicts)
    r = analyse(answers(age_group="26_35", sensitivity="slightly", reacts_to_products="sometimes",
                        oiliness="very_oily", oil_location="t_zone", breakouts="frequently",
                        goals=["acne_control"], **extra))
    assert r["routine"]["conflicts"] == []
