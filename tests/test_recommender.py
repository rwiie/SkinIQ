"""TF-IDF + cosine similarity ranking (proposal 3.6.4)."""

import numpy as np
import pytest

from skiniq.config import SKIN_CONDITIONS
from skiniq.fusion import ImageAnalysis, build_usce
from skiniq.products import Product, ingredient_functions, ingredient_groups
from skiniq.recommender import (
    RULE_SCORE_WEIGHT,
    ProductIndex,
    default_index,
    product_terms,
    query_terms,
    rank_products,
)


def product(pid, *ingredients, step="treatment", tags=()):
    ings = tuple(ingredients)
    functions, concerns = ingredient_functions(ings)
    return Product(id=pid, brand="B", name=f"P{pid}", category="X", step=step, ingredients=ings,
                   tags=frozenset(tags), groups=ingredient_groups(ings), functions=functions, concerns=concerns)


def photo(**probs):
    return ImageAnalysis(conditions={c: probs.get(c, 0.05) for c in SKIN_CONDITIONS}, independent=True)


@pytest.fixture
def toy_index():
    return ProductIndex([
        product(0, "Water", "Salicylic Acid", "Niacinamide"),          # acne
        product(1, "Water", "Sodium Hyaluronate", "Glycerin", "Ceramide NP"),  # hydration
        product(2, "Water", "Centella Asiatica Extract", "Panthenol"),  # soothing
        product(3, "Water", "Retinol", "Palmitoyl Tripeptide-1"),       # anti-aging
        product(4, "Water", "Zinc Oxide", step="sunscreen"),
    ])


# ---- documents ------------------------------------------------------------------------

def test_product_terms_include_ingredient_function_and_concern():
    t = product_terms(product(0, "Salicylic Acid"))
    assert {"ing:Salicylic Acid", "fn:exfoliant", "fn:acne_control", "concern:acne"} <= set(t)


def test_formulation_ingredients_are_left_out():
    t = product_terms(product(0, "Water", "Xanthan Gum", "Disodium EDTA", "Citric Acid"))
    assert t == {}


def test_earlier_ingredients_weigh_more():
    early = product_terms(product(0, "Niacinamide", *["Water"] * 20, "Sodium Hyaluronate"))
    assert early["ing:Niacinamide"] > early["ing:Hyaluronic Acid, Sodium Hyaluronate"]


def test_query_reflects_concerns_skin_type_goals_and_weather(answers, kl_afternoon):
    from skiniq.environment import EnvironmentData

    u = build_usce(answers(oiliness="dry", tightness="often", flakiness="yes", goals=["brightening"]),
                   photo(acne=0.9), EnvironmentData(5, 30, 9))
    q = query_terms(u)
    assert q["concern:acne"] > 0 and q["fn:acne_control"] > 0          # photo concern
    assert q["concern:dryness"] > 0 and q["fn:humectant"] > 0          # dry skin + dry air
    assert q["fn:brightening"] > 0                                      # goal
    assert q["fn:uv_filter"] > 0                                        # high UV
    assert "concern:wrinkles" not in q                                  # not a concern


def test_unsure_photo_concern_scales_query_weight(answers):
    strong = query_terms(build_usce(answers(), photo(redness=0.9)))
    weak = query_terms(build_usce(answers(), photo(redness=0.3)))
    assert strong["concern:redness"] > weak["concern:redness"] > 0


# ---- similarity -----------------------------------------------------------------------

def test_similarity_is_cosine_in_unit_range(toy_index):
    sims = toy_index.similarity({"fn:acne_control": 1.0})
    assert sims.shape == (5,) and np.all(sims >= 0) and np.all(sims <= 1 + 1e-9)


def test_identical_document_has_similarity_one(toy_index):
    terms = product_terms(toy_index.products[1])
    assert toy_index.similarity(terms)[1] == pytest.approx(1.0)


@pytest.mark.parametrize("query, best", [
    ({"concern:acne": 1, "fn:acne_control": 1}, 0),
    ({"fn:humectant": 1, "concern:dryness": 1}, 1),
    ({"fn:soothing": 1, "concern:redness": 1}, 2),
    ({"fn:anti_aging": 1, "concern:wrinkles": 1}, 3),
])
def test_most_similar_product_matches_the_need(toy_index, query, best):
    assert int(np.argmax(toy_index.similarity(query))) == best


def test_empty_query_scores_zero(toy_index):
    assert np.all(toy_index.similarity({}) == 0)


def test_top_terms_explain_the_match(toy_index):
    assert toy_index.top_terms(0, {"fn:acne_control": 1.0, "fn:uv_filter": 1.0}) == ["fn:acne_control"]


# ---- ranking with rules ---------------------------------------------------------------

def test_rank_combines_similarity_and_rule_score(answers, toy_index):
    recs = rank_products(build_usce(answers(goals=["acne_control"])), toy_index, per_step=5)
    for r in recs["treatment"]:
        assert r.score == pytest.approx(r.similarity + RULE_SCORE_WEIGHT * r.rule_score, abs=1e-3)
    assert recs["treatment"][0].product.id == 0
    assert [r.score for r in recs["treatment"]] == sorted((r.score for r in recs["treatment"]), reverse=True)


def test_excluded_products_are_never_ranked(answers, toy_index):
    recs = rank_products(build_usce(answers(pregnant="yes", goals=["anti_aging"])), toy_index, per_step=5)
    assert 3 not in {r.product.id for r in recs["treatment"]}  # retinol excluded in pregnancy


def test_products_grouped_by_step_and_limited(answers, toy_index):
    recs = rank_products(build_usce(answers()), toy_index, per_step=2)
    assert len(recs["treatment"]) == 2 and [r.product.id for r in recs["sunscreen"]] == [4]
    assert recs["cleanser"] == []


def test_real_catalogue_profiles():
    from tests.conftest import BASE_ANSWERS

    acne = build_usce({**BASE_ANSWERS, "breakouts": "frequently", "goals": ["acne_control"]}, photo(acne=0.9))
    recs = rank_products(acne, per_step=3)
    assert all(len(v) == 3 for k, v in recs.items() if k != "exfoliator")
    assert all("acne" in r.product.concerns for r in recs["treatment"])

    rosacea = build_usce({**BASE_ANSWERS, "known_conditions": ["rosacea"], "sensitivity": "highly"})
    for step in rank_products(rosacea, per_step=5).values():
        for r in step:
            assert not (r.product.has("fragrance") or r.product.has("essential_oil"))


def test_ranking_is_deterministic(answers):
    u = build_usce(answers(goals=["hydration"]))
    first = [r.product.id for r in rank_products(u)["moisturizer"]]
    assert first == [r.product.id for r in rank_products(u)["moisturizer"]]
    assert default_index() is default_index()  # cached
