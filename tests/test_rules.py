"""Knowledge-base rules (proposal 3.6.3, 3.8 rule validation).

Each rule has at least one test that fires it (and checks its action) and the suite
fails if a rule is added without one (test_every_rule_is_tested).
"""

import pytest

from skiniq.environment import EnvironmentData
from skiniq.fusion import ImageAnalysis, build_usce
from skiniq.products import Product, clean_catalogue, ingredient_groups, load_products, split_ingredients
from skiniq.rules import ALL_RULES, Action, apply_rules, check_routine
from skiniq.rules.base import assess

TESTED: set[str] = set()


def product(*ingredients, tags=(), step="moisturizer", name="Test"):
    ings = tuple(ingredients) or ("Water", "Glycerin")
    return Product(id=0, brand="Brand", name=name, category="X", step=step, ingredients=ings,
                   tags=frozenset(tags), groups=ingredient_groups(ings))


def fired(usce, prod, rule_id):
    """Return the finding of `rule_id` (or None), and record the rule as tested."""
    TESTED.add(rule_id)
    rules = [r for r in ALL_RULES if r.id == rule_id]
    assert rules, f"unknown rule {rule_id}"
    found = assess(prod, usce, rules).findings
    return found[0] if found else None


@pytest.fixture
def user(answers):
    def make(image=None, environment=None, **overrides):
        return build_usce(answers(**overrides), image=image, environment=environment)
    return make


# ---- ingredient detection -------------------------------------------------------------

def test_split_keeps_numbered_ingredients_together():
    assert split_ingredients("Water,1,2-Hexanediol,Glycerin") == ("Water", "1,2-Hexanediol", "Glycerin")


@pytest.mark.parametrize("ingredient, group", [
    ("Retinol", "retinoid"), ("Retinyl Palmitate", "retinoid"), ("Glycolic Acid", "aha"),
    ("Salicylic Acid 2%", "bha"), ("Betaine Salicylate", "bha"), ("Gluconolactone", "pha"),
    ("Ascorbic Acid", "pure_vitamin_c"), ("Parfum", "fragrance"), ("Alcohol Denat.", "drying_alcohol"),
    ("Lavandula Angustifolia Oil", "essential_oil"), ("Citrus Aurantium Bergamia Fruit Oil", "essential_oil"),
    ("Ethylhexyl Methoxycinnamate", "chemical_filter"), ("Zinc Oxide", "mineral_filter"),
])
def test_ingredient_groups_detected(ingredient, group):
    assert group in ingredient_groups((ingredient,))


@pytest.mark.parametrize("ingredient, not_group", [
    ("Citric Acid", "aha"),                    # pH adjuster
    ("Helianthus Annuus Seed Oil", "essential_oil"),  # carrier oil
    ("Cetearyl Alcohol", "drying_alcohol"),     # fatty alcohol, not drying
    ("Bakuchiol", "retinoid"),                  # retinol alternative
    ("Phloretin", "retinoid"),
])
def test_lookalike_ingredients_not_misdetected(ingredient, not_group):
    assert not_group not in ingredient_groups((ingredient,))


def test_catalogue_cleaning_drops_empty_and_duplicates():
    import pandas as pd
    raw = pd.DataFrame({
        "brand": ["A", "A", "B"], "product": ["X", "X", "Y"], "category": ["Toner"] * 3,
        "ingredients": ["Water", "Water", None], "afterUse": [None, None, "Drying"], "Unnamed: 5": [None] * 3,
    })
    df = clean_catalogue(raw)
    assert len(df) == 1 and "Unnamed: 5" not in df and df.afterUse.iloc[0] == ""


def test_real_catalogue_loads():
    products = load_products()
    assert len(products) > 1600
    assert {p.step for p in products} >= {"cleanser", "toner", "treatment", "moisturizer", "sunscreen"}


# ---- safety & medical -----------------------------------------------------------------

def test_pregnancy_excludes_retinoid(user):
    f = fired(user(pregnant="yes"), product("Water", "Retinol"), "S1_pregnancy_retinoid")
    assert f.action is Action.EXCLUDE and "Retinol" in f.reason


def test_pregnancy_does_not_exclude_bakuchiol(user):
    assert fired(user(pregnant="yes"), product("Bakuchiol"), "S1_pregnancy_retinoid") is None


def test_not_pregnant_retinoid_allowed(user):
    assert fired(user(), product("Retinol"), "S1_pregnancy_retinoid") is None


def test_pregnancy_excludes_hydroquinone(user):
    assert fired(user(pregnant="yes"), product("Hydroquinone"), "S1_pregnancy_hydroquinone").action is Action.EXCLUDE


def test_pregnancy_cautions_bha(user):
    f = fired(user(pregnant="yes"), product("Salicylic Acid"), "S1_pregnancy_bha")
    assert f.action is Action.CAUTION and f.weight < 0


@pytest.mark.parametrize("option, ingredient", [
    ("fragrance", "Parfum"), ("alcohol", "Alcohol Denat."), ("essential_oils", "Lavandula Angustifolia Oil"),
    ("retinol", "Retinal"), ("salicylic_acid", "Salicylic Acid"), ("chemical_sunscreens", "Octocrylene"),
])
def test_avoided_ingredient_excludes_product(user, option, ingredient):
    f = fired(user(avoid_ingredients=[option]), product("Water", ingredient), f"S2_avoid_{option}")
    assert f.action is Action.EXCLUDE and "asked to avoid" in f.reason


def test_avoid_none_excludes_nothing(user):
    assert fired(user(avoid_ingredients=["none"]), product("Parfum"), "S2_avoid_fragrance") is None


@pytest.mark.parametrize("condition", ["rosacea", "eczema", "dermatitis"])
def test_medical_condition_excludes_fragrance_and_essential_oils(user, condition):
    u = user(known_conditions=[condition])
    assert fired(u, product("Parfum"), "S3_medical_fragrance").action is Action.EXCLUDE
    assert fired(u, product("Lavandula Angustifolia Oil"), "S3_medical_fragrance").action is Action.EXCLUDE
    assert fired(u, product("Helianthus Annuus Seed Oil"), "S3_medical_fragrance") is None


def test_rosacea_cautions_rosacea_flagged_product(user):
    u = user(known_conditions=["rosacea"])
    assert fired(u, product(tags={"Rosacea"}), "S3_medical_aggravates").action is Action.CAUTION
    assert fired(u, product(tags={"Eczema"}), "S3_medical_aggravates") is None


def test_eczema_cautions_eczema_flagged_product(user):
    assert fired(user(known_conditions=["eczema"]), product(tags={"Eczema"}), "S3_medical_aggravates") is not None


def test_sensitive_cautions_irritating(user):
    u = user(sensitivity="highly")
    assert fired(u, product(tags={"Irritating"}), "S4_sensitive_irritating").action is Action.CAUTION
    assert fired(user(), product(tags={"Irritating"}), "S4_sensitive_irritating") is None


def test_sensitive_cautions_fragrance_without_medical_condition(user):
    assert fired(user(reacts_to_products="yes"), product("Parfum"), "S4_sensitive_fragrance") is not None
    # with a medical condition fragrance is already excluded, so no duplicate caution
    assert fired(user(known_conditions=["rosacea"]), product("Parfum"), "S4_sensitive_fragrance") is None


def test_sensitive_prefers_soothing(user):
    f = fired(user(sensitivity="moderately"), product("Centella Asiatica Extract"), "S4_sensitive_soothing")
    assert f.action is Action.PREFER and f.weight > 0


# ---- dermatological -------------------------------------------------------------------

def test_acne_cautions_acne_trigger_unless_also_acne_fighting(user, image):
    u = user(image=image(acne=0.9))
    assert fired(u, product(tags={"Acne Trigger"}), "D1_acne_trigger").action is Action.CAUTION
    assert fired(u, product(tags={"Acne Trigger", "Acne Fighting"}), "D1_acne_trigger") is None
    assert fired(user(), product(tags={"Acne Trigger"}), "D1_acne_trigger") is None  # no acne


def test_acne_goal_prefers_bha(user):
    f = fired(user(goals=["acne_control"]), product("Salicylic Acid"), "D1_acne_fighting")
    assert f.action is Action.PREFER and "Salicylic Acid" in f.reason


def test_blackheads_prefer_pore_clearing(user, image):
    assert fired(user(image=image(blackheads=0.8)), product(tags={"Reduces Large Pores"}), "D2_pores") is not None


def test_oily_skin_cautions_heavy_and_prefers_oil_balancing(user):
    u = user(oiliness="very_oily", oil_location="all_over", tightness="never")
    assert u.skin_type == "oily"
    assert fired(u, product(tags={"May Worsen Oily Skin"}), "D3_oily_worsen").action is Action.CAUTION
    assert fired(u, product(tags={"Good For Oily Skin"}), "D3_oily_good").action is Action.PREFER


def test_dry_skin_cautions_drying_and_prefers_hydrating(user):
    u = user(oiliness="dry", tightness="often", flakiness="yes", goals=["anti_aging"])
    assert u.skin_type == "dry"
    assert fired(u, product("Alcohol"), "D3_dry_drying").action is Action.CAUTION
    f = fired(u, product("Sodium Hyaluronate"), "D3_dry_hydrating")
    assert f.action is Action.PREFER and "Sodium Hyaluronate" in f.reason


def test_pigmentation_prefers_brightening(user):
    f = fired(user(known_conditions=["melasma"]), product("Niacinamide"), "D4_pigmentation")
    assert "niacinamide" in f.reason


def test_redness_prefers_centella(user, image):
    assert fired(user(image=image(redness=0.7)), product("Madecassoside"), "D5_redness") is not None


def test_wrinkle_goal_prefers_anti_aging(user):
    assert fired(user(goals=["anti_aging"]), product(tags={"Anti-Aging"}), "D6_wrinkles") is not None
    assert fired(user(), product(tags={"Anti-Aging"}), "D6_wrinkles") is None


# ---- ingredient compatibility ---------------------------------------------------------

def test_retinoid_with_acid_caution_and_sensitive_exclude(user):
    p = product("Retinol", "Glycolic Acid")
    assert fired(user(), p, "I1_retinoid_acid").action is Action.CAUTION
    assert fired(user(), p, "I1_retinoid_acid_sensitive") is None
    assert fired(user(sensitivity="highly"), p, "I1_retinoid_acid_sensitive").action is Action.EXCLUDE


def test_multi_acid_caution_for_sensitive(user):
    p = product("Glycolic Acid", "Salicylic Acid")
    assert fired(user(sensitivity="moderately"), p, "I2_multi_acid_sensitive") is not None
    assert fired(user(), p, "I2_multi_acid_sensitive") is None


def test_vitamin_c_with_acid_caution(user):
    assert fired(user(), product("Ascorbic Acid", "Lactic Acid"), "I3_vitc_acid") is not None
    assert fired(user(), product("Ascorbyl Glucoside", "Lactic Acid"), "I3_vitc_acid") is None  # derivative


def test_routine_conflicts():
    night = product("Retinol", name="Night")
    peel = product("Glycolic Acid", name="Peel")
    serum = product("Ascorbic Acid", name="Vit C")
    bha = product("Salicylic Acid", name="BHA")
    reasons = {c.reason for c in check_routine([night, peel, serum, bha])}
    assert reasons == {
        "Retinoid and exfoliating acid in the same routine",
        "Retinoid and pure vitamin C in the same routine",
        "Two exfoliating products in the same routine",
    }
    assert check_routine([product("Glycerin"), product("Niacinamide")]) == []


# ---- environmental --------------------------------------------------------------------

def test_high_uv_prefers_sunscreen_with_uv_in_reason(user, kl_afternoon):
    f = fired(user(environment=kl_afternoon), product(step="sunscreen"), "E1_uv_sunscreen")
    assert f.action is Action.PREFER and "UV index is 10" in f.reason


def test_sun_protection_goal_prefers_sunscreen_without_weather(user):
    assert fired(user(goals=["sun_protection"]), product(step="sunscreen"), "E1_uv_sunscreen") is not None


def test_low_uv_no_sunscreen_boost(user):
    assert fired(user(environment=EnvironmentData(25, 50, 2)), product(step="sunscreen"), "E1_uv_sunscreen") is None


def test_high_uv_cautions_photosensitising(user, kl_afternoon):
    assert fired(user(environment=kl_afternoon), product("Glycolic Acid"), "E1_uv_photosensitising") is not None


def test_humid_weather_cautions_heavy_moisturiser_prefers_light(user, kl_afternoon):
    u = user(environment=kl_afternoon)
    assert fired(u, product(tags={"May Worsen Oily Skin"}, step="moisturizer"), "E2_humid_heavy") is not None
    assert fired(u, product(tags={"May Worsen Oily Skin"}, step="cleanser"), "E2_humid_heavy") is None
    assert fired(u, product(tags={"Good For Oily Skin"}), "E2_humid_light") is not None


def test_dry_cold_air_prefers_hydrating(user):
    u = user(environment=EnvironmentData(temperature_c=8, humidity_pct=30, uv_index=1))
    assert fired(u, product("Ceramide NP"), "E3_dry_air") is not None


def test_no_weather_no_environmental_rules(user):
    u = user()
    for rid in ("E1_uv_photosensitising", "E2_humid_heavy", "E2_humid_light", "E3_dry_air"):
        assert fired(u, product("Retinol", "Ceramide NP", tags={"Good For Oily Skin", "May Worsen Oily Skin"}), rid) is None


# ---- edge-case user profiles (proposal 3.8) -------------------------------------------

def test_edge_pregnant_acne_user_with_high_uv(user, kl_afternoon):
    u = user(pregnant="yes", breakouts="frequently", goals=["acne_control"], environment=kl_afternoon)
    retinol_acid = assess(product("Retinol", "Salicylic Acid"), u, ALL_RULES)
    assert not retinol_acid.eligible
    assert any("pregnancy" in r for r in retinol_acid.exclusions)
    sunscreen = assess(product("Zinc Oxide", step="sunscreen"), u, ALL_RULES)
    assert sunscreen.eligible and sunscreen.rule_score > 0


def test_edge_rosacea_user_avoiding_everything(user):
    u = user(known_conditions=["rosacea"], sensitivity="highly",
             avoid_ingredients=["fragrance", "alcohol", "essential_oils", "retinol", "salicylic_acid",
                                "chemical_sunscreens"])
    bad = [product("Parfum"), product("Alcohol"), product("Octocrylene", step="sunscreen"),
           product("Retinol"), product("Salicylic Acid"), product("Rosmarinus Officinalis Leaf Oil")]
    assert all(not assess(p, u, ALL_RULES).eligible for p in bad)
    gentle = assess(product("Water", "Centella Asiatica Extract", "Ceramide NP"), u, ALL_RULES)
    assert gentle.eligible and gentle.rule_score > 0


def test_edge_user_with_no_photo_no_weather_keeps_whole_catalogue(user):
    results = apply_rules(user(goals=[]), load_products())
    assert all(r.eligible for r in results)


def test_rule_score_ignores_exclusions_and_sums_cautions_and_benefits(user):
    u = user(oiliness="very_oily", oil_location="all_over", tightness="never", pregnant="yes")
    a = assess(product("Retinol", tags={"Good For Oily Skin", "May Worsen Oily Skin"}), u, ALL_RULES)
    assert not a.eligible
    assert a.rule_score == pytest.approx(sum(f.weight for f in a.findings if f.action is not Action.EXCLUDE))


def test_rule_ids_unique():
    ids = [r.id for r in ALL_RULES]
    assert len(ids) == len(set(ids))


def test_every_rule_is_tested():
    """Runs last in this file: every rule must have been fired by a test above."""
    untested = {r.id for r in ALL_RULES} - TESTED
    assert not untested, f"Rules without a test: {sorted(untested)}"
