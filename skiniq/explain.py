"""Explanations (XAI) for the skin analysis and every recommended product (proposal 3.6.5).

All text is built from the data that produced the decision, so each explanation is
faithful to the system: questionnaire answers behind the skin type, the photo or
questionnaire evidence behind each concern, the ingredients and functions behind each
product match, and the knowledge-base rules that fired.
"""

from __future__ import annotations

from skiniq.fusion import USCE
from skiniq.products import Product, ingredient_info
from skiniq.recommender import FORMULATION_FUNCTIONS, Recommendation, query_terms

CONCERN_LABELS = {
    "acne": "acne", "blackheads": "blackheads", "hyperpigmentation": "dark spots / hyperpigmentation",
    "redness": "redness", "enlarged_pores": "enlarged pores", "wrinkles": "fine lines and wrinkles",
    "dryness": "dryness", "oiliness": "oiliness", "sensitivity": "sensitivity", "sun_damage": "sun damage",
}
FUNCTION_LABELS = {
    "exfoliant": "exfoliates", "gentle_exfoliant": "gently exfoliates", "acne_control": "fights acne",
    "pore_clearing": "clears pores", "anti_aging": "anti-aging", "brightening": "brightens",
    "hydration": "hydrates", "humectant": "draws in moisture", "barrier_repair": "repairs the skin barrier",
    "soothing": "soothes", "anti_inflammatory": "calms inflammation", "sebum_regulation": "controls oil",
    "antioxidant": "antioxidant", "collagen_support": "supports collagen", "emollient": "softens",
    "occlusive": "locks in moisture", "uv_filter": "UV protection", "oil_absorbing": "absorbs oil",
    "plumping": "plumps", "repair": "repairs",
}
STEP_LABELS = {
    "cleanser": "Cleanser", "toner": "Toner", "treatment": "Serum / treatment", "moisturizer": "Moisturizer",
    "sunscreen": "Sunscreen", "exfoliator": "Exfoliator", "mask": "Mask",
}
DISCLAIMER = ("SkinIQ gives general skincare guidance, not a medical diagnosis. For persistent or severe "
              "skin problems, or before using new actives while pregnant, consult a dermatologist.")


def _label(concern: str) -> str:
    return CONCERN_LABELS.get(concern, concern.replace("_", " "))


# ---- skin analysis ------------------------------------------------------------------------

def explain_skin_analysis(usce: USCE) -> dict:
    """Section 1: skin type, concerns and environment, each with its evidence."""
    concerns = []
    for name, c in sorted(usce.concerns.items(), key=lambda kv: -kv[1].score):
        if c.status == "none":
            continue
        evidence = []
        if "photo" in c.sources and c.photo is not None:
            evidence.append(f"photo analysis {c.photo:.0%}")
        if "questionnaire" in c.sources:
            evidence.append("your questionnaire answers")
        where = f" (seen on: {', '.join(l.replace('_', ' ') for l in c.locations)})" if c.locations else ""
        concerns.append({
            "concern": name,
            "label": _label(name),
            "status": c.status,
            "confidence": c.score,
            "text": f"{'Detected' if c.status == 'detected' else 'Possible'}: {_label(name)}{where}, "
                    f"based on {' and '.join(evidence)}.",
        })

    env = usce.environment
    if env:
        bits = [f"{env.temperature_c:g} °C", f"{env.humidity_pct:g}% humidity",
                f"UV index {env.uv_index:g} ({env.uv_level.replace('_', ' ')})"]
        env_text = (f"Weather{' in ' + env.location if env.location else ''}: " + ", ".join(bits) + ".")
    else:
        env_text = "Weather data unavailable, so no environmental adjustments were made."

    return {
        "skin_type": usce.skin_type,
        "skin_type_text": f"Your skin type is {usce.skin_type}, based on your answers.",
        "skin_type_reasons": usce.skin_type_reasons[:3],
        "concerns": concerns,
        "no_concerns_text": None if concerns else "No skin concerns were detected.",
        "sensitive": usce.is_sensitive,
        "environment_text": env_text,
        "photo_used": usce.has_photo,
    }


# ---- advice -------------------------------------------------------------------------------

def personalised_advice(usce: USCE) -> list[str]:
    """Section 2: short, rule-based skincare advice for this user."""
    tips = []
    flags = usce.env_flags
    if flags.get("high_uv"):
        tips.append(f"The UV index is {usce.environment.uv_index:g}: apply sunscreen every morning and reapply "
                    "every 2 hours outdoors.")
    else:
        tips.append("Use sunscreen every morning, even on cloudy days.")
    if flags.get("humid") or flags.get("hot"):
        tips.append("Hot, humid weather: choose light gel or lotion textures and cleanse gently twice a day.")
    if flags.get("dry_air") or flags.get("cold"):
        tips.append("Dry or cold air: layer a hydrating toner or serum under your moisturizer.")
    if usce.is_sensitive:
        tips.append("Sensitive skin: patch-test new products on your jaw for 2–3 days and introduce one at a time.")
    if usce.pregnant:
        tips.append("Pregnancy or breastfeeding: retinoids and hydroquinone were excluded; check new actives "
                    "with your doctor.")
    if usce.medical_conditions:
        tips.append("You reported " + ", ".join(c.replace("_", " ") for c in usce.medical_conditions)
                    + ": fragrance and essential oils were excluded. See a dermatologist if it flares.")
    detected = set(usce.detected_conditions)
    if "acne" in detected:
        tips.append("Acne: avoid picking, change pillowcases often, and give a new routine 6–8 weeks.")
    if "hyperpigmentation" in detected:
        tips.append("Dark spots fade slowly and return with sun exposure, so daily sunscreen matters most.")
    if usce.skin_type == "dry":
        tips.append("Dry skin: use lukewarm water and apply moisturizer to slightly damp skin.")
    if usce.skin_type in ("oily", "combination"):
        tips.append("Oily areas still need moisturizer: skipping it can increase oil production.")
    return tips


# ---- products ------------------------------------------------------------------------------

def key_ingredients(product: Product, usce: USCE, limit: int = 3) -> list[dict]:
    """Active ingredients of this product that serve this user's needs, most relevant first.

    Relevance = sum of the user's query weights for the ingredient's functions and concerns,
    scaled by list position (earlier = higher concentration), as in the ranking.
    """
    needs = query_terms(usce)
    found, seen = [], set()
    for pos, ingredient in enumerate(product.ingredients):
        for row in ingredient_info(ingredient):
            functions = [f for f in row.functions if f not in FORMULATION_FUNCTIONS]
            useful_f = [f for f in functions if f"fn:{f}" in needs]
            useful_c = [c for c in row.concerns if f"concern:{c}" in needs]
            if (useful_f or useful_c) and row.ingredient not in seen:
                seen.add(row.ingredient)
                weight = sum(needs[f"fn:{f}"] for f in useful_f) + sum(needs[f"concern:{c}"] for c in useful_c)
                relevance = weight / (1.0 + pos / 10.0)
                found.append((relevance, pos, {"ingredient": ingredient, "functions": useful_f or functions[:2],
                                               "helps_with": useful_c}))
    found.sort(key=lambda t: (-t[0], t[1]))
    return [item for _, _, item in found[:limit]]


def explain_product(rec: Recommendation, usce: USCE) -> dict:
    """Section 4: why this product, with its key ingredients, rule reasons and cautions."""
    keys = key_ingredients(rec.product, usce)
    reasons = []
    for k in keys:
        does = ", ".join(FUNCTION_LABELS.get(f, f.replace("_", " ")) for f in k["functions"])
        helps = f" — helps with {', '.join(_label(c) for c in k['helps_with'])}" if k["helps_with"] else ""
        reasons.append(f"{k['ingredient']}: {does}{helps}")
    benefits = [b for b in rec.assessment.benefits]
    return {
        "product": rec.product.title,
        "brand": rec.product.brand,
        "name": rec.product.name,
        "category": rec.product.category,
        "step": rec.product.step,
        "score": rec.score,
        "match": rec.similarity,
        "rule_score": rec.rule_score,
        "summary": _summary(keys, benefits),
        "key_ingredients": reasons,
        "why": benefits,
        "cautions": rec.assessment.cautions,
    }


def _summary(keys: list[dict], benefits: list[str]) -> str:
    concerns = []
    for k in keys:
        for c in k["helps_with"]:
            if c not in concerns:
                concerns.append(c)
    if concerns:
        names = ", ".join(k["ingredient"] for k in keys[:2])
        return f"Chosen for your {', '.join(_label(c) for c in concerns[:3])}, mainly thanks to {names}."
    if benefits:
        return benefits[0] + "."
    return "A gentle, well-matched option for this step of your routine."
