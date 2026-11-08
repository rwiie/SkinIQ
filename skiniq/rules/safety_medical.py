"""Safety & Medical Constraint KB (proposal Table 3.6.3).

Hard exclusions for pregnancy, ingredients the user asked to avoid, and known medical
skin conditions; cautions for sensitive skin.
"""

from __future__ import annotations

from skiniq.rules.base import Action, Rule, ingredient_list

KB = "safety_medical"

# Questionnaire Q11 option -> ingredient group
AVOID_GROUPS = {
    "fragrance": "fragrance",
    "alcohol": "drying_alcohol",
    "essential_oils": "essential_oil",
    "retinol": "retinoid",
    "salicylic_acid": "bha",
    "chemical_sunscreens": "chemical_filter",
}
AVOID_LABELS = {
    "fragrance": "fragrance", "alcohol": "drying alcohol", "essential_oils": "essential oils",
    "retinol": "retinoids", "salicylic_acid": "salicylic acid / BHA", "chemical_sunscreens": "chemical UV filters",
}
ECZEMA_LIKE = ("eczema", "dermatitis", "seborrheic_dermatitis", "psoriasis")


def _avoid_rule(option: str) -> Rule:
    group = AVOID_GROUPS[option]
    return Rule(
        id=f"S2_avoid_{option}",
        kb=KB,
        description=f"User avoids {AVOID_LABELS[option]} -> exclude products containing it",
        applies=lambda u, o=option: o in u.avoid_ingredients,
        matches=lambda p, u, g=group: p.has(g),
        action=Action.EXCLUDE,
        source="User preference (Appendix A, Q11)",
        reason=lambda p, u, o=option, g=group: (
            f"Contains {AVOID_LABELS[o]} ({ingredient_list(p, g)}), which you asked to avoid"),
    )


RULES: list[Rule] = [
    Rule(
        "S1_pregnancy_retinoid", KB,
        "Pregnant or breastfeeding -> exclude retinoids",
        applies=lambda u: u.pregnant,
        matches=lambda p, u: p.has("retinoid"),
        action=Action.EXCLUDE,
         source="Bozzo et al. (2011); Kaplan et al. (2015)",
        reason=lambda p, u: f"Contains a retinoid ({ingredient_list(p, 'retinoid')}), not advised during pregnancy or breastfeeding",
    ),
    Rule(
        "S1_pregnancy_hydroquinone", KB,
        "Pregnant or breastfeeding -> exclude hydroquinone",
        applies=lambda u: u.pregnant,
        matches=lambda p, u: p.has("hydroquinone"),
        action=Action.EXCLUDE,
         source="Bozzo et al. (2011)",
        reason="Contains hydroquinone, not advised during pregnancy or breastfeeding",
    ),
    Rule(
        "S1_pregnancy_bha", KB,
        "Pregnant or breastfeeding -> caution for salicylic acid / BHA",
        applies=lambda u: u.pregnant,
        matches=lambda p, u: p.has("bha"),
        action=Action.CAUTION, weight=-0.15,
         source="Precaution (Bialas et al., 2023; Vinardell, 2015)",
        reason=lambda p, u: f"Contains {ingredient_list(p, 'bha')}; check with your doctor before using BHA during pregnancy",
    ),
    *[_avoid_rule(o) for o in AVOID_GROUPS],
    Rule(
        "S3_medical_fragrance", KB,
        "Rosacea, eczema, psoriasis or dermatitis -> exclude fragrance and essential oils",
        applies=lambda u: bool(u.medical_conditions),
        matches=lambda p, u: p.has("fragrance") or p.has("essential_oil"),
        action=Action.EXCLUDE,
         source="Hakulinen (2025); Schachner et al. (2023)",
        reason=lambda p, u: (
            "Contains fragrance or essential oils, a common trigger for "
            + ", ".join(c.replace("_", " ") for c in u.medical_conditions)),
    ),
    Rule(
        "S3_medical_aggravates", KB,
        "Rosacea or eczema-type condition -> caution for products flagged as aggravating it",
        applies=lambda u: bool(u.medical_conditions),
        matches=lambda p, u: ("rosacea" in u.medical_conditions and "Rosacea" in p.tags)
        or (any(c in u.medical_conditions for c in ECZEMA_LIKE) and "Eczema" in p.tags),
        action=Action.CAUTION, weight=-0.25,
         source="Product dataset tags",
        reason="Contains ingredients flagged as possibly aggravating your skin condition",
    ),
    Rule(
        "S4_sensitive_irritating", KB,
        "Sensitive skin -> caution for products flagged as irritating",
        applies=lambda u: u.is_sensitive,
        matches=lambda p, u: "Irritating" in p.tags,
        action=Action.CAUTION, weight=-0.2,
         source="Product dataset tags",
        reason="Contains ingredients flagged as potentially irritating for sensitive skin",
    ),
    Rule(
        "S4_sensitive_fragrance", KB,
        "Sensitive skin -> caution for fragrance, essential oils and drying alcohol",
        applies=lambda u: u.is_sensitive and not u.medical_conditions,
        matches=lambda p, u: p.has("fragrance") or p.has("essential_oil") or p.has("drying_alcohol"),
        action=Action.CAUTION, weight=-0.15,
         source="Hakulinen (2025); Schachner et al. (2023)",
        reason="Contains fragrance, essential oils or drying alcohol, which can irritate sensitive skin",
    ),
    Rule(
        "S4_sensitive_soothing", KB,
        "Sensitive skin -> prefer soothing, barrier-repairing products",
        applies=lambda u: u.is_sensitive,
        matches=lambda p, u: "Reduces Irritation" in p.tags or p.has("centella") or p.has("ceramide"),
        action=Action.PREFER, weight=0.1,
         source="Schachner et al. (2023); product dataset tags",
        reason="Contains soothing or barrier-repairing ingredients (e.g. centella, ceramides)",
    ),
]
