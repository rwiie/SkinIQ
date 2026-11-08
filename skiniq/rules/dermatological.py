"""Dermatological KB (proposal Table 3.6.3): match products to skin type, detected
concerns and the user's goals."""

from __future__ import annotations

from skiniq.rules.base import Action, Rule, ingredient_list

KB = "dermatological"


def has_concern(u, condition: str) -> bool:
    """Concern detected or possible (photo or questionnaire)."""
    return u.concerns[condition].status in ("detected", "possible")


def _goal(u, goal: str) -> bool:
    return goal in u.goals


RULES: list[Rule] = [
    # Acne
    Rule("D1_acne_trigger", KB, "Acne concern -> caution for products flagged as acne triggers",
         applies=lambda u: has_concern(u, "acne"),
         matches=lambda p, u: "Acne Trigger" in p.tags and "Acne Fighting" not in p.tags,
         action=Action.CAUTION, weight=-0.25,
         source="Product dataset tags",
         reason="Contains ingredients that may clog pores or trigger breakouts"),
    Rule("D1_acne_fighting", KB, "Acne concern or acne-control goal -> prefer acne-fighting products",
         applies=lambda u: has_concern(u, "acne") or _goal(u, "acne_control"),
         matches=lambda p, u: "Acne Fighting" in p.tags or p.has("bha"),
         action=Action.PREFER, weight=0.25,
         source="Schachner et al. (2023); Li et al. (2020); product dataset tags",
         reason=lambda p, u: "Helps fight acne" + (f" ({ingredient_list(p, 'bha')})" if p.has("bha") else "")),
    # Blackheads and pores
    Rule("D2_pores", KB, "Blackheads or enlarged pores -> prefer pore-clearing products",
         applies=lambda u: has_concern(u, "blackheads") or has_concern(u, "enlarged_pores"),
         matches=lambda p, u: "Reduces Large Pores" in p.tags or p.has("bha"),
         action=Action.PREFER, weight=0.2,
         source="Schachner et al. (2023); product dataset tags",
         reason=lambda p, u: "Helps clear pores and blackheads"
         + (f" ({ingredient_list(p, 'bha')})" if p.has("bha") else "")),
    # Skin type
    Rule("D3_oily_worsen", KB, "Oily or combination skin -> caution for products that may worsen oiliness",
         applies=lambda u: u.skin_type in ("oily", "combination"),
         matches=lambda p, u: "May Worsen Oily Skin" in p.tags and "Good For Oily Skin" not in p.tags,
         action=Action.CAUTION, weight=-0.2,
         source="Product dataset tags",
         reason="May feel heavy or worsen oiliness"),
    Rule("D3_oily_good", KB, "Oily or combination skin, or oil-control goal -> prefer oil-balancing products",
         applies=lambda u: u.skin_type in ("oily", "combination") or _goal(u, "oil_control"),
         matches=lambda p, u: "Good For Oily Skin" in p.tags,
         action=Action.PREFER, weight=0.2,
         source="Product dataset tags",
         reason="Suited to oily and combination skin"),
    Rule("D3_dry_drying", KB, "Dry skin -> caution for drying products",
         applies=lambda u: u.skin_type == "dry",
         matches=lambda p, u: "Drying" in p.tags or p.has("drying_alcohol"),
         action=Action.CAUTION, weight=-0.25,
         source="Product dataset tags",
         reason="May dry out the skin"),
    Rule("D3_dry_hydrating", KB, "Dry skin or hydration goal -> prefer hydrating, barrier-supporting products",
         applies=lambda u: u.skin_type == "dry" or _goal(u, "hydration"),
         matches=lambda p, u: "Hydrating" in p.tags or p.has("hyaluronic_acid") or p.has("ceramide"),
         action=Action.PREFER, weight=0.2,
         source="Hakulinen (2025); Pavlou et al. (2021)",
         reason=lambda p, u: "Hydrates and supports the skin barrier ("
         + ingredient_list(p, "hyaluronic_acid" if p.has("hyaluronic_acid") else "ceramide", 2) + ")"
         if (p.has("hyaluronic_acid") or p.has("ceramide")) else "Hydrates the skin"),
    # Pigmentation and brightening
    Rule("D4_pigmentation", KB, "Hyperpigmentation or brightening/pigmentation goal -> prefer brightening products",
         applies=lambda u: has_concern(u, "hyperpigmentation") or _goal(u, "brightening")
         or _goal(u, "pigmentation_reduction"),
         matches=lambda p, u: bool({"Brightening", "Dark Spots"} & p.tags) or p.has("niacinamide") or p.has("vitamin_c"),
         action=Action.PREFER, weight=0.2,
         source="Hakulinen (2025); Li et al. (2020); Suarantika et al. (2025)",
         reason=lambda p, u: "Helps fade dark spots and even skin tone"
         + (" (niacinamide)" if p.has("niacinamide") else " (vitamin C)" if p.has("vitamin_c") else "")),
    # Redness
    Rule("D5_redness", KB, "Redness or sensitivity-reduction goal -> prefer redness-reducing products",
         applies=lambda u: has_concern(u, "redness") or _goal(u, "sensitivity_reduction"),
         matches=lambda p, u: "Redness Reducing" in p.tags or p.has("centella"),
         action=Action.PREFER, weight=0.2,
         source="Product dataset tags",
         reason=lambda p, u: "Helps calm redness" + (" (centella)" if p.has("centella") else "")),
    # Wrinkles
    Rule("D6_wrinkles", KB, "Wrinkles or anti-aging goal -> prefer anti-aging products",
         applies=lambda u: has_concern(u, "wrinkles") or _goal(u, "anti_aging"),
         matches=lambda p, u: "Anti-Aging" in p.tags or p.has("retinoid"),
         action=Action.PREFER, weight=0.2,
         source="Hakulinen (2025); Narsa et al. (2024)",
         reason=lambda p, u: "Targets fine lines and wrinkles"
         + (f" ({ingredient_list(p, 'retinoid', 1)})" if p.has("retinoid") else "")),
]
