"""Ingredient & Chemical Compatibility KB (proposal Table 3.6.3).

Product-level rules flag combinations inside one product. `check_routine` checks a
whole routine (several products), used when the routine is assembled (3.6.5).
"""

from __future__ import annotations

from dataclasses import dataclass

from skiniq.products import Product
from skiniq.rules.base import Action, Rule

KB = "ingredient_compatibility"


def _strong_actives(p: Product) -> int:
    return len(p.exfoliants) + p.has("retinoid") + p.has("pure_vitamin_c")


RULES: list[Rule] = [
    Rule("I1_retinoid_acid", KB, "Retinoid with AHA/BHA in one product -> exclude for sensitive skin, else caution",
         applies=lambda u: True,
         matches=lambda p, u: p.has("retinoid") and bool({"aha", "bha"} & p.groups),
         action=Action.CAUTION, weight=-0.2,
         reason="Combines a retinoid with exfoliating acids, which can over-exfoliate and irritate"),
    Rule("I1_retinoid_acid_sensitive", KB, "Retinoid with AHA/BHA, sensitive skin -> exclude",
         applies=lambda u: u.is_sensitive,
         matches=lambda p, u: p.has("retinoid") and bool({"aha", "bha"} & p.groups),
         action=Action.EXCLUDE,
         reason="Combines a retinoid with exfoliating acids, too harsh for sensitive skin"),
    Rule("I2_multi_acid_sensitive", KB, "Two or more acid types (AHA/BHA/PHA), sensitive skin -> caution",
         applies=lambda u: u.is_sensitive,
         matches=lambda p, u: len(p.exfoliants) >= 2,
         action=Action.CAUTION, weight=-0.2,
         reason=lambda p, u: "Contains several exfoliating acids ("
         + "/".join(e.upper() for e in p.exfoliants) + "), which may be too strong for sensitive skin"),
    Rule("I3_vitc_acid", KB, "Pure vitamin C with AHA/BHA in one product -> caution",
         applies=lambda u: True,
         matches=lambda p, u: p.has("pure_vitamin_c") and bool({"aha", "bha"} & p.groups),
         action=Action.CAUTION, weight=-0.1,
         reason="Combines pure vitamin C with exfoliating acids, which can sting"),
]


@dataclass(frozen=True)
class RoutineConflict:
    products: tuple[str, str]
    reason: str
    advice: str


def check_routine(products: list[Product]) -> list[RoutineConflict]:
    """Conflicts between products used in the same routine."""
    conflicts = []
    for i, a in enumerate(products):
        for b in products[i + 1:]:
            pair = (a.title, b.title)
            if (a.has("retinoid") and {"aha", "bha"} & b.groups) or (b.has("retinoid") and {"aha", "bha"} & a.groups):
                conflicts.append(RoutineConflict(pair, "Retinoid and exfoliating acid in the same routine",
                                                 "Use them on alternate nights"))
            if (a.has("retinoid") and b.has("pure_vitamin_c")) or (b.has("retinoid") and a.has("pure_vitamin_c")):
                conflicts.append(RoutineConflict(pair, "Retinoid and pure vitamin C in the same routine",
                                                 "Use vitamin C in the morning and the retinoid at night"))
            if a.exfoliants and b.exfoliants:
                conflicts.append(RoutineConflict(pair, "Two exfoliating products in the same routine",
                                                 "Use only one exfoliating product per routine"))
    return conflicts
