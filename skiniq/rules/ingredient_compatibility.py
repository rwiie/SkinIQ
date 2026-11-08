"""Ingredient & Chemical Compatibility KB (proposal Table 3.6.3, literature review 2.3.3).

Product-level rules flag combinations inside one product. `check_routine` checks a
whole routine (several products), used when the routine is assembled (3.6.5).
"""

from __future__ import annotations

from dataclasses import dataclass

from skiniq.products import Product
from skiniq.rules.base import Action, Rule

KB = "ingredient_compatibility"
OVERLOAD = "Poelhekken & van der Bent (2026)"


def _retinoid_with_acid(p: Product) -> bool:
    return p.has("retinoid") and bool({"aha", "bha"} & p.groups)


RULES: list[Rule] = [
    Rule("I1_retinoid_acid", KB, "Retinoid with AHA/BHA in one product -> caution",
         applies=lambda u: not u.is_sensitive,
         matches=lambda p, u: _retinoid_with_acid(p),
         action=Action.CAUTION, weight=-0.2, source=OVERLOAD,
         reason="Combines a retinoid with exfoliating acids, which can over-exfoliate and irritate"),
    Rule("I1_retinoid_acid_sensitive", KB, "Retinoid with AHA/BHA in one product, sensitive skin -> exclude",
         applies=lambda u: u.is_sensitive,
         matches=lambda p, u: _retinoid_with_acid(p),
         action=Action.EXCLUDE, source=OVERLOAD,
         reason="Combines a retinoid with exfoliating acids, too harsh for sensitive skin"),
    Rule("I2_multi_acid_sensitive", KB, "Two or more acid types (AHA/BHA/PHA), sensitive skin -> caution",
         applies=lambda u: u.is_sensitive,
         matches=lambda p, u: len(p.exfoliants) >= 2,
         action=Action.CAUTION, weight=-0.2, source=OVERLOAD,
         reason=lambda p, u: "Contains several exfoliating acids ("
         + "/".join(e.upper() for e in p.exfoliants) + "), which may be too strong for sensitive skin"),
    Rule("I3_hydroquinone_retinoid", KB, "Hydroquinone with a retinoid in one product -> caution",
         applies=lambda u: True,
         matches=lambda p, u: p.has("hydroquinone") and p.has("retinoid"),
         action=Action.CAUTION, weight=-0.25, source="Kim et al. (2017)",
         reason="Combines hydroquinone with a retinoid, which increases irritation and redness"),
    Rule("I4_sensitive_pha", KB, "Sensitive skin -> prefer gentle PHA exfoliants over AHA",
         applies=lambda u: u.is_sensitive,
         matches=lambda p, u: p.has("pha") and not p.has("aha"),
         action=Action.PREFER, weight=0.1, source="Hakulinen (2025); Schachner et al. (2023)",
         reason="Exfoliates with gentle PHAs (e.g. gluconolactone), milder than AHAs for sensitive skin"),
]


@dataclass(frozen=True)
class RoutineConflict:
    products: tuple[str, str]
    reason: str
    advice: str
    source: str = OVERLOAD


def check_routine(products: list[Product]) -> list[RoutineConflict]:
    """Conflicts between products used in the same routine (ingredient overload)."""
    conflicts = []
    for i, a in enumerate(products):
        for b in products[i + 1:]:
            pair = (a.title, b.title)
            if (a.has("retinoid") and {"aha", "bha"} & b.groups) or (b.has("retinoid") and {"aha", "bha"} & a.groups):
                conflicts.append(RoutineConflict(pair, "Retinoid and exfoliating acid in the same routine",
                                                 "Use them on alternate nights"))
            if (a.has("hydroquinone") and b.has("retinoid")) or (b.has("hydroquinone") and a.has("retinoid")):
                conflicts.append(RoutineConflict(pair, "Hydroquinone and retinoid in the same routine",
                                                 "Use them on alternate nights", source="Kim et al. (2017)"))
            if a.exfoliants and b.exfoliants:
                conflicts.append(RoutineConflict(pair, "Two exfoliating products in the same routine",
                                                 "Use only one exfoliating product per routine"))
    return conflicts
