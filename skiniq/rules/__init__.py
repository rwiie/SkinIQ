"""Knowledge-base condition gates (proposal 3.6.3): four knowledge bases applied to the
USCE and every product. See docs/rules.md for the full rule table."""

from __future__ import annotations

from skiniq.fusion import USCE
from skiniq.products import Product
from skiniq.rules import dermatological, environmental, ingredient_compatibility, safety_medical
from skiniq.rules.base import Action, Finding, ProductAssessment, Rule, assess
from skiniq.rules.ingredient_compatibility import RoutineConflict, check_routine

ALL_RULES: list[Rule] = [
    *safety_medical.RULES,
    *dermatological.RULES,
    *ingredient_compatibility.RULES,
    *environmental.RULES,
]


def apply_rules(usce: USCE, products: tuple[Product, ...] | list[Product],
                rules: list[Rule] | None = None) -> list[ProductAssessment]:
    """Assess every product; excluded products are kept in the list with eligible=False."""
    rules = ALL_RULES if rules is None else rules
    return [assess(p, usce, rules) for p in products]


__all__ = ["ALL_RULES", "Action", "Finding", "ProductAssessment", "Rule", "RoutineConflict",
           "apply_rules", "check_routine"]
