"""Rule engine for the knowledge-base condition gates (proposal 3.6.3).

A rule has two parts:
- `applies(usce)`: does the rule concern this user? (e.g. "user is pregnant")
- `matches(product, usce)`: does the product trigger it? (e.g. "contains a retinoid")

and one action when both hold:
- EXCLUDE: the product is removed from the eligible pool.
- CAUTION: the product stays, with a warning and a score penalty.
- PREFER:  the product stays, with a reason and a score bonus.

Every triggered rule records a plain-language reason, used by the explanations (3.6.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from skiniq.fusion import USCE
from skiniq.products import Product


class Action(str, Enum):
    EXCLUDE = "exclude"
    CAUTION = "caution"
    PREFER = "prefer"


@dataclass(frozen=True)
class Rule:
    id: str
    kb: str
    description: str
    applies: Callable[[USCE], bool]
    matches: Callable[[Product, USCE], bool]
    action: Action
    reason: Callable[[Product, USCE], str] | str
    weight: float = 0.0  # score adjustment: positive for PREFER, negative for CAUTION

    def explain(self, product: Product, usce: USCE) -> str:
        return self.reason(product, usce) if callable(self.reason) else self.reason


@dataclass
class Finding:
    rule_id: str
    kb: str
    action: Action
    reason: str
    weight: float


@dataclass
class ProductAssessment:
    product: Product
    findings: list[Finding] = field(default_factory=list)

    @property
    def eligible(self) -> bool:
        return not any(f.action is Action.EXCLUDE for f in self.findings)

    @property
    def rule_score(self) -> float:
        """Sum of CAUTION penalties and PREFER bonuses (used by the ranking, 3.6.4)."""
        return round(sum(f.weight for f in self.findings if f.action is not Action.EXCLUDE), 4)

    def by_action(self, action: Action) -> list[Finding]:
        return [f for f in self.findings if f.action is action]

    @property
    def exclusions(self) -> list[str]:
        return [f.reason for f in self.by_action(Action.EXCLUDE)]

    @property
    def cautions(self) -> list[str]:
        return [f.reason for f in self.by_action(Action.CAUTION)]

    @property
    def benefits(self) -> list[str]:
        return [f.reason for f in self.by_action(Action.PREFER)]


def assess(product: Product, usce: USCE, rules: list[Rule]) -> ProductAssessment:
    out = ProductAssessment(product)
    for rule in rules:
        if rule.applies(usce) and rule.matches(product, usce):
            out.findings.append(Finding(rule.id, rule.kb, rule.action, rule.explain(product, usce), rule.weight))
    return out


def ingredient_list(product: Product, group: str, limit: int = 3) -> str:
    names = product.matching(group)
    shown = ", ".join(names[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")
