"""Content-based ranking: TF-IDF + cosine similarity (proposal 3.6.4).

The USCE (about the user) and product ingredient lists (about products) live in
different spaces, so both are expressed in one vocabulary taken from the
ingredient-function mapping table (data/knowledge/ingredient_functions.csv):

- Product document: for each *active* mapped ingredient, three kinds of token:
  the ingredient (`ing:Niacinamide`), its functions (`fn:brightening`) and the
  concerns it addresses (`concern:hyperpigmentation`). Formulation-only ingredients
  (water, thickeners, preservatives ...) are left out. Tokens are weighted by the
  ingredient's position in the list (earlier usually means higher concentration).
- User query: weighted `fn:` / `concern:` / `ing:` tokens derived from the USCE
  (concern scores, skin type, goals, sensitivity, weather), see `query_terms`.

TF-IDF (sublinear tf, smooth idf) is fitted on all products; the query is weighted
with the same idf; cosine similarity scores each product. The final score combines
similarity with the knowledge-base rule score (3.6.3):

    score = similarity + RULE_SCORE_WEIGHT * rule_score

Only eligible products (not excluded by a rule) are ranked.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from skiniq.fusion import USCE
from skiniq.products import Product, ingredient_info, load_products
from skiniq.rules import ProductAssessment, apply_rules

RULE_SCORE_WEIGHT = 0.5
POSITION_DECAY = 10.0  # weight = 1 / (1 + position / POSITION_DECAY)
FORMULATION_FUNCTIONS = {
    "solvent", "thickener", "ph_adjuster", "emulsifier", "chelating", "preservative",
    "preservative_booster", "antioxidant_preservative", "cleansing",
}
ROUTINE_ORDER = ("cleanser", "toner", "treatment", "moisturizer", "sunscreen", "exfoliator", "mask")

# USCE need -> query tokens (functions / concerns / named ingredients)
SKIN_TYPE_TERMS = {
    "dry": {"concern:dryness": 1.0, "fn:humectant": 0.8, "fn:barrier_repair": 0.8, "fn:emollient": 0.6},
    "oily": {"concern:oiliness": 1.0, "fn:sebum_regulation": 0.8, "fn:oil_absorbing": 0.4},
    "combination": {"concern:oiliness": 0.6, "fn:sebum_regulation": 0.5, "fn:hydration": 0.6},
    "normal": {"fn:hydration": 0.6, "fn:antioxidant": 0.4},
}
GOAL_TERMS = {
    "hydration": {"fn:hydration": 1.0, "fn:humectant": 0.8, "concern:dryness": 0.6},
    "acne_control": {"fn:acne_control": 1.0, "concern:acne": 0.8},
    "oil_control": {"fn:sebum_regulation": 1.0, "concern:oiliness": 0.8},
    "brightening": {"fn:brightening": 1.0, "concern:hyperpigmentation": 0.6},
    "anti_aging": {"fn:anti_aging": 1.0, "fn:collagen_support": 0.6, "concern:wrinkles": 0.6},
    "sensitivity_reduction": {"fn:soothing": 1.0, "fn:barrier_repair": 0.8, "concern:sensitivity": 0.6},
    "pigmentation_reduction": {"fn:brightening": 1.0, "concern:hyperpigmentation": 1.0},
    "sun_protection": {"fn:uv_filter": 1.0, "concern:sun_damage": 0.8},
}
CONCERN_EXTRA = {  # functions that address each photo/questionnaire concern
    "acne": {"fn:acne_control": 0.8, "fn:anti_inflammatory": 0.4},
    "blackheads": {"fn:pore_clearing": 0.8, "fn:exfoliant": 0.5},
    "enlarged_pores": {"fn:pore_clearing": 0.8, "fn:sebum_regulation": 0.5},
    "hyperpigmentation": {"fn:brightening": 0.8},
    "redness": {"fn:soothing": 0.8, "fn:anti_inflammatory": 0.6},
    "wrinkles": {"fn:anti_aging": 0.8, "fn:collagen_support": 0.5},
}


# ---- documents ------------------------------------------------------------------------

def product_terms(product: Product) -> dict[str, float]:
    """Weighted tokens of one product (see module docstring)."""
    terms: dict[str, float] = defaultdict(float)
    for pos, ingredient in enumerate(product.ingredients):
        rows = [r for r in ingredient_info(ingredient) if not set(r.functions) <= FORMULATION_FUNCTIONS]
        if not rows:
            continue
        w = 1.0 / (1.0 + pos / POSITION_DECAY)
        for r in rows:
            terms[f"ing:{r.ingredient}"] += w
            for f in r.functions:
                if f not in FORMULATION_FUNCTIONS:
                    terms[f"fn:{f}"] += w
            for c in r.concerns:
                terms[f"concern:{c}"] += w
    return dict(terms)


def query_terms(usce: USCE) -> dict[str, float]:
    """Weighted tokens describing what this user's products should provide."""
    q: dict[str, float] = defaultdict(float)

    def add(terms: dict[str, float], scale: float = 1.0) -> None:
        for t, w in terms.items():
            q[t] += w * scale

    for c, concern in usce.concerns.items():
        if concern.status != "none":
            add({f"concern:{c}": 1.0, **CONCERN_EXTRA.get(c, {})}, concern.score)
    add(SKIN_TYPE_TERMS.get(usce.skin_type, {}))
    for g in usce.goals:
        add(GOAL_TERMS.get(g, {}))
    if usce.is_sensitive:
        add({"fn:soothing": 0.8, "fn:barrier_repair": 0.8, "concern:sensitivity": 0.8, "fn:gentle_exfoliant": 0.3})
    flags = usce.env_flags
    if flags.get("high_uv"):
        add({"fn:uv_filter": 1.0, "concern:sun_damage": 1.0, "fn:antioxidant": 0.4})
    if flags.get("dry_air") or flags.get("cold"):
        add({"fn:humectant": 0.6, "fn:barrier_repair": 0.6, "concern:dryness": 0.6})
    if flags.get("humid") or flags.get("hot"):
        add({"fn:sebum_regulation": 0.4, "concern:oiliness": 0.4})
    return dict(q)


# ---- TF-IDF index -----------------------------------------------------------------------

class ProductIndex:
    """TF-IDF vectors of all products, fitted once."""

    def __init__(self, products: tuple[Product, ...] | list[Product]) -> None:
        self.products = list(products)
        docs = [product_terms(p) for p in self.products]
        # Documents are token->weight dicts; the analyzer expands them to weighted counts.
        self.vectorizer = TfidfVectorizer(analyzer=lambda d: list(d), sublinear_tf=True, norm=None)
        self.vectorizer.fit(docs)
        self.vocab = self.vectorizer.vocabulary_
        self.idf = self.vectorizer.idf_
        self.matrix = normalize(self._weighted(docs))  # rows: products (L2-normalised TF-IDF)

    def _weighted(self, docs: list[dict[str, float]]):
        from scipy.sparse import csr_matrix

        rows, cols, vals = [], [], []
        for i, d in enumerate(docs):
            for t, w in d.items():
                j = self.vocab.get(t)
                if j is not None and w > 0:
                    rows.append(i)
                    cols.append(j)
                    vals.append((1.0 + np.log(w)) if w >= 1 else w)  # sublinear tf for weighted counts
        tf = csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.vocab)))
        return tf.multiply(self.idf).tocsr()

    def query_vector(self, terms: dict[str, float]) -> np.ndarray:
        v = np.asarray(self._weighted([terms]).todense()).ravel()
        n = np.linalg.norm(v)
        return v / n if n else v

    def similarity(self, terms: dict[str, float]) -> np.ndarray:
        return self.matrix @ self.query_vector(terms)

    def top_terms(self, product_pos: int, terms: dict[str, float], k: int = 3) -> list[str]:
        """Shared tokens contributing most to a product's similarity (for explanations)."""
        q = self.query_vector(terms)
        row = self.matrix[product_pos].toarray().ravel()
        contrib = row * q
        inv = {j: t for t, j in self.vocab.items()}
        best = np.argsort(contrib)[::-1][:k]
        return [inv[j] for j in best if contrib[j] > 0]


@lru_cache(maxsize=2)
def default_index() -> ProductIndex:
    return ProductIndex(load_products())


# ---- ranking ----------------------------------------------------------------------------

@dataclass
class Recommendation:
    product: Product
    similarity: float
    rule_score: float
    score: float
    matched: list[str]
    assessment: ProductAssessment = field(repr=False)


def rank_products(usce: USCE, index: ProductIndex | None = None, per_step: int = 3,
                  steps: tuple[str, ...] = ROUTINE_ORDER) -> dict[str, list[Recommendation]]:
    """Top `per_step` eligible products for each routine step, best first."""
    index = index or default_index()
    terms = query_terms(usce)
    sims = index.similarity(terms)
    assessments = apply_rules(usce, index.products)
    by_step: dict[str, list[Recommendation]] = defaultdict(list)
    for pos, (product, a) in enumerate(zip(index.products, assessments)):
        if not a.eligible or product.step not in steps:
            continue
        score = float(sims[pos]) + RULE_SCORE_WEIGHT * a.rule_score
        by_step[product.step].append(Recommendation(product, round(float(sims[pos]), 4), a.rule_score,
                                                    round(score, 4), [], a))
    out = {}
    for step in steps:
        ranked = sorted(by_step.get(step, []), key=lambda r: (-r.score, r.product.id))[:per_step]
        for r in ranked:
            r.matched = index.top_terms(index.products.index(r.product), terms)
        out[step] = ranked
    return out
