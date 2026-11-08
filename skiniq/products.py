"""Korean skincare product catalogue: cleaning and ingredient detection.

Source: data/raw/products.csv (brand, product, category, ingredients, afterUse).
Cleaning: drop products without ingredients, exact duplicates and the empty column.

Ingredient groups are detected per ingredient (not by substring of the whole list), so
"citric acid" (a pH adjuster in most formulas) is not counted as an exfoliating acid
and plant carrier oils are not counted as essential oils.

The afterUse tags come from an ingredient checker: positive tags (e.g. "Acne Fighting")
and warnings (e.g. "Acne Trigger", "Irritating", and "Rosacea"/"Eczema" meaning "may
aggravate") are triggered by single ingredients, so they are used as soft signals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from skiniq.config import DATA_DIR

PRODUCTS_CSV = DATA_DIR / "raw" / "products.csv"

# Dataset category -> routine step
ROUTINE_STEP = {
    "Face Cleanser": "cleanser",
    "Exfoliator": "exfoliator",
    "Toner": "toner",
    "Essence": "treatment",
    "Serum": "treatment",
    "Facial Treatment": "treatment",
    "Emulsion": "moisturizer",
    "General Moisturizer": "moisturizer",
    "Sunscreen": "sunscreen",
    "Sheet Mask": "mask",
    "Wet Mask": "mask",
    "Overnight Mask": "mask",
}

KNOWLEDGE_DIR = DATA_DIR / "knowledge"
ALIASES_CSV = KNOWLEDGE_DIR / "ingredient_aliases.csv"
FUNCTIONS_CSV = KNOWLEDGE_DIR / "ingredient_functions.csv"
_PERCENT = re.compile(r"\s*\d+(\.\d+)?\s*%.*$")


@dataclass(frozen=True)
class IngredientRule:
    """One row of the ingredient-function mapping table (proposal Table 3.5.2.2)."""

    ingredient: str
    pattern: re.Pattern
    group: str
    functions: tuple[str, ...]
    concerns: tuple[str, ...]
    source: str


@lru_cache(maxsize=1)
def load_aliases(path: str | Path = ALIASES_CSV) -> dict[str, str]:
    df = pd.read_csv(path)
    return {v.strip().lower(): s.strip() for v, s in zip(df.variant, df.standard_name)}


@lru_cache(maxsize=1)
def load_ingredient_table(path: str | Path = FUNCTIONS_CSV) -> tuple[IngredientRule, ...]:
    df = pd.read_csv(path).fillna("")
    return tuple(
        IngredientRule(r.ingredient, re.compile(r.pattern), r.group,
                       tuple(f for f in r.functions.split("|") if f), tuple(c for c in r.concerns.split("|") if c),
                       r.source)
        for r in df.itertuples()
    )


def standardize(name: str) -> str:
    """Standard ingredient name (proposal Table 3.5.2.1): strip concentrations and map
    common or non-INCI names to their INCI name, e.g. 'Vitamin C' -> 'Ascorbic Acid',
    'Salicylic Acid 2%' -> 'Salicylic Acid', 'Aqua' -> 'Water'."""
    clean = _PERCENT.sub("", name).strip(" .*")
    clean = re.sub(r"\s+", " ", clean)
    aliases = load_aliases()
    key = clean.lower()
    if key in aliases:
        return aliases[key]
    if "(" in key:  # e.g. "Fragrance (Parfum)", "Water (Aqua)"
        outer, inner = key.split("(", 1)
        standards = {s.lower(): s for s in aliases.values()}
        for part in (outer.strip(), inner.strip(" )")):
            if part in aliases:
                return aliases[part]
            if part in standards:
                return standards[part]
    return clean


def ingredient_info(name: str) -> list[IngredientRule]:
    """Rows of the mapping table that match one (standardized) ingredient."""
    low = name.lower()
    return [r for r in load_ingredient_table() if r.pattern.search(low)]


def _group_patterns() -> dict[str, list[re.Pattern]]:
    groups: dict[str, list[re.Pattern]] = {}
    for r in load_ingredient_table():
        if r.group:
            groups.setdefault(r.group, []).append(r.pattern)
    return groups


EXFOLIANT_GROUPS = ("aha", "bha", "pha")


@dataclass(frozen=True)
class Product:
    id: int
    brand: str
    name: str
    category: str
    step: str
    ingredients: tuple[str, ...]
    tags: frozenset[str]
    groups: frozenset[str] = field(default_factory=frozenset)
    functions: frozenset[str] = field(default_factory=frozenset)
    concerns: frozenset[str] = field(default_factory=frozenset)  # concerns its ingredients address

    def has(self, group: str) -> bool:
        return group in self.groups

    def matching(self, group: str) -> list[str]:
        """Ingredients of this product in a group (for explanations)."""
        patterns = _group_patterns()[group]
        return [i for i in self.ingredients if any(rx.search(i.lower()) for rx in patterns)]

    def with_function(self, function: str) -> list[str]:
        """Ingredients of this product with a function, e.g. 'humectant' (for explanations)."""
        return [i for i in self.ingredients if any(function in r.functions for r in ingredient_info(i))]

    @property
    def exfoliants(self) -> list[str]:
        return [g for g in EXFOLIANT_GROUPS if g in self.groups]

    @property
    def title(self) -> str:
        return f"{self.brand} {self.name}"


def split_ingredients(text: str) -> tuple[str, ...]:
    """Split on commas (keeping '1,2-Hexanediol' whole) and standardize each name."""
    parts = re.split(r"(?<!\d),|,(?!\d)", text)  # keep "1,2-Hexanediol" whole
    names = (standardize(p) for p in parts if p.strip())
    return tuple(dict.fromkeys(n for n in names if n))  # drop duplicates, keep order


def ingredient_groups(ingredients: tuple[str, ...]) -> frozenset[str]:
    return frozenset(r.group for i in ingredients for r in ingredient_info(standardize(i)) if r.group)


def ingredient_functions(ingredients: tuple[str, ...]) -> tuple[frozenset[str], frozenset[str]]:
    """(functions, concerns) of a product's ingredients from the mapping table."""
    rows = [r for i in ingredients for r in ingredient_info(standardize(i))]
    return frozenset(f for r in rows for f in r.functions), frozenset(c for r in rows for c in r.concerns)


def clean_catalogue(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.loc[:, ~raw.columns.str.startswith("Unnamed")].copy()
    for col in ("brand", "product", "category"):
        df[col] = df[col].astype(str).str.strip()
    df = df.dropna(subset=["ingredients"])
    df = df[df.ingredients.str.strip() != ""]
    df = df.drop_duplicates(subset=["brand", "product"]).reset_index(drop=True)
    df["afterUse"] = df["afterUse"].fillna("")
    return df


@lru_cache(maxsize=4)
def load_products(path: str | Path = PRODUCTS_CSV) -> tuple[Product, ...]:
    df = clean_catalogue(pd.read_csv(path))
    products = []
    for i, r in enumerate(df.itertuples()):
        ingredients = split_ingredients(r.ingredients)
        functions, concerns = ingredient_functions(ingredients)
        products.append(Product(
            id=i,
            brand=r.brand,
            name=r.product,
            category=r.category,
            step=ROUTINE_STEP.get(r.category, "other"),
            ingredients=ingredients,
            tags=frozenset(t.strip() for t in r.afterUse.split(",") if t.strip()),
            groups=ingredient_groups(ingredients),
            functions=functions,
            concerns=concerns,
        ))
    return tuple(products)
