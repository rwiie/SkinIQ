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

# Regexes matched against each ingredient name (lower case, whole ingredient).
INGREDIENT_GROUPS: dict[str, str] = {
    "retinoid": r"^(retinol|retinal|retinaldehyde|retinyl \w+|hydroxypinacolone retinoate|tretinoin|adapalene)\b",
    "aha": r"^(glycolic|lactic|mandelic|tartaric) acid\b",
    "bha": r"^(salicylic acid|betaine salicylate|capryloyl salicylic acid)\b",
    "pha": r"^(gluconolactone|lactobionic acid)\b",
    "vitamin_c": r"ascorbic acid|ascorbyl",
    "pure_vitamin_c": r"^(l-)?ascorbic acid\b",
    "niacinamide": r"^niacinamide\b",
    "hydroquinone": r"^hydroquinone\b",
    "fragrance": r"^(fragrance|parfum|perfume|aroma)\b",
    "drying_alcohol": r"^(alcohol|alcohol denat\.?|ethanol|sd alcohol[\w\- ]*|isopropyl alcohol)$",
    "essential_oil": (
        r"(lavandula|lavender|bergamia|bergamot|citrus [\w ]*(peel|fruit|leaf|flower) oil|lemon|orange|grapefruit|"
        r"lime|melaleuca|tea tree|mentha|peppermint|spearmint|eucalyptus|rosmarinus|rosemary|pelargonium|"
        r"geranium|rosa [\w ]*flower oil|ylang|cananga|chamomilla [\w ]*oil|anthemis [\w ]*oil|cinnamomum|"
        r"clove|eugenia caryophyllus|juniperus|cymbopogon|lemongrass|patchouli|pogostemon|santalum|"
        r"cupressus|thymus|origanum|salvia sclarea|citronella)[\w ]*oil\b"
    ),
    "chemical_filter": (
        r"ethylhexyl methoxycinnamate|octinoxate|avobenzone|butyl methoxydibenzoylmethane|homosalate|"
        r"octisalate|ethylhexyl salicylate|oxybenzone|benzophenone|octocrylene|bis-ethylhexyloxyphenol|"
        r"diethylamino hydroxybenzoyl|ethylhexyl triazone|drometrizole|polysilicone-15|"
        r"isoamyl p-methoxycinnamate|methylene bis-benzotriazolyl|diethylhexyl butamido triazone|"
        r"phenylbenzimidazole sulfonic acid|terephthalylidene dicamphor"
    ),
    "mineral_filter": r"^(zinc oxide|titanium dioxide)\b",
    "hyaluronic_acid": r"hyaluron",
    "ceramide": r"^ceramide",
    "centella": r"centella|madecassoside|asiaticoside|madecassic|asiatic acid",
    "panthenol": r"^panthenol\b",
}
_COMPILED = {k: re.compile(v) for k, v in INGREDIENT_GROUPS.items()}

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

    def has(self, group: str) -> bool:
        return group in self.groups

    def matching(self, group: str) -> list[str]:
        """Ingredients of this product in a group (for explanations)."""
        rx = _COMPILED[group]
        return [i for i in self.ingredients if rx.search(i.lower())]

    @property
    def exfoliants(self) -> list[str]:
        return [g for g in EXFOLIANT_GROUPS if g in self.groups]

    @property
    def title(self) -> str:
        return f"{self.brand} {self.name}"


def split_ingredients(text: str) -> tuple[str, ...]:
    """Split on commas, but keep numbers like '1,2-Hexanediol' together."""
    parts = re.split(r"(?<!\d),|,(?!\d)", text)  # keep "1,2-Hexanediol" whole
    return tuple(p.strip() for p in parts if p.strip())


def ingredient_groups(ingredients: tuple[str, ...]) -> frozenset[str]:
    found = set()
    for ing in ingredients:
        low = ing.lower()
        for group, rx in _COMPILED.items():
            if rx.search(low):
                found.add(group)
    return frozenset(found)


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
        products.append(Product(
            id=i,
            brand=r.brand,
            name=r.product,
            category=r.category,
            step=ROUTINE_STEP.get(r.category, "other"),
            ingredients=ingredients,
            tags=frozenset(t.strip() for t in r.afterUse.split(",") if t.strip()),
            groups=ingredient_groups(ingredients),
        ))
    return tuple(products)
