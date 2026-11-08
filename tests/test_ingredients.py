"""Ingredient standardization (proposal Table 3.5.2.1) and the ingredient-function
mapping table (Table 3.5.2.2)."""

import re

import pandas as pd
import pytest

from skiniq.config import SKIN_CONDITIONS
from skiniq.products import (
    ALIASES_CSV,
    FUNCTIONS_CSV,
    ingredient_functions,
    ingredient_groups,
    ingredient_info,
    load_ingredient_table,
    load_products,
    split_ingredients,
    standardize,
)
from skiniq.rules import ALL_RULES

EXTRA_CONCERNS = {"dryness", "oiliness", "sensitivity", "sun_damage"}


# ---- standardization ------------------------------------------------------------------

@pytest.mark.parametrize("raw, standard", [
    # proposal Table 3.5.2.1
    ("Vitamin C", "Ascorbic Acid"), ("L-Ascorbic Acid", "Ascorbic Acid"), ("Ascorbic Acid", "Ascorbic Acid"),
    ("Sodium Hyaluronate", "Sodium Hyaluronate"), ("Salicylic Acid", "Salicylic Acid"),
    # concentrations, INCI EU names, common names
    ("Salicylic Acid 2%", "Salicylic Acid"), ("Niacinamide 5%", "Niacinamide"), ("Retinol 0.1%", "Retinol"),
    ("Aqua", "Water"), ("Water/Aqua", "Water"), ("Water (Aqua)", "Water"), ("Fragrance", "Parfum"),
    ("Vitamin B3", "Niacinamide"), ("Cica", "Centella Asiatica Extract"),
    ("  Glycerin ", "Glycerin"), ("Glycerin*", "Glycerin"),
])
def test_standardize(raw, standard):
    assert standardize(raw) == standard


def test_bha_on_a_label_is_butylated_hydroxyanisole_not_salicylic_acid():
    assert standardize("BHA") == "Butylated Hydroxyanisole"
    assert "bha" not in ingredient_groups(("BHA",))
    assert "bha" in ingredient_groups(("Beta Hydroxy Acid",))


def test_split_standardizes_and_removes_duplicates():
    assert split_ingredients("Aqua,Water,Niacinamide 5%,1,2-Hexanediol") == ("Water", "Niacinamide", "1,2-Hexanediol")


def test_unknown_names_are_kept_as_is():
    assert standardize("Portulaca Oleracea Extract") == "Portulaca Oleracea Extract"


# ---- mapping table --------------------------------------------------------------------

def test_tables_are_well_formed():
    aliases = pd.read_csv(ALIASES_CSV)
    assert aliases.variant.str.lower().is_unique and aliases.standard_name.notna().all()
    table = pd.read_csv(FUNCTIONS_CSV).fillna("")
    assert table.ingredient.is_unique
    assert (table.functions != "").all(), "every row needs at least one function"
    assert (table.source != "").all(), "every row needs a source"
    for pattern in table.pattern:
        re.compile(pattern)


def test_concerns_use_known_names():
    allowed = set(SKIN_CONDITIONS) | EXTRA_CONCERNS
    for row in load_ingredient_table():
        assert set(row.concerns) <= allowed, (row.ingredient, row.concerns)


def test_every_group_used_by_rules_is_in_the_table():
    groups = {r.group for r in load_ingredient_table() if r.group}
    used = {"retinoid", "aha", "bha", "pha", "pure_vitamin_c", "vitamin_c", "niacinamide", "hydroquinone",
            "fragrance", "drying_alcohol", "essential_oil", "chemical_filter", "mineral_filter",
            "hyaluronic_acid", "ceramide", "centella"}
    assert used <= groups
    import inspect
    rule_source = "".join(inspect.getsource(__import__(f"skiniq.rules.{m}", fromlist=["x"]))
                          for m in ("dermatological", "safety_medical", "ingredient_compatibility", "environmental"))
    for g in re.findall(r'has\("(\w+)"\)', rule_source):
        assert g in groups, f"rules use group {g!r} that the table doesn't define"


@pytest.mark.parametrize("ingredient, functions, concerns", [
    # proposal Table 3.5.2.2
    ("Salicylic Acid", {"exfoliant", "acne_control"}, {"acne"}),
    ("Sodium Hyaluronate", {"hydration", "humectant"}, {"dryness"}),
    ("Niacinamide", {"brightening", "sebum_regulation"}, {"hyperpigmentation"}),
    # formulation ingredients
    ("Citric Acid", {"ph_adjuster"}, set()),
    ("Water", {"solvent"}, set()),
])
def test_mapping_examples(ingredient, functions, concerns):
    rows = ingredient_info(ingredient)
    assert rows
    assert functions <= {f for r in rows for f in r.functions}
    assert concerns <= {c for r in rows for c in r.concerns}


def test_citric_acid_is_not_an_exfoliant():
    assert not any(r.group for r in ingredient_info("Citric Acid"))


def test_product_functions_and_concerns():
    functions, concerns = ingredient_functions(("Water", "Salicylic Acid", "Glycerin"))
    assert {"exfoliant", "humectant", "solvent"} <= functions
    assert {"acne", "dryness"} <= concerns


def test_real_catalogue_coverage():
    products = load_products()
    occurrences = [i for p in products for i in p.ingredients]
    mapped = sum(1 for i in occurrences if ingredient_info(i))
    assert mapped / len(occurrences) > 0.5
    assert sum(bool(p.functions) for p in products) / len(products) > 0.95
    assert not any("%" in i for p in products for i in p.ingredients)
