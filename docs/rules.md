# Knowledge-base rules (proposal 3.6.3)

Generated from `skiniq/rules/` (`python -c "from skiniq.rules import ALL_RULES"`). Every rule has a test in `tests/test_rules.py`.

**Actions:** *exclude* removes the product; *caution* keeps it with a warning and a score penalty; *prefer* keeps it with a reason and a score bonus. Penalties and bonuses add up to each product's rule score, which the ranking (3.6.4) combines with TF-IDF similarity.

Hard exclusions are used only for safety (pregnancy, ingredients the user avoids, medical skin conditions, harsh combinations for sensitive skin). The product dataset's `afterUse` tags (e.g. "Acne Trigger", "Irritating") come from an ingredient checker and fire on single ingredients (37% of products are tagged "Acne Trigger"), so they are used only as cautions.

## Safety & Medical Constraint KB

| Rule | Condition → action | Action | Weight |
|---|---|---|---:|
| `S1_pregnancy_retinoid` | Pregnant or breastfeeding -> exclude retinoids | exclude | – |
| `S1_pregnancy_hydroquinone` | Pregnant or breastfeeding -> exclude hydroquinone | exclude | – |
| `S1_pregnancy_bha` | Pregnant or breastfeeding -> caution for salicylic acid / BHA | caution | -0.15 |
| `S2_avoid_fragrance` | User avoids fragrance -> exclude products containing it | exclude | – |
| `S2_avoid_alcohol` | User avoids drying alcohol -> exclude products containing it | exclude | – |
| `S2_avoid_essential_oils` | User avoids essential oils -> exclude products containing it | exclude | – |
| `S2_avoid_retinol` | User avoids retinoids -> exclude products containing it | exclude | – |
| `S2_avoid_salicylic_acid` | User avoids salicylic acid / BHA -> exclude products containing it | exclude | – |
| `S2_avoid_chemical_sunscreens` | User avoids chemical UV filters -> exclude products containing it | exclude | – |
| `S3_medical_fragrance` | Rosacea, eczema, psoriasis or dermatitis -> exclude fragrance and essential oils | exclude | – |
| `S3_medical_aggravates` | Rosacea or eczema-type condition -> caution for products flagged as aggravating it | caution | -0.25 |
| `S4_sensitive_irritating` | Sensitive skin -> caution for products flagged as irritating | caution | -0.20 |
| `S4_sensitive_fragrance` | Sensitive skin -> caution for fragrance, essential oils and drying alcohol | caution | -0.15 |
| `S4_sensitive_soothing` | Sensitive skin -> prefer soothing, barrier-repairing products | prefer | +0.10 |

## Dermatological KB

| Rule | Condition → action | Action | Weight |
|---|---|---|---:|
| `D1_acne_trigger` | Acne concern -> caution for products flagged as acne triggers | caution | -0.25 |
| `D1_acne_fighting` | Acne concern or acne-control goal -> prefer acne-fighting products | prefer | +0.25 |
| `D2_pores` | Blackheads or enlarged pores -> prefer pore-clearing products | prefer | +0.20 |
| `D3_oily_worsen` | Oily or combination skin -> caution for products that may worsen oiliness | caution | -0.20 |
| `D3_oily_good` | Oily or combination skin, or oil-control goal -> prefer oil-balancing products | prefer | +0.20 |
| `D3_dry_drying` | Dry skin -> caution for drying products | caution | -0.25 |
| `D3_dry_hydrating` | Dry skin or hydration goal -> prefer hydrating, barrier-supporting products | prefer | +0.20 |
| `D4_pigmentation` | Hyperpigmentation or brightening/pigmentation goal -> prefer brightening products | prefer | +0.20 |
| `D5_redness` | Redness or sensitivity-reduction goal -> prefer redness-reducing products | prefer | +0.20 |
| `D6_wrinkles` | Wrinkles or anti-aging goal -> prefer anti-aging products | prefer | +0.20 |

## Ingredient & Chemical Compatibility KB

| Rule | Condition → action | Action | Weight |
|---|---|---|---:|
| `I1_retinoid_acid` | Retinoid with AHA/BHA in one product -> exclude for sensitive skin, else caution | caution | -0.20 |
| `I1_retinoid_acid_sensitive` | Retinoid with AHA/BHA, sensitive skin -> exclude | exclude | – |
| `I2_multi_acid_sensitive` | Two or more acid types (AHA/BHA/PHA), sensitive skin -> caution | caution | -0.20 |
| `I3_vitc_acid` | Pure vitamin C with AHA/BHA in one product -> caution | caution | -0.10 |

## Environmental Adaptation KB

| Rule | Condition → action | Action | Weight |
|---|---|---|---:|
| `E1_uv_sunscreen` | High UV -> prefer broad-protection sunscreens | prefer | +0.30 |
| `E1_uv_photosensitising` | High UV -> caution for retinoids and AHAs (increase sun sensitivity) | caution | -0.10 |
| `E2_humid_heavy` | Hot or humid weather -> caution for heavy products | caution | -0.10 |
| `E2_humid_light` | Hot or humid weather -> prefer lightweight, oil-balancing products | prefer | +0.10 |
| `E3_dry_air` | Dry or cold air -> prefer hydrating, barrier-repairing products | prefer | +0.15 |

## Routine-level compatibility checks

`check_routine()` flags pairs of products in one routine: retinoid + AHA/BHA (alternate nights), retinoid + pure vitamin C (vitamin C morning, retinoid night), and two exfoliating products.

## Ingredient detection

Ingredients are matched one by one (`skiniq/products.py`): citric acid (a pH adjuster) is not an AHA, plant carrier oils are not essential oils, fatty alcohols (e.g. cetearyl alcohol) are not drying alcohol, and bakuchiol is not a retinoid.
