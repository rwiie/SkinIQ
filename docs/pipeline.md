# Pipeline and explanations (proposal 3.6.5)

Code: `skiniq/pipeline.py` (`analyse()`), `skiniq/explain.py`. Tests: `tests/test_pipeline.py`.
Full example output: [`example_result.json`](example_result.json).

```python
from skiniq.pipeline import analyse
from skiniq.environment import EnvironmentData

result = analyse(answers, image=pil_image, environment=EnvironmentData(32, 78, 10, "Shah Alam"))
```

## Steps

| # | Step | Module | Proposal |
|---|---|---|---|
| 1 | Skin concerns from the photo: multi-label EfficientNet-V2, whole image, per-concern thresholds | `vision/model.py` | 3.6.1 |
| 2 | Fuse photo, questionnaire and weather into the USCE | `fusion.py` | 3.6.2 |
| 3 | Knowledge-base rules: exclude / caution / prefer each of 1,676 products | `rules/` | 3.6.3 |
| 4 | TF-IDF + cosine similarity ranking per routine step | `recommender.py` | 3.6.4 |
| 5 | Morning and evening routines, compatibility checks, weekly extras | `pipeline.py` | 3.6.5 |
| 6 | Explanations in four sections; JSON output | `explain.py` | 3.6.5, 3.7 |

The photo and the weather are optional: without them the questionnaire drives the result and a note says so.
If the model file is missing, the photo is skipped with a note.

## Routine assembly

- **Morning:** cleanser, toner, treatment, moisturizer, sunscreen. Retinoids and AHAs are not allowed in the morning
  because they increase sun sensitivity (FDA, 2005).
- **Evening:** cleanser, toner, treatment, moisturizer.
- **Weekly:** one exfoliator (1–2× a week; once for sensitive skin) if it doesn't conflict with the evening routine,
  and one mask.
- For each step the best-ranked product is taken unless it conflicts with products already chosen
  (`check_routine`: retinoid + AHA/BHA, hydroquinone + retinoid, two exfoliants), in which case the next best is
  used. Conflicts that cannot be avoided are reported with advice (e.g. "use them on alternate nights").
- Two alternatives are kept for each step.

## Explanations (XAI): the four result sections

| Section | Content | Built from |
|---|---|---|
| 1. Skin analysis | Skin type with the answers behind it; each concern with its evidence ("photo analysis 92% and your questionnaire answers"); weather summary | USCE: questionnaire scoring, concern sources, environment |
| 2. Personalised advice | Short tips: sunscreen and UV, weather, sensitivity (patch test), pregnancy, medical conditions, concern- and skin-type-specific tips | USCE flags |
| 3. Routine | Morning, evening and weekly products, alternatives, routine conflicts with advice | Ranking + routine checks |
| 4. Why each product | One-line summary; key ingredients that serve *this* user's needs, most relevant first ("Niacinamide: controls oil, calms inflammation — helps with acne, oiliness"); rule reasons ("Suited to oily and combination skin"); cautions | Ingredient-function table, TF-IDF query terms, fired rules |

Explanations are faithful: they are generated from the same data and rules that produced each decision, not
written separately. The system goes by the real ingredient list, not product names: a cleanser named "AHA, BHA,
PHA" whose only AHA is citric acid (a pH adjuster) is correctly treated as BHA + PHA and allowed in the morning.

Every result ends with a disclaimer that SkinIQ gives general skincare guidance, not a medical diagnosis.
