# Product ranking: TF-IDF + cosine similarity (proposal 3.6.4)

Code: `skiniq/recommender.py`. Tests: `tests/test_recommender.py`. Evaluation: `training/evaluate_ranking.py`.

## Method

The USCE describes the **user** (36 numeric features) while TF-IDF vectors describe **ingredients**, so the two
cannot be compared directly. Both are expressed in one vocabulary taken from the ingredient-function mapping table
(`docs/ingredients.md`):

| Side | Document | Example tokens |
|---|---|---|
| Product | For each active mapped ingredient: the ingredient, its functions, the concerns it addresses. Formulation-only ingredients (water, thickeners, preservatives, pH adjusters, emulsifiers, chelators) are left out. Each token is weighted by list position, `1 / (1 + position / 10)`, since earlier ingredients are usually at higher concentration. | `ing:Niacinamide`, `fn:brightening`, `concern:hyperpigmentation` |
| User (query) | Built from the USCE: each concern (scaled by its score) with the functions that address it; skin type needs; goals; sensitivity; weather (high UV, dry/cold air, humid/hot). | `concern:acne` ×0.9, `fn:acne_control`, `fn:uv_filter` |

1. `TfidfVectorizer` (scikit-learn; sublinear tf, smoothed idf) is fitted on all 1,676 product documents
   (83-term vocabulary).
2. The query is weighted with the same idf and L2-normalised; **cosine similarity** scores every product.
3. Products excluded by a knowledge-base rule (3.6.3) are removed.
4. `score = similarity + 0.5 × rule score`, ranked within each routine step (cleanser, toner, treatment,
   moisturizer, sunscreen, exfoliator, mask).
5. The shared tokens contributing most to each product's similarity are kept for the explanations (3.6.5).

**Design choices:** position decay (10), rule weight (0.5) and query term weights are heuristic.

## Sanity evaluation

There are no ground-truth "correct products" for a user. As an independent check, the dataset's `afterUse` tags
(from an external ingredient checker, **not used by the TF-IDF similarity**) serve as weak relevance labels: for
single-concern users, precision@k is the share of the top-k treatments and moisturizers carrying the matching tag;
the base rate is the share among all of them (a random pick). Only the similarity is evaluated, since the rules also
read these tags.

| Profile | Relevant tag(s) | Base rate | P@5 | P@10 | P@20 | Lift@10 |
|---|---|---:|---:|---:|---:|---:|
| acne | Acne Fighting | 43% | 100% | 100% | 95% | 2.35× |
| hyperpigmentation | Brightening, Dark Spots | 70% | 100% | 100% | 100% | 1.42× |
| redness | Redness Reducing | 66% | 80% | 80% | 85% | 1.22× |
| wrinkles | Anti-Aging | 54% | 40% | 60% | 75% | 1.11× |
| enlarged pores | Reduces Large Pores | 18% | 100% | 70% | 65% | 3.98× |
| blackheads | Reduces Large Pores | 18% | 100% | 100% | 85% | 5.68× |
| dry skin | Hydrating | 15% | 20% | 10% | 10% | 0.67× |
| oily skin | Good For Oily Skin | 55% | 100% | 100% | 100% | 1.82× |
| mean |  | 42% | 80% | 78% | 77% | 2.28× |

On average the top 10 products are relevant **78%** of the time against a
**42%** base rate.

**Limitation (dry skin, wrinkles):** the "Hydrating" tag follows a few specific ingredients (e.g. sorbitol,
*Chondrus crispus* extract, ferments): only 137 of 902 products with hyaluronic acid carry it. The ranking picks
barrier creams with ceramides, humectants and emollients for dry skin, which fit the need but often lack the tag,
so precision for dry skin understates the ranking. The same applies, to a lesser degree, to "Anti-Aging". A user
study or expert rating of the top products would be a stronger evaluation (future work).
