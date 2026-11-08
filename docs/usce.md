# Unified Skin Context Embedding (USCE)

Proposal section 3.6.2. Built by `skiniq.fusion.build_usce(answers, image, environment)`.

## Inputs

| Input | Source | Used for |
|---|---|---|
| Visual | CNN probabilities for 6 skin conditions (`ImageAnalysis`) | condition concerns |
| Personal | Questionnaire answers, Appendix A (`skiniq/questionnaire.py`) | skin type, concerns, sensitivity, safety, goals, ingredient exclusions |
| Contextual | Temperature, humidity, UV index (`EnvironmentData`) | environmental adaptation |

The photo and the weather are optional: without a photo, concerns come from the
questionnaire; without weather, environmental features are zero and `env:known` is 0.

## Fusion rules

| Field | How it is derived |
|---|---|
| Skin type | Questionnaire Section C scoring. The photo's skin type is blended in with weight `SKIN_TYPE_IMAGE_WEIGHT` (0, because the image model performed near chance) |
| Condition concern score | max(photo probability, self-reported evidence) |
| Self-reported evidence | known acne / hyperpigmentation / melasma / rosacea → 0.8 for acne / hyperpigmentation / hyperpigmentation / redness; breakouts "frequently" → acne 0.7, "occasionally" → acne 0.4 |
| Concern status | detected ≥ 0.50, possible ≥ 0.20, otherwise none |
| Sensitivity (0–1) | max(sensitivity level / 3, product reactions: yes 0.67, sometimes 0.33) |
| Sensitive flag | moderately or highly sensitive, or reacts to products, or a known medical skin condition |
| Medical conditions | rosacea, eczema, psoriasis, dermatitis, seborrheic dermatitis (safety rules) |
| Environment flags | high UV ≥ 6 (WHO "high"), humid ≥ 70%, dry air < 40%, hot ≥ 30 °C, cold < 18 °C |

**Detection threshold.** The condition model has no "clear skin" class, so it always ranks
some condition first. On 240 held-out test images it was ≥ 50% confident for 224, and 97.8%
of those were correct; below 50% the top prediction was right only half the time. Only
confident predictions are therefore reported as detected.

## Embedding (36 features, all scaled 0–1)

| Features | Count |
|---|---:|
| `skin_type:normal / oily / dry / combination` (scores, sum to 1) | 4 |
| `concern:<condition>` (concern score) | 6 |
| `sensitivity`, `pregnant`, `age` (age group 0–1) | 3 |
| `goal:<goal>` (multi-hot) | 8 |
| `avoid:<ingredient>` (multi-hot) | 6 |
| `env:temperature` (/40 °C), `env:humidity` (/100), `env:uv_index` (/11) | 3 |
| `env:high_uv / humid / dry_air / hot / cold`, `env:known` | 6 |

`USCE.to_dict()` gives the JSON form, including the named vector, for the pipeline (3.6.5).
