# Multi-label annotation: method and agreement

## How the labels were made

| Photos | Annotator | Method |
|---|---|---|
| 22 (test sample) | Human (project author) | Colab labelling tool, `docs/annotation_guidelines.md` |
| 478 (rest of the 300 test + 200 validation sample) | AI vision assistant (Claude) | Viewed each photo on numbered contact sheets (`training/label_sheets.py`), six concerns per photo, same written guidelines. Close-ups and Roboflow photos started from their known concern(s); other concerns were added or removed by inspection |

| 405 (training, full-face selfies from the skin type dataset) | AI vision assistant (Claude) | Same method, nine photos per sheet. 242 have no concerns (clear skin); 110 are flagged unsure and 30 unusable |

All 905 photos are in `data/labels/multilabel_labels.csv` (`annotator` column: `raneem` or `claude`).
In test and validation, 106 photos are flagged `unsure` (blurry, small, filtered or heavily made-up) and are left out of the test metrics by default. Unsure training photos are left out of training.

## Agreement check (pilot, before the AI labels were made)

The AI labelled the 21 usable human-labelled photos **without seeing the human labels**.

| Concern | Human | AI | Agreement | Cohen's kappa |
|---|---:|---:|---:|---:|
| Acne | 6 | 5 | 95% | 0.88 |
| Blackheads | 4 | 3 | 95% | 0.83 |
| Enlarged pores | 4 | 6 | 90% | 0.74 |
| Redness | 6 | 3 | 86% | 0.59 |
| Wrinkles | 8 | 3 | 76% | 0.43 |
| Hyperpigmentation | 4 | 3 | 76% | 0.15 |

Overall label agreement 87%. After the pilot the wrinkle rule was clarified to include fine lines
(the main source of disagreement) before the 478 photos were labelled.

## Limitations

- Test and validation labels are mostly AI-produced, not dermatologist-confirmed.
- The agreement sample is small (21 photos); hyperpigmentation agreement was poor, so its
  results should be treated with caution.
- A second human annotator on the same test photos would strengthen the reliability estimate.
