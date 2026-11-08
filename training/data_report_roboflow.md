# Roboflow dataset preparation report

## Skin problem detection (Roboflow, multi-label)

- Images scanned: 3240
- Images kept: 2215 from 2015 original-photo groups
- Removed (no SkinIQ concern boxed (only dry/oily skin, eye bags or none)): 1020
- Removed (exact duplicate): 5

| Concern | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| acne | 1259 | 293 | 288 | 1840 |
| blackheads | 220 | 51 | 51 | 322 |
| hyperpigmentation | 45 | 10 | 10 | 65 |
| redness | 45 | 10 | 9 | 64 |
| enlarged_pores | 233 | 61 | 57 | 351 |
| wrinkles | 215 | 46 | 46 | 307 |

Images: train 1550, val 333, test 332; 617 images have two or more concerns.
