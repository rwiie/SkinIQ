# Data

Datasets are public Kaggle datasets (proposal 3.5) and are **not committed** (too large).
They are shared with the project via Google Drive.

| Folder | Contents | Source |
|---|---|---|
| `raw/skin_type/` | 4,093 facial images in `train/valid/test/<type>/` + two 0–5 severity label sheets (`*.xlsx`, 200 images) | Kaggle, olimov "Skin Type Dataset" (Roboflow export) |
| `raw/skin_condition/` | 8,843 images in `acne/ blackheades/ pigmentation/ pores/ redness/ wrinkles/` | Kaggle, drishya "Skin Issues Dataset" (Roboflow export) |
| `raw/skin_problem/` | 3,240 images (640×640) in `train/ valid/ test/` with COCO bounding boxes for 10 classes (`_annotations.coco.json`) | Roboflow Universe, *Skin-Problem-Detection-Relabel-Clean3* v1 (parin-kittipongdaja), **CC BY 4.0** |
| `raw/products.csv` | 1,682 Korean skincare products: brand, product, category, ingredients, afterUse tags | Product Ingredients List |
| `knowledge/ingredient_aliases.csv` | Ingredient standardization table (proposal Table 3.5.2.1) | Proposal + common/INCI names; see `docs/ingredients.md` |
| `knowledge/ingredient_functions.csv` | Ingredient-function mapping table (proposal Table 3.5.2.2) | Proposal literature review; rows marked *verify* need a check |
| `processed/<task>/{train,val,test}/<label>/` | Cleaned, leakage-free 70/15/15 split (hard links to `raw/`) | `training/prepare_data.py` |
| `processed/<task>/manifest.csv` | One row per kept image: label, split, photo group, size, low-res flag | `training/prepare_data.py` |
| `processed/<task>/removed.csv` | Every dropped image and why | `training/prepare_data.py` |

## Rebuild the processed data

1. Unzip the datasets into `raw/`, renaming the top folders to `skin_type/` and `skin_condition/`
   (skip the `__MACOSX/` folders). Unzip `Skin Problem Detection.zip` into `raw/skin_problem/`.
2. From the repo root: `python training/prepare_data.py` and `python training/prepare_roboflow.py`
   (report: `training/data_report_roboflow.md`).

## Citation (Roboflow dataset, CC BY 4.0 requires attribution)

> parin-kittipongdaja. (2024). *Skin-Problem-Detection-Relabel-Clean3* (Version 1) [Dataset].
> Roboflow Universe. https://universe.roboflow.com/parin-kittipongdaja-vwmn3/skin-problem-detection-relabel-clean3
> Licensed under CC BY 4.0.

Class counts per split are in [`training/data_report.md`](../training/data_report.md).

## Known dataset limitations (for the report)

- Roboflow dataset: boxes were converted to six image-level labels (Acne and Whiteheads → acne,
  Dark-Spots → hyperpigmentation, Skin-Redness → redness, Englarged-Pores → enlarged pores;
  Dry-Skin, Oily-Skin and Eyebags ignored). It merges sub-datasets that each labelled their own
  concerns, so a missing box is not a reliable "no": missing concerns are weak negatives in
  training, and the 1,020 images with only dry/oily/eye-bag boxes were dropped (some are not
  faces). Its own split leaked duplicates and had no redness in test, so it was re-split.
  Redness (64 images) and hyperpigmentation (65) are rare.

- Both datasets are Roboflow exports with several augmented copies per original photo, and the
  skin type dataset's own split leaked copies of 801 photos across train/valid/test. The data is
  re-split by original photo so every copy of a photo is in one split.
- Class imbalance: combination is 8.4% of skin type images; hyperpigmentation and redness are
  about 7% each of skin condition images. Use class weights when training.
- 1,339 skin condition images (mostly acne and blackheads) are under 256 px.
- Skin condition images are mostly close-ups, while the app takes full-face photos.
- Skin type labels are noisy in places (e.g. "normal" faces with visible redness).
