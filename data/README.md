# Data

Datasets are public Kaggle datasets (proposal 3.5) and are **not committed** (too large).
They are shared with the project via Google Drive.

| Folder | Contents | Source |
|---|---|---|
| `raw/skin_type/` | 4,093 facial images in `train/valid/test/<type>/` + two 0–5 severity label sheets (`*.xlsx`, 200 images) | Kaggle, olimov "Skin Type Dataset" (Roboflow export) |
| `raw/skin_condition/` | 8,843 images in `acne/ blackheades/ pigmentation/ pores/ redness/ wrinkles/` | Kaggle, drishya "Skin Issues Dataset" (Roboflow export) |
| `raw/products.csv` | 1,682 Korean skincare products: brand, product, category, ingredients, afterUse tags | Product Ingredients List |
| `processed/<task>/{train,val,test}/<label>/` | Cleaned, leakage-free 70/15/15 split (hard links to `raw/`) | `training/prepare_data.py` |
| `processed/<task>/manifest.csv` | One row per kept image: label, split, photo group, size, low-res flag | `training/prepare_data.py` |
| `processed/<task>/removed.csv` | Every dropped image and why | `training/prepare_data.py` |

## Rebuild the processed data

1. Unzip the datasets into `raw/`, renaming the top folders to `skin_type/` and `skin_condition/`
   (skip the `__MACOSX/` folders).
2. From the repo root: `python training/prepare_data.py`

Class counts per split are in [`training/data_report.md`](../training/data_report.md).

## Known dataset limitations (for the report)

- Both datasets are Roboflow exports with several augmented copies per original photo, and the
  skin type dataset's own split leaked copies of 801 photos across train/valid/test. The data is
  re-split by original photo so every copy of a photo is in one split.
- Class imbalance: combination is 8.4% of skin type images; hyperpigmentation and redness are
  about 7% each of skin condition images. Use class weights when training.
- 1,339 skin condition images (mostly acne and blackheads) are under 256 px.
- Skin condition images are mostly close-ups, while the app takes full-face photos.
- Skin type labels are noisy in places (e.g. "normal" faces with visible redness).
