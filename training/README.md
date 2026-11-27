# Training

Data preparation (proposal 3.5) and EfficientNet-V2 training (3.6.1), evaluated on the
held-out test split (3.8).

## Run on Google Colab (recommended)

Open [`train_colab.ipynb`](https://colab.research.google.com/github/rwiie/SkinIQ/blob/main/training/train_colab.ipynb)
in Colab, set **Runtime → T4 GPU**, then **Run all**. It mounts your Drive, unzips the two
datasets, runs `prepare_data.py`, trains both models and saves results to
`My Drive/SkinIQ/models/<task>/`.

## Skin type v2 experiment

The first skin type model reached only 31.6% test accuracy (chance is 25%): it memorised the
training images by epoch 6 while validation accuracy stayed at 30–35%. The skin condition model,
trained with the same code, reached 93.1%. Open
[`train_skin_type_v2.ipynb`](https://colab.research.google.com/github/rwiie/SkinIQ/blob/main/training/train_skin_type_v2.ipynb)
to retrain skin type with anti-overfitting settings (EfficientNet-V2-B0, lower learning rate,
3 frozen-backbone epochs, strong augmentation, dropout and stochastic depth). It saves to
`models/skin_type_v2/` and writes a v1 vs v2 comparison table and chart.

Result: 29.8% test accuracy (v1: 31.6%). See **Results** below for the decision.

## Multi-label skin concern model (current approach)

The single-label condition model picks one condition per image and has no "clear skin"
option, so it can't report several concerns on one face and labels healthy skin as some
condition. The multi-label model fixes both: **EfficientNet-V2 with six independent sigmoid
outputs, trained with binary cross-entropy**, so a face can have any number of concerns,
including none. **YuNet** face detection splits a face into forehead, nose, cheeks and chin.

Data: the two Kaggle datasets plus the Roboflow *Skin Problem Detection* dataset (CC BY 4.0),
whose bounding boxes are converted to six image-level labels by `prepare_roboflow.py`
(report: `data_report_roboflow.md`). Images are resized to 224 × 224 and ImageNet-normalised.

| Step | Notebook / script | Output |
|---|---|---|
| 1. Label photos (all six concerns) | [`annotate_colab.ipynb`](https://colab.research.google.com/github/rwiie/SkinIQ/blob/main/training/annotate_colab.ipynb), rules in [`docs/annotation_guidelines.md`](../docs/annotation_guidelines.md) | `My Drive/SkinIQ/labels/multilabel_labels.csv` |
| 2. Train + evaluate per concern | [`train_multilabel_colab.ipynb`](https://colab.research.google.com/github/rwiie/SkinIQ/blob/main/training/train_multilabel_colab.ipynb) → `train_multilabel.py` | `models/skin_concerns/` |
| 3. Whole image vs face regions | `evaluate_regions.py` (run by the same notebook) | `region_evaluation.csv` |

- **Annotation:** one label set per original photo (6,079 photos; copies share labels), in the
  priority order of `data/labels/annotation_queue.csv` (8,094 photos): a stratified 300-photo
  test sample (100 each from full faces, close-ups and Roboflow), a 200-photo validation sample,
  then training photos. Minimum: the 500 test + validation photos (≈1 hour). Training labels
  are optional but help, especially clear full faces (the strongest "no concern" examples).
- **v3 batch:** 216 more full-face training selfies labelled by Claude (`data/labels/review_batch_v3.txt`),
  raising the fully-labelled training files from 680 (265 photos) to 1,067 (406 photos). v3 was
  trained with `train_multilabel_colab.ipynb` (saved to `models/skin_concerns_v3/`, compared with v2).
  Afterwards the author relabelled all 1,099 Claude-labelled photos in `annotate_colab.ipynb` section 2b.
- **v2 with test-time augmentation (what the app runs).** The v2 weights are unchanged; the app averages
  each photo with its mirror image and uses thresholds re-tuned on the validation photos
  (`skiniq/vision/weights.py`, `TTA_THRESHOLDS`). On the same 237 test photos: macro F1 0.661 → **0.688**,
  micro F1 0.677 → 0.707, exact match 35.9% → 40.5%, clear-skin false alarms 17.2% → 10.3%
  (`results/skin_concerns_v2_tta/`). Adding 90% crops as well gave no further gain (0.685).
- **v3 result and decision: the app keeps v2.** On the same 237 test photos v3 tied v2 on macro F1
  (0.661) and mean ROC-AUC (0.850 vs 0.851); it cut clear-skin false alarms (17.2% → 6.9%) and improved
  acne (F1 0.77 → 0.82), but lowered hyperpigmentation (0.58 → 0.54) and enlarged pores (0.70 → 0.64).
  With no gain in overall detection, v2 stays in the app; v3 is reported as an experiment
  (`results/skin_concerns_v3/`).
- **v2 vs v4 on the same answer key.** v2 was trained and first scored on Claude's labels; v4 on the
  author's relabelled labels. `evaluate_multilabel.py` scored v2 against the author's labels (same 248
  test photos as v4, thresholds re-tuned on the author's validation photos):

  | | v2 | v4 |
  |---|---:|---:|
  | Macro F1 | 0.592 (0.593 with mirror averaging) | **0.627** |
  | Mean ROC-AUC | 0.786 | 0.790 |
  | Macro precision / recall | **0.685** / 0.573 | 0.602 / **0.689** |
  | Exact match | 16.9% | 16.1% |
  | Clear-skin false alarms (15 photos) | **33%** | 47% |

  Ranking ability is the same (ROC-AUC 0.79); v4's higher F1 comes almost all from hyperpigmentation
  (F1 0.43 → 0.67), which the author marks far more often (139 of 248 test photos). Results:
  `results/v2_on_author_labels/`.
- **Unlabelled close-ups and Roboflow images still train the model:** their folder concern or
  boxed concerns are known positives; their other concerns are weak negatives (loss weight 0.1).
  Unlabelled full faces are skipped.
- **Thresholds:** one per concern, chosen on validation photos to maximise F1.
- **Metrics (test photos only, copies averaged per photo):** per concern precision, recall,
  F1, ROC-AUC and PR-AUC; overall macro/micro F1, exact match, Hamming loss, and the
  false-alarm rate on clear-skin photos.

## Training experiments (`experiments_colab.ipynb`)

Three one-factor experiments around the v2 settings, all on the author's relabelled labels:
data augmentation (mild / **strong**), learning rate (1e-4 / **2e-4** / 3e-4) and dropout
(0.2 / **0.3** / 0.4). The baseline (bold values) is the v4 run; 5 new runs. The best value of each
setting is chosen by validation mean ROC-AUC; test results are reported for every run
(`models/experiments/experiment_results.csv`).

## Run locally (GPU)

```bash
python training/prepare_data.py
python training/train.py --task skin_type
python training/train.py --task skin_condition
```

## Settings

| Setting | Value | Why |
|---|---|---|
| Model | `tf_efficientnetv2_s.in21k_ft_in1k` (20.2M params) | EfficientNet-V2-S, ImageNet-21k pretrained; fits a free T4 |
| Input size | 300 × 300 | EfficientNet-V2-S training resolution |
| Split | 70 / 15 / 15 by original photo | No near-copies across splits; counts in `data_report.md` |
| Augmentation (train only) | random crop 70–100%, flip, ±10° rotation, mild brightness/contrast, blur (20%), Gaussian noise (30%) | Mild, because the images already carry Roboflow rotations and mosaics; no hue change, since colour carries redness and pigmentation |
| Class imbalance | inverse-frequency class weights | combination, hyperpigmentation and redness are under 9% each |
| Optimiser | AdamW, lr 3e-4, weight decay 0.05, 1 warm-up epoch then cosine decay | |
| Loss | cross-entropy, label smoothing 0.1 | noisy labels |
| Epochs | up to 20, early stop after 5 without validation macro-F1 gain | |
| Model selection | best validation macro-F1 | accuracy alone hides weak minority classes |

## Results

Result files from each Colab run (everything except `model.pt`) are copied to
[`results/`](results/) for the report. Model weights stay in Drive (`My Drive/SkinIQ/models/`).

| Run | Model | Test accuracy | Macro F1 |
|---|---|---:|---:|
| `skin_condition` | EfficientNet-V2-S | 93.1% | 93.2% |
| `skin_type` v1 | EfficientNet-V2-S | 31.6% | 29.5% |
| `skin_type` v2 | EfficientNet-V2-B0, anti-overfitting settings | 29.8% | 28.3% |

**Decision: skin type comes from the questionnaire** (`skiniq/questionnaire.py`,
`SKIN_TYPE_IMAGE_WEIGHT = 0.0` in `skiniq/config.py`). v2 halved the gap between validation
and training loss (1.14 → 0.60), so overfitting was reduced, but validation macro-F1 still
plateaued at about 0.33, the same as v1 (see `results/skin_type_v2/comparison_v1_v2.png`).
Two different set-ups reaching the same ceiling points to the data, not the training: with
these labels, skin type cannot be predicted from photos much better than chance (25%). The
CNN is used for the six skin conditions.

## Outputs (`models/<task>/`, not committed)

| File | Contents |
|---|---|
| `model.pt` | best checkpoint, class labels, model name, image size |
| `metrics.json` | test accuracy, macro/weighted precision, recall, F1, per-class scores |
| `history.csv` | train/val loss, val accuracy and macro-F1 per epoch |
| `confusion_matrix.csv` / `.png` | test confusion matrix (counts and row %) |
| `training_curves.png` | loss and validation curves (written by the notebook) |

The app loads a checkpoint with `skiniq.vision.model.SkinClassifier.load(path)`.
