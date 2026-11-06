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

Skin type is assessed from the questionnaire (`skiniq/questionnaire.py`), with the image model
at most a supporting signal; see the comparison results before deciding its weight.

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

## Outputs (`models/<task>/`, not committed)

| File | Contents |
|---|---|
| `model.pt` | best checkpoint, class labels, model name, image size |
| `metrics.json` | test accuracy, macro/weighted precision, recall, F1, per-class scores |
| `history.csv` | train/val loss, val accuracy and macro-F1 per epoch |
| `confusion_matrix.csv` / `.png` | test confusion matrix (counts and row %) |
| `training_curves.png` | loss and validation curves (written by the notebook) |

The app loads a checkpoint with `skiniq.vision.model.SkinClassifier.load(path)`.
