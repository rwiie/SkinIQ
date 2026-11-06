# Training

Data preparation (proposal 3.5) and EfficientNet-V2 training (3.6.1), evaluated on the
held-out test split (3.8).

## Run on Google Colab (recommended)

Open [`train_colab.ipynb`](https://colab.research.google.com/github/rwiie/SkinIQ/blob/main/training/train_colab.ipynb)
in Colab, set **Runtime → T4 GPU**, then **Run all**. It mounts your Drive, unzips the two
datasets, runs `prepare_data.py`, trains both models and saves results to
`My Drive/SkinIQ/models/<task>/`.

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

## Outputs (`models/<task>/`, not committed)

| File | Contents |
|---|---|
| `model.pt` | best checkpoint, class labels, model name, image size |
| `metrics.json` | test accuracy, macro/weighted precision, recall, F1, per-class scores |
| `history.csv` | train/val loss, val accuracy and macro-F1 per epoch |
| `confusion_matrix.csv` / `.png` | test confusion matrix (counts and row %) |
| `training_curves.png` | loss and validation curves (written by the notebook) |

The app loads a checkpoint with `skiniq.vision.model.SkinClassifier.load(path)`.
