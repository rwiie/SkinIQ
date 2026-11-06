# Training

Data preparation (proposal 3.5) and EfficientNet-V2 training (3.6.1).

- Split: 70% train / 15% validation / 15% test, by original photo (no leakage) — run `python training/prepare_data.py`. Counts: `data_report.md`.
- Augmentation (train only): light rotations, noise, blur. The source images already contain Roboflow rotations and mosaics, so keep it mild.
- Class imbalance: use class weights (combination, hyperpigmentation and redness are under 9% each).
- Model: EfficientNet-V2 via `timm`, fine-tuned from ImageNet weights.
- Output: weights saved to `models/` (not committed) plus a metrics report.

Training runs on Google Colab or a local GPU; scripts and notebooks go in this folder.
