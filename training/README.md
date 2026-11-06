# Training

Data preparation (proposal 3.5) and EfficientNet-V2 training (3.6.1).

- Split: 70% train / 15% validation / 15% test, stratified by class.
- Augmentation (train only): small rotations, noise, mosaic, blur.
- Model: EfficientNet-V2 via `timm`, fine-tuned from ImageNet weights.
- Output: weights saved to `models/` (not committed) plus a metrics report.

Training runs on Google Colab or a local GPU; scripts and notebooks go in this folder.
