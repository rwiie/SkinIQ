"""Shared constants and settings. Import labels from here; don't redefine them."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

SKIN_TYPES: tuple[str, ...] = ("normal", "oily", "dry", "combination")

SKIN_CONDITIONS: tuple[str, ...] = (
    "acne",
    "blackheads",
    "hyperpigmentation",
    "redness",
    "enlarged_pores",
    "wrinkles",
)

DATA_SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

IMAGE_SIZE = 300  # EfficientNet-V2-S training resolution (fits a free Colab T4)


def openweather_api_key() -> str | None:
    """Return the OpenWeatherMap key from the environment, or None if unset.

    The Streamlit app passes st.secrets["OPENWEATHER_API_KEY"] explicitly instead.
    """
    return os.environ.get("OPENWEATHER_API_KEY")

# Weight of the image model's skin type prediction when fused with the questionnaire
# (step 4). 0.0 = skin type comes from the questionnaire only.
# Evidence: both skin type models stayed near chance on the leakage-free test split
# (v1 EfficientNet-V2-S 31.6%, v2 EfficientNet-V2-B0 with anti-overfitting settings 29.8%;
# chance is 25%). See training/results/ and training/README.md.
SKIN_TYPE_IMAGE_WEIGHT = 0.0
