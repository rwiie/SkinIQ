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

IMAGE_SIZE = 384  # EfficientNet-V2-S default input resolution


def openweather_api_key() -> str | None:
    """Return the OpenWeatherMap key from the environment, or None if unset.

    The Streamlit app passes st.secrets["OPENWEATHER_API_KEY"] explicitly instead.
    """
    return os.environ.get("OPENWEATHER_API_KEY")
