"""Cached resources and secrets for the Streamlit app (UI-independent logic stays in skiniq/)."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from dataclasses import replace

import requests
import streamlit as st
from PIL import Image

from skiniq.environment import EnvironmentData, coordinates_label, fetch_weather_at
from skiniq.pipeline import load_classifier
from skiniq.product_images import tidy_photo
from skiniq.recommender import default_index
from skiniq.vision.quality import QualityReport, check_photo


class _NoModel(Exception):
    pass


@st.cache_resource(show_spinner="Loading the skin analysis model (downloaded on first use, about 80 MB)…")
def _cached_classifier():
    clf = load_classifier()
    if clf is None:
        raise _NoModel()  # exceptions are not cached, so a failed download is retried next time
    return clf


def get_classifier():
    try:
        return _cached_classifier()
    except _NoModel:
        return None


@st.cache_resource(show_spinner="Loading the product catalogue…")
def get_product_index():
    return default_index()


def openweather_key() -> str | None:
    try:
        key = st.secrets.get("OPENWEATHER_API_KEY")
    except FileNotFoundError:
        key = None
    return key or os.environ.get("OPENWEATHER_API_KEY") or None


@st.cache_data(ttl="30m", max_entries=500, show_spinner="Fetching today's weather for your location…")
def _weather(lat: float, lon: float, place: str | None) -> EnvironmentData:
    return fetch_weather_at(lat, lon, place, openweather_key())


def get_weather_at(lat: float, lon: float, place: str | None = None) -> EnvironmentData:
    # Cached on a ~100 m grid (same weather), but the exact detected position is reported
    w = _weather(round(lat, 3), round(lon, 3), place)
    location = w.location if place or w.location != coordinates_label(round(lat, 3), round(lon, 3)) \
        else coordinates_label(lat, lon)
    return replace(w, latitude=round(lat, 5), longitude=round(lon, 5), location=location)


@lru_cache(maxsize=512)
def product_photo(url: str) -> bytes | None:
    """One product photo, background removed and centred on a square (PNG); None if unavailable."""
    try:
        r = requests.get(url, timeout=8)
        if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image/"):
            return None
        return tidy_photo(r.content)
    except (requests.RequestException, ValueError):
        return None


def routine_photos(result: dict) -> dict[str, bytes]:
    """Tidied photos of every product in the routine, keyed by product title (fetched in parallel)."""
    steps = [s for part in ("morning", "evening", "weekly") for s in result["routine"][part]]
    urls = {s["product"]["product"]: s["product"]["image_url"] for s in steps if s["product"].get("image_url")}
    with ThreadPoolExecutor(max_workers=6) as pool:
        photos = dict(zip(urls, pool.map(product_photo, urls.values())))
    return {title: data for title, data in photos.items() if data}


@st.cache_data(max_entries=50, show_spinner="Checking your photo…")
def photo_quality(photo: bytes) -> QualityReport:
    import io

    return check_photo(Image.open(io.BytesIO(photo)))
