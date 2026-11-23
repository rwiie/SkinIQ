"""Cached resources and secrets for the Streamlit app (UI-independent logic stays in skiniq/)."""

from __future__ import annotations

import os
from dataclasses import replace

import streamlit as st
from PIL import Image

from skiniq.environment import EnvironmentData, coordinates_label, fetch_weather_at
from skiniq.pipeline import load_classifier
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


@st.cache_data(max_entries=50, show_spinner="Checking your photo…")
def photo_quality(photo: bytes) -> QualityReport:
    import io

    return check_photo(Image.open(io.BytesIO(photo)))
