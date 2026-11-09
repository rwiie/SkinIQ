"""Cached resources and secrets for the Streamlit app (UI-independent logic stays in skiniq/)."""

from __future__ import annotations

import os

import streamlit as st

from skiniq.environment import EnvironmentData, fetch_weather
from skiniq.pipeline import SKIN_CONCERN_MODEL, load_classifier
from skiniq.recommender import default_index


@st.cache_resource(show_spinner="Loading the skin analysis model…")
def get_classifier():
    return load_classifier(SKIN_CONCERN_MODEL)


@st.cache_resource(show_spinner="Loading the product catalogue…")
def get_product_index():
    return default_index()


def openweather_key() -> str | None:
    try:
        key = st.secrets.get("OPENWEATHER_API_KEY")
    except FileNotFoundError:
        key = None
    return key or os.environ.get("OPENWEATHER_API_KEY") or None


@st.cache_data(ttl="30m", max_entries=200, show_spinner="Fetching today's weather…")
def get_weather(place: str) -> EnvironmentData:
    return fetch_weather(place, openweather_key())
