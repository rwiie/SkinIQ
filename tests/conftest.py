"""Shared fixtures: questionnaire answers, CNN outputs and weather, reused by the
fusion (step 4), rule (step 5) and ranking (step 6) tests."""

import pytest

from skiniq.config import SKIN_CONDITIONS
from skiniq.environment import EnvironmentData
from skiniq.fusion import ImageAnalysis

BASE_ANSWERS = {
    "age_group": "18_24",
    "pregnant": "no",
    "known_conditions": ["none"],
    "sensitivity": "not",
    "reacts_to_products": "no",
    "tightness": "rarely",
    "oiliness": "balanced",
    "oil_location": "nowhere",
    "flakiness": "no",
    "breakouts": "rarely",
    "goals": ["hydration"],
    "avoid_ingredients": ["none"],
}


@pytest.fixture
def answers():
    """Factory: baseline normal-skin, non-sensitive user with optional overrides."""

    def make(**overrides):
        return {**BASE_ANSWERS, **overrides}

    return make


@pytest.fixture
def image():
    """Factory: CNN condition probabilities. Unlisted conditions share the remainder."""

    def make(**probs):
        rest = [c for c in SKIN_CONDITIONS if c not in probs]
        remainder = max(0.0, 1 - sum(probs.values()))
        full = {**probs, **{c: remainder / len(rest) for c in rest}} if rest else probs
        return ImageAnalysis(conditions=full)

    return make


@pytest.fixture
def kl_afternoon():
    """Typical Kuala Lumpur afternoon: hot, humid, very high UV."""
    return EnvironmentData(temperature_c=32, humidity_pct=78, uv_index=10, location="Kuala Lumpur")
