"""Environmental context: temperature, humidity and UV index (proposal 3.4, 3.6.2).

`EnvironmentData` is what the fusion layer uses. The OpenWeatherMap client that
fills it from the user's location is added with the Streamlit prototype (step 8).
"""

from __future__ import annotations

from dataclasses import dataclass

from skiniq.config import COLD_C, DRY_AIR_PCT, HOT_C, HUMID_PCT, UV_HIGH, UV_MODERATE


@dataclass(frozen=True)
class EnvironmentData:
    temperature_c: float
    humidity_pct: float
    uv_index: float
    location: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.humidity_pct <= 100:
            raise ValueError(f"humidity must be 0-100%, got {self.humidity_pct}")
        if self.uv_index < 0:
            raise ValueError(f"UV index cannot be negative, got {self.uv_index}")

    @property
    def uv_level(self) -> str:
        """WHO UV Index category."""
        uv = self.uv_index
        if uv < UV_MODERATE:
            return "low"
        if uv < UV_HIGH:
            return "moderate"
        if uv < 8:
            return "high"
        if uv < 11:
            return "very_high"
        return "extreme"

    @property
    def flags(self) -> dict[str, bool]:
        return {
            "high_uv": self.uv_index >= UV_HIGH,
            "humid": self.humidity_pct >= HUMID_PCT,
            "dry_air": self.humidity_pct < DRY_AIR_PCT,
            "hot": self.temperature_c >= HOT_C,
            "cold": self.temperature_c < COLD_C,
        }
