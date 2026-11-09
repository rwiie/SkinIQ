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


# ---- live weather (proposal 3.7) ----------------------------------------------------------
#
# OpenWeatherMap (proposal) gives location, temperature and humidity on its free plan, but
# the UV index needs its paid One Call API, so UV comes from Open-Meteo (free, no key).
# Without an OpenWeatherMap key, Open-Meteo provides everything.

import requests  # noqa: E402

OWM_GEOCODE = "https://api.openweathermap.org/geo/1.0/direct"
OWM_WEATHER = "https://api.openweathermap.org/data/2.5/weather"
OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_S = 8


class WeatherError(RuntimeError):
    """Weather could not be fetched (unknown place, network or API problem)."""


RETRIES = 1  # one retry for timeouts and rate limits (HTTP 429)


def _get(url: str, params: dict) -> dict | list:
    for attempt in range(RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT_S)
        except requests.RequestException as e:
            if attempt < RETRIES:
                continue
            raise WeatherError(f"Weather service unreachable ({type(e).__name__})") from None
        if resp.status_code == 429 and attempt < RETRIES:
            continue
        break
    if resp.status_code == 401:
        raise WeatherError("OpenWeatherMap rejected the API key (new keys can take a few hours to activate)")
    if resp.status_code != 200:
        raise WeatherError(f"Weather service returned HTTP {resp.status_code}")
    return resp.json()


def _open_meteo_place(place: str) -> tuple[float, float, str]:
    data = _get(OPEN_METEO_GEOCODE, {"name": place, "count": 1})
    results = data.get("results") or []
    if not results:
        raise WeatherError(f"Place not found: {place!r}")
    r = results[0]
    name = ", ".join(x for x in (r.get("name"), r.get("country")) if x)
    return r["latitude"], r["longitude"], name


def _open_meteo_current(lat: float, lon: float) -> dict:
    data = _get(OPEN_METEO_FORECAST, {"latitude": lat, "longitude": lon, "timezone": "auto",
                                      "current": "temperature_2m,relative_humidity_2m,uv_index",
                                      "daily": "uv_index_max", "forecast_days": 1})
    current = data.get("current") or {}
    daily = (data.get("daily") or {}).get("uv_index_max") or [None]
    return {"temperature_c": current.get("temperature_2m"), "humidity_pct": current.get("relative_humidity_2m"),
            "uv_now": current.get("uv_index"), "uv_max_today": daily[0]}


def fetch_weather(place: str, api_key: str | None = None) -> EnvironmentData:
    """Current weather for a place name (e.g. "Shah Alam").

    The UV index used is today's maximum, which is what a morning routine should plan
    for (current UV is 0 at night). Raises WeatherError on failure.
    """
    place = (place or "").strip()
    if not place:
        raise WeatherError("No location given")
    if api_key:
        found = _get(OWM_GEOCODE, {"q": place, "limit": 1, "appid": api_key})
        if not found:
            raise WeatherError(f"Place not found: {place!r}")
        lat, lon = found[0]["lat"], found[0]["lon"]
        w = _get(OWM_WEATHER, {"lat": lat, "lon": lon, "units": "metric", "appid": api_key})
        temp, humidity = w["main"]["temp"], w["main"]["humidity"]
        name = ", ".join(x for x in (w.get("name") or found[0].get("name"), found[0].get("country")) if x)
        uv = _open_meteo_current(lat, lon)
    else:
        lat, lon, name = _open_meteo_place(place)
        uv = _open_meteo_current(lat, lon)
        temp, humidity = uv["temperature_c"], uv["humidity_pct"]
    uv_index = uv["uv_max_today"] if uv["uv_max_today"] is not None else uv["uv_now"]
    if temp is None or humidity is None or uv_index is None:
        raise WeatherError("Weather service returned incomplete data")
    return EnvironmentData(temperature_c=round(float(temp), 1), humidity_pct=float(humidity),
                           uv_index=round(float(uv_index), 1), location=name)
