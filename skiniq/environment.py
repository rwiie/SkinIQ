"""Environmental context: today's average temperature, humidity and UV index at the
user's exact location (proposal 3.4, 3.6.2, 3.7).

The browser supplies the device's GPS / Wi-Fi position (latitude, longitude); weather
for that point is fetched from Open-Meteo (free, no key) as today's hourly forecast,
averaged over the local day. OpenWeatherMap (with an API key) is the fallback for
temperature and humidity; its free plan has no UV index.

UV: the average over daylight hours is reported as asked, and the day's peak is kept
too. Sun-protection rules use the peak, because the WHO UV Index categories describe the
day's maximum exposure: in Malaysia the daylight average is often ~4 ("moderate") on a
day that peaks at 10 ("very high").
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

from skiniq.config import COLD_C, DRY_AIR_PCT, HOT_C, HUMID_PCT, UV_HIGH, UV_MODERATE


def uv_category(uv: float) -> str:
    """WHO UV Index category."""
    if uv < UV_MODERATE:
        return "low"
    if uv < UV_HIGH:
        return "moderate"
    if uv < 8:
        return "high"
    if uv < 11:
        return "very_high"
    return "extreme"


@dataclass(frozen=True)
class EnvironmentData:
    temperature_c: float                # today's average
    humidity_pct: float                 # today's average relative humidity
    uv_index: float | None              # today's average over daylight hours (None if unavailable)
    location: str = ""
    uv_max: float | None = None         # today's peak UV
    latitude: float | None = None
    longitude: float | None = None
    source: str = ""
    date: str = ""                      # local calendar date of the forecast, ISO "YYYY-MM-DD"

    def __post_init__(self) -> None:
        if not 0 <= self.humidity_pct <= 100:
            raise ValueError(f"humidity must be 0-100%, got {self.humidity_pct}")
        for name in ("uv_index", "uv_max"):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise ValueError(f"UV index cannot be negative, got {v}")

    @property
    def uv_peak(self) -> float | None:
        """UV used for sun-protection decisions: the day's peak, else the given value."""
        return self.uv_max if self.uv_max is not None else self.uv_index

    @property
    def uv_level(self) -> str | None:
        return uv_category(self.uv_peak) if self.uv_peak is not None else None

    @property
    def flags(self) -> dict[str, bool]:
        return {
            "high_uv": self.uv_peak is not None and self.uv_peak >= UV_HIGH,
            "humid": self.humidity_pct >= HUMID_PCT,
            "dry_air": self.humidity_pct < DRY_AIR_PCT,
            "hot": self.temperature_c >= HOT_C,
            "cold": self.temperature_c < COLD_C,
        }

    @property
    def date_label(self) -> str:
        """'Monday, 23 November 2026' (empty if unknown)."""
        try:
            d = datetime.strptime(self.date, "%Y-%m-%d")
        except ValueError:
            return ""
        return f"{d:%A}, {d.day} {d:%B %Y}"

    def summary(self) -> str:
        """One line for the results page."""
        where = f" in {self.location}" if self.location else ""
        bits = [f"{self.temperature_c:g} °C", f"{self.humidity_pct:g}% humidity"]
        if self.uv_index is not None and self.uv_max is not None:
            bits.append(f"UV index {self.uv_index:g} on average, peaking at {self.uv_max:g} "
                        f"({self.uv_level.replace('_', ' ')})")
        elif self.uv_peak is not None:
            bits.append(f"UV index {self.uv_peak:g} ({self.uv_level.replace('_', ' ')})")
        else:
            bits.append("UV index unavailable")
        return f"Today's average{where}: " + ", ".join(bits) + "."


# ---- live weather by coordinates ----------------------------------------------------------

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OWM_FORECAST = "https://api.openweathermap.org/data/2.5/forecast"
OWM_REVERSE = "https://api.openweathermap.org/geo/1.0/reverse"
TIMEOUT_S = 8
RETRIES = 1  # one retry for timeouts and rate limits (HTTP 429)


class WeatherError(RuntimeError):
    """Weather could not be fetched (network or API problem)."""


def _get(url: str, params: dict) -> dict | list:
    resp = None
    for attempt in range(RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT_S)
        except requests.RequestException as e:
            if attempt < RETRIES:
                continue
            raise WeatherError(f"weather service unreachable ({type(e).__name__})") from None
        if resp.status_code == 429 and attempt < RETRIES:
            continue
        break
    if resp.status_code == 401:
        raise WeatherError("OpenWeatherMap rejected the API key (new keys can take a few hours to activate)")
    if resp.status_code == 429:
        raise WeatherError("weather service rate limit reached; try again later")
    if resp.status_code != 200:
        raise WeatherError(f"weather service returned HTTP {resp.status_code}")
    return resp.json()


def _mean(values) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _open_meteo_today(lat: float, lon: float) -> dict:
    data = _get(OPEN_METEO_FORECAST, {
        "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 1,
        "hourly": "temperature_2m,relative_humidity_2m,uv_index", "daily": "uv_index_max",
    })
    hourly = data.get("hourly") or {}
    uv_hours = [v for v in hourly.get("uv_index") or [] if v is not None]
    daylight = [v for v in uv_hours if v > 0]
    daily = data.get("daily") or {}
    peak = (daily.get("uv_index_max") or [None])[0]
    day = (daily.get("time") or [""])[0] or ((hourly.get("time") or [""])[0][:10])
    temp = _mean(hourly.get("temperature_2m") or [])
    humidity = _mean(hourly.get("relative_humidity_2m") or [])
    if temp is None or humidity is None:
        raise WeatherError("weather service returned incomplete data")
    return {"temperature_c": temp, "humidity_pct": humidity,
            "uv_index": _mean(daylight) if daylight else (0.0 if uv_hours else None),
            "uv_max": peak if peak is not None else (max(uv_hours) if uv_hours else None), "date": day}


def _owm_today(lat: float, lon: float, api_key: str) -> dict:
    """Free OpenWeatherMap 5-day / 3-hour forecast, averaged over the local calendar day."""
    data = _get(OWM_FORECAST, {"lat": lat, "lon": lon, "units": "metric", "appid": api_key})
    offset = timedelta(seconds=(data.get("city") or {}).get("timezone", 0))
    today = (datetime.now(timezone.utc) + offset).date()
    items = [i for i in data.get("list", []) if (datetime.fromtimestamp(i["dt"], timezone.utc) + offset).date() == today]
    items = items or data.get("list", [])[:8]  # late evening: next 24 h
    temp = _mean(i["main"]["temp"] for i in items)
    humidity = _mean(i["main"]["humidity"] for i in items)
    if temp is None or humidity is None:
        raise WeatherError("OpenWeatherMap returned incomplete data")
    return {"temperature_c": temp, "humidity_pct": humidity, "uv_index": None, "uv_max": None,
            "date": today.isoformat()}


def _owm_place(lat: float, lon: float, api_key: str) -> str:
    try:
        found = _get(OWM_REVERSE, {"lat": lat, "lon": lon, "limit": 1, "appid": api_key})
    except WeatherError:
        return ""
    if not found:
        return ""
    f = found[0]
    return ", ".join(x for x in (f.get("name"), f.get("state"), f.get("country")) if x)


def coordinates_label(lat: float, lon: float) -> str:
    return f"{abs(lat):.4f}° {'N' if lat >= 0 else 'S'}, {abs(lon):.4f}° {'E' if lon >= 0 else 'W'}"


def fetch_weather_at(lat: float, lon: float, place: str | None = None, api_key: str | None = None
                     ) -> EnvironmentData:
    """Today's average weather at a point. `place` is a name for display (e.g. from the
    browser's reverse geocoding). Raises WeatherError if no source works."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise WeatherError("invalid coordinates")
    errors = []
    w = source = None
    try:
        w, source = _open_meteo_today(lat, lon), "Open-Meteo"
    except WeatherError as e:
        errors.append(f"Open-Meteo: {e}")
        if api_key:
            try:
                w, source = _owm_today(lat, lon, api_key), "OpenWeatherMap (UV unavailable)"
            except WeatherError as e2:
                errors.append(f"OpenWeatherMap: {e2}")
    if w is None:
        raise WeatherError("; ".join(errors))
    name = place or (_owm_place(lat, lon, api_key) if api_key else "") or coordinates_label(lat, lon)

    def r(v):
        return None if v is None else round(float(v), 1)

    return EnvironmentData(temperature_c=r(w["temperature_c"]), humidity_pct=round(float(w["humidity_pct"])),
                           uv_index=r(w["uv_index"]), location=name, uv_max=r(w["uv_max"]),
                           latitude=round(lat, 5), longitude=round(lon, 5), source=source,
                           date=w.get("date") or "")
