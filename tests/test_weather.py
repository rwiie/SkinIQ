"""Live weather at the user's coordinates (daily averages), with HTTP mocked."""

import pytest
import requests

from skiniq import environment as env
from skiniq.environment import EnvironmentData, WeatherError, coordinates_label, fetch_weather_at

SHAH_ALAM = (3.0733, 101.5185)


class Resp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def json(self):
        return self._data


def router(routes, calls=None):
    def fake_get(url, params=None, timeout=None):
        if calls is not None:
            calls.append((url, params))
        for prefix, resp in routes.items():
            if url.startswith(prefix):
                return resp() if callable(resp) else resp
        raise AssertionError(f"unexpected URL {url}")
    return fake_get


# 24 hourly values for one local day: night UV 0, daylight 07:00-18:00
TEMPS = [25.0] * 12 + [31.0] * 12            # mean 28.0
HUMIDITY = [90] * 12 + [70] * 12             # mean 80
UV = [0] * 7 + [1, 2, 4, 6, 8, 10, 10, 8, 6, 4, 2, 1] + [0] * 5   # daylight mean 5.17, peak 10
METEO = Resp({"hourly": {"temperature_2m": TEMPS, "relative_humidity_2m": HUMIDITY, "uv_index": UV},
              "daily": {"uv_index_max": [10.2]}})


def test_daily_averages_at_coordinates(monkeypatch):
    calls = []
    monkeypatch.setattr(requests, "get", router({env.OPEN_METEO_FORECAST: METEO}, calls))
    w = fetch_weather_at(*SHAH_ALAM, place="Seksyen 13, Shah Alam")
    assert (w.temperature_c, w.humidity_pct) == (28.0, 80)
    assert w.uv_index == 5.2 and w.uv_max == 10.2           # daylight average, peak
    assert (w.latitude, w.longitude, w.location) == (3.0733, 101.5185, "Seksyen 13, Shah Alam")
    params = calls[0][1]
    assert params["latitude"] == 3.0733 and params["timezone"] == "auto" and params["forecast_days"] == 1


def test_summary_line_shows_average_and_peak(monkeypatch):
    monkeypatch.setattr(requests, "get", router({env.OPEN_METEO_FORECAST: METEO}))
    text = fetch_weather_at(*SHAH_ALAM, place="Seksyen 13, Shah Alam").summary()
    assert text == ("Today's average in Seksyen 13, Shah Alam: 28 °C, 80% humidity, "
                    "UV index 5.2 on average, peaking at 10.2 (very high).")


def test_without_place_name_shows_coordinates(monkeypatch):
    monkeypatch.setattr(requests, "get", router({env.OPEN_METEO_FORECAST: METEO}))
    assert fetch_weather_at(*SHAH_ALAM).location == "3.0733° N, 101.5185° E"
    assert coordinates_label(-33.8688, -151.2093) == "33.8688° S, 151.2093° W"


def test_owm_fallback_averages_today_and_reverse_geocodes(monkeypatch):
    import time

    now = int(time.time())
    owm = Resp({"city": {"timezone": 0},
                "list": [{"dt": now, "main": {"temp": 30.0, "humidity": 70}},
                         {"dt": now + 1, "main": {"temp": 26.0, "humidity": 90}}]})
    monkeypatch.setattr(requests, "get", router({
        env.OPEN_METEO_FORECAST: Resp({}, status=429),
        env.OWM_FORECAST: owm,
        env.OWM_REVERSE: Resp([{"name": "Shah Alam", "state": "Selangor", "country": "MY"}]),
    }))
    w = fetch_weather_at(*SHAH_ALAM, api_key="KEY")
    assert (w.temperature_c, w.humidity_pct, w.uv_index, w.uv_max) == (28.0, 80, None, None)
    assert w.location == "Shah Alam, Selangor, MY" and "OpenWeatherMap" in w.source
    assert w.summary().endswith("UV index unavailable.")


def test_all_sources_failing_raises_with_reasons(monkeypatch):
    monkeypatch.setattr(requests, "get", router({env.OPEN_METEO_FORECAST: Resp({}, status=500),
                                                 env.OWM_FORECAST: Resp({}, status=401)}))
    with pytest.raises(WeatherError, match="Open-Meteo.*HTTP 500.*rejected the API key"):
        fetch_weather_at(*SHAH_ALAM, api_key="bad")


def test_one_retry_after_rate_limit(monkeypatch):
    responses = iter([Resp({}, status=429), METEO])
    monkeypatch.setattr(requests, "get", lambda *a, **k: next(responses))
    assert fetch_weather_at(*SHAH_ALAM).uv_max == 10.2


def test_rate_limit_twice_is_reported(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp({}, status=429))
    with pytest.raises(WeatherError, match="rate limit"):
        fetch_weather_at(*SHAH_ALAM)


def test_network_failure_retries_once_then_gives_up(monkeypatch):
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise requests.ConnectionError("down")
    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(WeatherError, match="unreachable"):
        fetch_weather_at(*SHAH_ALAM)
    assert len(calls) == 2


def test_invalid_coordinates():
    with pytest.raises(WeatherError, match="invalid"):
        fetch_weather_at(120, 10)


def test_environment_backwards_compatible_with_single_uv():
    e = EnvironmentData(30, 80, 9)
    assert e.uv_peak == 9 and e.flags["high_uv"] and e.summary().endswith("UV index 9 (very high).")
