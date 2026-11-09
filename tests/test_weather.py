"""Live weather client, with HTTP mocked."""

import pytest
import requests

from skiniq import environment as env
from skiniq.environment import WeatherError, fetch_weather


class Resp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def json(self):
        return self._data


def router(routes, calls):
    def fake_get(url, params=None, timeout=None):
        calls.append((url, params))
        for prefix, resp in routes.items():
            if url.startswith(prefix):
                return resp
        raise AssertionError(f"unexpected URL {url}")
    return fake_get


METEO = Resp({"current": {"temperature_2m": 31.4, "relative_humidity_2m": 76, "uv_index": 0.0},
              "daily": {"uv_index_max": [10.25]}})


def test_open_meteo_only_without_key(monkeypatch):
    calls = []
    monkeypatch.setattr(requests, "get", router({
        env.OPEN_METEO_GEOCODE: Resp({"results": [{"name": "Shah Alam", "country": "Malaysia",
                                                   "latitude": 3.07, "longitude": 101.52}]}),
        env.OPEN_METEO_FORECAST: METEO,
    }, calls))
    w = fetch_weather("Shah Alam")
    assert (w.temperature_c, w.humidity_pct, w.uv_index, w.location) == (31.4, 76.0, 10.2, "Shah Alam, Malaysia")
    assert not any("openweathermap" in u for u, _ in calls)


def test_openweathermap_with_key_and_uv_from_open_meteo(monkeypatch):
    calls = []
    monkeypatch.setattr(requests, "get", router({
        env.OWM_GEOCODE: Resp([{"name": "Shah Alam", "country": "MY", "lat": 3.07, "lon": 101.52}]),
        env.OWM_WEATHER: Resp({"name": "Shah Alam", "main": {"temp": 32.0, "humidity": 78}}),
        env.OPEN_METEO_FORECAST: METEO,
    }, calls))
    w = fetch_weather("Shah Alam", api_key="KEY")
    assert (w.temperature_c, w.humidity_pct, w.uv_index) == (32.0, 78.0, 10.2)
    owm = [p for u, p in calls if "openweathermap" in u]
    assert owm and all(p["appid"] == "KEY" for p in owm)
    assert any(u == env.OPEN_METEO_FORECAST for u, _ in calls)


def test_uv_uses_today_max_not_current(monkeypatch):
    monkeypatch.setattr(requests, "get", router({
        env.OPEN_METEO_GEOCODE: Resp({"results": [{"name": "X", "latitude": 0, "longitude": 0}]}),
        env.OPEN_METEO_FORECAST: Resp({"current": {"temperature_2m": 25, "relative_humidity_2m": 50,
                                                   "uv_index": 0.0}, "daily": {"uv_index_max": [7.0]}}),
    }, []))
    assert fetch_weather("X").uv_index == 7.0


@pytest.mark.parametrize("routes, message", [
    ({env.OPEN_METEO_GEOCODE: Resp({"results": []})}, "Place not found"),
    ({env.OPEN_METEO_GEOCODE: Resp({}, status=500)}, "HTTP 500"),
])
def test_errors_are_reported_clearly(monkeypatch, routes, message):
    monkeypatch.setattr(requests, "get", router(routes, []))
    with pytest.raises(WeatherError, match=message):
        fetch_weather("Nowhere")


def test_bad_key_message(monkeypatch):
    monkeypatch.setattr(requests, "get", router({env.OWM_GEOCODE: Resp({}, status=401)}, []))
    with pytest.raises(WeatherError, match="rejected the API key"):
        fetch_weather("Shah Alam", api_key="bad")


def test_network_failure(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(WeatherError, match="unreachable"):
        fetch_weather("Shah Alam")


def test_empty_place():
    with pytest.raises(WeatherError, match="No location"):
        fetch_weather("  ")


def test_one_retry_after_rate_limit(monkeypatch):
    responses = iter([Resp({}, status=429),
                      Resp({"results": [{"name": "X", "latitude": 0, "longitude": 0}]}), METEO])
    monkeypatch.setattr(requests, "get", lambda *a, **k: next(responses))
    assert fetch_weather("X").uv_index == 10.2


def test_retry_after_timeout_then_gives_up(monkeypatch):
    calls = []

    def slow(*a, **k):
        calls.append(1)
        raise requests.Timeout()
    monkeypatch.setattr(requests, "get", slow)
    with pytest.raises(WeatherError, match="unreachable"):
        fetch_weather("X")
    assert len(calls) == 2
