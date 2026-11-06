import math

from skiniq import config


def test_skin_types_match_proposal():
    assert set(config.SKIN_TYPES) == {"normal", "oily", "dry", "combination"}


def test_six_skin_conditions():
    assert len(config.SKIN_CONDITIONS) == 6
    assert len(set(config.SKIN_CONDITIONS)) == 6


def test_data_split_sums_to_one():
    assert math.isclose(sum(config.DATA_SPLIT.values()), 1.0)


def test_api_key_read_from_env(monkeypatch):
    monkeypatch.setenv("OPENWEATHER_API_KEY", "dummy")
    assert config.openweather_api_key() == "dummy"
    monkeypatch.delenv("OPENWEATHER_API_KEY")
    assert config.openweather_api_key() is None
