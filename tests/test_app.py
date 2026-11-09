"""Streamlit app (proposal 3.7), driven with st.testing.AppTest. Weather is mocked."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app.services as services
from skiniq.environment import EnvironmentData, WeatherError
from skiniq.questionnaire import QUESTIONS_BY_ID

ROOT = Path(__file__).resolve().parent.parent
PAGE = str(ROOT / "app" / "app_pages" / "analyse.py")
FULL = dict(age_group="25_34", pregnant="no", sensitivity="not", reacts_to_products="no",
            tightness="rarely", oiliness="very_oily", oil_location="all_over", flakiness="no",
            breakouts="frequently")


@pytest.fixture(autouse=True)
def fake_weather(monkeypatch):
    services.get_weather.clear()
    monkeypatch.setattr(services, "fetch_weather",
                        lambda place, key=None: EnvironmentData(31, 80, 9, place.title()))


def fill(at, answers, place="Shah Alam", **multi):
    at.segmented_control(key="photo_source").set_value("Skip").run()
    for qid, value in answers.items():
        widget = at.radio(key=f"q_{qid}") if qid == "oil_location" else at.segmented_control(key=f"q_{qid}")
        widget.set_value(value)
    for qid, values in multi.items():
        at.pills(key=f"q_{qid}").set_value(values)
    at.text_input(key="place").set_value(place)
    at.button[0].click().run()
    return at


def texts(at):
    return " ".join(str(getattr(e, "value", "")) for e in at.main.children.values()) + " " + " ".join(
        str(m.value) for m in at.markdown) + " " + " ".join(str(c.value) for c in at.caption)


def test_page_renders_all_questions():
    at = AppTest.from_file(PAGE, default_timeout=120).run()
    assert not at.exception
    keys = {w.key for w in [*at.segmented_control, *at.pills, *at.radio]}
    assert {f"q_{q}" for q in QUESTIONS_BY_ID} <= keys


def test_missing_answers_are_listed():
    at = fill(AppTest.from_file(PAGE, default_timeout=120).run(), {"age_group": "18_24"})
    assert not at.exception
    assert at.error and "4, 5, 6, 7, 7b, 8, 9" in at.error[0].value
    assert not any("Your results" in h.value for h in at.header)


def test_full_submission_shows_results_with_weather():
    at = fill(AppTest.from_file(PAGE, default_timeout=180).run(), FULL,
              goals=["acne_control"], avoid_ingredients=["fragrance"])
    assert not at.exception
    assert any("Your results" in h.value for h in at.header)
    assert at.metric[0].value == "Oily"
    body = texts(at)
    assert "Weather in Shah Alam: 31 °C, 80% humidity, UV index 9" in body
    assert "acne" in body
    assert len(at.tabs) == 3
    assert at.session_state.result["routine"]["morning"][-1]["step"] == "sunscreen"


def test_weather_failure_shows_a_single_clear_note(monkeypatch):
    def fail(place, key=None):
        raise WeatherError("Place not found: 'Atlantis'")
    monkeypatch.setattr(services, "fetch_weather", fail)
    at = fill(AppTest.from_file(PAGE, default_timeout=180).run(), FULL, place="Atlantis")
    assert not at.exception
    notes = [w.value for w in at.warning if "eather" in w.value]
    assert notes == ["Weather unavailable (Place not found: 'Atlantis'), so no weather adjustments were made."]


def test_about_page():
    at = AppTest.from_file(str(ROOT / "app" / "app_pages" / "about.py")).run()
    assert not at.exception and at.title[0].value == "About SkinIQ"
