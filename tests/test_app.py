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


SHAH_ALAM = {"status": "ok", "lat": 3.0733, "lon": 101.5185, "accuracy_m": 18, "place": "Seksyen 13, Shah Alam"}


@pytest.fixture(autouse=True)
def fake_weather(monkeypatch):
    services._weather.clear()
    monkeypatch.setattr(services, "fetch_weather_at",
                        lambda lat, lon, place=None, key=None: EnvironmentData(
                            28.4, 81, 4.6, place or "?", uv_max=10.1, latitude=lat, longitude=lon,
                            source="Open-Meteo"))


def fill(at, answers, location=SHAH_ALAM, **multi):
    if location is not None:
        at.session_state["geo"] = {"location": location}
    at.segmented_control(key="photo_source").set_value("Skip").run()
    for qid, value in answers.items():
        widget = at.radio(key=f"q_{qid}") if qid == "oil_location" else at.segmented_control(key=f"q_{qid}")
        widget.set_value(value)
    for qid, values in multi.items():
        at.pills(key=f"q_{qid}").set_value(values)
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


def test_full_submission_shows_location_and_daily_weather():
    at = fill(AppTest.from_file(PAGE, default_timeout=180).run(), FULL,
              goals=["acne_control"], avoid_ingredients=["fragrance"])
    assert not at.exception
    assert any("Your results" in h.value for h in at.header)
    assert not at.text_input  # no location box: location is detected automatically
    assert at.metric[0].value == "Oily"
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Avg temperature"] == "28.4 °C" and metrics["Avg humidity"] == "81%"
    assert metrics["Avg UV index"] == "4.6" and metrics["Peak UV index"] == "10.1"
    body = texts(at)
    assert "Seksyen 13, Shah Alam" in body and "3.0733, 101.5185" in body
    assert "Today's average in Seksyen 13, Shah Alam" in body
    assert at.session_state.result["routine"]["morning"][-1]["step"] == "sunscreen"


def test_weather_failure_shows_a_single_clear_note(monkeypatch):
    def fail(lat, lon, place=None, key=None):
        raise WeatherError("weather service rate limit reached; try again later")
    monkeypatch.setattr(services, "fetch_weather_at", fail)
    at = fill(AppTest.from_file(PAGE, default_timeout=180).run(), FULL)
    assert not at.exception
    notes = [w.value for w in at.warning if "eather" in w.value]
    assert notes == ["Weather unavailable (weather service rate limit reached; try again later), "
                     "so no weather adjustments were made."]


def test_location_denied_still_gives_results(monkeypatch):
    denied = {"status": "error", "code": 1, "message": "Location access was blocked."}
    at = fill(AppTest.from_file(PAGE, default_timeout=180).run(), FULL, location=denied)
    assert not at.exception and any("Your results" in h.value for h in at.header)
    assert any("location access was not available" in w.value for w in at.warning)


def test_about_page():
    at = AppTest.from_file(str(ROOT / "app" / "app_pages" / "about.py")).run()
    assert not at.exception and at.title[0].value == "About SkinIQ"


def test_rejected_upload_is_not_analysed(monkeypatch):
    from skiniq.vision.quality import QualityReport

    monkeypatch.setattr(services, "check_photo", lambda img: QualityReport(False, ["No face was detected."]))
    services.photo_quality.clear()
    import io

    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", (400, 400), (200, 150, 120)).save(buf, format="PNG")
    report = services.photo_quality(buf.getvalue())
    assert not report.ok and report.message == "No face was detected."


def test_coordinates_are_exact_when_no_place_name(monkeypatch):
    from skiniq.environment import coordinates_label

    monkeypatch.setattr(services, "fetch_weather_at", lambda lat, lon, place=None, key=None: EnvironmentData(
        28, 80, 4, place or coordinates_label(lat, lon), uv_max=10, latitude=lat, longitude=lon))
    w = services.get_weather_at(3.07334, 101.51851)
    assert w.location == "3.0733° N, 101.5185° E" and (w.latitude, w.longitude) == (3.07334, 101.51851)


def test_browser_face_detector_is_bundled():
    from app.components.camera import STATIC_BASE

    static = ROOT / STATIC_BASE
    for name in ["ort.wasm.bundle.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm",
                 "yunet_2023mar_dynamic.onnx"]:
        assert (static / name).is_file(), name


def test_browser_yunet_accepts_the_camera_input_size():
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    from app.components.camera import STATIC_BASE

    session = ort.InferenceSession(str(ROOT / STATIC_BASE / "yunet_2023mar_dynamic.onnx"))
    out = dict(zip([o.name for o in session.get_outputs()],
                   session.run(None, {"input": np.zeros((1, 3, 256, 320), np.float32)})))
    assert out["cls_8"].shape == (1, (256 // 8) * (320 // 8), 1) and out["bbox_32"].shape == (1, 80, 4)
