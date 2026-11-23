"""Streamlit app (proposal 3.7), driven with st.testing.AppTest. Weather is mocked."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app.services as services
from skiniq.environment import EnvironmentData, WeatherError
from skiniq.questionnaire import QUESTIONS_BY_ID

ROOT = Path(__file__).resolve().parent.parent
PAGE = str(ROOT / "app" / "app_pages" / "analyse.py")
SECTION_A = dict(age_group="26_35", gender="female", pregnant="no", sensitivity="not")
SECTION_B = dict(reacts_to_products="no", tightness="rarely", oiliness="very_oily", oil_location="all_over",
                 flakiness="no", breakouts="frequently")


SHAH_ALAM = {"status": "ok", "lat": 3.0733, "lon": 101.5185, "accuracy_m": 18, "place": "Seksyen 13, Shah Alam"}


@pytest.fixture(autouse=True)
def fake_weather(monkeypatch):
    services._weather.clear()
    monkeypatch.setattr(services, "fetch_weather_at",
                        lambda lat, lon, place=None, key=None: EnvironmentData(
                            28.4, 81, 4.6, place or "?", uv_max=10.1, latitude=lat, longitude=lon,
                            source="Open-Meteo", date="2026-11-23"))


def answer(at, values: dict):
    for qid, value in values.items():
        if isinstance(value, list):
            at.pills(key=f"q_{qid}").set_value(value)
        elif any(w.key == f"q_{qid}" for w in at.radio):
            at.radio(key=f"q_{qid}").set_value(value)
        else:
            at.segmented_control(key=f"q_{qid}").set_value(value)
    return at.run()


def nxt(at):
    return at.button(key="next").click().run()


def start(location=SHAH_ALAM):
    at = AppTest.from_file(PAGE, default_timeout=180).run()
    if location is not None:
        at.session_state["geo"] = {"location": location}
    at.segmented_control(key="photo_source").set_value("Skip").run()
    return at


def complete(at, a=None, b=None, cd=None):
    nxt(answer(nxt(at), {**SECTION_A, **(a or {})}))   # photo -> A -> B
    nxt(answer(at, {**SECTION_B, **(b or {})}))         # B -> C & D
    nxt(answer(at, cd or {"goals": ["acne_control"], "avoid_ingredients": ["fragrance"]}))  # -> results
    return at


def texts(at):
    return " ".join(str(getattr(e, "value", "")) for e in at.main.children.values()) + " " + " ".join(
        str(m.value) for m in at.markdown) + " " + " ".join(str(c.value) for c in at.caption)


def test_photo_step_shows_photo_choice_and_location():
    at = AppTest.from_file(PAGE, default_timeout=120).run()
    assert not at.exception
    assert at.segmented_control(key="photo_source").options == ["Upload a photo", "Take a photo", "Skip"]
    assert any(s.value == "Your location" for s in at.subheader)
    assert not any(b.key == "back" for b in at.button)  # no Back on the first step


def test_next_needs_a_photo_or_skip():
    at = AppTest.from_file(PAGE, default_timeout=120).run()
    nxt(at)
    assert at.session_state.step == 0 and "choose Skip" in at.error[0].value


def test_each_section_is_its_own_step():
    at = nxt(start())
    keys = {w.key for w in [*at.segmented_control, *at.pills, *at.radio]}
    assert keys == {"q_age_group", "q_gender", "q_pregnant", "q_known_conditions", "q_sensitivity"}
    nxt(answer(at, SECTION_A))
    keys = {w.key for w in [*at.segmented_control, *at.pills, *at.radio]}
    assert keys == {f"q_{k}" for k in SECTION_B}
    nxt(answer(at, SECTION_B))
    assert {w.key for w in at.pills} == {"q_goals", "q_avoid_ingredients"}
    assert at.button(key="next").label == "Analyse my skin"


def test_missing_answers_block_next_and_are_listed():
    at = nxt(answer(nxt(start()), {"age_group": "18_25"}))
    assert at.session_state.step == 1
    assert "2, 3, 5" in at.error[0].value


def test_back_keeps_answers():
    at = nxt(answer(nxt(start()), SECTION_A))
    at.button(key="back").click().run()
    assert at.session_state.step == 1
    assert at.segmented_control(key="q_age_group").value == "26_35"


def test_male_users_are_not_asked_about_pregnancy():
    at = answer(nxt(start()), {"gender": "male"})
    assert "q_pregnant" not in {w.key for w in at.segmented_control}
    nxt(answer(at, {"age_group": "18_25", "sensitivity": "not"}))
    assert at.session_state.step == 2


def test_other_shows_a_text_box():
    at = answer(nxt(start()), {"known_conditions": ["other"]})
    assert at.text_input(key="q_known_conditions_other")


def test_full_flow_shows_results_with_date_and_pdf():
    at = complete(start())
    assert not at.exception
    assert at.session_state.step == 4 and at.title[0].value == "Your results"
    assert at.metric[0].value == "Oily"
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Avg temperature"] == "28.4 °C" and metrics["Peak UV index"] == "10.1"
    body = texts(at)
    assert "Your answers point to oily skin" in body
    assert "?" not in at.session_state.result["skin_analysis"]["skin_type_text"]
    assert "Monday, 23 November 2026" in body and "Source: Open-Meteo." in body
    assert "Today's average" not in body  # no repeated weather sentence
    assert "Seksyen 13, Shah Alam" in body
    assert "Skin care for your age (26–35)" in [s.value for s in at.subheader]
    assert at.session_state.result_pdf.startswith(b"%PDF")
    assert at.session_state.result["routine"]["morning"][-1]["step"] == "sunscreen"


def test_start_again_clears_answers():
    at = complete(start())
    at.button(key="restart").click().run()
    assert at.session_state.step == 0 and at.session_state.result is None
    assert "q_age_group" not in at.session_state


def test_weather_failure_shows_a_single_clear_note(monkeypatch):
    def fail(lat, lon, place=None, key=None):
        raise WeatherError("weather service rate limit reached; try again later")
    monkeypatch.setattr(services, "fetch_weather_at", fail)
    at = complete(start())
    assert not at.exception
    notes = [w.value for w in at.warning if "eather" in w.value]
    assert notes == ["Weather unavailable (weather service rate limit reached; try again later), "
                     "so no weather adjustments were made."]


def test_location_denied_still_gives_results():
    denied = {"status": "error", "code": 1, "message": "Location access was blocked."}
    at = complete(start(location=denied))
    assert not at.exception and at.title[0].value == "Your results"
    assert any("location access was not available" in w.value for w in at.warning)


def test_welcome_page():
    at = AppTest.from_file(str(ROOT / "app" / "app_pages" / "welcome.py")).run()
    assert not at.exception
    body = texts(at)
    assert "Welcome to SkinIQ" in body and "Personalized Korean Skincare Advisory Platform" in body
    assert at.button(key="start").label == "Let's Start"


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
