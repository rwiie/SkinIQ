import io
import json

import streamlit as st
from PIL import Image, UnidentifiedImageError

from app.components.camera import face_camera
from app.components.geolocation import detect_location
from app.services import get_classifier, get_product_index, get_weather_at, photo_quality
from skiniq.environment import WeatherError
from skiniq.pipeline import analyse
from skiniq.questionnaire import QUESTIONS

SINGLE_CHOICE_INLINE_MAX = 22  # longest option label shown as a segmented control
REQUIRED = [q for q in QUESTIONS if not q.multi]

if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.photo_bytes = None

st.title("Analyse my skin")
st.caption("Add a photo and answer a few questions to get a personalised Korean skincare routine. "
           "Your location is detected automatically for today's weather.")

# ---- 1. Photo ------------------------------------------------------------------------------
st.subheader("1. Your photo")
source = st.segmented_control(
    "Photo", ["Upload a photo", "Take a photo", "Skip"], default="Upload a photo", key="photo_source",
    label_visibility="collapsed",
)
photo_bytes = None
if source == "Upload a photo":
    upload = st.file_uploader("Face photo (front-facing, no makeup or filters, good light)",
                              type=["jpg", "jpeg", "png", "webp"], key="photo_upload")
    photo_bytes = upload.getvalue() if upload is not None else None
elif source == "Take a photo":
    st.caption("Fit your face inside the outline. It turns green when your face is detected and positioned well.")
    photo_bytes = face_camera(key="face_camera")
else:
    st.caption("Without a photo, your skin concerns come from your answers only.")

# Reject unreadable, low-quality or face-less photos right away
image = None
if photo_bytes:
    try:
        image = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
    except UnidentifiedImageError:
        st.error("That file isn't a readable image. Upload a JPG, PNG or WEBP photo.",
                 title="Photo not accepted", icon=":material/no_photography:")
    else:
        report = photo_quality(photo_bytes)
        if report.ok:
            st.success("Photo accepted: one clear face detected.", icon=":material/check_circle:")
        else:
            st.error(report.message, title="Photo not accepted", icon=":material/no_photography:")
            image = None


# ---- 2. Questionnaire + 3. Location (one form, submitted together) --------------------------
def question_widget(q):
    key = f"q_{q.id}"
    labels = q.options
    label = f"{q.number}. {q.text}"
    if q.multi:
        return st.pills(label, list(labels), selection_mode="multi", format_func=labels.get, key=key)
    if max(len(v) for v in labels.values()) <= SINGLE_CHOICE_INLINE_MAX and len(labels) <= 5:
        return st.segmented_control(label, list(labels), format_func=labels.get, key=key)
    return st.radio(label, list(labels), index=None, format_func=labels.get, key=key)


# ---- Location (automatic) ----------------------------------------------------------------------
st.subheader("Your location")
location = detect_location(key="geo")


with st.form("questionnaire", border=False):
    st.subheader("2. About your skin")
    answers = {}
    for section in dict.fromkeys(q.section for q in QUESTIONS):
        with st.container(border=True):
            st.markdown(f"**{section}**")
            for q in (q for q in QUESTIONS if q.section == section):
                answers[q.id] = question_widget(q)

    submitted = st.form_submit_button("Analyse my skin", type="primary", icon=":material/search:",
                                      width="stretch")

if submitted:
    missing = [q.number for q in REQUIRED if answers.get(q.id) is None]
    if missing:
        st.error("Please answer question" + ("s " if len(missing) > 1 else " ") + ", ".join(missing) + ".",
                 title="Some answers are missing")
    else:
        environment = None
        weather_error = None
        if location and location.get("status") == "ok":
            try:
                environment = get_weather_at(location["lat"], location["lon"], location.get("place") or None)
            except WeatherError as e:
                weather_error = str(e)
        elif location:
            weather_error = "location access was not available"
        else:
            weather_error = "your location is still being detected"
        with st.spinner("Analysing your skin and choosing products…"):
            get_product_index()
            result = analyse({**answers, "known_conditions": answers["known_conditions"] or ["none"],
                              "goals": answers["goals"] or [],
                              "avoid_ingredients": answers["avoid_ingredients"] or ["none"]},
                             image=image, environment=environment,
                             classifier=get_classifier() if image is not None else None)
        if weather_error:  # replace the pipeline's generic note with the specific reason
            result["notes"] = [n for n in result["notes"] if not n.startswith("Weather data")]
            result["notes"].insert(0, f"Weather unavailable ({weather_error}), so no weather adjustments were made.")
        st.session_state.result = result
        st.session_state.photo_bytes = photo_bytes if image is not None else None

# ---- Results ----------------------------------------------------------------------------------
result = st.session_state.result
if result is None:
    st.stop()

st.divider()
st.header("Your results")
for note in result["notes"]:
    st.warning(note)

# Section 1: skin analysis
sa = result["skin_analysis"]
st.subheader("Skin analysis")
with st.container(border=True):
    left, right = st.columns([1, 2], vertical_alignment="center")
    with left:
        if st.session_state.photo_bytes:
            st.image(st.session_state.photo_bytes, alt="Your uploaded face photo", width="stretch")
        else:
            st.caption("No photo analysed")
    with right:
        st.metric("Skin type", sa["skin_type"].title())
        for reason in sa["skin_type_reasons"]:
            st.caption(reason)
    if sa["concerns"]:
        for c in sa["concerns"]:
            badge = ":red-badge[Detected]" if c["status"] == "detected" else ":orange-badge[Possible]"
            st.markdown(f"{badge} {c['text'].split(': ', 1)[1]}")
    else:
        st.markdown(":green-badge[Clear] " + sa["no_concerns_text"])

    env = result["usce"]["environment"]
    st.markdown("**Your location and today's weather**")
    if env:
        coords = f"{env['latitude']:.4f}, {env['longitude']:.4f}" if env.get("latitude") is not None else ""
        named = coords and not env["location"].endswith(("E", "W"))  # a place name, not just coordinates
        st.markdown(f":material/location_on: {env['location']}" + (f"  \n:gray[{coords}]" if named else ""))
        with st.container(horizontal=True):
            st.metric("Avg temperature", f"{env['temperature_c']:g} °C", border=True)
            st.metric("Avg humidity", f"{env['humidity_pct']:g}%", border=True)
            if env.get("uv_index") is not None:
                st.metric("Avg UV index", f"{env['uv_index']:g}", border=True,
                          help="Average over daylight hours today.")
            if env.get("uv_max") is not None:
                st.metric("Peak UV index", f"{env['uv_max']:g}", border=True,
                          help="Today's maximum. Sun-protection advice uses the peak (WHO UV Index).")
        st.caption(sa["environment_text"] + (f" Source: {env['source']}." if env.get("source") else ""))
    else:
        st.caption(sa["environment_text"])

# Section 2: advice
st.subheader("Advice for you")
st.info("\n".join(f"- {tip}" for tip in result["advice"]), icon=":material/lightbulb:")


# Sections 3 and 4: routine with explanations
def product_card(step: dict) -> None:
    p = step["product"]
    with st.container(border=True):
        st.caption(step["label"])
        st.markdown(f"**{p['product']}**")
        st.write(p["summary"])
        with st.expander("Why this product"):
            if p["key_ingredients"]:
                st.markdown("**Key ingredients for you**")
                st.markdown("\n".join(f"- {k}" for k in p["key_ingredients"]))
            if p["why"]:
                st.markdown("**Why it suits you**")
                st.markdown("\n".join(f"- {w}" for w in p["why"]))
            for caution in p["cautions"]:
                st.warning(caution, icon=":material/warning:")
            if step.get("alternatives"):
                st.caption("Alternatives: " + "; ".join(a["product"] for a in step["alternatives"]))


st.subheader("Your routine")
routine = result["routine"]
morning, evening, weekly = st.tabs(["Morning", "Evening", "Weekly extras"])
with morning:
    for step in routine["morning"]:
        product_card(step)
with evening:
    for step in routine["evening"]:
        product_card(step)
with weekly:
    if not routine["weekly"]:
        st.caption("No weekly extras suggested.")
    for step in routine["weekly"]:
        product_card(step)
        st.caption(f"Use {step['how_often']}.")
for c in routine["conflicts"]:
    st.warning(f"{c['reason']}: {c['advice']} ({' + '.join(c['products'])}).", icon=":material/sync_problem:")

st.download_button("Download results (JSON)", json.dumps(result, indent=2, ensure_ascii=False),
                   file_name="skiniq_results.json", mime="application/json", icon=":material/download:")
st.caption(result["disclaimer"])
