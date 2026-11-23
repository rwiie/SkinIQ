import io

import streamlit as st
from PIL import Image, UnidentifiedImageError

from app import wizard
from app.components.camera import face_camera
from app.components.geolocation import detect_location
from app.services import get_classifier, get_product_index, get_weather_at, photo_quality, routine_photos
from skiniq.environment import WeatherError
from skiniq.pipeline import analyse
from skiniq.product_images import SOURCE_NAME
from skiniq.questionnaire import QUESTIONS
from skiniq.report import build_pdf

SINGLE_CHOICE_INLINE_MAX = 22  # longest option label shown as a segmented control

wizard.init()
ss = st.session_state
step = ss.step


# ---- shared pieces --------------------------------------------------------------------------

def question_widget(q) -> None:
    key = f"q_{q.id}"
    labels = q.options
    label = f"{q.number}. {q.text}" if q.number else q.text
    if q.multi:
        st.pills(label, list(labels), selection_mode="multi", format_func=labels.get, key=key,
                 persist_state="session", help=q.help or None)
    elif max(len(v) for v in labels.values()) <= SINGLE_CHOICE_INLINE_MAX and len(labels) <= 5:
        st.segmented_control(label, list(labels), format_func=labels.get, key=key, persist_state="session",
                             help=q.help or None)
    else:
        st.radio(label, list(labels), index=None, format_func=labels.get, key=key, persist_state="session",
                 help=q.help or None)
    if q.other_id and "other" in (ss.get(key) or []):
        st.text_input("Please specify", key=f"q_{q.other_id}", persist_state="session",
                      placeholder="e.g. perioral dermatitis" if q.id == "known_conditions"
                      else "e.g. niacinamide, parabens", max_chars=200)


def section(name: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{name}**")
        for q in QUESTIONS:
            if q.section != name:
                continue
            if q.id == "pregnant" and not wizard.asks_pregnancy():
                continue
            question_widget(q)


def location_section() -> None:
    st.subheader("Your location")
    found = detect_location(key="geo")
    if found:
        ss.location = found


def nav(next_label: str = "Next", next_icon: str = ":material/arrow_forward:", skip: bool = False) -> None:
    if ss.nav_error:  # shown after a blocked "Next", until the problem is fixed
        ss.nav_error = wizard.step_problem(step)
        if ss.nav_error:
            st.error(ss.nav_error, icon=":material/error:")
    with st.container(horizontal=True, vertical_alignment="center"):
        if step > wizard.PHOTO:
            st.button("Back", icon=":material/arrow_back:", on_click=wizard.go_back, key="back")
        st.space("stretch")
        if skip:
            st.button("Skip", on_click=wizard.skip_photo, key="skip",
                      help="Continue without a photo: your skin concerns then come from your answers only.")
        st.button(next_label, type="primary", icon=next_icon, icon_position="right", on_click=wizard.go_next,
                  key="next")


def progress() -> None:
    names = [name for name, _ in wizard.STEPS[:-1]]
    st.caption(f"Step {step + 1} of {len(names)} · {names[step]}")
    st.progress((step + 1) / len(names))


# ---- step 1: photo ----------------------------------------------------------------------------

def photo_step() -> None:
    st.title("Analyse my skin")
    progress()
    st.subheader("Your photo")
    source = st.segmented_control(
        "Photo", ["Upload a photo", "Take a photo"], default="Upload a photo", key="photo_source",
        label_visibility="collapsed", persist_state="session",
    )
    if ss.photo_bytes:
        st.image(ss.photo_bytes, width=260, caption="Your photo")
        st.success("Photo accepted: one clear face detected.", icon=":material/check_circle:")
        st.button("Use a different photo", icon=":material/refresh:", on_click=wizard.clear_photo)
    else:
        if source == "Take a photo":
            st.caption("Fit your face inside the outline. It turns green when your face is detected and "
                       "positioned well.")
            photo = face_camera(key=f"face_camera_{ss.photo_nonce}")
        else:
            upload = st.file_uploader("Face photo (front-facing, no makeup or filters, good light)",
                                      type=["jpg", "jpeg", "png", "webp"], key=f"photo_upload_{ss.photo_nonce}")
            photo = upload.getvalue() if upload is not None else None
        if photo:
            try:
                Image.open(io.BytesIO(photo)).verify()
            except (UnidentifiedImageError, OSError):
                st.error("That file isn't a readable image. Upload a JPG, PNG or WEBP photo.",
                         title="Photo not accepted", icon=":material/no_photography:")
            else:
                report = photo_quality(photo)
                if report.ok:
                    ss.photo_bytes = photo
                    ss.nav_error = None
                    st.success("Photo accepted: one clear face detected.", icon=":material/check_circle:")
                else:
                    st.error(report.message, title="Photo not accepted", icon=":material/no_photography:")
    location_section()
    nav(skip=True)


# ---- steps 2-4: questionnaire -----------------------------------------------------------------

def question_step() -> None:
    st.title("Analyse my skin")
    progress()
    for name in wizard.STEPS[step][1]:
        section(name)
    last = step == wizard.RESULTS - 1
    if last and not ss.location:
        location_section()
    if last:
        nav("Analyse my skin", ":material/search:")
    else:
        nav()


# ---- step 5: results --------------------------------------------------------------------------

def run_analysis() -> None:
    environment, weather_error = None, None
    loc = ss.location
    if loc and loc.get("status") == "ok":
        try:
            environment = get_weather_at(loc["lat"], loc["lon"], loc.get("place") or None)
        except WeatherError as e:
            weather_error = str(e)
    elif loc:
        weather_error = "location access was not available"
    else:
        weather_error = "your location could not be detected"
    photo = wizard.photo_for_analysis()
    image = Image.open(io.BytesIO(photo)).convert("RGB") if photo else None
    with st.spinner("Analysing your skin and choosing products…"):
        get_product_index()
        result = analyse(wizard.answers(), image=image, environment=environment,
                         classifier=get_classifier() if image is not None else None)
        if weather_error:  # replace the pipeline's generic note with the specific reason
            result["notes"] = [n for n in result["notes"] if not n.startswith("Weather data")]
            result["notes"].insert(0, f"Weather unavailable ({weather_error}), so no weather adjustments were made.")
        ss.result = result
        ss.result_photo = photo
        ss.result_pdf = build_pdf(result, photo, product_photos=routine_photos(result))


def product_card(step_: dict, weekly: bool = False) -> None:
    p = step_["product"]
    with st.container(border=True):
        photo, text = st.columns([1, 3], vertical_alignment="center")
        with photo:
            if p.get("image_url"):
                st.image(p["image_url"], alt=p["product"], width="stretch")
                st.caption(f"Photo: [{SOURCE_NAME}]({p['image_source']})")
            else:
                with st.container(height=120, horizontal_alignment="center", vertical_alignment="center",
                                  border=False):
                    st.markdown(":gray[:material/image_not_supported:]  \n:gray[No photo]",
                                text_alignment="center")
        with text:
            st.caption(step_["label"])
            st.markdown(f"**{p['product']}**")
            st.write(p["summary"])
            if weekly:
                st.caption(f"Use {step_['how_often']}.")
        with st.expander("Why this product"):
            if p["key_ingredients"]:
                st.markdown("**Key ingredients for you**")
                st.markdown("\n".join(f"- {k}" for k in p["key_ingredients"]))
            if p["why"]:
                st.markdown("**Why it suits you**")
                st.markdown("\n".join(f"- {w}" for w in p["why"]))
            for caution in p["cautions"]:
                st.warning(caution, icon=":material/warning:")
            if step_.get("alternatives"):
                st.caption("Alternatives: " + "; ".join(a["product"] for a in step_["alternatives"]))


def results_step() -> None:
    if ss.result is None:
        run_analysis()
    result = ss.result
    st.title("Your results")
    for note in result["notes"]:
        st.warning(note)

    sa = result["skin_analysis"]
    st.subheader("Skin analysis")
    with st.container(border=True):
        left, right = st.columns([1, 2], vertical_alignment="center")
        with left:
            if ss.get("result_photo"):
                st.image(ss.result_photo, alt="Your face photo", width="stretch")
            else:
                st.caption("No photo analysed")
        with right:
            st.metric("Skin type", sa["skin_type"].title())
            st.write(sa["skin_type_text"])
        st.markdown("**Skin concerns**")
        if sa["concerns"]:
            for c in sa["concerns"]:
                badge = ":red-badge[Detected]" if c["status"] == "detected" else ":orange-badge[Possible]"
                st.markdown(f"{badge} {c['text'].split(': ', 1)[1]}")
        else:
            st.markdown(":green-badge[Clear] " + sa["no_concerns_text"])

    g = sa["age_guidance"]
    st.subheader(f"Skin care for your age ({g['label'].replace('Ages ', '')})")
    with st.container(border=True):
        st.markdown(f"**Common concerns:** {g['common_concerns']}")
        st.markdown(f"**Skin changes:** {g['skin_changes']}")
        st.markdown(f"**Care focus:** {g['care_focus']}")

    env = result["usce"]["environment"]
    st.subheader("Your location and today's weather")
    with st.container(border=True):
        if env:
            loc = sa["location"]
            coords = f"{env['latitude']:.4f}, {env['longitude']:.4f}" if env.get("latitude") is not None else ""
            named = coords and not env["location"].endswith(("E", "W"))  # a place name, not just coordinates
            st.markdown(f":material/location_on: {env['location']}" + (f"  \n:gray[{coords}]" if named else ""))
            if loc.get("date_label"):
                st.markdown(f":material/calendar_today: {loc['date_label']}")
            with st.container(horizontal=True):
                st.metric("Avg temperature", f"{env['temperature_c']:g} °C", border=True)
                st.metric("Avg humidity", f"{env['humidity_pct']:g}%", border=True)
                if env.get("uv_index") is not None:
                    st.metric("Avg UV index", f"{env['uv_index']:g}", border=True,
                              help="Average over daylight hours today.")
                if env.get("uv_max") is not None:
                    st.metric("Peak UV index", f"{env['uv_max']:g}", border=True,
                              help="Today's maximum. Sun-protection advice uses the peak (WHO UV Index).")
            if env.get("source"):
                st.caption(f"Source: {env['source']}.")
        else:
            st.caption(sa["environment_text"])

    st.subheader("Advice for you")
    st.info("\n".join(f"- {tip}" for tip in result["advice"]), icon=":material/lightbulb:")

    routine = result["routine"]
    st.subheader("Your routine")
    morning, night, weekly = st.tabs(["Morning", "Night", "Weekly"])
    with morning:
        for s in routine["morning"]:
            product_card(s)
    with night:
        for s in routine["evening"]:
            product_card(s)
    with weekly:
        if not routine["weekly"]:
            st.caption("No weekly extras suggested.")
        for s in routine["weekly"]:
            product_card(s, weekly=True)
    for c in routine["conflicts"]:
        st.warning(f"{c['reason']}: {c['advice']} ({' + '.join(c['products'])}).", icon=":material/sync_problem:")

    st.divider()
    with st.container(horizontal=True, vertical_alignment="center"):
        st.button("Back", icon=":material/arrow_back:", on_click=wizard.go_back, key="back")
        st.space("stretch")
        st.download_button("Download results (PDF)", ss.result_pdf, file_name="skiniq_results.pdf",
                           mime="application/pdf", icon=":material/download:", on_click="ignore")
        st.button("Start again", icon=":material/restart_alt:", on_click=wizard.restart, key="restart")
    st.caption(result["disclaimer"])


if step == wizard.PHOTO:
    photo_step()
elif step == wizard.RESULTS:
    results_step()
else:
    question_step()
