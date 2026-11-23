"""Step-by-step flow of the "Analyse my skin" page: state, steps and per-step validation.

Widget values use persist_state="session", so answers survive Back/Next. The accepted
photo and the detected location are copied into plain session-state keys because the
uploader, camera and location components are only shown on some steps.
"""

from __future__ import annotations

import streamlit as st

from skiniq.questionnaire import QUESTIONS, SECTION_A, SECTION_B, SECTION_C, SECTION_D

STEPS = (
    ("Your photo", ()),
    ("Section A", (SECTION_A,)),
    ("Section B", (SECTION_B,)),
    ("Sections C & D", (SECTION_C, SECTION_D)),
    ("Your results", ()),
)
PHOTO, RESULTS = 0, len(STEPS) - 1

ANSWER_KEYS = [f"q_{q.id}" for q in QUESTIONS] + [f"q_{q.other_id}" for q in QUESTIONS if q.other_id]


def init() -> None:
    ss = st.session_state
    ss.setdefault("step", PHOTO)
    ss.setdefault("photo_bytes", None)   # accepted photo (passed the quality check)
    ss.setdefault("photo_nonce", 0)      # bumps to reset the uploader / camera
    ss.setdefault("location", None)      # last result of the location component
    ss.setdefault("result", None)
    ss.setdefault("result_pdf", None)
    ss.setdefault("nav_error", None)


def restart() -> None:
    """Clear answers, photo and results; keep the detected location."""
    ss = st.session_state
    for key in ANSWER_KEYS + ["photo_source"]:
        ss.pop(key, None)
    ss.photo_bytes = None
    ss.photo_nonce = ss.get("photo_nonce", 0) + 1
    ss.result = ss.result_pdf = ss.nav_error = None
    ss.step = PHOTO


def clear_photo() -> None:
    st.session_state.photo_bytes = None
    st.session_state.photo_nonce += 1


def is_male() -> bool:
    return st.session_state.get("q_gender") == "male"


def answers() -> dict:
    """Current answers keyed by question id (multi-selects default to 'none'/empty)."""
    ss = st.session_state
    out = {q.id: ss.get(f"q_{q.id}") for q in QUESTIONS}
    for q in QUESTIONS:
        if q.other_id:
            out[q.other_id] = (ss.get(f"q_{q.other_id}") or "").strip()
    if is_male():
        out["pregnant"] = "no"
    out["known_conditions"] = out["known_conditions"] or ["none"]
    out["goals"] = out["goals"] or []
    out["avoid_ingredients"] = out["avoid_ingredients"] or ["none"]
    return out


def missing(step: int) -> list[str]:
    """Question numbers still unanswered on a step (single-choice questions are required)."""
    a = answers()
    out = []
    for q in QUESTIONS:
        if q.section not in STEPS[step][1] or q.multi:
            continue
        if q.id == "pregnant" and is_male():
            continue
        if a.get(q.id) is None:
            out.append(q.number)
    return out


def step_problem(step: int) -> str | None:
    if step == PHOTO:
        if st.session_state.get("photo_source") != "Skip" and not st.session_state.photo_bytes:
            return "Add a photo that passes the check, or choose Skip to continue without one."
        return None
    gaps = missing(step)
    if gaps:
        return "Please answer question" + ("s " if len(gaps) > 1 else " ") + ", ".join(gaps) + "."
    return None


def go_next() -> None:
    ss = st.session_state
    problem = step_problem(ss.step)
    ss.nav_error = problem
    if problem is None:
        if ss.step == RESULTS - 1:
            ss.result = ss.result_pdf = None  # analyse on the results step
        ss.step = min(ss.step + 1, RESULTS)


def go_back() -> None:
    st.session_state.nav_error = None
    st.session_state.step = max(st.session_state.step - 1, PHOTO)
