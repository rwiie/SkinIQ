"""SkinIQ Streamlit prototype (proposal 3.7). Full UI is built in step 8."""

import sys
from pathlib import Path

import streamlit as st

# `streamlit run app/streamlit_app.py` only puts app/ on sys.path; add the repo root
# so the skiniq package is importable locally and on Streamlit Community Cloud.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skiniq.config import SKIN_CONDITIONS, SKIN_TYPES

st.set_page_config(page_title="SkinIQ", page_icon="🌸", layout="centered")

st.title("SkinIQ")
st.caption("Personalized Korean Skincare Advisory Platform")

st.info("Prototype under development. Skin analysis and recommendations are coming in later steps.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Skin types")
    st.write(", ".join(t.title() for t in SKIN_TYPES))
with col2:
    st.subheader("Skin conditions")
    st.write(", ".join(c.replace("_", " ").title() for c in SKIN_CONDITIONS))
