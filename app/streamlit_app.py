"""SkinIQ Streamlit prototype (proposal 3.7). Run: streamlit run app/streamlit_app.py"""

import sys
from pathlib import Path

import streamlit as st

# `streamlit run app/streamlit_app.py` only puts app/ on sys.path; add the repo root
# so the skiniq package is importable locally and on Streamlit Community Cloud.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="SkinIQ", page_icon=":material/face_retouching_natural:", layout="centered")

page = st.navigation([
    st.Page("app_pages/analyse.py", title="Analyse my skin", icon=":material/face_retouching_natural:",
            default=True,
            help="Answer a short questionnaire, add a face photo and your location, and get a personalised "
                 "Korean skincare routine with reasons for every product."),
    st.Page("app_pages/about.py", title="About", icon=":material/info:",
            help="How SkinIQ works: the photo model, questionnaire, weather, safety rules and product ranking, "
                 "plus data sources and limitations."),
])
page.run()
