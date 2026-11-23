import streamlit as st

from app.wizard import restart

st.space("large")
with st.container(horizontal_alignment="center"):
    st.markdown("# :material/face_retouching_natural: Welcome to SkinIQ", text_alignment="center")
    st.markdown("### Personalized Korean Skincare Advisory Platform", text_alignment="center")
    st.space("small")
    st.markdown(
        "Get a Korean skincare routine made for your skin. Add a face photo, answer a few short questions, "
        "and SkinIQ combines them with today's weather where you are to choose products and explain why "
        "each one suits you.",
        text_alignment="center",
    )
    st.space("small")
    with st.container(horizontal=True, horizontal_alignment="center"):
        for icon, text in ((":material/photo_camera:", "Photo analysis"),
                           (":material/checklist:", "Short questionnaire"),
                           (":material/partly_cloudy_day:", "Today's weather"),
                           (":material/spa:", "Explained routine")):
            st.markdown(f"{icon} {text}")
    st.space("medium")
    if st.button("Let's Start", type="primary", icon=":material/arrow_forward:", key="start"):
        restart()
        st.switch_page("app_pages/analyse.py")
    st.caption("Takes about 3 minutes. General skincare guidance, not a medical diagnosis.", text_alignment="center")
