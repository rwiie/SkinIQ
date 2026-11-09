import streamlit as st

st.title("About SkinIQ")
st.caption("Personalized Korean Skincare Advisory Platform — Final Year Project, UiTM")

st.markdown("""
SkinIQ combines three kinds of information to suggest a Korean skincare routine:

- **Your photo:** an EfficientNet-V2 model checks for six skin concerns (acne, blackheads, dark spots,
  redness, enlarged pores and wrinkles). It can find several at once, or none.
- **Your answers:** a short questionnaire gives your skin type, sensitivity, goals and ingredients to avoid.
  Skin type comes from your answers because photos alone could not predict it reliably.
- **Your weather:** temperature, humidity and today's UV index for your location.
""")

with st.container(border=True):
    st.markdown("**How products are chosen**")
    st.markdown("""
1. **Safety rules** remove unsuitable products: for example retinoids during pregnancy, ingredients you avoid,
   and fragrance for rosacea or eczema. 34 rules from four knowledge bases.
2. **Matching:** the remaining 1,676 products are ranked by how well their ingredients match your needs
   (TF-IDF and cosine similarity), plus bonuses and penalties from the rules.
3. **Routine:** the best product per step, with retinoids and AHAs kept out of the morning and clashing
   products swapped out.
""")

st.markdown("**Data sources**")
st.markdown("""
- Facial skin type images (Olimov, Kaggle) and skin condition images (Kaggle)
- Skin-Problem-Detection-Relabel-Clean3 (Roboflow Universe, CC BY 4.0)
- Korean skincare product data (Skinsort via Kaggle)
- Weather: OpenWeatherMap and Open-Meteo
""")

st.warning(
    "SkinIQ gives general skincare guidance, not a medical diagnosis. The photo model can be wrong, especially "
    "with makeup, filters or poor lighting. See a dermatologist for persistent or severe skin problems.",
    title="Limitations",
)
