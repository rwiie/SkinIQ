# SkinIQ Streamlit prototype (proposal 3.7)

Code: `app/streamlit_app.py` (navigation), `app/app_pages/analyse.py`, `app/app_pages/about.py`,
`app/services.py` (cached model, product index, weather). Tests: `tests/test_app.py`, `tests/test_weather.py`.
Screenshots: [`screenshots/`](screenshots/).

## Run it on your computer

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Open it at **http://localhost:8501**: the camera and location need a secure page, and `localhost` counts as one.

1. **Model weights:** downloaded automatically on first use (82 MB, from the project's Google Drive, checked by
   SHA-256) to `models/skin_concerns/model.pt`. If the download fails, the photo is skipped and the results note
   says why.
2. **Weather key (optional):** copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add an
   OpenWeatherMap key; it is used only as a fallback. Never commit `secrets.toml`.

## Pages

| Page | Content |
|---|---|
| Welcome | "Welcome to SkinIQ: Personalized Korean Skincare Advisory Platform" and a **Let's Start** button that opens Analyse my skin |
| Analyse my skin | Step-by-step pages with **Back** (bottom left) and **Next** (bottom right): 1. Photo (upload, guided camera or skip) and automatic location · 2. Section A · 3. Section B · 4. Sections C & D, then **Analyse my skin** · 5. **Your results** (one scrolling page) |
| About | How SkinIQ works, data sources, limitations and disclaimer |

The questions come from the updated questionnaire in `skiniq/questionnaire.py` (Q1–12, sections A–D).
Single-choice questions are required: "Next" lists any that are missing by number. Answers are kept when going
Back (`persist_state="session"`). Male users are not asked the pregnancy question. Choosing "Other" in Q4
(known conditions) or Q12 (ingredients to avoid) shows a text box: a typed condition gets the same precautions as
the listed medical conditions, and typed ingredients are standardised and excluded from the routine (rule
`S2_avoid_custom`).

**Your results:** skin type explained in plain sentences (not the questions), skin concerns, skin care for your
age group, your location with today's date, the weather boxes with their source, advice, the morning, evening and
weekly routines with "Why this product", and **Download results (PDF)** (`skiniq/report.py`). "Start again"
clears the answers.

## Photo: guided camera and quality check

- **Guided camera** (`app/components/camera.py`, a Streamlit custom component): live video with a head-and-shoulders
  outline. Face detection runs in the browser (YuNet via ONNX Runtime Web on the CPU, no WebGL needed; bundled in `app/static/facedetect/` and
  served with `server.enableStaticServing`). The outline is **red** with a hint (no face, more than one face,
  centre your face, move closer/back, too dark/bright) and turns **green** when exactly one face is centred at
  the right distance in enough light; only then is "Take photo" enabled.
- **Quality check on every photo**, uploaded or taken (`skiniq/vision/quality.py`, YuNet on the server): rejected
  with a message if there is no face, more than one face, the face is too small, or the photo is blurry, too
  dark, overexposed or under 240 px. Thresholds were calibrated on the test photos: 88% of 150 real selfies are
  accepted; close-up skin photos without a face are rejected.

## Location and weather

- **Location is detected automatically** (`app/components/geolocation.py`): the browser's Geolocation API (GPS /
  Wi-Fi) gives the exact position after the user allows it once. The place name comes from BigDataCloud's free
  reverse geocoding, called from the browser. There is no location box. If access is blocked, a "Try again"
  button and instructions are shown, and the analysis runs without weather adjustments.
- **Today's averages at that point** (`fetch_weather_at`): Open-Meteo's hourly forecast for the local day,
  averaged: temperature, relative humidity, and UV index over daylight hours. The day's **peak UV** is also
  shown, and the sun-protection rules use it, because WHO UV categories describe the day's maximum (in Malaysia
  the daylight average is often about 4 on a day peaking at 10).
- **Fallback:** with an OpenWeatherMap key, its free 5-day/3-hour forecast gives today's average temperature
  and humidity (no UV) if Open-Meteo fails. One retry on timeout or rate limit; results cached 30 minutes on a
  ~100 m grid; the exact detected coordinates are always shown.
- The results show the place, coordinates, average temperature, humidity and UV, and peak UV.

## Performance

The model and the TF-IDF product index are loaded once per server (`st.cache_resource`); weather is cached per
place (`st.cache_data`, 30 min). After the first load an analysis takes a few seconds on CPU.
