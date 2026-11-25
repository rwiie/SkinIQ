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
| Analyse my skin | Step-by-step pages with **Back** (bottom left) and **Next** (bottom right): 1. Photo (upload or guided camera; a **Skip** button sits left of Next) and automatic location · 2. Section A · 3. Section B · 4. Sections C & D, then **Analyse my skin** · 5. **Your results** (one scrolling page) |
| About | How SkinIQ works, data sources, limitations and disclaimer |

The questions come from the updated questionnaire in `skiniq/questionnaire.py` (Q1–12, sections A–D).
Single-choice questions are required: "Next" lists any that are missing by number. Answers are kept when going
Back (`persist_state="session"`). The pregnancy question appears under Q2 only when the answer is Female (it is an unnumbered follow-up, so known conditions is Q3). Choosing "Other" in Q3
(known conditions) or Q12 (ingredients to avoid) shows a text box: a typed condition gets the same precautions as
the listed medical conditions, and typed ingredients are standardised and excluded from the routine (rule
`S2_avoid_custom`).

**Your results:** skin type explained in plain sentences (not the questions), skin concerns, skin care for your
age group, your location with today's date, the weather boxes with their source, advice, the routine in Morning / Night /
Weekly tabs with "Why this product", and **Download results (PDF)** (`skiniq/report.py`). "Start again"
clears the answers.

**Product photos:** each product card shows the product's photo from INKEEDecoder (formerly INCIDecoder),
with a "Photo: INKEEDecoder" link to its page; products without a photo show a placeholder. The links are in
`data/knowledge/product_images.csv`, built once with `python tools/fetch_product_images.py` (robots.txt
allows product pages; about one request per second; resumable). A product is matched by its page address or
the site's search, and only when the page's own title names the same product: every word of the catalogue
name present with at most two extra words, or one word of a long name dropped with nothing added (a
sub-brand such as "Dermalogy"). A swapped word means another product or variant, so "Moisturizing Foam
Cleanser" is never used for "Moisturizing Cream Cleanser" nor "Sheet Mask (Lime)" for "Sheet Mask (Olive)".
The page title is checked (not just the search result name or address) because some pages on the site were
renamed; `--recheck` re-validates every stored match and the CSV keeps each page's title.
When the results are shown, the routine's photos are downloaded (in parallel, kept in memory) and tidied by
`tidy_photo` (`skiniq/product_images.py`): a plain white/grey background is made transparent (near-white
pixels connected to the photo's border, stopped at the product's outline so white bottles and labels are kept),
the product is trimmed and centred on a square with the same margin, so every card and the PDF show the
products at the same size. Photos with a non-plain background are trimmed and centred but keep it.

All product photos can be browsed by category in the [product photo gallery](gallery/README.md)
(rebuild it with `python tools/make_gallery.py` after the photo list changes).

## Photo: guided camera and quality check

- **Same checks for camera and upload** (`skiniq/vision/quality.py`, `skiniq/vision/face_models.py`; details,
  models, licences and evaluation in [photo_check.md](photo_check.md)). A photo is accepted only if it is a real
  colour photo (not a drawing) of exactly one face that is framed **head and shoulders** like the camera outline,
  **straight on**, with **both eyes open**, **no glasses or sunglasses**, **no mask** (face mask, sheet or clay mask),
  **no hand** on the face, the **forehead uncovered**, **evenly lit** (not dark, blown out or half in shadow), sharp,
  and at least 240 px. Face parts are checked with a trained face-parsing model, hands with MediaPipe's palm and hand
  models, drawings with CLIP. Each problem has its own message telling the user what to fix.
- **Guided camera** (`app/components/camera.py`, a Streamlit custom component): live video with a head-and-shoulders
  outline. Face detection runs in the browser (YuNet via ONNX Runtime Web on the CPU, no WebGL needed; bundled in
  `app/static/facedetect/` and served with `server.enableStaticServing`). Every frame runs the same checks with the
  same thresholds (passed from `quality.settings()`): YuNet every frame, face parsing and the palm detector about
  every 0.6 s (in the browser, CPU). The outline stays red ("Loading face checks…") until those models have loaded,
  is **red** with a hint while anything is wrong, and turns **green** only after three frames in a row pass with a
  fresh face-part result; only then is "Take photo" enabled. After a capture the photo stays in the camera frame
  with **Retake**, which restarts the live camera directly. The captured photo is exactly the picture inside the
  frame and is checked again on the server.
- **Upload**: the same check runs as soon as a photo is uploaded; a rejected photo shows "Photo not accepted"
  with every problem found, and the user cannot continue with it (they can upload another or press Skip).

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
