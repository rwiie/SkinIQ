# SkinIQ Streamlit prototype (proposal 3.7)

Code: `app/streamlit_app.py` (navigation), `app/app_pages/analyse.py`, `app/app_pages/about.py`,
`app/services.py` (cached model, product index, weather). Tests: `tests/test_app.py`, `tests/test_weather.py`.
Screenshots: [`screenshots/`](screenshots/).

## Run it on your computer

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

1. **Model weights:** put the multi-label skin concern model at `models/skin_concerns/model.pt`
   (download `model.pt` from `My Drive/SkinIQ/models/skin_concerns/`). Without it the app still works: the photo
   is skipped with a note.
2. **Weather key (optional):** copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your
   OpenWeatherMap key. Without a key, Open-Meteo is used. Never commit `secrets.toml`.

## Pages

| Page | Content |
|---|---|
| Analyse my skin | 1. Photo: upload, webcam (`st.camera_input`) or skip · 2. The Appendix A questionnaire (with question 7b), generated from `skiniq/questionnaire.py` · 3. Location · Results in the four sections of proposal 3.7: skin analysis, advice, routine (morning / evening / weekly tabs), and "Why this product" for every product · Download results as JSON |
| About | How SkinIQ works, data sources, limitations and disclaimer |

The questionnaire is one form, so the analysis runs only when "Analyse my skin" is pressed. All single-choice
questions are required; missing ones are listed by number.

## Weather

| | Location, temperature, humidity | UV index |
|---|---|---|
| With an OpenWeatherMap key | OpenWeatherMap (geocoding + current weather) | Open-Meteo |
| Without a key | Open-Meteo | Open-Meteo |

OpenWeatherMap's free plan has no UV index (it needs the paid One Call API), so UV always comes from Open-Meteo
(free, no key). The UV used is **today's maximum**, since current UV is 0 at night and the morning routine should
plan for the day. Requests time out after 8 s with one retry (timeouts, HTTP 429); results are cached for 30
minutes. If weather fails the analysis continues without weather adjustments and says why.

## Performance

The model and the TF-IDF product index are loaded once per server (`st.cache_resource`); weather is cached per
place (`st.cache_data`, 30 min). After the first load an analysis takes a few seconds on CPU.
