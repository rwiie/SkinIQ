# SkinIQ

Final Year Project: Personalized Korean Skincare Advisory Platform (Streamlit, Python 3.11+).
See README.md for the pipeline, project layout and roadmap.

## Conventions

- Library code lives in the `skiniq/` package; the Streamlit app in `app/` only handles UI and calls `skiniq.pipeline`.
- Class labels (skin types, conditions) are defined once in `skiniq/config.py`; import them, don't redefine.
- Knowledge-base rules: one module per knowledge base in `skiniq/rules/`. Every rule needs a pytest case in `tests/`, including edge-case user profiles (proposal 3.8).
- Secrets (OpenWeatherMap key) come from `st.secrets` or the `OPENWEATHER_API_KEY` env var. Never hard-code or commit them.
- Raw images and model weights are not committed (`data/raw/`, `models/` are gitignored).
- Run `pytest` before committing.

## Skills to use

| Work | Skill |
|---|---|
| Streamlit UI, caching, deployment | `developing-with-streamlit` |
| Dataset prep, EfficientNet-V2 training (timm) | `huggingface-vision-trainer` |
| TF-IDF, cosine similarity, evaluation metrics, bias testing | `scikit-learn` |
| Rule and pipeline tests | `python-testing-patterns` |
| Any bug or failing test | `systematic-debugging` |
| Finding more skills | `find-skills` |
