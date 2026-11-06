# SkinIQ — Personalized Korean Skincare Advisory Platform

Final Year Project, Bachelor of Information Systems (Hons.) Intelligent Systems Engineering, UiTM.

SkinIQ analyses a facial image with an EfficientNet-V2 CNN (skin type + skin conditions),
fuses the result with the user's profile and real-time weather (temperature, humidity,
UV index), filters Korean skincare products through knowledge-based rules, ranks them
with TF-IDF + cosine similarity, and explains every recommendation.

## Pipeline

```
facial image ──► EfficientNet-V2 ──┐
user questionnaire ────────────────┼─► USCE (fusion) ─► KB rule filters ─► TF-IDF ranking ─► explanations ─► Streamlit UI
OpenWeatherMap (temp/humidity/UV) ─┘
```

| Output | Classes |
|---|---|
| Skin type | normal, oily, dry, combination |
| Skin conditions | acne, blackheads, hyperpigmentation, redness, enlarged pores, wrinkles |

## Project layout

| Path | Purpose | Proposal section |
|---|---|---|
| `skiniq/vision/` | EfficientNet-V2 loading and inference | 3.6.1 |
| `skiniq/environment.py` | OpenWeatherMap client | 3.7 |
| `skiniq/fusion.py` | Unified Skin Context Embedding (USCE) | 3.6.2 |
| `skiniq/rules/` | Knowledge-base condition gates | 3.6.3 |
| `skiniq/recommender.py` | TF-IDF + cosine similarity ranking | 3.6.4 |
| `skiniq/explain.py` | Explanations (XAI) for each recommendation | 3.6.5 |
| `skiniq/pipeline.py` | End-to-end pipeline, JSON output | 3.6.5 |
| `app/streamlit_app.py` | Streamlit prototype | 3.7 |
| `training/` | Data preparation and model training | 3.5, 3.6.1 |
| `tests/` | Rule validation, recommender and pipeline tests | 3.8 |
| `data/` | Datasets (raw images are not committed) | 3.5 |
| `models/` | Trained model weights (not committed) | 3.6.1 |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
```

Add your OpenWeatherMap key to `.streamlit/secrets.toml` (copy `.streamlit/secrets.toml.example`).
Never commit that file.

## Run

```bash
streamlit run app/streamlit_app.py
```

## Test

```bash
pytest
```

## Development roadmap

1. Project setup ✅
2. Data preparation (3.5)
3. Model training (3.6.1)
4. USCE fusion layer (3.6.2)
5. Knowledge-base rules (3.6.3)
6. TF-IDF recommendation ranking (3.6.4)
7. Explanations + pipeline integration (3.6.5)
8. Streamlit prototype (3.7)
9. Debugging and code review
10. Testing and evaluation (3.8)
11. Deployment
12. Report and presentation
