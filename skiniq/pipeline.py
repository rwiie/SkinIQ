"""End-to-end pipeline: photo + questionnaire + weather -> JSON results (proposal 3.6.5).

    result = analyse(answers, image=pil_image, environment=EnvironmentData(...))
    json.dumps(result)

Steps: CNN skin concerns (whole image, multi-label) -> USCE fusion (3.6.2) ->
knowledge-base rules (3.6.3) -> TF-IDF ranking (3.6.4) -> morning/evening routine with
compatibility checks -> explanations (3.6.5). The photo and the weather are optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image

from skiniq.config import MODELS_DIR
from skiniq.environment import EnvironmentData
from skiniq.explain import DISCLAIMER, STEP_LABELS, explain_product, explain_skin_analysis, personalised_advice
from skiniq.fusion import USCE, ImageAnalysis, build_usce
from skiniq.products import Product
from skiniq.recommender import Recommendation, rank_products
from skiniq.rules import check_routine

SKIN_CONCERN_MODEL = MODELS_DIR / "skin_concerns" / "model.pt"
CANDIDATES_PER_STEP = 8
ALTERNATIVES = 2
MORNING = ("cleanser", "toner", "treatment", "moisturizer", "sunscreen")
EVENING = ("cleanser", "toner", "treatment", "moisturizer")
PHOTOSENSITISING = ("retinoid", "aha")


@lru_cache(maxsize=1)
def load_classifier(path: str | Path = SKIN_CONCERN_MODEL):
    """The multi-label skin concern model, or None if the weights are not available."""
    if not Path(path).exists():
        return None
    from skiniq.vision.model import SkinClassifier

    return SkinClassifier.load(path)


def analyse_image(image: Image.Image, classifier) -> ImageAnalysis:
    """Whole-image multi-label analysis (face regions tested worse, see training/README.md)."""
    probs = classifier.predict(image.convert("RGB"))
    return ImageAnalysis(conditions=probs, independent=bool(getattr(classifier, "multilabel", False)),
                         thresholds=getattr(classifier, "thresholds", None))


# ---- routine assembly -------------------------------------------------------------------

@dataclass
class RoutineStep:
    step: str
    pick: Recommendation
    alternatives: list[Recommendation]


def _allowed(rec: Recommendation, routine: str) -> bool:
    # Retinoids and AHAs increase sun sensitivity: evening only (FDA, 2005)
    return not (routine == "morning" and any(rec.product.has(g) for g in PHOTOSENSITISING))


def build_routine(ranked: dict[str, list[Recommendation]], steps: tuple[str, ...], routine: str,
                  ) -> tuple[list[RoutineStep], list[dict]]:
    """Pick the best product per step so that no two products in the routine conflict."""
    chosen: list[RoutineStep] = []
    for step in steps:
        options = [r for r in ranked.get(step, []) if _allowed(r, routine)]
        if not options:
            continue
        picked = next((r for r in options
                       if not check_routine([c.pick.product for c in chosen] + [r.product])), options[0])
        alts = [r for r in options if r is not picked][:ALTERNATIVES]
        chosen.append(RoutineStep(step, picked, alts))
    conflicts = [{"products": list(c.products), "reason": c.reason, "advice": c.advice, "source": c.source}
                 for c in check_routine([c.pick.product for c in chosen])]
    return chosen, conflicts


def _routine_json(steps: list[RoutineStep], usce: USCE) -> list[dict]:
    return [{
        "step": s.step,
        "label": STEP_LABELS[s.step],
        "product": explain_product(s.pick, usce),
        "alternatives": [{"product": a.product.title, "score": a.score} for a in s.alternatives],
    } for s in steps]


def _weekly(ranked: dict[str, list[Recommendation]], usce: USCE, evening: list[Product]) -> list[dict]:
    """Optional weekly extras: one exfoliator (if compatible with the evening routine) and one mask."""
    extras = []
    for step, freq in (("exfoliator", "1–2 times a week, in the evening"), ("mask", "1–3 times a week")):
        options = ranked.get(step, [])
        if step == "exfoliator":
            options = [r for r in options if not check_routine(evening + [r.product])] or []
            if usce.is_sensitive:
                freq = "once a week at most, in the evening"
        if options:
            extras.append({"step": step, "label": STEP_LABELS[step], "how_often": freq,
                           "product": explain_product(options[0], usce)})
    return extras


# ---- main entry point -----------------------------------------------------------------------

def analyse(answers: dict, image: Image.Image | None = None, environment: EnvironmentData | None = None,
            classifier=None) -> dict:
    """Run the full pipeline and return a JSON-serialisable result.

    `classifier` defaults to the trained multi-label model; if it is unavailable the photo
    is skipped and a note is added (the questionnaire still drives the result).
    """
    notes = []
    image_analysis = None
    if image is not None:
        classifier = classifier or load_classifier()
        if classifier is None:
            notes.append("The photo could not be analysed (model not available); results use your answers only.")
        else:
            image_analysis = analyse_image(image, classifier)
    if environment is None:
        notes.append("Weather data was not available; no environmental adjustments were made.")

    usce = build_usce(answers, image=image_analysis, environment=environment)
    ranked = rank_products(usce, per_step=CANDIDATES_PER_STEP)
    morning, am_conflicts = build_routine(ranked, MORNING, "morning")
    evening, pm_conflicts = build_routine(ranked, EVENING, "evening")

    return {
        "skin_analysis": explain_skin_analysis(usce),
        "advice": personalised_advice(usce),
        "routine": {
            "morning": _routine_json(morning, usce),
            "evening": _routine_json(evening, usce),
            "weekly": _weekly(ranked, usce, [s.pick.product for s in evening]),
            "conflicts": am_conflicts + pm_conflicts,
        },
        "notes": notes,
        "disclaimer": DISCLAIMER,
        "usce": usce.to_dict(),
    }
