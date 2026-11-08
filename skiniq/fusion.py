"""Unified Skin Context Embedding (USCE): multimodal data fusion (proposal 3.6.2).

Combines the three inputs into one structure used by the knowledge-base rules
(3.6.3), the TF-IDF ranking (3.6.4) and the explanations (3.6.5):

- Visual: CNN probabilities for the six skin conditions.
- Personal: questionnaire answers (Appendix A), including skin type.
- Contextual: temperature, humidity and UV index.

`USCE.vector()` is the fixed-length numeric embedding; `USCE.to_dict()` is the
JSON form passed to the app.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from skiniq.config import (
    CONDITION_DETECTED_THRESHOLD,
    CONDITION_POSSIBLE_THRESHOLD,
    SKIN_CONDITIONS,
    SKIN_TYPE_IMAGE_WEIGHT,
    SKIN_TYPES,
)
from skiniq.environment import EnvironmentData
from skiniq.questionnaire import QUESTIONS_BY_ID, assess_skin_type

# Questionnaire answer -> skin condition concern it indicates, with a score on the
# same 0-1 scale as the CNN probabilities. A known (diagnosed) condition is strong
# evidence; breakout frequency is weaker evidence of acne.
KNOWN_CONDITION_CONCERNS = {
    "acne": "acne",
    "hyperpigmentation": "hyperpigmentation",
    "melasma": "hyperpigmentation",
    "rosacea": "redness",
}
KNOWN_CONDITION_SCORE = 0.8
BREAKOUT_ACNE_SCORE = {"frequently": 0.7, "occasionally": 0.4, "rarely": 0.0, "never": 0.0}

# Known conditions that call for safety precautions (gentle, fragrance-free products).
MEDICAL_CONDITIONS = ("rosacea", "eczema", "psoriasis", "dermatitis", "seborrheic_dermatitis")

SENSITIVITY_LEVEL = {"not": 0, "slightly": 1, "moderately": 2, "highly": 3}
REACTION_SCORE = {"yes": 2 / 3, "sometimes": 1 / 3, "no": 0.0}
AGE_ORDER = ("18_24", "25_34", "35_44", "45_54", "55_plus")

GOALS = tuple(QUESTIONS_BY_ID["goals"].options)
AVOIDABLE = tuple(k for k in QUESTIONS_BY_ID["avoid_ingredients"].options if k != "none")
ENV_FLAGS = ("high_uv", "humid", "dry_air", "hot", "cold")

REQUIRED_SINGLE = ("age_group", "pregnant", "sensitivity", "reacts_to_products", "breakouts")
MULTI_SELECT = ("known_conditions", "goals", "avoid_ingredients")


@dataclass(frozen=True)
class ImageAnalysis:
    """CNN output for one facial image."""

    conditions: dict[str, float]               # probability per skin condition
    skin_type: dict[str, float] | None = None  # optional; ignored while SKIN_TYPE_IMAGE_WEIGHT is 0

    def __post_init__(self) -> None:
        _check_distribution(self.conditions, SKIN_CONDITIONS, "conditions")
        if self.skin_type is not None:
            _check_distribution(self.skin_type, SKIN_TYPES, "skin_type")


@dataclass
class Concern:
    condition: str
    score: float           # 0-1: max of photo probability and self-reported evidence
    status: str            # "detected", "possible" or "none"
    photo: float | None    # CNN probability (None if no photo)
    self_reported: float   # questionnaire evidence
    sources: list[str] = field(default_factory=list)


@dataclass
class USCE:
    skin_type: str
    skin_type_scores: dict[str, float]
    skin_type_source: str
    skin_type_reasons: list[str]
    concerns: dict[str, Concern]
    sensitivity: float               # 0-1
    is_sensitive: bool
    pregnant: bool
    age_group: str
    medical_conditions: list[str]
    goals: list[str]
    avoid_ingredients: list[str]
    environment: EnvironmentData | None
    has_photo: bool

    @property
    def detected_conditions(self) -> list[str]:
        return [c for c, v in self._concerns_by_score() if v.status == "detected"]

    @property
    def possible_conditions(self) -> list[str]:
        return [c for c, v in self._concerns_by_score() if v.status == "possible"]

    @property
    def env_flags(self) -> dict[str, bool]:
        return self.environment.flags if self.environment else {f: False for f in ENV_FLAGS}

    def _concerns_by_score(self):
        return sorted(self.concerns.items(), key=lambda kv: kv[1].score, reverse=True)

    # ---- numeric embedding --------------------------------------------------------
    @staticmethod
    def feature_names() -> list[str]:
        return [
            *(f"skin_type:{t}" for t in SKIN_TYPES),
            *(f"concern:{c}" for c in SKIN_CONDITIONS),
            "sensitivity",
            "pregnant",
            "age",
            *(f"goal:{g}" for g in GOALS),
            *(f"avoid:{a}" for a in AVOIDABLE),
            "env:temperature",
            "env:humidity",
            "env:uv_index",
            *(f"env:{f}" for f in ENV_FLAGS),
            "env:known",
        ]

    def vector(self) -> list[float]:
        """Fixed-length embedding, every value scaled to 0-1, in `feature_names()` order."""
        env = self.environment
        values = [
            *(self.skin_type_scores[t] for t in SKIN_TYPES),
            *(self.concerns[c].score for c in SKIN_CONDITIONS),
            self.sensitivity,
            float(self.pregnant),
            AGE_ORDER.index(self.age_group) / (len(AGE_ORDER) - 1),
            *(float(g in self.goals) for g in GOALS),
            *(float(a in self.avoid_ingredients) for a in AVOIDABLE),
            _clip01(env.temperature_c / 40) if env else 0.0,
            env.humidity_pct / 100 if env else 0.0,
            _clip01(env.uv_index / 11) if env else 0.0,
            *(float(self.env_flags[f]) for f in ENV_FLAGS),
            float(env is not None),
        ]
        return [round(v, 4) for v in values]

    def to_dict(self) -> dict:
        """JSON-ready form (proposal 3.6.5)."""
        d = asdict(self)
        d["concerns"] = {c: asdict(v) for c, v in self._concerns_by_score()}
        d["environment"] = (
            {**asdict(self.environment), "uv_level": self.environment.uv_level, "flags": self.environment.flags}
            if self.environment else None
        )
        d["detected_conditions"] = self.detected_conditions
        d["possible_conditions"] = self.possible_conditions
        d["vector"] = dict(zip(self.feature_names(), self.vector()))
        return d


def build_usce(
    answers: dict,
    image: ImageAnalysis | None = None,
    environment: EnvironmentData | None = None,
    skin_type_image_weight: float = SKIN_TYPE_IMAGE_WEIGHT,
) -> USCE:
    """Fuse questionnaire answers, the CNN output and weather into a USCE.

    Raises ValueError for missing or invalid answers. `image` and `environment`
    may be None (no photo, or the weather service is unavailable).
    """
    answers = _validate_answers(answers)

    # Skin type: questionnaire, optionally blended with the image model
    assessment = assess_skin_type(answers)
    scores = dict(assessment.scores)
    source = "questionnaire"
    if image is not None and image.skin_type is not None and skin_type_image_weight > 0:
        w = skin_type_image_weight
        scores = {t: round((1 - w) * scores[t] + w * image.skin_type[t], 4) for t in SKIN_TYPES}
        source = f"questionnaire ({1 - w:.0%}) + photo ({w:.0%})"
    skin_type = max(SKIN_TYPES, key=lambda t: scores[t])

    # Skin condition concerns: photo probability and self-reported evidence
    self_reported = {c: 0.0 for c in SKIN_CONDITIONS}
    for known in answers["known_conditions"]:
        if known in KNOWN_CONDITION_CONCERNS:
            c = KNOWN_CONDITION_CONCERNS[known]
            self_reported[c] = max(self_reported[c], KNOWN_CONDITION_SCORE)
    self_reported["acne"] = max(self_reported["acne"], BREAKOUT_ACNE_SCORE[answers["breakouts"]])

    concerns = {}
    for c in SKIN_CONDITIONS:
        photo = image.conditions[c] if image else None
        score = max(photo or 0.0, self_reported[c])
        sources = []
        if photo is not None and photo >= CONDITION_POSSIBLE_THRESHOLD:
            sources.append("photo")
        if self_reported[c] >= CONDITION_POSSIBLE_THRESHOLD:
            sources.append("questionnaire")
        concerns[c] = Concern(
            condition=c,
            score=round(score, 4),
            status=_status(score),
            photo=None if photo is None else round(photo, 4),
            self_reported=self_reported[c],
            sources=sources,
        )

    # Sensitivity and safety
    medical = [k for k in answers["known_conditions"] if k in MEDICAL_CONDITIONS]
    level = SENSITIVITY_LEVEL[answers["sensitivity"]]
    sensitivity = max(level / 3, REACTION_SCORE[answers["reacts_to_products"]])
    is_sensitive = level >= 2 or answers["reacts_to_products"] == "yes" or bool(medical)

    return USCE(
        skin_type=skin_type,
        skin_type_scores=scores,
        skin_type_source=source,
        skin_type_reasons=assessment.reasons,
        concerns=concerns,
        sensitivity=round(sensitivity, 4),
        is_sensitive=is_sensitive,
        pregnant=answers["pregnant"] == "yes",
        age_group=answers["age_group"],
        medical_conditions=medical,
        goals=list(answers["goals"]),
        avoid_ingredients=list(answers["avoid_ingredients"]),
        environment=environment,
        has_photo=image is not None,
    )


# ---- helpers ------------------------------------------------------------------------

def _status(score: float) -> str:
    if score >= CONDITION_DETECTED_THRESHOLD:
        return "detected"
    if score >= CONDITION_POSSIBLE_THRESHOLD:
        return "possible"
    return "none"


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def _check_distribution(probs: dict[str, float], labels: tuple[str, ...], name: str) -> None:
    if set(probs) != set(labels):
        raise ValueError(f"{name} must have exactly {list(labels)}, got {sorted(probs)}")
    if any(not 0 <= p <= 1 for p in probs.values()):
        raise ValueError(f"{name} probabilities must be between 0 and 1")
    if abs(sum(probs.values()) - 1) > 0.01:
        raise ValueError(f"{name} probabilities must sum to 1, got {sum(probs.values()):.3f}")


def _validate_answers(answers: dict) -> dict:
    """Check single-choice answers and normalise multi-select ones to lists without 'none'."""
    clean = dict(answers)
    for qid in REQUIRED_SINGLE:
        options = QUESTIONS_BY_ID[qid].options
        if clean.get(qid) not in options:
            raise ValueError(f"'{qid}' needs one of {sorted(options)}, got {clean.get(qid)!r}")
    for qid in MULTI_SELECT:
        chosen = clean.get(qid) or []
        if isinstance(chosen, str):
            chosen = [chosen]
        options = QUESTIONS_BY_ID[qid].options
        bad = [c for c in chosen if c not in options]
        if bad:
            raise ValueError(f"'{qid}' has invalid options {bad}")
        clean[qid] = [c for c in dict.fromkeys(chosen) if c != "none"]
    return clean
