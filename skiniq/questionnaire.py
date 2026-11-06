"""User questionnaire (proposal Appendix A) and skin type scoring.

The questions are defined here once so the Streamlit UI can render them and the
pipeline can read the answers by id.

Skin type is assessed from the questionnaire: the image model could not learn
skin type from photos (31.6% test accuracy, near chance), while oiliness,
tightness and flakiness are what skin type actually measures. Question 7b
("where does oil appear?") is added to Appendix A because nothing else separates
combination skin (oily T-zone only) from oily skin.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from skiniq.config import SKIN_TYPES


@dataclass(frozen=True)
class Question:
    id: str
    section: str
    number: str
    text: str
    options: dict[str, str]  # answer key -> label shown to the user
    multi: bool = False
    help: str = ""


QUESTIONS: tuple[Question, ...] = (
    # Section A: Personal characteristics
    Question("age_group", "A. Personal characteristics", "1", "Which age group do you belong to?",
             {"18_24": "18–24", "25_34": "25–34", "35_44": "35–44", "45_54": "45–54", "55_plus": "55 and above"}),
    Question("pregnant", "A. Personal characteristics", "2", "Are you currently pregnant or breastfeeding?",
             {"yes": "Yes", "no": "No"}),
    # Section B: Skin profile
    Question("known_conditions", "B. Skin profile", "3", "Do you have any known skin conditions?",
             {"acne": "Acne", "rosacea": "Rosacea", "eczema": "Eczema", "psoriasis": "Psoriasis",
              "hyperpigmentation": "Hyperpigmentation", "melasma": "Melasma", "dermatitis": "Dermatitis",
              "seborrheic_dermatitis": "Seborrheic dermatitis", "none": "None known"},
             multi=True),
    Question("sensitivity", "B. Skin profile", "4", "How would you describe your skin sensitivity?",
             {"not": "Not sensitive", "slightly": "Slightly sensitive",
              "moderately": "Moderately sensitive", "highly": "Highly sensitive"}),
    Question("reacts_to_products", "B. Skin profile", "5",
             "Does your skin easily react to skincare products (e.g., redness, burning, itching, or irritation)?",
             {"yes": "Yes", "sometimes": "Sometimes", "no": "No"}),
    # Section C: Skin behaviour (used for skin type)
    Question("tightness", "C. Skin behaviour", "6", "After washing your face, does your skin feel tight or dry?",
             {"often": "Yes, often", "sometimes": "Sometimes", "rarely": "Rarely", "never": "Never"}),
    Question("oiliness", "C. Skin behaviour", "7", "How oily does your skin become throughout the day?",
             {"very_oily": "Very oily", "slightly_oily": "Slightly oily", "balanced": "Balanced", "dry": "Dry"}),
    Question("oil_location", "C. Skin behaviour", "7b", "Where does your skin get oily or shiny?",
             {"t_zone": "Only my forehead, nose and chin (T-zone)", "all_over": "All over my face",
              "nowhere": "It doesn't get oily", "not_sure": "Not sure"},
             help="Added to Appendix A: separates combination skin from oily skin."),
    Question("flakiness", "C. Skin behaviour", "8", "Do you frequently experience dry patches or flakiness?",
             {"yes": "Yes", "sometimes": "Sometimes", "no": "No"}),
    Question("breakouts", "C. Skin behaviour", "9", "How often do you experience breakouts or pimples?",
             {"frequently": "Frequently", "occasionally": "Occasionally", "rarely": "Rarely", "never": "Never"}),
    # Section D: Skincare goals
    Question("goals", "D. Skincare goals", "10", "What are your main skincare goals?",
             {"hydration": "Hydration", "acne_control": "Acne control", "oil_control": "Oil control",
              "brightening": "Brightening", "anti_aging": "Anti-aging",
              "sensitivity_reduction": "Sensitivity reduction", "pigmentation_reduction": "Pigmentation reduction",
              "sun_protection": "Sun protection"},
             multi=True),
    # Section E: Ingredient preferences and restrictions
    Question("avoid_ingredients", "E. Ingredient preferences & restrictions", "11",
             "Do you prefer to avoid any of the following ingredients?",
             {"fragrance": "Fragrance", "alcohol": "Alcohol", "essential_oils": "Essential oils",
              "retinol": "Retinol", "salicylic_acid": "Salicylic acid",
              "chemical_sunscreens": "Chemical sunscreens", "none": "None"},
             multi=True),
)

QUESTIONS_BY_ID = {q.id: q for q in QUESTIONS}

# Evidence points each answer gives to each skin type (Section C).
# Oiliness and where it appears carry the most weight; tightness and flakiness
# separate dry from normal and add to combination when oil is also present.
SKIN_TYPE_EVIDENCE: dict[str, dict[str, dict[str, float]]] = {
    "oiliness": {
        "very_oily": {"oily": 3, "combination": 1},
        "slightly_oily": {"oily": 1, "normal": 1, "combination": 1},
        "balanced": {"normal": 3},
        "dry": {"dry": 3},
    },
    "oil_location": {
        "t_zone": {"combination": 4},
        "all_over": {"oily": 2},
        "nowhere": {"normal": 1, "dry": 1},
        "not_sure": {},
    },
    "tightness": {
        "often": {"dry": 2, "combination": 0.5},
        "sometimes": {"dry": 1, "combination": 0.5},
        "rarely": {"normal": 1, "oily": 0.5},
        "never": {"oily": 1, "normal": 1},
    },
    "flakiness": {
        "yes": {"dry": 2, "combination": 1},
        "sometimes": {"dry": 1, "combination": 0.5},
        "no": {"normal": 0.5, "oily": 0.5},
    },
}
SKIN_TYPE_QUESTIONS = tuple(SKIN_TYPE_EVIDENCE)

# Combination skin is "oily in some areas but normal or dry in others" (proposal 2.1.2),
# so oil together with clear dryness signs adds extra combination evidence.
MIXED_SIGNS_POINTS = 2.0
MIXED_SIGNS_REASON = "Oily areas together with dry patches or tightness"


def _has_mixed_signs(answers: dict) -> bool:
    oily = answers["oiliness"] in ("very_oily", "slightly_oily") or answers["oil_location"] in ("t_zone", "all_over")
    dry = answers["flakiness"] == "yes" or answers["tightness"] == "often"
    return oily and dry


@dataclass
class SkinTypeAssessment:
    label: str
    scores: dict[str, float]  # share of evidence per skin type, sums to 1
    reasons: list[str] = field(default_factory=list)  # answers supporting the label, for explanations


def assess_skin_type(answers: dict) -> SkinTypeAssessment:
    """Score skin type from Section C answers.

    Raises ValueError if a required answer is missing or not a valid option.
    """
    points: dict[str, float] = defaultdict(float)
    support: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for qid, table in SKIN_TYPE_EVIDENCE.items():
        answer = answers.get(qid)
        if answer not in table:
            raise ValueError(f"'{qid}' needs one of {sorted(table)}, got {answer!r}")
        q = QUESTIONS_BY_ID[qid]
        for skin_type, pts in table[answer].items():
            points[skin_type] += pts
            support[skin_type].append((pts, f"{q.text} — {q.options[answer]}"))

    if _has_mixed_signs(answers):
        points["combination"] += MIXED_SIGNS_POINTS
        support["combination"].append((MIXED_SIGNS_POINTS, MIXED_SIGNS_REASON))

    total = sum(points.values())
    scores = {t: round(points[t] / total, 4) for t in SKIN_TYPES}
    # Ties go to the earlier type in SKIN_TYPES order (normal, oily, dry, combination)
    label = max(SKIN_TYPES, key=lambda t: points[t])
    reasons = [text for _, text in sorted(support[label], key=lambda s: s[0], reverse=True)]
    return SkinTypeAssessment(label=label, scores=scores, reasons=reasons)
