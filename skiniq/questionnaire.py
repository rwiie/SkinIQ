"""User questionnaire (updated Appendix A) and skin type scoring.

The questions are defined here once so the Streamlit UI can render them and the
pipeline can read the answers by id.

Skin type is assessed from the questionnaire: the image model could not learn
skin type from photos (31.6% test accuracy, near chance), while oiliness,
tightness and flakiness are what skin type actually measures. Question 8b
("where does your skin get oily?") separates combination skin (oily T-zone only)
from oily skin.
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
    other_id: str = ""  # answer id of the free-text box shown when "Other" is chosen


SECTION_A = "A. Personal information and skin background"
SECTION_B = "B. Skin characteristics and behaviour"
SECTION_C = "C. Skincare goals"
SECTION_D = "D. Ingredient preferences & restrictions"
SECTIONS = (SECTION_A, SECTION_B, SECTION_C, SECTION_D)

QUESTIONS: tuple[Question, ...] = (
    # Section A: Personal information and skin background
    Question("age_group", SECTION_A, "1", "Which age group do you belong to?",
             {"18_25": "18–25", "26_35": "26–35", "36_49": "36–49", "50_plus": "50 and above"}),
    Question("gender", SECTION_A, "2", "What is your gender?",
             {"male": "Male", "female": "Female"}),
    Question("pregnant", SECTION_A, "3", "Are you currently pregnant or breastfeeding?",
             {"yes": "Yes", "no": "No"}),
    Question("known_conditions", SECTION_A, "4", "Do you have any known skin conditions?",
             {"none": "None known", "rosacea": "Rosacea", "eczema": "Eczema", "psoriasis": "Psoriasis",
              "melasma": "Melasma", "dermatitis": "Dermatitis", "seborrheic_dermatitis": "Seborrheic dermatitis",
              "other": "Other"},
             multi=True, other_id="known_conditions_other"),
    Question("sensitivity", SECTION_A, "5", "How would you describe your skin sensitivity?",
             {"not": "Not sensitive", "slightly": "Slightly sensitive",
              "moderately": "Moderately sensitive", "highly": "Highly sensitive"}),
    # Section B: Skin characteristics and behaviour (Q7-9 decide skin type)
    Question("reacts_to_products", SECTION_B, "6",
             "Does your skin easily react to skincare products (e.g., redness, burning, itching, or irritation)?",
             {"yes": "Yes", "sometimes": "Sometimes", "no": "No"}),
    Question("tightness", SECTION_B, "7", "After washing your face, does your skin feel tight or dry?",
             {"often": "Yes, often", "sometimes": "Sometimes", "rarely": "Rarely", "never": "Never"}),
    Question("oiliness", SECTION_B, "8a", "How oily does your skin become throughout the day?",
             {"very_oily": "Very oily", "slightly_oily": "Slightly oily", "balanced": "Balanced", "dry": "Dry"}),
    Question("oil_location", SECTION_B, "8b", "Where does your skin get oily or shiny?",
             {"t_zone": "T-zone only", "all_over": "All over", "nowhere": "Doesn’t get oily",
              "not_sure": "Not sure"},
             help="The T-zone is your forehead, nose and chin."),
    Question("flakiness", SECTION_B, "9", "Do you frequently experience dry patches or flakiness?",
             {"yes": "Yes", "sometimes": "Sometimes", "no": "No"}),
    Question("breakouts", SECTION_B, "10", "How often do you experience breakouts or pimples?",
             {"frequently": "Frequently", "occasionally": "Occasionally", "rarely": "Rarely", "never": "Never"}),
    # Section C: Skincare goals
    Question("goals", SECTION_C, "11", "What are your main skincare goals?",
             {"hydration": "Hydration", "acne_control": "Acne control", "oil_control": "Oil control",
              "brightening": "Brightening", "anti_aging": "Anti-aging",
              "sensitivity_reduction": "Sensitivity reduction", "pigmentation_reduction": "Pigmentation reduction",
              "sun_protection": "Sun protection"},
             multi=True),
    # Section D: Ingredient preferences and restrictions
    Question("avoid_ingredients", SECTION_D, "12", "Do you prefer to avoid any of the following ingredients?",
             {"none": "None", "fragrance": "Fragrance", "alcohol": "Alcohol", "essential_oils": "Essential oils",
              "retinol": "Retinol", "salicylic_acid": "Salicylic acid",
              "chemical_sunscreens": "Chemical sunscreens", "other": "Other"},
             multi=True, other_id="avoid_other"),
)

QUESTIONS_BY_ID = {q.id: q for q in QUESTIONS}

# Evidence points each answer gives to each skin type (Section B, Q7-9).
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
    """Score skin type from the Section B answers (Q7-9).

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


# ---- plain-language explanation (results page) ----------------------------------------------

_OILINESS_TEXT = {
    "very_oily": "your skin becomes very oily",
    "slightly_oily": "your skin gets slightly oily",
    "balanced": "your skin stays balanced",
    "dry": "your skin stays dry",
}
_OIL_LOCATION_TEXT = {"t_zone": " on the T-zone (forehead, nose and chin)", "all_over": " all over your face",
                      "nowhere": "", "not_sure": ""}
_TIGHTNESS_TEXT = {"often": "often feels tight after washing", "sometimes": "sometimes feels tight after washing",
                   "rarely": "rarely feels tight after washing", "never": "never feels tight after washing"}
_FLAKINESS_TEXT = {"yes": "you often get dry or flaky patches", "sometimes": "you sometimes get dry patches",
                   "no": "you don't get dry or flaky patches"}
SKIN_TYPE_MEANING = {
    "normal": "Normal skin has a good balance of oil and moisture, so a simple routine that keeps it hydrated "
              "and protected from the sun is enough.",
    "oily": "Oily skin produces more sebum than it needs, so it does best with gentle cleansing and light, "
            "oil-free hydration that won't clog pores.",
    "dry": "Dry skin makes too little oil to hold in moisture, so it needs gentle cleansing and richer, "
           "barrier-repairing moisturisers.",
    "combination": "Combination skin is oily in some areas but normal or dry in others, so it needs oil control "
                   "where it shines and extra hydration where it feels tight.",
}


def explain_skin_type(answers: dict, label: str) -> str:
    """Two or three sentences explaining the skin type from the user's answers (no question text)."""
    oil = _OILINESS_TEXT[answers["oiliness"]]
    if answers["oiliness"] != "dry":
        oil += _OIL_LOCATION_TEXT[answers["oil_location"]]
    oil += " during the day"
    first = (f"Your answers point to {label} skin: {oil}, it {_TIGHTNESS_TEXT[answers['tightness']]}, "
             f"and {_FLAKINESS_TEXT[answers['flakiness']]}.")
    sentences = [first, SKIN_TYPE_MEANING[label]]
    if label == "combination" and _has_mixed_signs(answers):
        sentences.insert(1, "Having oily areas alongside tight or flaky ones is the typical sign of combination skin.")
    return " ".join(sentences)
