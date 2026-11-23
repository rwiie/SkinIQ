"""Age-group skincare guidance shown with the results (updated questionnaire, Q1).

Text from the project's updated questionnaire document. Keys match the Q1 answer keys
in skiniq.questionnaire.
"""

from __future__ import annotations

INTRO = ("Skin health shifts from acne and barrier protection in your late teens and 20s to collagen support, "
         "hydration and pigmentation care in your 30s and beyond, with year-round UV protection essential in "
         "sunny climates like Malaysia.")

AGE_GUIDANCE: dict[str, dict[str, str]] = {
    "18_25": {
        "label": "Ages 18–25",
        "common_concerns": "Ongoing hormonal acne, excess oil, uneven texture, and early sun exposure.",
        "skin_changes": "Cell turnover is still quick and natural collagen is high, though early UV damage and "
                        "subtle aging processes can begin.",
        "care_focus": "Keep routines simple with a gentle cleanser, non-comedogenic moisturizer, and daily "
                      "broad-spectrum sunscreen, avoiding over-exfoliation that harms the skin barrier.",
    },
    "26_35": {
        "label": "Ages 26–35",
        "common_concerns": "Early fine lines, dullness, dehydration, and dark spots or uneven tone.",
        "skin_changes": "Natural collagen and hyaluronic acid levels gradually decline while cell renewal slows "
                        "slightly.",
        "care_focus": "Prioritize daily SPF, antioxidants like vitamin C, barrier hydration, and targeted "
                      "cell-turnover ingredients such as retinoids if well tolerated.",
    },
    "36_49": {
        "label": "Ages 36–49",
        "common_concerns": "Deeper expression lines, loss of firmness or elasticity, dryness, and sun-induced "
                           "pigmentation.",
        "skin_changes": "Collagen and elastin break down faster, and hormonal shifts—including perimenopause—can "
                        "reduce skin density and moisture.",
        "care_focus": "Use richer moisturizers, peptides, retinoids, and consistent sun protection on the face, "
                      "neck, and chest.",
    },
    "50_plus": {
        "label": "Ages 50+",
        "common_concerns": "Noticeable dryness, thinning skin, laxity, deeper wrinkles, and age spots or "
                           "sun-related lesions.",
        "skin_changes": "Lower estrogen, reduced oil production, and fat loss make the skin thinner, drier, and "
                        "more prone to irritation.",
        "care_focus": "Focus on intensive moisture, gentle non-stripping cleansers, daily sunscreen, and regular "
                      "skin checks with a doctor or dermatologist.",
    },
}


def age_guidance(age_group: str) -> dict[str, str]:
    """Guidance for one Q1 answer, e.g. '26_35'. Raises KeyError for an unknown group."""
    return {"age_group": age_group, **AGE_GUIDANCE[age_group]}
