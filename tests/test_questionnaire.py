import itertools

import pytest

from skiniq.config import SKIN_TYPES
from skiniq.questionnaire import (
    QUESTIONS,
    QUESTIONS_BY_ID,
    SKIN_TYPE_EVIDENCE,
    assess_skin_type,
)


def test_appendix_a_questions_present():
    numbers = [q.number for q in QUESTIONS]
    assert numbers == ["1", "2", "3", "4", "5", "6", "7", "7b", "8", "9", "10", "11"]
    assert len(QUESTIONS_BY_ID) == len(QUESTIONS)


def test_evidence_uses_valid_options_and_skin_types():
    for qid, table in SKIN_TYPE_EVIDENCE.items():
        assert set(table) == set(QUESTIONS_BY_ID[qid].options), qid
        for pts in table.values():
            assert set(pts) <= set(SKIN_TYPES)


@pytest.mark.parametrize(
    "answers, expected",
    [
        ({"oiliness": "very_oily", "oil_location": "all_over", "tightness": "never", "flakiness": "no"}, "oily"),
        ({"oiliness": "slightly_oily", "oil_location": "t_zone", "tightness": "sometimes", "flakiness": "sometimes"},
         "combination"),
        ({"oiliness": "dry", "oil_location": "nowhere", "tightness": "often", "flakiness": "yes"}, "dry"),
        ({"oiliness": "balanced", "oil_location": "nowhere", "tightness": "rarely", "flakiness": "no"}, "normal"),
        # Edge cases
        ({"oiliness": "very_oily", "oil_location": "t_zone", "tightness": "never", "flakiness": "no"}, "combination"),
        # Oil plus dry patches is combination by definition
        ({"oiliness": "slightly_oily", "oil_location": "all_over", "tightness": "often", "flakiness": "yes"},
         "combination"),
        # Says "dry" but also flaky with no oil anywhere: dry, not combination
        ({"oiliness": "dry", "oil_location": "not_sure", "tightness": "sometimes", "flakiness": "yes"}, "dry"),
        ({"oiliness": "balanced", "oil_location": "not_sure", "tightness": "never", "flakiness": "no"}, "normal"),
    ],
)
def test_typical_and_edge_profiles(answers, expected):
    assert assess_skin_type(answers).label == expected


def test_scores_sum_to_one_for_every_combination():
    for combo in itertools.product(*(SKIN_TYPE_EVIDENCE[q] for q in SKIN_TYPE_EVIDENCE)):
        result = assess_skin_type(dict(zip(SKIN_TYPE_EVIDENCE, combo)))
        assert sum(result.scores.values()) == pytest.approx(1.0, abs=1e-3)
        assert result.label in SKIN_TYPES
        assert result.scores[result.label] == max(result.scores.values())


def test_every_skin_type_is_reachable():
    labels = {
        assess_skin_type(dict(zip(SKIN_TYPE_EVIDENCE, combo))).label
        for combo in itertools.product(*(SKIN_TYPE_EVIDENCE[q] for q in SKIN_TYPE_EVIDENCE))
    }
    assert labels == set(SKIN_TYPES)


def test_reasons_explain_the_label():
    result = assess_skin_type(
        {"oiliness": "dry", "oil_location": "nowhere", "tightness": "often", "flakiness": "yes"}
    )
    assert result.reasons[0] == "How oily does your skin become throughout the day? — Dry"
    assert len(result.reasons) == 4


def test_mixed_signs_reason_is_given():
    result = assess_skin_type(
        {"oiliness": "slightly_oily", "oil_location": "all_over", "tightness": "often", "flakiness": "yes"}
    )
    assert "Oily areas together with dry patches or tightness" in result.reasons


@pytest.mark.parametrize("missing", list(SKIN_TYPE_EVIDENCE))
def test_missing_answer_is_rejected(missing):
    answers = {"oiliness": "balanced", "oil_location": "nowhere", "tightness": "rarely", "flakiness": "no"}
    del answers[missing]
    with pytest.raises(ValueError, match=missing):
        assess_skin_type(answers)


def test_invalid_answer_is_rejected():
    with pytest.raises(ValueError):
        assess_skin_type({"oiliness": "greasy", "oil_location": "nowhere", "tightness": "rarely", "flakiness": "no"})
