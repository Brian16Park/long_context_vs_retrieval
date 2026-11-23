"""Checks for scoring (lcvr/scoring.py), against answers worked out by hand."""
import pytest

from lcvr.scoring import clean_prediction, normalize, score


def test_normalize():
    assert normalize("The Eiffel Tower.") == "eiffel tower"
    assert normalize("  An  apple, a day ") == "apple day"


def test_exact_match_ignores_case_articles_and_punctuation():
    assert score("the Beatles!", ["The Beatles"]) == (1.0, 1.0)


def test_best_over_accepted_answers():
    assert score("New York City", ["NYC", "New York City"]) == (1.0, 1.0)


def test_partial_credit():
    # "barack obama" vs "obama": 1 word in common; precision 1/2, recall 1/1 -> F1 2/3
    em, f = score("Barack Obama", ["Obama"])
    assert em == 0.0
    assert f == pytest.approx(2 / 3)


def test_repeated_words_count_once_each():
    # "cat cat" vs "cat": only one "cat" can match -> precision 1/2, recall 1 -> F1 2/3
    assert score("the cat the cat", ["cat"])[1] == pytest.approx(2 / 3)


def test_wrong_and_empty_answers_score_zero():
    assert score("Paris", ["London"]) == (0.0, 0.0)
    assert score("", ["London"]) == (0.0, 0.0)


def test_clean_prediction():
    assert clean_prediction("Answer: 1990\n\nBecause the film came out then.") == "1990"
    assert clean_prediction("**Answer:** Paris") == "Paris"
    assert clean_prediction("\n  Leonardo da Vinci  \n") == "Leonardo da Vinci"