"""Checks for the prompt styles (lcvr/prompts.py)."""
import pytest

from lcvr.prompts import (CLOSED_BOOK_INSTRUCTIONS, CLOSED_BOOK_REASONING_INSTRUCTIONS, INSTRUCTIONS,
                          MAX_ANSWER_TOKENS, MAX_REASONING_TOKENS, REASONING_INSTRUCTIONS,
                          prompt_settings, style_label)


def test_each_style_has_its_instructions_and_token_cap():
    assert prompt_settings("answer_only") == (INSTRUCTIONS, MAX_ANSWER_TOKENS)
    assert prompt_settings("reasoning") == (REASONING_INSTRUCTIONS, MAX_REASONING_TOKENS)
    assert prompt_settings("answer_only", closed_book=True) == (CLOSED_BOOK_INSTRUCTIONS, MAX_ANSWER_TOKENS)
    assert prompt_settings("reasoning", closed_book=True) == (CLOSED_BOOK_REASONING_INSTRUCTIONS,
                                                              MAX_REASONING_TOKENS)


def test_unknown_style_is_an_error():
    with pytest.raises(ValueError):
        prompt_settings("verbose")


def test_only_the_reasoning_style_adds_a_label():
    assert style_label("answer_only") == {}
    assert style_label("reasoning") == {"prompt": "reasoning"}
