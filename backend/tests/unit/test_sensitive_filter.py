import pytest

from backend.app.services.sensitive_filter import SensitiveFilter


@pytest.fixture
def filter_():
    return SensitiveFilter(("禁止词", "bad-word", "bad"))


def test_filter_blocks_spaced_and_mixed_case_terms(filter_):
    assert filter_.check("禁 止 词").blocked is True
    assert filter_.check("BaD-WoRd").blocked is True


def test_filter_prefers_longest_term_and_does_not_store_input(filter_):
    result = filter_.check("This is bad-word")

    assert result.matches == ("bad-word",)
    assert result.normalized_text == "thisisbadword"
    assert "This" not in result.matches
