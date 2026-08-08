import pytest

from src.rag.evaluation import (
    citation_precision,
    refusal_correct,
    source_hit,
)


def test_source_hit_requires_expected_source():
    assert source_hit(["policy.md"], "policy.md") is True
    assert source_hit(["other.md"], "policy.md") is False


def test_citation_precision_ignores_unknown_sources():
    assert citation_precision(["policy.md", "other.md"], ["policy.md"]) == 1.0
    assert citation_precision([], ["policy.md"]) == 0.0


def test_refusal_correct_matches_expected_behaviour():
    assert refusal_correct(True, True) is True
    assert refusal_correct(False, False) is True
    assert refusal_correct(True, False) is False
