"""Small, dependency-light evaluation helpers for RAG experiments."""

from typing import Iterable, Optional, Set


def source_hit(expected_source: Optional[str], retrieved_sources: Iterable[str]) -> bool:
    """Return whether retrieval included the expected source."""
    if expected_source is None:
        return False
    return expected_source in set(retrieved_sources)


def citation_coverage(expected_sources: Iterable[str], cited_sources: Iterable[str]) -> float:
    """Return the fraction of expected sources represented by citations."""
    expected: Set[str] = set(expected_sources)
    if not expected:
        return 1.0
    cited = set(cited_sources)
    return len(expected.intersection(cited)) / len(expected)


def refusal_correct(expected_refusal: bool, actual_status: str) -> bool:
    """Check whether the answer status matches the evaluation expectation."""
    refusal_statuses = {"refused", "insufficient_evidence"}
    if expected_refusal:
        return actual_status in refusal_statuses
    return actual_status not in refusal_statuses
