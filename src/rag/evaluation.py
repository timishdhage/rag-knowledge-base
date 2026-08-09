from typing import Iterable, Optional


def source_hit(retrieved_sources: Iterable[str], expected_source: Optional[str]) -> bool:
    """Return whether the expected source appears in retrieved sources."""
    if not expected_source:
        return False
    return expected_source in set(retrieved_sources)


def citation_precision(
    cited_sources: Iterable[str], expected_sources: Iterable[str]
) -> float:
    """Return the fraction of expected sources covered by citations."""
    cited = set(cited_sources)
    expected = set(expected_sources)
    if not expected:
        return 0.0
    overlap = cited.intersection(expected)
    return len(overlap) / len(expected)


def refusal_correct(actual_refusal: bool, expected_refusal: bool) -> bool:
    """Return whether observed refusal behavior matches the evaluation label."""
    return actual_refusal is expected_refusal
