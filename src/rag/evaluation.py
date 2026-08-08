from typing import Iterable, Optional


def source_hit(retrieved_sources: Iterable[str], expected_source: Optional[str]) -> bool:
    """Return whether the expected source appears in retrieved sources."""
    if not expected_source:
        return False
    return expected_source in set(retrieved_sources)


def citation_precision(
    cited_sources: Iterable[str], expected_sources: Iterable[str]
) -> float:
    """Return the fraction of citations that refer to expected sources."""
    cited = list(cited_sources)
    expected = set(expected_sources)
    if not cited:
        return 0.0
    return sum(source in expected for source in cited) / len(cited)


def refusal_correct(actual_refusal: bool, expected_refusal: bool) -> bool:
    """Return whether observed refusal behavior matches the evaluation label."""
    return actual_refusal is expected_refusal
