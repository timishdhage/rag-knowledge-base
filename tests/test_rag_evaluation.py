from rag.evaluation import citation_coverage, refusal_correct, source_hit


def test_source_hit_requires_expected_source():
    assert source_hit("policy.md", ["policy.md", "guide.md"])
    assert not source_hit("policy.md", ["guide.md"])
    assert not source_hit(None, ["policy.md"])


def test_citation_coverage_is_fraction_of_expected_sources():
    assert citation_coverage(["a.md", "b.md"], ["a.md"]) == 0.5
    assert citation_coverage(["a.md"], ["a.md", "b.md"]) == 1.0
    assert citation_coverage([], []) == 1.0


def test_refusal_correctness_handles_safe_statuses():
    assert refusal_correct(True, "refused")
    assert refusal_correct(True, "insufficient_evidence")
    assert not refusal_correct(True, "answered")
    assert refusal_correct(False, "answered")
    assert not refusal_correct(False, "refused")
