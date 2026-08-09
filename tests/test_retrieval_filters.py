from src.rag.retrieval import filter_chunks


def test_filter_chunks_matches_all_metadata_filters():
    chunks = [
        {"id": "a", "source_file": "policy.md", "strategy": "fixed", "text": "A"},
        {"id": "b", "source_file": "guide.md", "strategy": "fixed", "text": "B"},
    ]

    result = filter_chunks(chunks, {"source_file": "policy.md", "strategy": "fixed"})

    assert [chunk["id"] for chunk in result] == ["a"]


def test_filter_chunks_returns_no_chunks_for_non_matching_filter():
    chunks = [{"id": "a", "source_file": "policy.md", "text": "A"}]

    assert filter_chunks(chunks, {"source_file": "missing.md"}) == []
