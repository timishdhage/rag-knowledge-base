import os
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from src.rag.api import _answer_with_contract


def test_answer_with_contract_returns_validated_response():
    fused = [
        {
            "score": 0.9,
            "chunk": {
                "id": "sample.md::0",
                "source_file": "sample.md",
                "chunk_index": 0,
                "strategy": "fixed",
                "text": "Relevant evidence text",
            },
        }
    ]

    with patch("src.rag.api.answer", return_value={"answer": "Grounded answer"}):
        response = _answer_with_contract("What is the answer?", fused)

    assert response.answer == "Grounded answer"
    assert response.status.value == "answered"
    assert response.metadata.retrieved_documents == 1
    assert response.citations[0].source == "sample.md"
    assert response.citations[0].chunk_id == "sample.md::0"


def test_empty_retrieval_is_insufficient_evidence():
    with patch(
        "src.rag.api.answer",
        return_value={"answer": "I don't know based on the provided documents."},
    ):
        response = _answer_with_contract("Unknown question", [])

    assert response.status.value == "insufficient_evidence"
    assert response.citations == []
    assert response.metadata.retrieved_documents == 0
