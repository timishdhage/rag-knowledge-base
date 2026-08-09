from unittest.mock import patch

from fastapi.testclient import TestClient

from src.rag.api import app


client = TestClient(app)


@patch("src.rag.api._retrieve")
@patch("src.rag.api.answer")
def test_health_and_query_endpoint(mock_answer, mock_retrieve):
    mock_retrieve.return_value = ([], [], [{
        "score": 0.9,
        "chunk": {
            "id": "sample.md::0",
            "source_file": "sample.md",
            "chunk_index": 0,
            "strategy": "fixed",
            "text": "Evidence text",
        },
    }])
    mock_answer.return_value = {"answer": "Grounded answer"}

    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/v1/query", json={"question": "What is supported?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Grounded answer"
    assert body["status"] == "answered"
    assert body["metadata"]["retrieved_documents"] == 1
    assert body["citations"][0]["chunk_id"] == "sample.md::0"


@patch("src.rag.api._retrieve", return_value=([], [], []))
@patch("src.rag.api.answer")
def test_query_returns_insufficient_evidence(mock_answer, mock_retrieve):
    mock_answer.return_value = {
        "answer": "I don't know based on the provided documents."
    }

    response = client.post("/v1/query", json={"question": "Unknown question"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "insufficient_evidence"
    assert body["citations"] == []
    assert body["metadata"]["retrieved_documents"] == 0


def test_query_rejects_invalid_request():
    response = client.post("/v1/query", json={"question": "", "top_k": 5})

    assert response.status_code == 422
