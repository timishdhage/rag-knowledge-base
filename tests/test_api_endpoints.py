import os
from unittest.mock import patch

os.environ.pop("API_AUTH_KEY", None)

from fastapi.testclient import TestClient

from src.rag.api import app
from src.rag.config import settings
from src.rag.rate_limit import limiter


client = TestClient(app)


@patch("src.rag.api._retrieve")
@patch("src.rag.api.answer")
def test_health_and_query_endpoint(mock_answer, mock_retrieve):
    mock_retrieve.return_value = ([], [], [{"score": 0.9, "chunk": {"id": "sample.md::0", "source_file": "sample.md", "chunk_index": 0, "strategy": "fixed", "text": "Evidence text"}}])
    mock_answer.return_value = {"answer": "Grounded answer"}
    health = client.get("/health")
    assert health.json() == {"status": "ok"}
    response = client.post("/v1/query", headers={"X-Request-ID": "req-test-123"}, json={"question": "What is supported?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Grounded answer"
    assert body["metadata"]["latency_ms"] >= 0


def test_query_requires_api_key_when_configured():
    original = settings.api_auth_key
    settings.api_auth_key = "test-secret"
    try:
        response = client.post("/v1/query", json={"question": "Question"})
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHORIZED"
    finally:
        settings.api_auth_key = original


def test_rate_limit_returns_structured_error():
    original_requests = settings.rate_limit_requests
    settings.rate_limit_requests = 1
    limiter._requests.clear()
    try:
        first = client.post("/v1/query", json={"question": "Question"})
        second = client.post("/v1/query", json={"question": "Question"})
        assert first.status_code != 429
        assert second.status_code == 429
        assert second.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"
    finally:
        settings.rate_limit_requests = original_requests
        limiter._requests.clear()
