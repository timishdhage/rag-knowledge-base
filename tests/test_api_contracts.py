import pytest
from pydantic import ValidationError

from src.rag.contracts import (
    AnswerStatus,
    Citation,
    ErrorDetails,
    ErrorResponse,
    QueryRequest,
    QueryResponse,
)


def test_query_request_has_safe_defaults():
    request = QueryRequest(question="What is the escalation process?")
    assert request.top_k == 5
    assert request.filters is None


def test_query_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        QueryRequest(question="")


def test_query_request_bounds_top_k():
    with pytest.raises(ValidationError):
        QueryRequest(question="Question", top_k=21)


def test_answer_response_supports_citations():
    response = QueryResponse(
        answer="Escalate to an authorised reviewer.",
        citations=[
            Citation(
                source="policy.md",
                chunk_id="policy-001",
                text="Escalate unresolved requests.",
            )
        ],
        status=AnswerStatus.ANSWERED,
    )
    assert response.citations[0].source == "policy.md"
    assert response.metadata.retrieved_documents == 0


def test_error_response_has_stable_shape():
    response = ErrorResponse(
        error=ErrorDetails(
            code="INVALID_REQUEST",
            message="question must not be empty",
            request_id="req-123",
        )
    )
    assert response.error.code == "INVALID_REQUEST"
    assert response.error.request_id == "req-123"
