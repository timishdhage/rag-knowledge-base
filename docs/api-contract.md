# API Contract

This document defines the intended public contract for the RAG service. It separates the stable client-facing behaviour from internal retrieval and model implementations.

## Query endpoint

The primary operation should accept a user question and return an evidence-aware answer.

```json
{
  "question": "What does the available documentation say about escalation?",
  "top_k": 5,
  "filters": {
    "document_type": "policy"
  }
}
```

### Request rules

- `question` is required and must contain non-whitespace text.
- `top_k` is optional and must be bounded to prevent excessive retrieval.
- `filters` is optional and must be validated before it reaches the retrieval layer.
- Credentials and raw provider-specific settings must not be accepted from the client.

## Response shape

```json
{
  "answer": "The available evidence indicates that the request should be escalated to an authorised reviewer.",
  "citations": [
    {
      "source": "escalation-policy.md",
      "chunk_id": "escalation-policy-003",
      "text": "..."
    }
  ],
  "status": "answered",
  "metadata": {
    "retrieved_documents": 3,
    "latency_ms": 142,
    "model_version": "example-version"
  }
}
```

## Response statuses

- `answered`: sufficient evidence was retrieved and an answer was generated.
- `insufficient_evidence`: the system could not support a reliable answer.
- `refused`: the request is outside the permitted scope or involves protected information.
- `error`: the request could not be processed safely.

## Error contract

Errors should use a consistent structure:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "question must not be empty",
    "request_id": "request-identifier"
  }
}
```

The service must not expose stack traces, credentials, provider secrets, or private document contents in error responses.

## Contract tests to add

- Empty questions return a validation error.
- Invalid `top_k` values are rejected.
- Successful answers contain an answer, status, and citations when evidence is available.
- Insufficient evidence produces a safe response instead of an invented answer.
- Provider failures are converted into a stable error response.
