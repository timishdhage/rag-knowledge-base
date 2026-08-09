# Query API Contract

## Endpoint

`POST /v1/query`

## Authentication

Set `API_AUTH_KEY` in the deployment environment to protect `/v1/query`, `/v1/ask`, and `/v1/ingest`. Send the configured value in the `X-API-Key` header. `/health` remains public for health checks. If `API_AUTH_KEY` is absent, local development and CI remain open for compatibility.

## Request

```json
{
  "question": "What is in the knowledge base?",
  "top_k": 5,
  "filters": {
    "source_file": "policy.md"
  }
}
```

- `question` is required and must be a non-empty string.
- `top_k` controls the maximum number of retrieved chunks.
- `filters` is optional metadata filtering. Every supplied key/value pair must match a chunk before retrieval is performed.

## Response

```json
{
  "answer": "Grounded answer text",
  "citations": [],
  "status": "answered",
  "metadata": {
    "retrieved_documents": 1,
    "request_id": "req-123",
    "latency_ms": 42.7
  }
}
```

The `X-Request-ID` header is accepted or generated and returned on every response. `status` is `answered` when evidence is available and `insufficient_evidence` when retrieval returns no chunks after filtering. Invalid requests return HTTP 422 with an `ErrorResponse`; missing or invalid API keys return HTTP 401.
