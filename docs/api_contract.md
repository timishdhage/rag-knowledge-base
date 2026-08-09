# Query API Contract

## Authentication

Set `API_AUTH_KEY` to protect query, ask, and ingestion routes. Send it with `X-API-Key`. `/health` remains public. Missing or invalid keys return HTTP 401 with `error.code` equal to `UNAUTHORIZED`.

## Rate limiting

Protected routes use an in-process per-client limiter. Configure `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW_SECONDS`. When exceeded, the API returns HTTP 429 with `error.code` equal to `RATE_LIMIT_EXCEEDED`. This is suitable for a single-process deployment; distributed deployments should use a shared store.

## Provider gateway

Answer generation uses a `ModelGateway` protocol and an `OpenAIModelGateway` adapter. This keeps API logic independent from the model provider and allows future Anthropic or Bedrock adapters.

## Request

```json
{"question":"What is in the knowledge base?","top_k":5,"filters":{"source_file":"policy.md"}}
```

## Response

```json
{"answer":"Grounded answer text","citations":[],"status":"answered","metadata":{"retrieved_documents":1,"request_id":"req-123","latency_ms":42.7}}
```
