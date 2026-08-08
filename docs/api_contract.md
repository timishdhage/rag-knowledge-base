# Query API Contract

## Endpoint

`POST /v1/query`

## Request

```json
{
  "question": "What is in the knowledge base?",
  "top_k": 5,
  "filters": {}
}
```

- `question` is required and must be a non-empty string.
- `top_k` controls the maximum number of retrieved chunks.
- `filters` is reserved for metadata filtering as retrieval evolves.

## Response

```json
{
  "answer": "Grounded answer text",
  "citations": [
    {
      "source": "sample.md",
      "chunk_id": "sample.md::0",
      "text": "Evidence used by the answer"
    }
  ],
  "status": "answered",
  "metadata": {
    "retrieved_documents": 1
  }
}
```

`status` is `answered` when evidence is available and `insufficient_evidence` when retrieval returns no chunks. The existing `POST /v1/ask` endpoint remains available for compatibility with the current UI.
