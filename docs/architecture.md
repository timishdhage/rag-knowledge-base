# Production Agentic RAG Platform

## Purpose

This project is a document-grounded AI assistant. It retrieves relevant passages from a document collection and uses them to produce an answer with evidence. The design is intended to evolve from a local RAG service into a testable, observable, and secure production application.

## Current flow

1. Documents enter through the ingestion layer.
2. Loaders read supported document formats.
3. Chunking converts documents into retrieval units.
4. Embeddings represent chunks for semantic search.
5. The vector store persists searchable representations.
6. Retrieval selects relevant context for a user query.
7. Generation produces an answer from the retrieved context.
8. The API exposes the workflow to clients.
9. The UI provides a simple interaction layer.

## Target architecture

```text
Client/UI
   |
   v
FastAPI service
   |
   +--> request validation and authentication
   |
   +--> retrieval service
   |       +--> vector search
   |       +--> keyword search
   |       +--> metadata filtering
   |       +--> reranking/fusion
   |
   +--> model gateway
   |       +--> Anthropic or Amazon Bedrock
   |       +--> prompt versioning
   |       +--> structured output validation
   |
   +--> safety and observability
           +--> citations and refusal rules
           +--> structured logs
           +--> latency and cost metrics
           +--> audit events
```

## Planned capabilities

- Hybrid retrieval combining semantic and keyword search.
- Citation-grounded answers.
- Evaluation for retrieval quality, faithfulness, citation correctness, latency, and cost.
- MCP tools for controlled document search and evidence operations.
- Optional agent orchestration with explicit tool permissions and escalation paths.
- Containerised deployment and continuous integration.

## Security principles

- Keep credentials in environment variables or a secrets manager.
- Never commit API keys or personal data.
- Validate all external inputs.
- Restrict tools to the minimum permissions required.
- Log decisions and tool calls without logging sensitive document contents unnecessarily.
- Escalate when evidence is missing or confidence is insufficient.

## Quality gates

A change is ready for review when tests pass, the behaviour is documented, evaluation results are recorded, and the change does not introduce secrets or unsupported CV claims.
