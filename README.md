# Production Agentic RAG Platform

A Python-based document-grounded question-answering system that retrieves relevant document context and generates evidence-based responses. The project is being developed as a reusable portfolio system for production AI and ML engineering roles.

## What it does

The current application provides the foundations for a retrieval-augmented generation workflow:

1. Load documents from the project knowledge base.
2. Split documents into searchable chunks.
3. Create embeddings for semantic retrieval.
4. Store and retrieve document representations.
5. Generate an answer using retrieved context.
6. Expose the workflow through an API and a lightweight UI.

The system is designed to evolve toward hybrid retrieval, citation-grounded answers, controlled tool use, evaluation, monitoring, and secure cloud deployment.

## Current repository structure

```text
.
├── docs/                  # Project documentation and sample knowledge files
├── evaluation/            # Versioned RAG evaluation dataset and guidance
├── src/rag/               # Application package
│   ├── api.py             # API layer
│   ├── build_index.py     # Index-building entry point
│   ├── chunking.py        # Document chunking
│   ├── config.py          # Runtime configuration
│   ├── embeddings.py      # Embedding integration
│   ├── generator.py       # Answer generation
│   ├── ingest.py          # Ingestion workflow
│   ├── loaders.py         # Document loaders
│   ├── retrieval.py       # Context retrieval
│   └── vectorstore.py     # Vector-store integration
├── tests/                 # Automated tests
├── ui/                    # Lightweight user interface
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick start

### Local environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
pytest -q
```

### Docker

```bash
docker compose up --build
```

Do not commit `.env` files, API keys, private documents, or customer data. Use `.env.example` only for documenting required variable names.

## Architecture

```text
Documents
   |
   v
Loaders -> Chunking -> Embeddings -> Vector store
                                      |
User query -> Retrieval -> Context assembly -> Generator -> Answer + evidence
                                      |
                                      v
                               API / UI layer
```

The target architecture is documented in [`docs/architecture.md`](docs/architecture.md).

## Evaluation

The repository contains a synthetic evaluation contract in [`evaluation/questions.json`](evaluation/questions.json). It includes answerable questions and refusal cases. The current tests validate the dataset structure; application-level retrieval and answer-quality metrics are the next implementation step.

Planned metrics include:

- Retrieval hit rate and recall at `k`.
- Citation correctness.
- Answer faithfulness.
- Refusal correctness.
- Latency and token/cost usage.

The project does not claim production accuracy until these metrics are measured against a larger, versioned evaluation set.

## Development quality gates

Every meaningful change should:

- Add or update tests.
- Document behavioural changes.
- Avoid committing secrets or sensitive data.
- Record evaluation results where retrieval or generation changes.
- Keep current capabilities separate from planned roadmap items.

CI runs Python compilation checks and the test suite for pushes and pull requests targeting `main` or `production-agentic-rag`.

## Roadmap

### Foundation

- Strengthen API request and response schemas.
- Add retrieval and generation tests.
- Add evaluation execution and result storage.
- Improve error handling and configuration validation.

### GenAI engineering

- Add hybrid keyword and vector retrieval.
- Add citation-grounded response schemas.
- Add prompt versioning and regression tests.
- Add an Anthropic or Amazon Bedrock model gateway.
- Add LangChain or LlamaIndex where it improves maintainability.

### Agentic capabilities

- Add MCP tools with explicit permissions.
- Add controlled tool selection and escalation.
- Add audit events for tool calls and sensitive operations.

### Production deployment

- Add a managed vector store such as OpenSearch, Pinecone, or Weaviate.
- Add API authentication, rate limits, and input validation.
- Add structured logs, metrics, tracing, and cost monitoring.
- Add cloud deployment documentation and least-privilege IAM.

## Project status

The project is an active engineering build. The current codebase is an early RAG scaffold. The roadmap items above are planned capabilities and should not be presented as completed experience until they are implemented, tested, and documented.

## Portfolio positioning

This project demonstrates the progression from a basic RAG prototype toward a reliable AI service: retrieval quality, evaluation, API design, testing, security, observability, and deployment are treated as engineering requirements rather than optional additions.