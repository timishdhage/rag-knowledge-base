# RAG Knowledge Base

A production-oriented retrieval-augmented generation (RAG) knowledge-base service built with FastAPI. It ingests documents, indexes them for hybrid retrieval, and generates grounded answers with citations, safety controls, and observability.

## Highlights

- **Hybrid retrieval:** combines vector similarity search with BM25 keyword search, metadata filtering, and reranking/fusion.
- **Document ingestion:** parses source files, creates retrieval-friendly chunks, generates embeddings, and stores them in ChromaDB.
- **Grounded generation:** routes requests through a model gateway that supports Anthropic / Amazon Bedrock, prompt versioning, and structured outputs.
- **Operational controls:** captures citations, refusals, structured logs, latency, cost, and audit events.
- **Deployment-ready:** includes Docker, Docker Compose, AWS Lambda support, and GitHub Actions workflows.

## Architecture

The FastAPI service coordinates retrieval, model access, and operational safeguards. The retrieval service searches indexed content, while the model gateway prepares structured, grounded responses. Safety and observability capture the evidence and runtime signals needed to operate the system responsibly.

![RAG target architecture](docs/images/Screenshot%202026-08-28%20at%2000.54.50.png)

### Retrieval workflow

The system separates offline ingestion from online question answering. During ingestion, documents are parsed, chunked, embedded, and stored in ChromaDB. At query time, hybrid retrieval supplies relevant context to the generator before the API/UI returns the final response.

![RAG ingestion and query pipeline](docs/images/Screenshot%202026-08-28%20at%2000.54.26.png)

## Project structure

```text
.
├── src/                  # FastAPI application and RAG services
├── ui/                   # Web UI
├── docs/                 # Architecture, API, configuration, and auth documentation
│   └── images/           # README architecture diagrams
├── evaluation/           # Retrieval and response evaluation assets
├── tests/                # Automated tests
├── Dockerfile            # Container build for the application
├── Dockerfile.lambda     # Container build for AWS Lambda
├── docker-compose.yml    # Local multi-service development setup
└── requirements.txt      # Python dependencies
```

## Getting started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (recommended for local services)
- An LLM provider credential configured for the selected model gateway

### Local setup

```bash
git clone https://github.com/timishdhage/rag-knowledge-base.git
cd rag-knowledge-base

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
```

Set the required provider and application variables in `.env`. Refer to the configuration documentation for the complete environment-variable reference.

```bash
uvicorn src.main:app --reload
```

The API will be available locally at `http://127.0.0.1:8000`. When the application is running, FastAPI's interactive API documentation is available at `/docs`.

### Docker

```bash
docker compose up --build
```

## How it works

1. **Ingest documents:** source files are loaded and parsed into text.
2. **Create chunks and embeddings:** parsed content is split into manageable units, embedded, and persisted in ChromaDB with metadata.
3. **Retrieve evidence:** a user question is evaluated using hybrid vector and keyword retrieval, with optional metadata filters and reranking.
4. **Generate a grounded response:** retrieved passages are passed to the configured LLM with prompt controls and output validation.
5. **Return citations and telemetry:** the service returns the answer with source evidence while recording structured operational events.

## Documentation

- [Architecture](docs/architecture.md)
- [API contract](docs/api-contract.md)
- [Configuration](docs/configuration.md)
- [Cognito authentication](docs/cognito-auth.md)
- [Live storage](docs/live-storage.md)

## Testing and evaluation

Run the automated test suite:

```bash
pytest
```

The `evaluation/` directory contains assets for measuring retrieval and generation quality. Use it to validate changes to chunking, embeddings, retrieval, prompts, or model configuration before deploying.

## Deployment

The repository includes both a standard Dockerfile and a Lambda-specific Dockerfile. This supports local container development as well as AWS-oriented deployment workflows. Review the configuration and architecture documentation before deploying, and ensure no secrets are committed to source control.

## License

Add a license file before distributing or accepting external contributions.
