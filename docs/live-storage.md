# Live storage foundation

The live deployment will use PostgreSQL with the pgvector extension for durable documents, chunks, metadata, and embeddings. The initial schema is in `migrations/001_initial.sql`.

## Configuration

Set `RAG_STORAGE_BACKEND=local` for the existing local test backend. Set `RAG_STORAGE_BACKEND=postgres` and provide `RAG_DATABASE_URL` when using PostgreSQL.

The current commit adds backend validation and the initial schema without changing the existing API path. The next step is wiring document ingestion and retrieval through the storage interface, followed by owner-scoped authorization tests.

## Local database

Use a PostgreSQL image that includes pgvector, apply the migration once, and keep the database URL in an uncommitted environment file. Do not use production credentials locally or commit secrets.
