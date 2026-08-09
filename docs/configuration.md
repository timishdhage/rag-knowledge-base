# Configuration and Secrets

## Configuration principles

Configuration should be provided through environment variables or a managed secrets service. Application code should read configuration through one validated settings layer rather than accessing environment variables throughout the codebase.

## Local setup

1. Copy `.env.example` to `.env`.
2. Set only the values required for the selected provider and vector store.
3. Keep `.env` out of version control.
4. Use synthetic or public documents for local development.

## Expected configuration categories

The exact variable names must match the implementation. Categories should include:

- Model provider and model name.
- Embedding provider and embedding model.
- Vector-store connection and collection/index name.
- Document directory or object-store location.
- API host and port.
- Logging level.
- Retrieval defaults such as top-k and chunk limits.
- Timeouts and retry limits.

## Secret handling

- Never commit API keys, access tokens, private certificates, or customer data.
- Keep `.env.example` limited to variable names and safe placeholders.
- Use least-privilege cloud credentials.
- Rotate credentials when exposure is suspected.
- Redact secrets and sensitive content from logs.
- Use separate credentials for local development, CI, staging, and production.

## CI configuration

GitHub Actions should receive secrets through repository or environment secrets. The workflow must not print secret values or write them into build artifacts.

## Deployment readiness checklist

- Configuration is validated at startup.
- Missing required settings produce a clear non-secret error.
- Provider timeouts and retries are bounded.
- Logs contain request identifiers but not sensitive prompts or documents.
- Vector-store and model connections can be replaced through configuration.
- The application can run with a mock provider for tests.
