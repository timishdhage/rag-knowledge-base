# RAG Knowledge Base

## Run locally

1. Copy `.env.example` to `.env` and add your `OPENAI_API_KEY`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Build the index: `python -m src.rag.build_index`.
4. Start the API: `uvicorn src.rag.api:app --reload`.
5. Open docs at `http://127.0.0.1:8000/docs`.
6. Start the UI: `streamlit run ui/app.py`.

## Docker

- API + UI: `docker compose up --build`

## Endpoints

- `POST /v1/ingest` to index documents.
- `POST /v1/ask` to ask questions.
- `GET /health` for health checks.
