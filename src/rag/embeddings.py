from openai import OpenAI

from .config import settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for embeddings")
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    response = _get_client().embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]
