from openai import OpenAI

from .config import settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for answer generation")
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def answer(question: str, chunks):
    if not chunks:
        return {
            "answer": "I don't know based on the provided documents.",
            "citations": [],
            "confidence": 0.0,
        }
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] Source: {chunk['source_file']} | {chunk['text']}"
        )
    prompt = (
        "Answer the question using only the context. Cite sources like [1], [2]. "
        "If the context is insufficient, say you don't know.\n\n"
        f"Question: {question}\n\nContext:\n" + "\n".join(blocks)
    )
    response = _get_client().responses.create(
        model=settings.generation_model,
        input=prompt,
    )
    text = getattr(response, "output_text", str(response))
    return {
        "answer": text,
        "citations": [f"[{i}]" for i in range(1, len(chunks) + 1)],
        "confidence": 0.7,
    }
