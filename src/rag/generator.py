from .provider import ModelGateway, OpenAIModelGateway

_gateway: ModelGateway | None = None


def get_gateway() -> ModelGateway:
    global _gateway
    if _gateway is None:
        _gateway = OpenAIModelGateway()
    return _gateway


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
    text = get_gateway().generate(prompt)
    return {
        "answer": text,
        "citations": [f"[{i}]" for i in range(1, len(chunks) + 1)],
        "confidence": 0.7,
    }
