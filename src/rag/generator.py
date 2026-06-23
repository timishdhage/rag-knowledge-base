import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def answer(question: str, chunks):
    if not chunks:
        return {'answer': "I don't know based on the provided documents.", 'citations': [], 'confidence': 0.0}
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] Source: {c['source_file']} | {c['text']}")
    prompt = "Answer the question using only the context. Cite sources like [1], [2]. If the context is insufficient, say you don't know.\n\nQuestion: " + question + "\n\nContext:\n" + "\n".join(blocks)
    resp = client.responses.create(model='gpt-4o-mini', input=prompt)
    text = getattr(resp, 'output_text', str(resp))
    return {'answer': text, 'citations': [f'[{i}]' for i in range(1, len(chunks)+1)], 'confidence': 0.7}
