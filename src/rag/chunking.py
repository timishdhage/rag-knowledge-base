from dataclasses import dataclass
from typing import List

@dataclass
class Chunk:
    text: str
    chunk_index: int
    strategy: str
    source_file: str

def fixed_size_chunks(text: str, source_file: str, chunk_size: int = 800, overlap: int = 120) -> List[Chunk]:
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(Chunk(text=chunk, chunk_index=idx, strategy='fixed', source_file=source_file))
            idx += 1
        if end >= len(text):
            break
        start = max(end - overlap, 0)
    return chunks
