from .ingest import ingest_folder
from .vectorstore import VectorStore


def build(folder='docs', owner_id: str | None = None):
    if not owner_id or not owner_id.strip():
        raise ValueError('owner_id is required for indexing')
    chunks = ingest_folder(folder)
    normalized_owner = owner_id.strip()
    for c in chunks:
        c['id'] = f"{c['source_file']}::{c['chunk_index']}"
        c['owner_id'] = normalized_owner
    store = VectorStore()
    if chunks:
        store.add_chunks(chunks)
    return len(chunks)


if __name__ == '__main__':
    build()
