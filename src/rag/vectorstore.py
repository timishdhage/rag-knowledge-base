from __future__ import annotations

from chromadb import PersistentClient

from .config import settings
from .embeddings import embed_texts


def owner_where(owner_id: str) -> dict[str, str]:
    normalized = owner_id.strip()
    if not normalized:
        raise ValueError("owner_id must not be empty")
    return {"owner_id": normalized}


class VectorStore:
    def __init__(self):
        self.client = PersistentClient(path=settings.chroma_path)
        self.collection = self.client.get_or_create_collection(name="rag_docs")

    def add_chunks(self, chunks):
        ids = [c["id"] for c in chunks]
        docs = [c["text"] for c in chunks]
        metas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
        embeds = embed_texts(docs)
        self.collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)

    def query(self, query: str, k: int = 5, owner_id: str | None = None):
        q = embed_texts([query])[0]
        kwargs = {"query_embeddings": [q], "n_results": k}
        if owner_id is not None:
            kwargs["where"] = owner_where(owner_id)
        return self.collection.query(**kwargs)
