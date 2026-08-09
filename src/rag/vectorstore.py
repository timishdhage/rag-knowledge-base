from chromadb import PersistentClient

from .config import settings
from .embeddings import embed_texts


class VectorStore:
    def __init__(self):
        self.client = PersistentClient(path=str(settings.chroma_directory))
        self.collection = self.client.get_or_create_collection(name="rag_docs")

    def add_chunks(self, chunks):
        if not chunks:
            return
        ids = [chunk["id"] for chunk in chunks]
        docs = [chunk["text"] for chunk in chunks]
        metadata = [
            {key: value for key, value in chunk.items() if key != "text"}
            for chunk in chunks
        ]
        embeddings = embed_texts(docs)
        self.collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metadata,
            embeddings=embeddings,
        )

    def query(self, query: str, k: int = 5):
        embedding = embed_texts([query])[0]
        return self.collection.query(query_embeddings=[embedding], n_results=k)
