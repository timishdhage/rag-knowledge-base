from chromadb import PersistentClient
from .embeddings import embed_texts
from .config import settings

class VectorStore:
    def __init__(self):
        self.client = PersistentClient(path=settings.chroma_path)
        self.collection = self.client.get_or_create_collection(name='rag_docs')

    def add_chunks(self, chunks):
        ids = [c['id'] for c in chunks]
        docs = [c['text'] for c in chunks]
        metas = [{k: v for k, v in c.items() if k != 'text'} for c in chunks]
        embeds = embed_texts(docs)
        self.collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)

    def query(self, query: str, k: int = 5):
        q = embed_texts([query])[0]
        return self.collection.query(query_embeddings=[q], n_results=k)
