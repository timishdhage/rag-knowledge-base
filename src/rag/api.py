from fastapi import FastAPI
from pydantic import BaseModel
from .build_index import build
from .vectorstore import VectorStore
from .ingest import ingest_folder
from .retrieval import HybridRetriever
from .generator import answer

app = FastAPI(title='RAG Knowledge Base')
store = VectorStore()
CACHE = {'chunks': []}

class AskRequest(BaseModel):
    question: str

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/v1/ingest')
def ingest(payload: dict):
    folder = payload.get('folder', 'docs')
    count = build(folder)
    CACHE['chunks'] = []
    for c in ingest_folder(folder):
        c['id'] = f"{c['source_file']}::{c['chunk_index']}"
        CACHE['chunks'].append(c)
    return {'status': 'ok', 'chunks_indexed': count}

@app.post('/v1/ask')
def ask(req: AskRequest):
    chunks = CACHE['chunks']
    retriever = HybridRetriever(chunks, store)
    sparse = retriever.sparse(req.question, k=5)
    dense = retriever.dense(req.question, k=5)
    fused = retriever.fuse(dense, sparse, k=5)
    response = answer(req.question, [x['chunk'] for x in fused])
    return {'question': req.question, 'dense': dense, 'sparse': sparse, 'retrieved': fused, **response}
