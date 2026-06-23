#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$HOME/Projects/rag_project"
mkdir -p "$PROJECT_DIR/src/rag" "$PROJECT_DIR/docs" "$PROJECT_DIR/tests" "$PROJECT_DIR/ui"
cd "$PROJECT_DIR"

cat > .gitignore <<'EOF'
__pycache__/
*.pyc
.env
.venv/
output/
chroma/
EOF

cat > requirements.txt <<'EOF'
fastapi
uvicorn
pydantic
python-dotenv
chromadb
openai
rank-bm25
numpy
pandas
pypdf
beautifulsoup4
markdown
streamlit
requests
EOF

cat > .env.example <<'EOF'
OPENAI_API_KEY=your_key_here
CHROMA_PATH=./chroma
EOF

cat > README.md <<'EOF'
# RAG Knowledge Base

## Run locally

1. Copy `.env.example` to `.env` and add your `OPENAI_API_KEY`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Build the index: `python -m src.rag.build_index`.
4. Start the API: `uvicorn src.rag.api:app --reload`.
5. Open docs at `http://127.0.0.1:8000/docs`.
6. Start the UI: `streamlit run ui/app.py`.

## Docker

- API + UI: `docker compose up --build`

## Endpoints

- `POST /v1/ingest` to index documents.
- `POST /v1/ask` to ask questions.
- `GET /health` for health checks.
EOF

cat > docs/sample.md <<'EOF'
# Sample Document

This is a sample internal document for the RAG system.
EOF

cat > tests/test_basic.py <<'EOF'
def test_placeholder():
    assert True
EOF

cat > src/rag/__init__.py <<'EOF'
EOF

cat > src/rag/config.py <<'EOF'
from pydantic import BaseModel

class Settings(BaseModel):
    chroma_path: str = './chroma'
    embedding_model: str = 'text-embedding-3-small'
    top_k_dense: int = 10
    top_k_sparse: int = 10
    top_k_final: int = 5

settings = Settings()
EOF

cat > src/rag/loaders.py <<'EOF'
from pathlib import Path
from bs4 import BeautifulSoup
from pypdf import PdfReader

def load_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')

def load_html(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser')
    return soup.get_text('\n', strip=True)

def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return '\n'.join(page.extract_text() or '' for page in reader.pages)

def load_document(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix in {'.txt', '.md'}:
        text = load_text(path)
    elif suffix in {'.html', '.htm'}:
        text = load_html(path)
    elif suffix == '.pdf':
        text = load_pdf(path)
    else:
        raise ValueError(f'Unsupported format: {suffix}')
    return {'source_file': path.name, 'text': text}
EOF

cat > src/rag/chunking.py <<'EOF'
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
EOF

cat > src/rag/ingest.py <<'EOF'
from pathlib import Path
from .loaders import load_document
from .chunking import fixed_size_chunks

def ingest_folder(folder: str):
    docs = []
    for path in Path(folder).glob('*'):
        if path.is_file():
            try:
                doc = load_document(path)
                chunks = fixed_size_chunks(doc['text'], doc['source_file'])
                for c in chunks:
                    docs.append({'source_file': c.source_file, 'chunk_index': c.chunk_index, 'strategy': c.strategy, 'text': c.text})
            except Exception:
                pass
    return docs
EOF

cat > src/rag/embeddings.py <<'EOF'
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def embed_texts(texts):
    resp = client.embeddings.create(model='text-embedding-3-small', input=texts)
    return [item.embedding for item in resp.data]
EOF

cat > src/rag/vectorstore.py <<'EOF'
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
EOF

cat > src/rag/retrieval.py <<'EOF'
from rank_bm25 import BM25Okapi

def tokenize(text: str):
    return [t.lower() for t in text.split()]

class HybridRetriever:
    def __init__(self, chunks, vectorstore):
        self.chunks = chunks
        self.vectorstore = vectorstore
        self.corpus = [tokenize(c['text']) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus) if chunks else None

    def sparse(self, query: str, k: int = 5):
        if not self.chunks:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return [{'chunk': self.chunks[i], 'score': float(score), 'source': 'bm25', 'rank': r+1} for r, (i, score) in enumerate(ranked)]

    def dense(self, query: str, k: int = 5):
        out = self.vectorstore.query(query, k=k)
        ids = out.get('ids', [[]])[0]
        scores = out.get('distances', [[]])[0]
        results = []
        for idx, cid in enumerate(ids):
            chunk = next((c for c in self.chunks if c['id'] == cid), None)
            if chunk:
                results.append({'chunk': chunk, 'score': float(scores[idx]) if idx < len(scores) else 0.0, 'source': 'dense', 'rank': idx+1})
        return results

    def fuse(self, dense_results, sparse_results, k: int = 5):
        scores = {}
        by_id = {c['id']: c for c in self.chunks}
        for rank, item in enumerate(dense_results, start=1):
            key = item['chunk']['id']
            scores[key] = scores.get(key, 0) + 1.0 / (60 + rank)
        for rank, item in enumerate(sparse_results, start=1):
            key = item['chunk']['id']
            scores[key] = scores.get(key, 0) + 1.0 / (60 + rank)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [{'chunk': by_id[cid], 'score': score} for cid, score in ranked]
EOF

cat > src/rag/generator.py <<'EOF'
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
EOF

cat > src/rag/build_index.py <<'EOF'
from .ingest import ingest_folder
from .vectorstore import VectorStore

def build(folder='docs'):
    chunks = ingest_folder(folder)
    for c in chunks:
        c['id'] = f"{c['source_file']}::{c['chunk_index']}"
    store = VectorStore()
    if chunks:
        store.add_chunks(chunks)
    return len(chunks)

if __name__ == '__main__':
    print(build())
EOF

cat > src/rag/api.py <<'EOF'
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
        c['id'] = #!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$HOME/Projects/rag_project"
mkdir -p "$PROJECT_DIR/src/rag" "$PROJECT_DIR/docs" "$PROJECT_DIR/tests" "$PROJECT_DIR/ui"
cd "$PROJECT_DIR"

cat > .gitignore <<'EOF'
__pycache__/
*.pyc
.env
.venv/
output/
chroma/
EOF

cat > requirements.txt <<'EOF'
fastapi
uvicorn
pydantic
python-dotenv
chromadb
openai
rank-bm25
numpy
pandas
pypdf
beautifulsoup4
markdown
streamlit
requests
EOF

cat > .env.example <<'EOF'
OPENAI_API_KEY=your_key_here
CHROMA_PATH=./chroma
EOF

cat > README.md <<'EOF'
# RAG Knowledge Base

## Run locally

1. Copy `.env.example` to `.env` and add your `OPENAI_API_KEY`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Build the index: `python -m src.rag.build_index`.
4. Start the API: `uvicorn src.rag.api:app --reload`.
5. Open docs at `http://127.0.0.1:8000/docs`.
6. Start the UI: `streamlit run ui/app.py`.

## Docker

- API + UI: `docker compose up --build`

## Endpoints

- `POST /v1/ingest` to index documents.
- `POST /v1/ask` to ask questions.
- `GET /health` for health checks.
EOF

cat > docs/sample.md <<'EOF'
# Sample Document

This is a sample internal document for the RAG system.
EOF

cat > tests/test_basic.py <<'EOF'
def test_placeholder():
    assert True
EOF

cat > src/rag/__init__.py <<'EOF'
EOF

cat > src/rag/config.py <<'EOF'
from pydantic import BaseModel

class Settings(BaseModel):
    chroma_path: str = './chroma'
    embedding_model: str = 'text-embedding-3-small'
    top_k_dense: int = 10
    top_k_sparse: int = 10
    top_k_final: int = 5

settings = Settings()
EOF

cat > src/rag/loaders.py <<'EOF'
from pathlib import Path
from bs4 import BeautifulSoup
from pypdf import PdfReader

def load_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')

def load_html(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser')
    return soup.get_text('\n', strip=True)

def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return '\n'.join(page.extract_text() or '' for page in reader.pages)

def load_document(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix in {'.txt', '.md'}:
        text = load_text(path)
    elif suffix in {'.html', '.htm'}:
        text = load_html(path)
    elif suffix == '.pdf':
        text = load_pdf(path)
    else:
        raise ValueError(f'Unsupported format: {suffix}')
    return {'source_file': path.name, 'text': text}
EOF

cat > src/rag/chunking.py <<'EOF'
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
EOF

cat > src/rag/ingest.py <<'EOF'
from pathlib import Path
from .loaders import load_document
from .chunking import fixed_size_chunks

def ingest_folder(folder: str):
    docs = []
    for path in Path(folder).glob('*'):
        if path.is_file():
            try:
                doc = load_document(path)
                chunks = fixed_size_chunks(doc['text'], doc['source_file'])
                for c in chunks:
                    docs.append({'source_file': c.source_file, 'chunk_index': c.chunk_index, 'strategy': c.strategy, 'text': c.text})
            except Exception:
                pass
    return docs
EOF

cat > src/rag/embeddings.py <<'EOF'
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def embed_texts(texts):
    resp = client.embeddings.create(model='text-embedding-3-small', input=texts)
    return [item.embedding for item in resp.data]
EOF

cat > src/rag/vectorstore.py <<'EOF'
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
EOF

cat > src/rag/retrieval.py <<'EOF'
from rank_bm25 import BM25Okapi

def tokenize(text: str):
    return [t.lower() for t in text.split()]

class HybridRetriever:
    def __init__(self, chunks, vectorstore):
        self.chunks = chunks
        self.vectorstore = vectorstore
        self.corpus = [tokenize(c['text']) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus) if chunks else None

    def sparse(self, query: str, k: int = 5):
        if not self.chunks:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return [{'chunk': self.chunks[i], 'score': float(score), 'source': 'bm25', 'rank': r+1} for r, (i, score) in enumerate(ranked)]

    def dense(self, query: str, k: int = 5):
        out = self.vectorstore.query(query, k=k)
        ids = out.get('ids', [[]])[0]
        scores = out.get('distances', [[]])[0]
        results = []
        for idx, cid in enumerate(ids):
            chunk = next((c for c in self.chunks if c['id'] == cid), None)
            if chunk:
                results.append({'chunk': chunk, 'score': float(scores[idx]) if idx < len(scores) else 0.0, 'source': 'dense', 'rank': idx+1})
        return results

    def fuse(self, dense_results, sparse_results, k: int = 5):
        scores = {}
        by_id = {c['id']: c for c in self.chunks}
        for rank, item in enumerate(dense_results, start=1):
            key = item['chunk']['id']
            scores[key] = scores.get(key, 0) + 1.0 / (60 + rank)
        for rank, item in enumerate(sparse_results, start=1):
            key = item['chunk']['id']
            scores[key] = scores.get(key, 0) + 1.0 / (60 + rank)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [{'chunk': by_id[cid], 'score': score} for cid, score in ranked]
EOF

cat > src/rag/generator.py <<'EOF'
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
EOF

cat > src/rag/build_index.py <<'EOF'
from .ingest import ingest_folder
from .vectorstore import VectorStore

def build(folder='docs'):
    chunks = ingest_folder(folder)
    for c in chunks:
        c['id'] = f"{c['source_file']}::{c['chunk_index']}"
    store = VectorStore()
    if chunks:
        store.add_chunks(chunks)
    return len(chunks)

if __name__ == '__main__':
    print(build())
EOF

cat > src/rag/api.py <<'EOF'
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
EOF

cat > src/rag/main.py <<'EOF'
from .api import app
EOF

cat > ui/app.py <<'EOF'
import streamlit as st
import requests

st.set_page_config(page_title='RAG Knowledge Base', layout='wide')
st.title('RAG Knowledge Base')

api = st.text_input('API URL', 'http://localhost:8000')
question = st.text_area('Ask a question')

if st.button('Ask') and question:
    r = requests.post(f'{api}/v1/ask', json={'question': question}, timeout=60)
    data = r.json()
    st.subheader('Answer')
    st.write(data.get('answer'))
    st.subheader('Retrieved Chunks')
    st.json(data.get('retrieved'))
EOF

cat > Dockerfile <<'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "src.rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

cat > docker-compose.yml <<'EOF'
version: '3.9'
services:
  api:
    build: .
    ports:
      - '8000:8000'
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
  ui:
    image: python:3.11-slim
    working_dir: /app
    volumes:
      - ./:/app
    command: sh -c "pip install -r requirements.txt && streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0"
    ports:
      - '8501:8501'
    depends_on:
      - api
EOF


