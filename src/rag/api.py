from functools import lru_cache
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .build_index import build
from .contracts import (
    AnswerStatus,
    Citation,
    ErrorDetails,
    ErrorResponse,
    QueryRequest,
    QueryResponse,
    ResponseMetadata,
)
from .generator import answer
from .ingest import ingest_folder
from .retrieval import HybridRetriever
from .vectorstore import VectorStore

app = FastAPI(title="Production Agentic RAG Platform")
CACHE = {"chunks": []}


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", str(uuid4()))
    payload = ErrorResponse(
        error=ErrorDetails(
            code="INVALID_REQUEST",
            message="Request validation failed",
            request_id=request_id,
        )
    )
    return JSONResponse(status_code=422, content=payload.model_dump())


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    return VectorStore()


class AskRequest(BaseModel):
    question: str


def _load_cached_chunks(folder: str) -> int:
    CACHE["chunks"] = []
    for chunk in ingest_folder(folder):
        chunk["id"] = f"{chunk['source_file']}::{chunk['chunk_index']}"
        CACHE["chunks"].append(chunk)
    return len(CACHE["chunks"])


def _retrieve(question: str, top_k: int):
    chunks = CACHE["chunks"]
    retriever = HybridRetriever(chunks, get_store())
    sparse = retriever.sparse(question, k=top_k)
    dense = retriever.dense(question, k=top_k)
    fused = retriever.fuse(dense, sparse, k=top_k)
    return dense, sparse, fused


def _answer_with_contract(question: str, fused, request_id: str | None = None):
    response = answer(question, [item["chunk"] for item in fused])
    citations = [
        Citation(
            source=item["chunk"]["source_file"],
            chunk_id=item["chunk"]["id"],
            text=item["chunk"]["text"],
        )
        for item in fused
    ]
    status = AnswerStatus.ANSWERED if fused else AnswerStatus.INSUFFICIENT_EVIDENCE
    return QueryResponse(
        answer=response.get("answer", ""),
        citations=citations,
        status=status,
        metadata=ResponseMetadata(
            retrieved_documents=len(fused),
            request_id=request_id,
        ),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/ingest")
def ingest(payload: dict):
    folder = payload.get("folder", "docs")
    count = build(folder)
    _load_cached_chunks(folder)
    return {"status": "ok", "chunks_indexed": count}


@app.post("/v1/ask")
def ask(req: AskRequest):
    dense, sparse, fused = _retrieve(req.question, top_k=5)
    response = answer(req.question, [item["chunk"] for item in fused])
    return {
        "question": req.question,
        "dense": dense,
        "sparse": sparse,
        "retrieved": fused,
        **response,
    }


@app.post("/v1/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request):
    _, _, fused = _retrieve(req.question, top_k=req.top_k)
    return _answer_with_contract(
        req.question,
        fused,
        request_id=request.state.request_id,
    )
