import time
from functools import lru_cache
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .auth import require_api_key
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
from .rate_limit import enforce_rate_limit
from .retrieval import HybridRetriever, filter_chunks
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


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", str(uuid4()))
    payload = ErrorResponse(
        error=ErrorDetails(
            code="INTERNAL_ERROR",
            message="An unexpected internal error occurred",
            request_id=request_id,
        )
    )
    return JSONResponse(status_code=500, content=payload.model_dump())


@app.exception_handler(__import__("fastapi").HTTPException)
async def http_exception_handler(request: Request, exc):
    request_id = getattr(request.state, "request_id", str(uuid4()))
    detail = exc.detail if isinstance(exc.detail, dict) else None
    if detail and "error" in detail:
        content = detail
    else:
        code = "UNAUTHORIZED" if exc.status_code == 401 else "RATE_LIMIT_EXCEEDED" if exc.status_code == 429 else "HTTP_ERROR"
        payload = ErrorResponse(
            error=ErrorDetails(code=code, message=str(exc.detail), request_id=request_id)
        )
        content = payload.model_dump()
    return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)


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


def _retrieve(question: str, top_k: int, filters=None):
    chunks = filter_chunks(CACHE["chunks"], filters)
    retriever = HybridRetriever(chunks, get_store())
    sparse = retriever.sparse(question, k=top_k)
    dense = retriever.dense(question, k=top_k)
    fused = retriever.fuse(dense, sparse, k=top_k)
    return dense, sparse, fused


def _answer_with_contract(question: str, fused, request_id: str | None = None, started_at: float | None = None):
    response = answer(question, [item["chunk"] for item in fused])
    citations = [
        Citation(source=item["chunk"]["source_file"], chunk_id=item["chunk"]["id"], text=item["chunk"]["text"])
        for item in fused
    ]
    status = AnswerStatus.ANSWERED if fused else AnswerStatus.INSUFFICIENT_EVIDENCE
    latency_ms = None if started_at is None else round((time.perf_counter() - started_at) * 1000, 3)
    return QueryResponse(answer=response.get("answer", ""), citations=citations, status=status, metadata=ResponseMetadata(retrieved_documents=len(fused), request_id=request_id, latency_ms=latency_ms))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/ingest", dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)])
def ingest(payload: dict):
    folder = payload.get("folder", "docs")
    count = build(folder)
    _load_cached_chunks(folder)
    return {"status": "ok", "chunks_indexed": count}


@app.post("/v1/ask", dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)])
def ask(req: AskRequest):
    dense, sparse, fused = _retrieve(req.question, top_k=5)
    response = answer(req.question, [item["chunk"] for item in fused])
    return {"question": req.question, "dense": dense, "sparse": sparse, "retrieved": fused, **response}


@app.post("/v1/query", response_model=QueryResponse, dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)])
def query(req: QueryRequest, request: Request):
    started_at = time.perf_counter()
    _, _, fused = _retrieve(req.question, top_k=req.top_k, filters=req.filters)
    return _answer_with_contract(req.question, fused, request_id=request.state.request_id, started_at=started_at)
