"""Stable request and response models for the RAG service."""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    REFUSED = "refused"
    ERROR = "error"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, gt=0, le=20)
    filters: Optional[Dict[str, str]] = None


class Citation(BaseModel):
    source: str = Field(..., min_length=1)
    chunk_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class ResponseMetadata(BaseModel):
    retrieved_documents: int = Field(default=0, ge=0)
    latency_ms: Optional[float] = Field(default=None, ge=0)
    model_version: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    status: AnswerStatus
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


class ErrorDetails(BaseModel):
    code: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    request_id: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetails
