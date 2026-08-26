from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    display_name: str
    mime_type: str
    size_bytes: int
    status: str
    page_count: int
    chunk_count: int
    created_at: datetime


class UploadResponse(BaseModel):
    documents: list[DocumentOut]
    duplicates: list[DocumentOut] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4_000)
    document_ids: list[str] | None = Field(default=None, max_length=50)
    conversation_id: str | None = None
    top_k: int = Field(default=6, ge=1, le=12)

    @field_validator("question")
    @classmethod
    def question_must_contain_text(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if len(cleaned) < 3:
            raise ValueError("Question must contain at least 3 characters")
        return cleaned


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2_000)
    document_ids: list[str] | None = Field(default=None, max_length=50)
    clause_types: list[str] | None = Field(default=None, max_length=20)
    top_k: int = Field(default=8, ge=1, le=25)


class SummarizeRequest(BaseModel):
    document_id: str
    focus: str | None = Field(default=None, max_length=1_000)


class SourceCitation(BaseModel):
    source_id: str
    document_id: str
    document_name: str
    chunk_id: str
    page: int | None = None
    section: str | None = None
    clause_type: str
    quote: str
    explanation: str = ""
    relevance: float = Field(ge=0, le=1)


class RiskFlag(BaseModel):
    severity: Literal["high", "medium", "low"]
    title: str
    detail: str


class AskResponse(BaseModel):
    answer: str
    citations: list[SourceCitation]
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    warning: str
    source_documents: list[str]
    conversation_id: str
    retrieval_used: bool
    mode: Literal["mistral", "openai", "local-demo"]


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page: int | None
    section: str | None
    clause_type: str
    text: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    mode: Literal["openai", "local-demo"]


class HealthResponse(BaseModel):
    status: str
    mode: Literal["mistral", "openai", "local-demo"]
    model: str
    embedding_model: str
    vector_engine: str
