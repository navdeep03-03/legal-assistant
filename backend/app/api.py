from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile, status
from sqlalchemy import select

from .dependencies import RequestContext, get_request_context
from .models import Document
from .schemas import (
    AskRequest,
    AskResponse,
    DocumentOut,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SummarizeRequest,
    UploadResponse,
)
from .services.ingestion import IngestionError

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    embeddings = request.app.state.embeddings
    orchestrator = request.app.state.orchestrator
    return HealthResponse(
        status="ok",
        mode="openai" if orchestrator.generator else "local-demo",
        model=settings.openai_model,
        embedding_model=embeddings.model_name,
        vector_engine=request.app.state.retriever.vector_engine,
    )


@router.get("/documents", response_model=list[DocumentOut], tags=["documents"])
def list_documents(
    request: Request,
    context: Annotated[RequestContext, Depends(get_request_context)],
) -> list[Document]:
    database = request.app.state.database
    with database.session() as session:
        return list(
            session.scalars(
                select(Document)
                .where(Document.tenant_id == context.tenant_id)
                .order_by(Document.created_at.desc())
            ).all()
        )


@router.post(
    "/upload-documents",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
async def upload_documents(
    request: Request,
    context: Annotated[RequestContext, Depends(get_request_context)],
    files: Annotated[list[UploadFile], File(description="PDF, DOCX, TXT, or Markdown files")],
) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Select at least one file")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Upload at most 20 files per request")

    ingestion = request.app.state.ingestion
    created: list[Document] = []
    duplicates: list[Document] = []
    try:
        for upload in files:
            document, duplicate = await ingestion.ingest(upload, context.tenant_id, context.user_id)
            (duplicates if duplicate else created).append(document)
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return UploadResponse(documents=created, duplicates=duplicates)


@router.delete(
    "/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["documents"]
)
def delete_document(
    document_id: str,
    request: Request,
    context: Annotated[RequestContext, Depends(get_request_context)],
) -> Response:
    deleted = request.app.state.ingestion.delete(document_id, context.tenant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/search", response_model=SearchResponse, tags=["legal-search"])
def search(
    payload: SearchRequest,
    request: Request,
    context: Annotated[RequestContext, Depends(get_request_context)],
) -> SearchResponse:
    try:
        results = request.app.state.retriever.search(
            tenant_id=context.tenant_id,
            query=payload.query,
            document_ids=payload.document_ids,
            clause_types=payload.clause_types,
            top_k=payload.top_k,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return SearchResponse(
        results=[
            SearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                document_name=result.document_name,
                page=result.page_number,
                section=result.section,
                clause_type=result.clause_type,
                text=result.text,
                score=round(result.score, 4),
            )
            for result in results
        ],
        mode=request.app.state.embeddings.mode,
    )


@router.post("/ask", response_model=AskResponse, tags=["legal-analysis"])
def ask(
    payload: AskRequest,
    request: Request,
    context: Annotated[RequestContext, Depends(get_request_context)],
) -> AskResponse:
    try:
        return request.app.state.orchestrator.ask(
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            question=payload.question,
            document_ids=payload.document_ids,
            conversation_id=payload.conversation_id,
            top_k=payload.top_k,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/summarize-document", response_model=AskResponse, tags=["legal-analysis"])
def summarize_document(
    payload: SummarizeRequest,
    request: Request,
    context: Annotated[RequestContext, Depends(get_request_context)],
) -> AskResponse:
    focus = f" Focus especially on: {payload.focus}" if payload.focus else ""
    question = (
        "Summarize this document's purpose, parties, key obligations, payment terms, "
        "duration and termination, liability allocation, dispute terms, and material risks." + focus
    )
    try:
        return request.app.state.orchestrator.ask(
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            question=question,
            document_ids=[payload.document_id],
            conversation_id=None,
            top_k=12,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
