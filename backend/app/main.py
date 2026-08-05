from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import router
from .config import Settings, get_settings
from .database import Database
from .services.embeddings import create_embedding_provider
from .services.ingestion import IngestionService
from .services.orchestration import AnswerOrchestrator
from .services.retrieval import Retriever


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        resolved_settings.prepare_directories()
        database = Database(resolved_settings.database_url)
        database.create_all()
        embeddings = create_embedding_provider(resolved_settings)
        retriever = Retriever(database, embeddings)

        application.state.settings = resolved_settings
        application.state.database = database
        application.state.embeddings = embeddings
        application.state.retriever = retriever
        application.state.ingestion = IngestionService(database, embeddings, resolved_settings)
        application.state.orchestrator = AnswerOrchestrator(database, retriever, resolved_settings)
        try:
            yield
        finally:
            database.dispose()

    application = FastAPI(
        title=resolved_settings.app_name,
        version="1.0.0",
        description=(
            "Tenant-scoped legal document ingestion, semantic retrieval, and grounded Q&A. "
            "Answers are informational and are not legal advice."
        ),
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Tenant-ID", "X-User-ID", "X-API-Key"],
    )
    application.include_router(router, prefix=resolved_settings.api_prefix)

    frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if frontend_dist.exists():
        application.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    return application


app = create_app()
