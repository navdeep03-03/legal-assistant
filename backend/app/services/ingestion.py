from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import select

from ..config import Settings
from ..database import Database
from ..models import Chunk, Document
from .chunking import chunk_pages
from .embeddings import EmbeddingProvider
from .text_extraction import SUPPORTED_EXTENSIONS, DocumentExtractionError, extract_document


class IngestionError(ValueError):
    pass


class IngestionService:
    def __init__(self, database: Database, embeddings: EmbeddingProvider, settings: Settings):
        self.database = database
        self.embeddings = embeddings
        self.settings = settings

    async def ingest(
        self, upload: UploadFile, tenant_id: str, user_id: str
    ) -> tuple[Document, bool]:
        safe_filename = Path(upload.filename or "document").name
        extension = Path(safe_filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise IngestionError("Unsupported file type. Upload PDF, DOCX, TXT, or Markdown files.")

        content = await upload.read(self.settings.max_upload_mb * 1024 * 1024 + 1)
        if len(content) > self.settings.max_upload_mb * 1024 * 1024:
            raise IngestionError(
                f"{safe_filename} exceeds the {self.settings.max_upload_mb} MB upload limit."
            )
        digest = hashlib.sha256(content).hexdigest()

        with self.database.session() as session:
            duplicate = session.scalar(
                select(Document).where(
                    Document.tenant_id == tenant_id,
                    Document.sha256 == digest,
                    Document.status == "ready",
                )
            )
            if duplicate:
                return duplicate, True

        try:
            pages = extract_document(safe_filename, content)
        except DocumentExtractionError as exc:
            raise IngestionError(str(exc)) from exc
        text_chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        if not text_chunks:
            raise IngestionError(f"No searchable text could be created from {safe_filename}.")

        vectors = self.embeddings.embed_texts([chunk.text for chunk in text_chunks])
        document_id = str(uuid4())
        tenant_dir = self.settings.upload_dir / tenant_id
        tenant_dir.mkdir(parents=True, exist_ok=True)
        stored_path = tenant_dir / f"{document_id}{extension}"
        stored_path.write_bytes(content)

        document = Document(
            id=document_id,
            tenant_id=tenant_id,
            uploaded_by=user_id,
            filename=safe_filename,
            display_name=Path(safe_filename).stem.replace("_", " ").replace("-", " ").strip(),
            mime_type=upload.content_type
            or mimetypes.guess_type(safe_filename)[0]
            or "application/octet-stream",
            file_path=str(stored_path.resolve()),
            sha256=digest,
            size_bytes=len(content),
            status="ready",
            page_count=len(pages),
            chunk_count=len(text_chunks),
        )

        try:
            with self.database.session() as session:
                session.add(document)
                session.add_all(
                    [
                        Chunk(
                            id=str(uuid4()),
                            document_id=document_id,
                            tenant_id=tenant_id,
                            chunk_index=chunk.chunk_index,
                            page_number=chunk.page_number,
                            section=chunk.section,
                            clause_type=chunk.clause_type,
                            text=chunk.text,
                            embedding_json=json.dumps(vector, separators=(",", ":")),
                        )
                        for chunk, vector in zip(text_chunks, vectors, strict=True)
                    ]
                )
        except Exception:
            stored_path.unlink(missing_ok=True)
            raise
        return document, False

    def delete(self, document_id: str, tenant_id: str) -> bool:
        with self.database.session() as session:
            document = session.scalar(
                select(Document).where(Document.id == document_id, Document.tenant_id == tenant_id)
            )
            if not document:
                return False
            file_path = Path(document.file_path)
            session.delete(document)
        file_path.unlink(missing_ok=True)
        return True
