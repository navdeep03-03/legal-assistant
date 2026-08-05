from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select

from ..database import Database
from ..models import Chunk, Document
from .embeddings import EmbeddingProvider

try:  # FAISS is optional because some platforms do not publish wheels for every Python version.
    import faiss  # type: ignore

    VECTOR_ENGINE = "faiss"
except ImportError:  # pragma: no cover - environment dependent
    faiss = None
    VECTOR_ENGINE = "numpy"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None
    section: str | None
    clause_type: str
    text: str
    score: float


class Retriever:
    def __init__(self, database: Database, embeddings: EmbeddingProvider):
        self.database = database
        self.embeddings = embeddings
        self.vector_engine = VECTOR_ENGINE

    def search(
        self,
        *,
        tenant_id: str,
        query: str,
        document_ids: list[str] | None = None,
        clause_types: list[str] | None = None,
        top_k: int = 6,
    ) -> list[RetrievedChunk]:
        statement = (
            select(Chunk, Document.display_name)
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.tenant_id == tenant_id, Document.status == "ready")
        )
        if document_ids:
            statement = statement.where(Chunk.document_id.in_(document_ids))
        if clause_types:
            statement = statement.where(Chunk.clause_type.in_(clause_types))

        with self.database.session() as session:
            rows = list(session.execute(statement).all())
        if not rows:
            return []

        chunks: list[Chunk] = []
        names: list[str] = []
        vectors: list[list[float]] = []
        for chunk, document_name in rows:
            try:
                vector = json.loads(chunk.embedding_json)
            except (TypeError, json.JSONDecodeError):
                continue
            chunks.append(chunk)
            names.append(document_name)
            vectors.append(vector)
        if not vectors:
            return []

        matrix = np.asarray(vectors, dtype="float32")
        query_vector = np.asarray([self.embeddings.embed_query(query)], dtype="float32")
        if matrix.shape[1] != query_vector.shape[1]:
            raise RuntimeError(
                "Stored embeddings were created by a different provider. Re-ingest the documents "
                "after changing EMBEDDING_PROVIDER or EMBEDDING_MODEL."
            )
        matrix = _normalize(matrix)
        query_vector = _normalize(query_vector)
        count = min(top_k, len(chunks))

        if faiss is not None:
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            scores, indices = index.search(query_vector, count)
            ranked = zip(indices[0].tolist(), scores[0].tolist(), strict=False)
        else:
            scores = matrix @ query_vector[0]
            indices = np.argsort(scores)[::-1][:count]
            ranked = ((int(index), float(scores[index])) for index in indices)

        return [
            RetrievedChunk(
                chunk_id=chunks[index].id,
                document_id=chunks[index].document_id,
                document_name=names[index],
                page_number=chunks[index].page_number,
                section=chunks[index].section,
                clause_type=chunks[index].clause_type,
                text=chunks[index].text,
                score=max(0.0, min(1.0, float(score))),
            )
            for index, score in ranked
            if index >= 0
        ]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms
