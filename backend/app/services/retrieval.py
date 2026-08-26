from __future__ import annotations

import json
import math
import re
from collections import Counter
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


TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9'-]*")
QUOTED_PHRASE_PATTERN = re.compile(r"['\"]([^'\"]{2,120})['\"]")
CASE_NAME_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z0-9'.-]*(?:\s+(?:of|and|the|for|in|[A-Z][A-Za-z0-9'.-]*)){0,8})"
    r"\s+(?:v\.?|V\.?|vs\.?|Vs\.?|VS\.?|versus|Versus)\s+"
    r"([A-Z][A-Za-z0-9'.-]*(?:\s+(?:of|and|the|for|in|[A-Z][A-Za-z0-9'.-]*)){0,8})\b"
)
PROPER_PHRASE_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z0-9'.-]*(?:\s+(?:of|and|the|for|in|[A-Z][A-Za-z0-9'.-]*)){1,6}\b"
)

STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "answer",
        "are",
        "as",
        "about",
        "by",
        "case",
        "correct",
        "did",
        "do",
        "does",
        "document",
        "explain",
        "for",
        "from",
        "give",
        "how",
        "in",
        "is",
        "it",
        "its",
        "judgment",
        "me",
        "of",
        "on",
        "or",
        "passage",
        "passages",
        "please",
        "question",
        "regarding",
        "retrieved",
        "summary",
        "tell",
        "that",
        "the",
        "this",
        "to",
        "v",
        "vs",
        "was",
        "were",
        "what",
        "when",
        "where",
        "whether",
        "which",
        "who",
        "why",
        "with",
    }
)


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

        dense_scores = _dense_scores(query, vectors, self.embeddings)
        searchable_texts = [
            _searchable_text(chunk=chunk, document_name=document_name)
            for chunk, document_name in zip(chunks, names, strict=True)
        ]
        bm25_scores = _bm25_scores(query, searchable_texts)
        exact_phrases = _extract_exact_phrases(query)
        exact_scores = _exact_match_scores(exact_phrases, searchable_texts)
        hybrid_scores = _hybrid_scores(dense_scores, bm25_scores, exact_scores, exact_phrases)

        count = min(top_k, len(chunks))
        ranked = sorted(
            range(len(chunks)),
            key=lambda index: (
                hybrid_scores[index],
                exact_scores[index],
                bm25_scores[index],
                dense_scores[index],
            ),
            reverse=True,
        )[:count]

        return [
            RetrievedChunk(
                chunk_id=chunks[index].id,
                document_id=chunks[index].document_id,
                document_name=names[index],
                page_number=chunks[index].page_number,
                section=chunks[index].section,
                clause_type=chunks[index].clause_type,
                text=chunks[index].text,
                score=max(0.0, min(1.0, float(hybrid_scores[index]))),
            )
            for index in ranked
            if index >= 0
        ]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def _dense_scores(
    query: str, vectors: list[list[float]], embeddings: EmbeddingProvider
) -> list[float]:
    matrix = np.asarray(vectors, dtype="float32")
    query_vector = np.asarray([embeddings.embed_query(query)], dtype="float32")
    if matrix.shape[1] != query_vector.shape[1]:
        raise RuntimeError(
            "Stored embeddings were created by a different provider. Re-ingest the documents "
            "after changing EMBEDDING_PROVIDER or EMBEDDING_MODEL."
        )
    matrix = _normalize(matrix)
    query_vector = _normalize(query_vector)
    if faiss is not None:
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        scores, indices = index.search(query_vector, len(vectors))
        dense_scores = [0.0] * len(vectors)
        for vector_index, score in zip(indices[0].tolist(), scores[0].tolist(), strict=False):
            if vector_index >= 0:
                dense_scores[vector_index] = float(score)
        return dense_scores
    return (matrix @ query_vector[0]).astype(float).tolist()


def _searchable_text(*, chunk: Chunk, document_name: str) -> str:
    return "\n".join(
        part
        for part in (
            document_name,
            chunk.section or "",
            chunk.clause_type,
            chunk.text,
        )
        if part
    )


def _bm25_scores(query: str, documents: list[str]) -> list[float]:
    query_terms = _query_terms(query)
    if not query_terms:
        return [0.0] * len(documents)

    tokenized_documents = [_tokenize(document) for document in documents]
    if not tokenized_documents:
        return []

    document_frequencies: Counter[str] = Counter()
    term_counts: list[Counter[str]] = []
    for tokens in tokenized_documents:
        counts = Counter(tokens)
        term_counts.append(counts)
        document_frequencies.update(counts.keys())

    document_lengths = [len(tokens) for tokens in tokenized_documents]
    average_length = sum(document_lengths) / len(document_lengths) if document_lengths else 1.0
    average_length = average_length or 1.0

    k1 = 1.5
    b = 0.75
    total_documents = len(documents)
    scores = [0.0] * total_documents
    for term, query_frequency in Counter(query_terms).items():
        document_frequency = document_frequencies.get(term, 0)
        if document_frequency == 0:
            continue
        idf = math.log(1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5))
        for index, counts in enumerate(term_counts):
            term_frequency = counts.get(term, 0)
            if term_frequency == 0:
                continue
            denominator = term_frequency + k1 * (
                1 - b + b * document_lengths[index] / average_length
            )
            scores[index] += (
                idf
                * (term_frequency * (k1 + 1))
                / denominator
                * min(query_frequency, 3)
            )
    return _normalize_scores(scores)


def _hybrid_scores(
    dense_scores: list[float],
    bm25_scores: list[float],
    exact_scores: list[float],
    exact_phrases: list[str],
) -> list[float]:
    dense = _normalize_scores(dense_scores)
    has_exact_query = bool(exact_phrases)
    if has_exact_query:
        dense_weight = 0.35
        bm25_weight = 0.40
        exact_weight = 0.25
    else:
        dense_weight = 0.65
        bm25_weight = 0.35
        exact_weight = 0.0

    scores: list[float] = []
    for index, dense_score in enumerate(dense):
        score = (
            dense_weight * dense_score
            + bm25_weight * bm25_scores[index]
            + exact_weight * exact_scores[index]
        )
        if has_exact_query and exact_scores[index] >= 0.95:
            score = max(score, 0.9 + 0.1 * dense_score)
        elif has_exact_query and exact_scores[index] >= 0.5:
            score = max(score, 0.65 + 0.2 * exact_scores[index])
        scores.append(max(0.0, min(1.0, score)))
    return scores


def _extract_exact_phrases(query: str) -> list[str]:
    phrases: list[str] = []
    phrases.extend(match.group(1) for match in QUOTED_PHRASE_PATTERN.finditer(query))
    for match in CASE_NAME_PATTERN.finditer(query):
        phrases.extend([match.group(0), match.group(1), match.group(2)])
    phrases.extend(match.group(0) for match in PROPER_PHRASE_PATTERN.finditer(query))

    deduped: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        cleaned = _clean_phrase(phrase)
        if not cleaned:
            continue
        key = _normalized_text(cleaned)
        if key and key not in seen:
            seen.add(key)
            deduped.append(cleaned)
    return deduped


def _exact_match_scores(phrases: list[str], documents: list[str]) -> list[float]:
    weighted_phrases = [
        (phrase, terms, max(2, len(terms)))
        for phrase in phrases
        if (terms := sorted(set(_query_terms(phrase))))
    ]
    if not weighted_phrases:
        return [0.0] * len(documents)

    total_weight = sum(weight for _, _, weight in weighted_phrases) or 1
    scores: list[float] = []
    for document in documents:
        normalized_document = f" {_normalized_text(document)} "
        document_terms = set(_tokenize(document))
        score = 0.0
        for phrase, terms, weight in weighted_phrases:
            normalized_phrase = _normalized_text(phrase)
            if normalized_phrase and f" {normalized_phrase} " in normalized_document:
                score += weight
                continue

            overlap = sum(1 for term in terms if term in document_terms)
            if overlap == len(terms):
                score += weight * 0.8
            elif overlap >= 2:
                score += weight * 0.35 * (overlap / len(terms))
        scores.append(min(1.0, score / total_weight))
    return scores


def _query_terms(text: str) -> list[str]:
    tokens = _tokenize(text)
    terms = [
        token
        for token in tokens
        if token not in STOP_WORDS and (len(token) > 1 or token.isdigit())
    ]
    return terms or tokens


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _clean_phrase(phrase: str) -> str | None:
    cleaned = " ".join(phrase.split()).strip(" .,:;()[]{}")
    if len(_query_terms(cleaned)) < 2:
        return None
    return cleaned


def _normalized_text(text: str) -> str:
    text = re.sub(r"\b(?:versus|vs\.?|v\.?)\b", " vs ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(text.split())


def _normalize_scores(scores: list[float]) -> list[float]:
    positive = [max(0.0, float(score)) for score in scores]
    maximum = max(positive, default=0.0)
    if maximum == 0:
        return [0.0] * len(scores)
    return [score / maximum for score in positive]
