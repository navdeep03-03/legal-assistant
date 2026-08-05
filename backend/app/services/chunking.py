from __future__ import annotations

import re
from dataclasses import dataclass

from .text_extraction import ExtractedPage


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    text: str
    page_number: int | None
    section: str | None
    clause_type: str


CLAUSE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "termination": ("terminat", "cancel", "expiry", "expiration", "notice period"),
    "confidentiality": ("confidential", "non-disclosure", "proprietary information"),
    "indemnity": ("indemn", "hold harmless", "defend against"),
    "limitation of liability": (
        "limitation of liability",
        "liability cap",
        "consequential damages",
    ),
    "payment": ("payment", "invoice", "fees", "late charge", "compensation"),
    "governing law": ("governing law", "jurisdiction", "venue", "arbitration"),
    "privacy": ("personal data", "data protection", "privacy", "processor", "controller"),
    "intellectual property": ("intellectual property", "copyright", "trademark", "license"),
    "force majeure": ("force majeure", "act of god", "beyond reasonable control"),
    "warranty": ("warrant", "as is", "merchantability", "fitness for a particular"),
}


def classify_clause(text: str) -> str:
    lowered = text.lower()
    matches = [
        (name, sum(lowered.count(keyword) for keyword in keywords))
        for name, keywords in CLAUSE_KEYWORDS.items()
    ]
    best_name, best_score = max(matches, key=lambda item: item[1])
    return best_name if best_score else "general"


def chunk_pages(pages: list[ExtractedPage], chunk_size: int, overlap: int) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    current_section: str | None = None

    for page in pages:
        paragraphs = [
            part.strip() for part in re.split(r"\n\s*\n|(?<=\.)\s*\n", page.text) if part.strip()
        ]
        buffer = ""
        buffer_section = current_section

        for paragraph in paragraphs:
            if _looks_like_heading(paragraph):
                if buffer:
                    chunks.extend(
                        _split_buffer(
                            buffer,
                            page.page_number,
                            buffer_section,
                            chunk_size,
                            overlap,
                            len(chunks),
                        )
                    )
                    buffer = ""
                current_section = paragraph[:255]
                buffer_section = current_section

            candidate = f"{buffer}\n\n{paragraph}".strip()
            if buffer and len(candidate) > chunk_size:
                chunks.extend(
                    _split_buffer(
                        buffer, page.page_number, buffer_section, chunk_size, overlap, len(chunks)
                    )
                )
                carry = buffer[-overlap:].lstrip() if overlap else ""
                buffer = f"{carry}\n\n{paragraph}".strip()
                buffer_section = current_section
            else:
                buffer = candidate

        if buffer:
            chunks.extend(
                _split_buffer(
                    buffer, page.page_number, buffer_section, chunk_size, overlap, len(chunks)
                )
            )

    return [
        TextChunk(index, chunk.text, chunk.page_number, chunk.section, chunk.clause_type)
        for index, chunk in enumerate(chunks)
        if chunk.text.strip()
    ]


def _split_buffer(
    text: str,
    page_number: int | None,
    section: str | None,
    chunk_size: int,
    overlap: int,
    start_index: int,
) -> list[TextChunk]:
    if len(text) <= chunk_size:
        return [TextChunk(start_index, text, page_number, section, classify_clause(text))]

    result: list[TextChunk] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            result.append(
                TextChunk(
                    start_index + len(result), piece, page_number, section, classify_clause(piece)
                )
            )
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return result


def _looks_like_heading(text: str) -> bool:
    one_line = "\n" not in text and len(text) <= 140
    if not one_line:
        return False
    if re.match(r"^(section\s+)?\d+(\.\d+)*[.):\s-]+\S+", text, re.IGNORECASE):
        return True
    words = text.split()
    return 1 <= len(words) <= 12 and (text.isupper() or text.istitle()) and not text.endswith(".")
