from __future__ import annotations

import re
from dataclasses import dataclass

from .text_extraction import ExtractedPage

TOKEN_PATTERN = re.compile(r"\S+")
NUMBERED_ENTRY_START_PATTERN = re.compile(
    r"(?m)^\s*\d{1,3}\.\s*(?:\n\s*)*(?=\d{2}-\d{2}-\d{4}\b)"
)


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
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    overlap = max(0, min(overlap, chunk_size - 1))
    chunks: list[TextChunk] = []
    current_section: str | None = None

    for page in pages:
        for segment in _structured_segments(page.text):
            segment_section = _numbered_entry_section(segment) or current_section
            paragraphs = [
                part.strip()
                for part in re.split(r"\n\s*\n|(?<=\.)\s*\n", segment)
                if part.strip()
            ]
            buffer = ""
            buffer_section = segment_section

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
                if buffer and _token_count(candidate) > chunk_size:
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
                    carry = _tail_tokens(buffer, overlap)
                    buffer = f"{carry}\n\n{paragraph}".strip()
                    buffer_section = segment_section or current_section
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
    tokens = list(TOKEN_PATTERN.finditer(text))
    if not tokens:
        return []
    if len(tokens) <= chunk_size:
        return [TextChunk(start_index, text, page_number, section, classify_clause(text))]

    result: list[TextChunk] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        if end < len(tokens):
            boundary = _sentence_boundary(tokens, start, end)
            if boundary and boundary > start + chunk_size // 2:
                end = boundary
        piece = text[tokens[start].start() : tokens[end - 1].end()].strip()
        if piece:
            result.append(
                TextChunk(
                    start_index + len(result), piece, page_number, section, classify_clause(piece)
                )
            )
        if end >= len(tokens):
            break
        start = max(end - overlap, start + 1)
    return result


def _structured_segments(text: str) -> list[str]:
    entry_starts = [match.start() for match in NUMBERED_ENTRY_START_PATTERN.finditer(text)]
    if not entry_starts:
        return [text]

    segments: list[str] = []
    if entry_starts[0] > 0:
        _append_segment(segments, text[: entry_starts[0]])

    for index, start in enumerate(entry_starts):
        end = entry_starts[index + 1] if index + 1 < len(entry_starts) else len(text)
        _append_segment(segments, text[start:end])

    return segments or [text]


def _append_segment(segments: list[str], text: str) -> None:
    segment = text.strip()
    if _token_count(segment) >= 3:
        segments.append(segment)


def _numbered_entry_section(text: str) -> str | None:
    normalized = " ".join(text.split())
    match = re.match(
        r"^\d{1,3}\.\s+(?P<date>\d{2}-\d{2}-\d{4})\s+(?P<body>.+)$",
        normalized,
    )
    if not match:
        return None

    body = match.group("body")
    stop = re.search(
        r"\b(?:W\.P|C\.A|CRL\.A|SLP|T\.P|Diary|Plea|Challenge|Whether|Question\(s\):|"
        r"\d{4}\s+INSC)\b",
        body,
        re.IGNORECASE,
    )
    title = body[: stop.start()].strip(" .") if stop else body.strip(" .")
    if not title:
        return None
    return f"{match.group('date')} {title}"[:255]


def _token_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def _tail_tokens(text: str, count: int) -> str:
    if count <= 0:
        return ""
    tokens = list(TOKEN_PATTERN.finditer(text))
    if not tokens:
        return ""
    start = max(0, len(tokens) - count)
    return text[tokens[start].start() : tokens[-1].end()].strip()


def _sentence_boundary(tokens: list[re.Match[str]], start: int, end: int) -> int | None:
    for index in range(end - 1, start, -1):
        if tokens[index].group(0).endswith((".", "?", "!", ";", ":")):
            return index + 1
    return None


def _looks_like_heading(text: str) -> bool:
    one_line = "\n" not in text and len(text) <= 140
    if not one_line:
        return False
    if re.match(r"^(section\s+)?\d+(\.\d+)*[.):\s-]+\S+", text, re.IGNORECASE):
        return True
    words = text.split()
    return 1 <= len(words) <= 12 and (text.isupper() or text.istitle()) and not text.endswith(".")
