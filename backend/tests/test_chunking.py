from backend.app.services.chunking import chunk_pages, classify_clause
from backend.app.services.text_extraction import ExtractedPage


def test_clause_classification() -> None:
    assert classify_clause("Either party may terminate on 30 days notice") == "termination"
    assert (
        classify_clause("The receiving party must keep all information confidential")
        == "confidentiality"
    )


def test_chunker_keeps_page_and_section_metadata() -> None:
    pages = [
        ExtractedPage(
            4,
            "7. TERMINATION\n\nEither party may terminate this agreement on written notice.",
        )
    ]
    chunks = chunk_pages(pages, chunk_size=200, overlap=20)
    assert chunks
    assert chunks[0].page_number == 4
    assert chunks[0].section == "7. TERMINATION"
    assert chunks[0].clause_type == "termination"
