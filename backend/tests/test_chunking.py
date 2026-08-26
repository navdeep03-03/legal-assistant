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


def test_chunker_uses_token_windows_with_overlap() -> None:
    pages = [ExtractedPage(1, " ".join(f"token{index}" for index in range(1, 13)))]
    chunks = chunk_pages(pages, chunk_size=5, overlap=2)

    assert len(chunks) == 4
    assert chunks[0].text.split() == ["token1", "token2", "token3", "token4", "token5"]
    assert chunks[1].text.split()[:2] == ["token4", "token5"]


def test_chunker_splits_numbered_case_digest_entries() -> None:
    pages = [
        ExtractedPage(
            2,
            """S. No. Date of Judgment Cause Title/Case No. Subject Judgment Summary

1.

03-01-2024
Vishal Tiwari Vs Union of
India

W.P.(C) No. 162/2023
Plea challenging SEBI's investigation into the
Adani Group and seeking constitution of
Special Investigation Team (SIT).

Vishal Tiwari Union of India
2024 INSC 3 (3 January 2024)

2.

08-01-2024 Bilkis Yakub Rasool Vs
Union Of India

W.P.(Crl.) No. 491/2022
Challenge to the remission orders passed by
the State of Gujarat.""",
        )
    ]

    chunks = chunk_pages(pages, chunk_size=500, overlap=50)
    vishal_chunks = [chunk for chunk in chunks if "Vishal Tiwari" in chunk.text]

    assert vishal_chunks
    assert all("Bilkis" not in chunk.text for chunk in vishal_chunks)
    assert any("Adani Group" in chunk.text for chunk in vishal_chunks)
    assert vishal_chunks[0].section == "03-01-2024 Vishal Tiwari Vs Union of India"


def test_chunker_does_not_mix_continuation_text_with_next_numbered_entry() -> None:
    pages = [
        ExtractedPage(
            3,
            """osed for the offence of murder?

View Judgment
Full Judgment

10.

21-03-2024 Noble M Paikada Vs Union
Of India

C.A. No. 1628-1629/2021
Challenge to exemptions from environmental
clearances for roads and pipelines.

Question(s):
Whether Item 6 in the notification dated 28 March 2020
was arbitrary and unconstitutional?""",
        )
    ]

    chunks = chunk_pages(pages, chunk_size=500, overlap=50)
    noble_chunks = [chunk for chunk in chunks if "Noble M Paikada" in chunk.text]

    assert noble_chunks
    assert all("murder" not in chunk.text.lower() for chunk in noble_chunks)
    assert noble_chunks[0].section == "21-03-2024 Noble M Paikada Vs Union Of India"
