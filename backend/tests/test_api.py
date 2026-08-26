from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.app.services.orchestration import (
    _extract_response_text,
    _focused_excerpt,
    _generation_fallback_reason,
    _openai_fallback_reason,
)

HEADERS = {"X-Tenant-ID": "acme", "X-User-ID": "lawyer-1"}


def upload_contract(client: TestClient, text: str, headers: dict[str, str] | None = None) -> dict:
    response = client.post(
        "/api/v1/upload-documents",
        headers=headers or HEADERS,
        files={"files": ("Vendor_MSA.txt", text.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 201, response.text
    return response.json()["documents"][0]


def test_health_reports_local_mode(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["mode"] == "local-demo"
    assert payload["vector_engine"] in {"faiss", "numpy"}


def test_upload_search_and_grounded_answer(client: TestClient, contract_text: str) -> None:
    document = upload_contract(client, contract_text)
    assert document["status"] == "ready"
    assert document["chunk_count"] >= 1

    search = client.post(
        "/api/v1/search",
        headers=HEADERS,
        json={"query": "termination notice period", "document_ids": [document["id"]]},
    )
    assert search.status_code == 200, search.text
    results = search.json()["results"]
    assert results
    assert any("thirty days" in result["text"] for result in results)

    answer = client.post(
        "/api/v1/ask",
        headers=HEADERS,
        json={
            "question": "What are the termination clauses in this contract?",
            "document_ids": [document["id"]],
        },
    )
    assert answer.status_code == 200, answer.text
    payload = answer.json()
    assert payload["retrieval_used"] is True
    assert payload["mode"] == "local-demo"
    assert payload["citations"]
    assert payload["citations"][0]["document_id"] == document["id"]
    assert "not legal advice" in payload["warning"].lower()


def test_search_prioritizes_exact_case_entities(client: TestClient) -> None:
    document = upload_contract(
        client,
        """S. No. Date of Judgment Cause Title Subject Judgment Summary

1. 03-01-2024 Vishal Tiwari Vs Union of India W.P.(C) No. 162/2023
Plea challenging SEBI's investigation into the Adani Group and seeking constitution
of a Special Investigation Team. The questions concern judicial review over the
regulator's investigation and whether a court-monitored SIT should be appointed.
\f
24. 23-07-2024 Gene Campaign Vs Union Of India W.P.(C) No. 115/2004
Challenge against the approval of hybrid transgenic mustard DMH-11 for environmental
release. The questions concern biosafety, public consultation, and regulatory approvals.
\f
25. 09-08-2024 Manish Sisodia Vs Directorate Of Enforcement CRL.A. No. 3295/2024
Whether prolonged incarceration and delay in trial justify bail under stringent bail
conditions.
""",
    )

    search = client.post(
        "/api/v1/search",
        headers=HEADERS,
        json={
            "query": "What is the Vishal Tiwari Vs Union of India case about regarding Adani Group?",
            "document_ids": [document["id"]],
            "top_k": 3,
        },
    )

    assert search.status_code == 200, search.text
    results = search.json()["results"]
    assert results
    first_text = results[0]["text"].lower()
    assert "vishal tiwari" in first_text
    assert "adani group" in first_text

    answer = client.post(
        "/api/v1/ask",
        headers=HEADERS,
        json={
            "question": (
                "What was the core legal challenge brought forward in the case "
                "Vishal Tiwari Vs Union of India regarding the Adani Group?"
            ),
            "document_ids": [document["id"]],
        },
    )

    assert answer.status_code == 200, answer.text
    payload = answer.json()
    assert "The core legal challenge was SEBI's investigation into the Adani Group" in (
        payload["answer"]
    )
    assert "Special Investigation Team" in payload["answer"]
    assert len(payload["citations"]) == 1
    assert "SEBI's investigation into the Adani Group" in payload["citations"][0]["quote"]
    assert "Chief Justice" not in payload["citations"][0]["quote"]


def test_local_answer_focuses_excerpt_within_mixed_chunk(client: TestClient) -> None:
    document = upload_contract(
        client,
        """9. 18-03-2024 Navas @ Mulanavas Vs State Of Kerala
Question(s): What is the suitable term of imprisonment that should be imposed
for the offence of murder?

View Judgment Full Judgment

10. 21-03-2024 Noble M Paikada Vs Union Of India
C.A. No. 1628-1629/2021
Challenge to exemptions from environmental clearances for roads and pipelines.

NOBLE M PAIKADA V. UNION OF INDIA
2024 INSC 241 (21 March 2024)
Question(s): Whether Item 6 in the notification dated 28 March 2020,
which granted a complete exemption from needing prior Environmental Clearance
to unearth soil for creating roads, pipelines etc., was arbitrary and unconstitutional?
""",
    )

    response = client.post(
        "/api/v1/ask",
        headers=HEADERS,
        json={
            "question": (
                "What specific notification and item number were challenged for granting "
                "exemptions from environmental clearances in Noble M Paikada Vs Union of India?"
            ),
            "document_ids": [document["id"]],
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert "The challenged exemption was Item 6 in the notification dated 28 March 2020." in (
        payload["answer"]
    )
    assert "Item 6" in payload["answer"]
    assert "28 March 2020" in payload["answer"]
    assert "Item 6" in payload["citations"][0]["quote"]
    assert "murder" not in payload["citations"][0]["quote"].lower()


def test_focused_excerpt_prefers_question_passage() -> None:
    text = """osed for the offence of murder?

View Judgment
Full Judgment

10.

21-03-2024 Noble M Paikada Vs Union
Of India

C.A. No. 1628-1629/2021
Challenge to exemptions from environmental
clearances for roads and pipelines.

NOBLE M PAIKADA V. UNION OF INDIA
2024 INSC 241 (21 March 2024)
Justices:
Justice Abhay S. Oka and Justice Sanjay Karol
Question(s):
Whether Item 6 in the notification dated 28 March 2020,
which granted a complete exemption from needing prior
Environmental Clearance to unearth soil for creating
roads, pipelines etc., was arbitrary and unconstitutional?

View Judgment
Full Judgment"""
    question = (
        "What specific notification and item number were challenged for granting exemptions "
        "from environmental clearances in Noble M Paikada Vs Union of India?"
    )

    excerpt = _focused_excerpt(question, text)

    assert excerpt.startswith("Whether Item 6")
    assert "28 March 2020" in excerpt
    assert "murder" not in excerpt.lower()


def test_focused_excerpt_prefers_core_challenge_over_case_heading() -> None:
    text = """S. No. Date of Judgment Cause Title/Case No. Subject Judgment Summary
1. 03-01-2024 Vishal Tiwari Vs Union of India
W.P.(C) No. 162/2023
Plea challenging SEBI's investigation into the Adani Group and seeking constitution
of Special Investigation Team (SIT).

Vishal Tiwari Union of India
2024 INSC 3 (3 January 2024)
Chief Justice (Dr.) Dhananjaya Y Chandrachud, Justice Jamshed B. Pardiwala
Question(s): (i) What is the scope of judicial review over the regulatory functions
of the Securities and Exchange Board of India ("SEBI")? (ii) Whether the Supreme Court
should transfer the investigation into the Adani Group from SEBI to a Special
Investigation Team ("SIT")..."""
    question = (
        "What was the core legal challenge brought forward in the case "
        "Vishal Tiwari Vs Union of India regarding the Adani Group?"
    )

    excerpt = _focused_excerpt(question, text)

    assert "SEBI's investigation into the Adani Group" in excerpt
    assert "Special Investigation Team" in excerpt
    assert "Chief Justice" not in excerpt


def test_specific_factual_background_answer_prunes_weak_sources(client: TestClient) -> None:
    document = upload_contract(
        client,
        """35. 05-11-2024 Property Owners Association Vs State Of Maharashtra
Question(s): (i) What is the correct interpretation of Article 31C after Minerva Mills?
(ii) Whether privately owned property constitutes material resources of the community?
\f
36. 05-11-2024 Anjum Kadari Vs Union Of India
SLP(C) No. 8541/2024
Challenge to the constitutional validity of the Uttar Pradesh Board of Madarsa Education Act, 2004
ANJUM KADARI V. UNION OF INDIA
2024 INSC 831 (5 November 2024)
Question(s): Whether the Uttar Pradesh Board of Madarsa Education Act, 2004
("Madarsa Act") is constitutional?
Factual Background:
The Madarsa Act established a Board of Madarsa Education to regulate standards
of education for students studying in Madarsas in the state. There are over
13,000 Madaras in Uttar Pradesh with over 12,00,000 students.
\f
1. 03-01-2024 Vishal Tiwari Vs Union of India
Question(s): Whether the Supreme Court should transfer the investigation into
the Adani Group from SEBI to a Special Investigation Team (SIT)?
""",
    )

    response = client.post(
        "/api/v1/ask",
        headers=HEADERS,
        json={
            "question": (
                "What are the key details of the factual background regarding the number "
                "of impacted institutions in Anjum Kadari Vs Union of India concerning "
                "the Uttar Pradesh Board of Madarsa Education Act?"
            ),
            "document_ids": [document["id"]],
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert "over 13,000 Madrasas" in payload["answer"]
    assert "over 12,00,000 students" in payload["answer"]
    assert len(payload["citations"]) == 1
    assert "Madarsa Act established" in payload["citations"][0]["quote"]
    assert "Minerva Mills" not in payload["answer"]
    assert "Adani Group" not in payload["answer"]


def test_openai_fallback_reason_reports_exhausted_quota() -> None:
    exc = RuntimeError(
        "Error code: 429 - {'error': {'code': 'credit_balance_exhausted', "
        "'message': 'You have no credits remaining.'}}"
    )

    assert _openai_fallback_reason(exc) == "OpenAI quota is exhausted"


def test_mistral_fallback_reason_reports_exhausted_quota() -> None:
    exc = RuntimeError("429 RESOURCE_EXHAUSTED: You exceeded your current quota.")

    assert _generation_fallback_reason(exc, "mistral") == "Mistral quota is exhausted"


def test_extract_response_text_reads_mistral_chat_choices() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"ok"}'))]
    )

    assert _extract_response_text(response) == '{"answer":"ok"}'


def test_duplicate_upload_is_reported(client: TestClient, contract_text: str) -> None:
    upload_contract(client, contract_text)
    response = client.post(
        "/api/v1/upload-documents",
        headers=HEADERS,
        files={"files": ("copy.txt", contract_text.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 201
    assert response.json()["documents"] == []
    assert len(response.json()["duplicates"]) == 1


def test_tenant_isolation(client: TestClient, contract_text: str) -> None:
    upload_contract(client, contract_text, {"X-Tenant-ID": "tenant-a", "X-User-ID": "user-a"})
    response = client.get(
        "/api/v1/documents",
        headers={"X-Tenant-ID": "tenant-b", "X-User-ID": "user-b"},
    )
    assert response.status_code == 200
    assert response.json() == []

    search = client.post(
        "/api/v1/search",
        headers={"X-Tenant-ID": "tenant-b", "X-User-ID": "user-b"},
        json={"query": "termination"},
    )
    assert search.status_code == 200
    assert search.json()["results"] == []


def test_rejects_unsupported_files(client: TestClient) -> None:
    response = client.post(
        "/api/v1/upload-documents",
        headers=HEADERS,
        files={"files": ("malware.exe", b"not really", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
