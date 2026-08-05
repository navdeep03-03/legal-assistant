from __future__ import annotations

from fastapi.testclient import TestClient

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
