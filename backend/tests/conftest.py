from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=f"sqlite:///{(tmp_path / 'test.sqlite3').as_posix()}",
        data_dir=tmp_path / "data",
        openai_api_key=None,
        embedding_provider="local",
        local_embedding_dimensions=128,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def contract_text() -> str:
    return """MASTER SERVICES AGREEMENT

1. TERM AND TERMINATION

This Agreement begins on 1 January 2026 and continues for twelve months. Either party may terminate this Agreement for convenience by giving thirty days' written notice.

The Customer may terminate immediately if the Supplier commits a material breach and fails to cure that breach within ten business days after receiving written notice.

2. CONFIDENTIALITY

Each party must protect the other party's Confidential Information using at least reasonable care. This obligation survives termination for three years.

3. LIMITATION OF LIABILITY

Each party's aggregate liability is limited to the fees paid in the twelve months before the claim. Neither party is liable for indirect or consequential damages, except for fraud or wilful misconduct.

4. GOVERNING LAW

This Agreement is governed by the laws of Karnataka, India. The courts in Bengaluru have exclusive jurisdiction.
"""
