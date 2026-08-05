from __future__ import annotations

import re
import secrets
from dataclasses import dataclass

from fastapi import Header, HTTPException, Request, status

from .config import Settings

IDENTITY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


@dataclass(frozen=True)
class RequestContext:
    tenant_id: str
    user_id: str


def _validate_identity(value: str, label: str) -> str:
    if not IDENTITY_PATTERN.fullmatch(value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {label}. Use letters, numbers, dots, dashes, or underscores.",
        )
    return value


def get_request_context(
    request: Request,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    x_user_id: str | None = Header(default=None, alias="X-User-ID"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> RequestContext:
    settings: Settings = request.app.state.settings
    if settings.app_api_key and not secrets.compare_digest(x_api_key or "", settings.app_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    if settings.require_identity_headers and (not x_tenant_id or not x_user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Tenant-ID and X-User-ID headers are required",
        )

    return RequestContext(
        tenant_id=_validate_identity(x_tenant_id or "demo-tenant", "tenant ID"),
        user_id=_validate_identity(x_user_id or "demo-user", "user ID"),
    )
