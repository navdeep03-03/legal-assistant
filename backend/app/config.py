from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _default_data_dir() -> Path:
    if os.getenv("VERCEL"):
        return Path("/tmp/legal_assistant")
    return Path("./data")


def _default_database_url() -> str:
    if os.getenv("VERCEL"):
        return "sqlite:////tmp/legal_assistant/legal_assistant.sqlite3"
    return "sqlite:///./data/legal_assistant.sqlite3"


class Settings(BaseSettings):
    app_name: str = "Counsel Legal Assistant"
    environment: Literal["development", "test", "production"] = "development"
    api_prefix: str = "/api/v1"

    llm_provider: Literal["auto", "mistral", "openai", "local"] = "auto"
    mistral_api_key: str | None = None
    mistral_model: str = "mistral-large-latest"
    mistral_reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "none"
    mistral_temperature: float = 0.2
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.6-sol"
    openai_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "medium"
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: Literal["auto", "openai", "local"] = "auto"
    local_embedding_dimensions: int = 384

    database_url: str = Field(default_factory=_default_database_url)
    data_dir: Path = Field(default_factory=_default_data_dir)
    app_api_key: str | None = None
    require_identity_headers: bool = False
    allowed_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"]
    )

    max_upload_mb: int = 20
    top_k: int = 6
    chunk_size: int = 500
    chunk_overlap: int = 100
    max_context_characters: int = 28_000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def using_openai(self) -> bool:
        if self.embedding_provider == "local":
            return bool(self.openai_api_key)
        return bool(self.openai_api_key)

    def prepare_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.prepare_directories()
    return settings
