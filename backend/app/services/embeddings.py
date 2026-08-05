from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

from ..config import Settings


class EmbeddingProvider(ABC):
    mode: str
    model_name: str

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class LocalHashEmbeddingProvider(EmbeddingProvider):
    mode = "local-demo"
    model_name = "local-hashing-v1"

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9][a-z0-9'-]*", text.lower())
        if not tokens:
            return vector
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(digest, "little")
            index = number % self.dimensions
            sign = -1.0 if number & 1 else 1.0
            vector[index] += sign * (1.5 if "_" in feature else 1.0)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    mode = "openai"

    def __init__(self, api_key: str, model: str):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model_name = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):
            batch = [text.replace("\n", " ") for text in texts[start : start + 64]]
            response = self.client.embeddings.create(model=self.model_name, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    wants_openai = settings.embedding_provider in {"auto", "openai"}
    if wants_openai and settings.openai_api_key:
        return OpenAIEmbeddingProvider(settings.openai_api_key, settings.embedding_model)
    if settings.embedding_provider == "openai" and not settings.openai_api_key:
        raise RuntimeError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
    return LocalHashEmbeddingProvider(settings.local_embedding_dimensions)
