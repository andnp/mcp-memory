from __future__ import annotations

import pytest

from mcp_memory.integrations.searchkernel_record_pipeline import (
    MemoryQueryEmbeddingProvider,
)
from searchkernel.runtime import clear_query_embedding_cache


pytestmark = pytest.mark.small


class FakeEmbedder:
    model_name = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0] for _ in texts]


class QueryAwareFakeEmbedder(FakeEmbedder):
    encoder_namespace = "fake-model|query-aware-v1"

    def __init__(self) -> None:
        super().__init__()
        self.query_calls = 0

    def embed_query(self, text: str) -> list[float]:
        assert text == "asymmetric query"
        self.query_calls += 1
        return [0.0, 1.0]


@pytest.mark.asyncio
async def test_query_embedding_provider_reuses_exact_query_cache() -> None:
    clear_query_embedding_cache()
    embedder = FakeEmbedder()
    provider = MemoryQueryEmbeddingProvider(embedder, 2)

    try:
        first = await provider.embed_query("cached query")
        second = await provider.embed_query("cached query")
    finally:
        clear_query_embedding_cache()

    assert first == second == [1.0, 0.0]
    assert embedder.calls == 1


@pytest.mark.asyncio
async def test_query_embedding_provider_prefers_query_aware_contract_and_namespace() -> None:
    clear_query_embedding_cache()
    embedder = QueryAwareFakeEmbedder()
    provider = MemoryQueryEmbeddingProvider(embedder, 2)

    try:
        first = await provider.embed_query("asymmetric query")
        second = await provider.embed_query("asymmetric query")
    finally:
        clear_query_embedding_cache()

    assert first == second == [0.0, 1.0]
    assert embedder.query_calls == 1
    assert embedder.calls == 0
    assert provider.encoder_namespace == "fake-model|query-aware-v1"
