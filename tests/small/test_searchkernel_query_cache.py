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
