from __future__ import annotations

import asyncio

import pytest

from mcp_memory.integrations.searchkernel_record_pipeline import (
    MemoryQueryEmbeddingProvider,
)
from searchkernel.runtime import QueryEmbeddingCache


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
    """Reuse one pipeline-owned cache for repeated normalized query text."""
    cache = QueryEmbeddingCache()
    embedder = FakeEmbedder()
    provider = MemoryQueryEmbeddingProvider(embedder, 2)

    first = await cache.async_get_or_compute(
        encoder_namespace=provider.encoder_namespace,
        query="cached query",
        compute=lambda: provider.embed_query("cached query"),
    )
    second = await cache.async_get_or_compute(
        encoder_namespace=provider.encoder_namespace,
        query="cached query",
        compute=lambda: provider.embed_query("cached query"),
    )

    assert first == second == [1.0, 0.0]
    assert embedder.calls == 1
    assert cache.metrics.hits == 1
    assert cache.metrics.misses == 1


@pytest.mark.asyncio
async def test_query_embedding_provider_prefers_query_aware_contract_and_namespace() -> None:
    """Keep query-aware encoding and its explicit namespace through the cache."""
    cache = QueryEmbeddingCache()
    embedder = QueryAwareFakeEmbedder()
    provider = MemoryQueryEmbeddingProvider(embedder, 2)

    first = await cache.async_get_or_compute(
        encoder_namespace=provider.encoder_namespace,
        query="asymmetric query",
        compute=lambda: provider.embed_query("asymmetric query"),
    )
    second = await cache.async_get_or_compute(
        encoder_namespace=provider.encoder_namespace,
        query="asymmetric query",
        compute=lambda: provider.embed_query("asymmetric query"),
    )

    assert first == second == [0.0, 1.0]
    assert embedder.query_calls == 1
    assert embedder.calls == 0
    assert provider.encoder_namespace == "fake-model|query-aware-v1"


@pytest.mark.asyncio
async def test_query_embedding_cache_separates_encoder_namespaces() -> None:
    """Never reuse a query vector across distinct encoder namespaces."""
    cache = QueryEmbeddingCache()
    first_embedder = QueryAwareFakeEmbedder()
    second_embedder = QueryAwareFakeEmbedder()
    first_embedder.encoder_namespace = "fake-model|query-aware-v1"
    second_embedder.encoder_namespace = "fake-model|query-aware-v2"
    first_provider = MemoryQueryEmbeddingProvider(first_embedder, 2)
    second_provider = MemoryQueryEmbeddingProvider(second_embedder, 2)

    await cache.async_get_or_compute(
        encoder_namespace=first_provider.encoder_namespace,
        query="same query",
        compute=lambda: first_provider.embed_query("asymmetric query"),
    )
    await cache.async_get_or_compute(
        encoder_namespace=second_provider.encoder_namespace,
        query="same query",
        compute=lambda: second_provider.embed_query("asymmetric query"),
    )

    assert first_embedder.query_calls == 1
    assert second_embedder.query_calls == 1
    assert cache.metrics.misses == 2


@pytest.mark.asyncio
async def test_query_embedding_cache_separates_model_namespaces() -> None:
    """Never reuse a query vector across distinct model names."""
    cache = QueryEmbeddingCache()
    first_embedder = FakeEmbedder()
    second_embedder = FakeEmbedder()
    first_embedder.model_name = "fake-model-v1"
    second_embedder.model_name = "fake-model-v2"
    first_provider = MemoryQueryEmbeddingProvider(first_embedder, 2)
    second_provider = MemoryQueryEmbeddingProvider(second_embedder, 2)

    for provider in (first_provider, second_provider):
        await cache.async_get_or_compute(
            encoder_namespace=provider.encoder_namespace,
            query="same query",
            compute=lambda provider=provider: provider.embed_query("same query"),
        )

    assert first_embedder.calls == 1
    assert second_embedder.calls == 1
    assert cache.metrics.misses == 2


@pytest.mark.asyncio
async def test_query_embedding_cache_does_not_store_failed_computations() -> None:
    """Retry failed embedding computation instead of caching an error."""
    cache = QueryEmbeddingCache()
    calls = 0

    def compute() -> list[float]:
        nonlocal calls
        calls += 1
        raise RuntimeError("encoder unavailable")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="encoder unavailable"):
            await cache.async_get_or_compute(
                encoder_namespace="fake-model|dim=2",
                query="failed query",
                compute=compute,
            )

    assert calls == 2
    assert cache.metrics.misses == 2
    assert cache.metrics.hits == 0


@pytest.mark.asyncio
async def test_query_embedding_cache_coalesces_concurrent_computations() -> None:
    """Coalesce concurrent misses while preserving one embedding computation."""
    cache = QueryEmbeddingCache()
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def compute() -> list[float]:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return [1.0, 0.0]

    tasks = [
        asyncio.create_task(
            cache.async_get_or_compute(
                encoder_namespace="fake-model|dim=2",
                query="concurrent query",
                compute=compute,
            )
        )
        for _ in range(3)
    ]
    await started.wait()
    release.set()

    assert await asyncio.gather(*tasks) == [[1.0, 0.0]] * 3
    assert calls == 1
    assert cache.metrics.misses == 1
    assert cache.metrics.coalesced_waiters == 2
