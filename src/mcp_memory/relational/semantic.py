"""Vector/embedding adapter used by relational search orchestration."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from mcp_memory.core.ports.memory import MemoryRecord
from searchkernel.ports import CandidateFilterSupport, EmbeddingBatchProvider
from searchkernel.runtime import get_or_compute_query_embedding


class RelationalSemanticSearchAdapter:
    """Translate relational semantic requests to the embedding/vector ports."""

    def __init__(self, embedder: EmbeddingBatchProvider, vector_store: Any) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    @property
    def model_name(self) -> str:
        return self._embedder.model_name

    def matches(self, embedder: EmbeddingBatchProvider, vector_store: Any) -> bool:
        return self._embedder is embedder and self._vector_store is vector_store

    def get_query_embedding(self, query: str) -> list[float]:
        return get_or_compute_query_embedding(
            model_name=self._embedder.model_name,
            query=query,
            compute=lambda: list(self._embedder.embed([query])[0]),
        )

    def search(
        self,
        query: str,
        candidates: list[MemoryRecord],
        *,
        candidate_ids: Sequence[str] | None,
        limit: int,
        semantic_timing_ms: dict[str, float] | None,
        vector_search_diagnostics: dict[str, object] | None,
    ) -> list[tuple[str, float]]:
        query_embedding_started = time.perf_counter()
        query_embedding = self.get_query_embedding(query)
        _record_timing_ms(semantic_timing_ms, "semantic_query_embedding", query_embedding_started)
        search_kwargs: dict[str, object] = {
            "source_kind": "memory",
            "model_name": self._embedder.model_name,
            "query_embedding": query_embedding,
            "limit": limit,
        }
        supports_candidate_filtering = isinstance(self._vector_store, CandidateFilterSupport)
        if candidate_ids and supports_candidate_filtering:
            search_kwargs["candidate_ids"] = list(candidate_ids)
        if vector_search_diagnostics is not None:
            search_kwargs["diagnostics"] = vector_search_diagnostics
        vector_search_started = time.perf_counter()
        matches = self._vector_store.search(**search_kwargs)
        _record_timing_ms(semantic_timing_ms, "semantic_vector_search", vector_search_started)
        return matches


def _record_timing_ms(timing_ms: dict[str, float] | None, key: str, started_at: float) -> None:
    if timing_ms is None:
        return
    elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 3)
    timing_ms[key] = round(timing_ms.get(key, 0.0) + elapsed_ms, 3)
