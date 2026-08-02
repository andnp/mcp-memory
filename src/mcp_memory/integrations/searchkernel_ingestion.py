"""Memory-owned application integration for searchkernel record ingestion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from searchkernel.domain import Cursor, Record
from searchkernel.ingestion import SemanticRecordIngestor
from searchkernel.ports import KeywordStore, VectorStore
from searchkernel.ports.content_source import (
    IngestionFailureMode,
    IngestionReceipt,
)

from mcp_memory.core.ports.memory import MemoryRecord
from mcp_memory.integrations.searchkernel_adapters import MemoryRecordAdapter, MemoryVectorBackend


class _NoopEmbeddingCache:
    """Provide the kernel cache contract when the provider owns caching."""

    def __init__(self, encoder_namespace: str) -> None:
        self.encoder_namespace = encoder_namespace

    def get_many(self, content_hashes: Sequence[str]) -> Mapping[str, Sequence[float]]:
        del content_hashes
        return {}

    def put_many(self, vectors: Mapping[str, Sequence[float]]) -> None:
        del vectors


class _MemoryEmbeddingProvider:
    """Add searchkernel's explicit dimension contract to a memory embedder."""

    def __init__(self, embedder: Any, dimension: int) -> None:
        self._embedder = embedder
        self.dim = dimension

    @property
    def model_name(self) -> str:
        return cast(str, self._embedder.model_name)

    @property
    def encoder_namespace(self) -> str:
        namespace = getattr(self._embedder, "encoder_namespace", None)
        return namespace if isinstance(namespace, str) and namespace else self.model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(vector) for vector in self._embedder.embed(texts)]


class _MemoryKeywordIndex:
    """Satisfy the kernel's synchronous indexing port; SQL remains authoritative."""

    def index(self, records: list[Record]) -> None:
        if any(record.source_kind != MemoryRecordAdapter.source_kind for record in records):
            raise ValueError("memory ingestion only accepts memory records")

    def search(self, query: str, k: int, filters: dict[str, Any] | None = None) -> list[Any]:
        del query, k, filters
        return []


class _MemoryVectorIndex:
    """Expose the memory vector backend through searchkernel's sync port."""

    def __init__(self, vector_store: MemoryVectorBackend) -> None:
        self._vector_store = vector_store

    def upsert(self, records: list[Record], model_name: str, dim: int) -> None:
        for record in records:
            if record.embedding is None:
                continue
            if len(record.embedding) != dim:
                raise ValueError(
                    f"Embedding for {record.source_id!r} has dimension "
                    f"{len(record.embedding)}, expected {dim}"
                )
            self._vector_store.upsert(
                source_kind=record.source_kind,
                source_id=record.source_id,
                workspace_id=record.workspace_id,
                model_name=model_name,
                embedding=list(record.embedding),
                source_updated_at=record.updated_at.isoformat(),
            )

    def search(
        self,
        query_vector: list[float],
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: dict[str, Any] | None = None,
    ) -> list[Any]:
        del query_vector, k, model_name, dim, filters
        return []

    def delete(self, record_ids: list[str]) -> None:
        for record_id in record_ids:
            self._vector_store.delete(
                source_kind=MemoryRecordAdapter.source_kind,
                source_id=record_id,
            )

    def epoch(self) -> int:
        return 0


class MemoryRecordIngestor:
    """Expose searchkernel's receipt-based ingestion boundary for memories."""

    def __init__(
        self,
        repository: Any,
        vector_store: MemoryVectorBackend,
        embedder: Any,
        *,
        embedding_cache: Any | None = None,
    ) -> None:
        provider = _MemoryEmbeddingProvider(embedder, _embedding_dimension(embedder))
        self._provider = provider
        keyword_index = _MemoryKeywordIndex()
        vector_index = _MemoryVectorIndex(vector_store)
        self._ingestor = SemanticRecordIngestor(
            embedding_provider=provider,
            keyword_store=cast(KeywordStore, keyword_index),
            vector_store=cast(VectorStore, vector_index),
            embedding_cache=embedding_cache or _NoopEmbeddingCache(provider.encoder_namespace),
        )

    async def index_records(
        self,
        records: Sequence[MemoryRecord],
        *,
        checkpoint: Cursor = None,
        failure_mode: IngestionFailureMode = "strict",
    ) -> IngestionReceipt:
        kernel_records = [_to_kernel_record(record) for record in records]
        return await self._ingestor.index_records(
            kernel_records,
            checkpoint=checkpoint,
            failure_mode=failure_mode,
        )


def _to_kernel_record(memory: MemoryRecord) -> Record:
    record = MemoryRecordAdapter.to_record(memory)
    record.indexed_text = _memory_embedding_text(memory)
    return record


def _memory_embedding_text(memory: MemoryRecord) -> str:
    return "\n".join(
        part
        for part in [memory.title, memory.summary or "", memory.content, ", ".join(memory.tags)]
        if part
    )


def _embedding_dimension(embedder: Any) -> int:
    for attribute in ("embedding_dimension", "dim", "_dimensions"):
        value = getattr(embedder, attribute, None)
        if isinstance(value, int) and value > 0:
            return value
    embeddings = embedder.embed(["searchkernel dimension probe"])
    if len(embeddings) != 1 or not embeddings[0]:
        raise ValueError("memory embedder did not return a usable dimension")
    return len(embeddings[0])


__all__ = ["MemoryRecordIngestor"]
