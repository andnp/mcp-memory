"""Searchkernel ports backed by mcp-memory's authoritative stores."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from searchkernel.domain import (
    Record,
    RecordHit,
    RecordIdentity,
    RecordStatus,
    Vector,
    canonical_storage_key,
)
from searchkernel.ports import (
    AsyncGraphStore,
    AsyncKeywordStore,
    AsyncVectorStore,
    CandidateFilterSupport,
    EmbeddingSink,
)

from mcp_memory.core.ports.memory import (
    MemoryMaintenanceReadPort,
    MemoryReadPort,
    MemoryRecord,
)

_GRAPH_EDGE_DISCOUNTS = {
    "DEPENDS_ON": 0.7,
    "AMENDS": 0.6,
    "CONTRADICTS": 0.35,
}


class MemoryVectorBackend(Protocol):
    """Memory-owned vector backend consumed by the read-only adapter."""

    def upsert(
        self,
        *,
        source_kind: str,
        source_id: str,
        workspace_id: str | None,
        model_name: str,
        embedding: list[float],
        source_updated_at: str | None = None,
    ) -> bool: ...

    def search(
        self,
        *,
        source_kind: str,
        model_name: str,
        query_embedding: list[float],
        candidate_ids: list[str] | None = None,
        diagnostics: dict[str, object] | None = None,
        workspace_id: str | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]: ...

    def delete(
        self,
        *,
        source_kind: str,
        source_id: str,
        model_name: str | None = None,
    ) -> int: ...


class MemoryRecordAdapter:
    """Translate an authoritative memory record to a kernel Record."""

    source_kind = "memory"

    @classmethod
    def to_record(cls, memory: MemoryRecord) -> Record:
        metadata = dict(memory.metadata)
        workspace_id = _first_workspace_id(memory.workspace_ids)
        metadata.update(
            {
                "memory_type": memory.type,
                "memory_status": memory.status,
                "summary": memory.summary,
                "workspace_ids": list(memory.workspace_ids),
                "tags": list(memory.tags),
                "read_count": memory.read_count,
                "access_score": memory.access_score,
                "canonical_id": canonical_storage_key(
                    workspace_id, cls.source_kind, memory.id
                ),
            }
        )
        return Record(
            source_kind=cls.source_kind,
            source_id=memory.id,
            title=memory.title,
            body=memory.content,
            created_at=_parse_timestamp(memory.created_at),
            updated_at=_parse_timestamp(memory.updated_at),
            metadata=metadata,
            status=_kernel_status(memory.status),
            workspace_id=workspace_id,
        )


class MemoryKeywordStore(AsyncKeywordStore):
    """Read-only keyword store over the memory-owned relational index."""

    def __init__(self, repository: MemoryReadPort) -> None:
        self._repository = repository

    def index(self, records: list[Record]) -> None:
        """Validate source ownership; relational indexing remains authoritative."""
        if any(record.source_kind != MemoryRecordAdapter.source_kind for record in records):
            raise ValueError("MemoryKeywordStore only accepts memory records")

    async def search(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RecordHit | tuple[str, float]]:
        filters = filters or {}
        memory_ids = await asyncio.to_thread(
            self._repository.search_keyword_memory_ids,
            query,
            workspace_id=_string_filter(filters, "workspace_id"),
            memory_type=_string_filter(filters, "memory_type"),
            status=_memory_status_filter(filters.get("status")),
            include_superseded=bool(filters.get("include_superseded", False)),
            limit=k,
        )
        signal_context = filters.get("_mcp_memory_signal_context")
        if hasattr(signal_context, "keyword_candidates_present"):
            setattr(signal_context, "keyword_candidates_present", bool(memory_ids))
        return [
            (memory_id, 1.0 / rank)
            for rank, memory_id in enumerate(memory_ids, start=1)
        ]


class MemoryVectorStore(AsyncVectorStore):
    """Adapt the memory vector stores to searchkernel's VectorStore port."""

    def __init__(self, vector_store: MemoryVectorBackend) -> None:
        self._vector_store = vector_store
        self.supports_candidate_filtering = isinstance(
            vector_store, CandidateFilterSupport
        )
        self._epoch = 0

    def upsert(self, records: list[Record], model_name: str, dim: int) -> None:
        for record in records:
            if record.embedding is None:
                continue
            if len(record.embedding) != dim:
                raise ValueError(
                    f"Embedding for {record.source_id!r} has dimension "
                    f"{len(record.embedding)}, expected {dim}"
                )
            accepted = self._vector_store.upsert(
                source_kind=MemoryRecordAdapter.source_kind,
                source_id=record.source_id,
                workspace_id=_workspace_id(record),
                model_name=model_name,
                embedding=list(record.embedding),
                source_updated_at=record.updated_at.isoformat(),
            )
            if accepted is False:
                continue
        self._epoch += 1

    async def search(
        self,
        query_vector: Vector,
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RecordHit | tuple[str, float]]:
        if len(query_vector) != dim:
            raise ValueError(
                f"Query vector has dimension {len(query_vector)}, expected {dim}"
            )
        filters = filters or {}
        kwargs: dict[str, Any] = {
            "source_kind": MemoryRecordAdapter.source_kind,
            "model_name": model_name,
            "query_embedding": list(query_vector),
            "limit": k,
        }
        diagnostics = filters.get("diagnostics")
        if isinstance(diagnostics, dict):
            kwargs["diagnostics"] = diagnostics
        candidate_ids = _string_sequence_filter(filters.get("candidate_ids"))
        if candidate_ids and self.supports_candidate_filtering:
            kwargs["candidate_ids"] = candidate_ids
        workspace_id = _string_filter(filters, "workspace_id")
        if workspace_id is not None:
            kwargs["workspace_id"] = workspace_id
        return cast(
            list[RecordHit | tuple[str, float]],
            await asyncio.to_thread(self._vector_store.search, **kwargs),
        )

    def delete(self, record_ids: list[str]) -> None:
        for record_id in record_ids:
            self._vector_store.delete(
                source_kind=MemoryRecordAdapter.source_kind,
                source_id=record_id,
            )
        if record_ids:
            self._epoch += 1

    def epoch(self) -> int:
        return self._epoch


class MemoryGraphStore(AsyncGraphStore):
    """Read-only graph view over authoritative memory links."""

    def __init__(self, repository: MemoryReadPort) -> None:
        self._repository = repository

    def upsert_edges(self, edges: list[tuple[str, str, str, float]]) -> None:
        raise NotImplementedError(
            "Memory links must be written through the memory mutation port"
        )

    def set(self, key: str, value: Any, epoch: int) -> None:
        raise NotImplementedError("MemoryGraphStore does not own cache state")

    def invalidate_epoch(self, epoch: int) -> None:
        raise NotImplementedError("MemoryGraphStore does not own cache state")

    async def neighbors(
        self,
        record_id: str | RecordIdentity,
        edge_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[tuple[str, str, float]]:
        return await asyncio.to_thread(
            self._neighbors_sync,
            record_id.source_id if isinstance(record_id, RecordIdentity) else record_id,
            edge_types,
            depth,
        )

    def _neighbors_sync(
        self,
        record_id: str,
        edge_types: list[str] | None,
        depth: int,
    ) -> list[tuple[str, str, float]]:
        if depth < 1:
            return []
        allowed_types = (
            set(edge_types)
            if edge_types is not None
            else set(_GRAPH_EDGE_DISCOUNTS)
        )
        results: list[tuple[str, str, float]] = []
        frontier = [record_id]
        visited = {record_id}
        for distance in range(1, depth + 1):
            next_frontier: list[str] = []
            for source_id in frontier:
                links = self._repository.get_links(source_id, direction="outgoing")
                for link in links:
                    if allowed_types is not None and link.link_type not in allowed_types:
                        continue
                    if link.target_id in visited:
                        continue
                    discount = _GRAPH_EDGE_DISCOUNTS.get(link.link_type)
                    if discount is None:
                        continue
                    visited.add(link.target_id)
                    next_frontier.append(link.target_id)
                    results.append(
                        (link.target_id, link.link_type, discount / distance)
                    )
            frontier = next_frontier
            if not frontier:
                break
        return results


class MemoryHydrator:
    """Hydrate authoritative memory records without retrieval side effects."""

    def __init__(self, repository: MemoryMaintenanceReadPort) -> None:
        self._repository = repository

    async def hydrate_context(self, memory_id: str):
        return await asyncio.to_thread(self._repository.peek_memory, memory_id)

    async def hydrate_record(
        self,
        identity: RecordIdentity | str,
    ) -> Record | None:
        workspace_id = getattr(identity, "workspace_id", None)
        source_id = getattr(identity, "source_id", identity)
        if not isinstance(source_id, str):
            return None
        context = await self.hydrate_context(source_id)
        if context is None:
            return None
        record = MemoryRecordAdapter.to_record(context.record)
        if isinstance(workspace_id, str):
            record.workspace_id = workspace_id
        return record


class MemoryEmbeddingSink(EmbeddingSink):
    """Constrain generic embedding writes to the memory source namespace."""

    source_kind = "memory"

    def __init__(self, vector_store: MemoryVectorBackend) -> None:
        self._vector_store = vector_store

    def upsert(
        self,
        *,
        source_kind: str,
        source_id: str,
        workspace_id: str | None,
        model_name: str,
        embedding: Vector,
        source_updated_at: str | None = None,
    ) -> bool:
        if source_kind != self.source_kind:
            raise ValueError(
                f"MemoryEmbeddingSink does not accept source kind {source_kind!r}"
            )
        return bool(
            self._vector_store.upsert(
                source_kind=self.source_kind,
                source_id=source_id,
                workspace_id=workspace_id,
                model_name=model_name,
                embedding=list(embedding),
                source_updated_at=source_updated_at,
            )
        )


class MemoryRepairQueue:
    """Thin boundary adapter preserving mcp-memory repair queue persistence."""

    def __init__(self, queue: Any) -> None:
        self._queue = queue

    def enqueue_unique(self, **kwargs: Any) -> Any:
        return self._queue.enqueue_unique(**kwargs)

    def claim_batch(self, **kwargs: Any) -> Any:
        return self._queue.claim_batch(**kwargs)

    def complete_item(self, item_id: str, **kwargs: Any) -> Any:
        return self._queue.complete_item(item_id, **kwargs)

    def release_item(self, item_id: str, **kwargs: Any) -> Any:
        return self._queue.release_item(item_id, **kwargs)

    def backlog_snapshot(self, **kwargs: Any) -> Any:
        return self._queue.backlog_snapshot(**kwargs)


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _kernel_status(status: str) -> RecordStatus:
    if status == "archived":
        return RecordStatus.ARCHIVED
    if status in {"stale", "degraded"}:
        return RecordStatus.STALE
    return RecordStatus.ACTIVE


def _memory_status_filter(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value if value in {"active", "stale", "degraded", "archived"} else None


def _string_filter(filters: Mapping[str, Any], key: str) -> str | None:
    value = filters.get(key)
    return value if isinstance(value, str) else None


def _string_sequence_filter(value: object) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _workspace_id(record: Record) -> str | None:
    if isinstance(record.workspace_id, str):
        return record.workspace_id
    value = record.metadata.get("workspace_id")
    if isinstance(value, str):
        return value
    workspace_ids = record.metadata.get("workspace_ids")
    if isinstance(workspace_ids, Sequence) and not isinstance(workspace_ids, str):
        for workspace_id in workspace_ids:
            if isinstance(workspace_id, str):
                return workspace_id
    return None


def _first_workspace_id(workspace_ids: Sequence[str]) -> str | None:
    return next((workspace_id for workspace_id in workspace_ids if workspace_id), None)


__all__ = [
    "MemoryEmbeddingSink",
    "MemoryGraphStore",
    "MemoryHydrator",
    "MemoryKeywordStore",
    "MemoryRecordAdapter",
    "MemoryRepairQueue",
    "MemoryVectorBackend",
    "MemoryVectorStore",
]
