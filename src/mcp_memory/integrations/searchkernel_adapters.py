"""Searchkernel ports backed by mcp-memory's authoritative stores."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from searchkernel.domain import (
    GraphNeighbor,
    Record,
    RecordHit,
    RecordIdentity,
    RecordStatus,
    Vector,
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
    MemoryReadContext,
    MemoryReadPort,
    MemoryRecord,
)

_GRAPH_EDGE_DISCOUNTS = {
    "DEPENDS_ON": 0.7,
    "AMENDS": 0.6,
    "CONTRADICTS": 0.35,
}
_MEMORY_QUERY_SYNONYMS = {
    "memory": ("memories", "record"),
    "memories": ("memory", "records"),
    "decision": ("decisions", "choice"),
    "architecture": ("design",),
    "implementation": ("code", "implementation"),
    "task": ("tasks", "work"),
    "bug": ("defect", "issue"),
    "fix": ("repair", "correction"),
    "search": ("retrieval", "lookup"),
    "curation": ("maintenance", "cleanup"),
}
_MAX_QUERY_SYNONYMS = 8


MemoryBackendHit = RecordHit | tuple[str, float]


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
    def identity(
        cls,
        memory: MemoryRecord,
        *,
        workspace_id: str | None = None,
    ) -> RecordIdentity:
        return RecordIdentity(
            workspace_id
            if workspace_id is not None
            else _first_workspace_id(memory.workspace_ids),
            cls.source_kind,
            memory.id,
        )

    @classmethod
    def to_record(
        cls,
        memory: MemoryRecord,
        *,
        identity: RecordIdentity | None = None,
    ) -> Record:
        if identity is not None and (
            identity.source_kind != cls.source_kind
            or identity.source_id != memory.id
        ):
            raise ValueError(
                "MemoryRecord identity does not match the requested memory"
            )
        record_identity = cls.identity(
            memory,
            workspace_id=(identity.workspace_id if identity is not None else None),
        )
        metadata = dict(memory.metadata)
        metadata.update(
            {
                "memory_type": memory.type,
                "memory_status": memory.status,
                "summary": memory.summary,
                "workspace_ids": list(memory.workspace_ids),
                "memory_ref": memory.memory_ref,
                "tags": list(memory.tags),
                "read_count": memory.read_count,
                "access_score": memory.access_score,
                "canonical_id": record_identity.storage_key,
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
            workspace_id=record_identity.workspace_id,
        )


class MemoryKeywordStore(AsyncKeywordStore):
    """Read-only keyword store over the memory-owned relational index."""

    def __init__(
        self,
        repository: MemoryReadPort,
        *,
        prefetch: Callable[[Sequence[str]], None] | None = None,
    ) -> None:
        self._repository = repository
        self._prefetch = prefetch

    def epochs(self) -> Mapping[str, int]:
        return _repository_search_epochs(self._repository)

    def index(self, records: list[Record]) -> None:
        """Validate source ownership; relational indexing remains authoritative."""
        if any(record.source_kind != MemoryRecordAdapter.source_kind for record in records):
            raise ValueError("MemoryKeywordStore only accepts memory records")

    def keyword_epoch(self) -> int:
        return _repository_search_epochs(self._repository)["keyword"]

    async def search(
        self,
        query: str,
        k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[RecordHit]:
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
        workspace_id = _string_filter(filters, "workspace_id")
        records_by_id = await asyncio.to_thread(
            _load_memory_records,
            self._repository,
            memory_ids,
            status=_memory_status_filter(filters.get("status")),
            include_superseded=bool(filters.get("include_superseded", False)),
            scalar_fallback=False,
        )
        if self._prefetch is not None and memory_ids:
            await asyncio.to_thread(self._prefetch, memory_ids)
        signal_context = filters.get("_mcp_memory_signal_context")
        if hasattr(signal_context, "keyword_candidates_present"):
            setattr(signal_context, "keyword_candidates_present", bool(memory_ids))
        return [
            RecordHit(
                _memory_identity(
                    memory_id,
                    records_by_id.get(memory_id),
                    workspace_id=workspace_id,
                ),
                1.0 / rank,
            )
            for rank, memory_id in enumerate(memory_ids, start=1)
        ]


class MemoryVectorStore(AsyncVectorStore):
    """Adapt the memory vector stores to searchkernel's VectorStore port."""

    def __init__(
        self,
        vector_store: MemoryVectorBackend,
        *,
        prefetch: Callable[[Sequence[str]], None] | None = None,
        ensure_embeddings: Callable[[Sequence[str], Mapping[str, Any]], None]
        | None = None,
        repository: MemoryReadPort | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._prefetch = prefetch
        self._ensure_embeddings = ensure_embeddings
        self._repository = repository
        self.supports_candidate_filtering = isinstance(
            vector_store, CandidateFilterSupport
        )

    def epochs(self) -> Mapping[str, int]:
        if self._repository is None:
            raise RuntimeError("MemoryVectorStore requires an authoritative repository epoch source")
        return _repository_search_epochs(self._repository)

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

    def vector_epoch(self) -> int:
        return _repository_search_epochs(self._repository)["vector"]

    def expand_query(
        self,
        query: str,
        *,
        top_k: int = _MAX_QUERY_SYNONYMS,
        similarity_threshold: float = 0.0,
    ) -> str:
        """Provide bounded memory vocabulary to searchkernel's expansion hook."""
        del similarity_threshold
        tokens = query.split()
        lowered = [
            token.casefold().strip(".,!?;:")
            for token in tokens
            if token.casefold().strip(".,!?;:")
        ]
        additions: list[str] = []
        for token in lowered:
            for synonym in _MEMORY_QUERY_SYNONYMS.get(token, ()):
                if synonym not in lowered and synonym not in additions:
                    additions.append(synonym)
                if len(additions) >= min(max(top_k, 0), _MAX_QUERY_SYNONYMS):
                    break
            if len(additions) >= min(max(top_k, 0), _MAX_QUERY_SYNONYMS):
                break
        return " ".join([query, *additions]) if additions else query

    async def search(
        self,
        query_vector: Vector,
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[RecordHit]:
        if len(query_vector) != dim:
            raise ValueError(
                f"Query vector has dimension {len(query_vector)}, expected {dim}"
            )
        filters = filters or {}
        if self._ensure_embeddings is not None:
            candidate_ids = _string_sequence_filter(filters.get("candidate_ids"))
            await asyncio.to_thread(
                self._ensure_embeddings,
                candidate_ids,
                filters,
            )
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
        results = cast(
            list[MemoryBackendHit],
            await asyncio.to_thread(self._vector_store.search, **kwargs),
        )
        records_by_id = (
            await asyncio.to_thread(
                _load_memory_records,
                self._repository,
                [
                    result.source_id if isinstance(result, RecordHit) else result[0]
                    for result in results
                ],
                include_superseded=True,
                scalar_fallback=False,
            )
            if self._repository is not None
            else {}
        )
        normalized_results: list[RecordHit] = [
            result
            if isinstance(result, RecordHit)
            else RecordHit(
                _memory_identity(
                    result[0],
                    records_by_id.get(result[0]),
                    workspace_id=workspace_id,
                ),
                result[1],
            )
            for result in results
        ]
        if self._prefetch is not None and normalized_results:
            record_ids = [
                result.source_id if isinstance(result, RecordHit) else result[0]
                for result in normalized_results
            ]
            await asyncio.to_thread(self._prefetch, record_ids)
        return normalized_results

    def delete(self, record_ids: list[str]) -> None:
        for record_id in record_ids:
            self._vector_store.delete(
                source_kind=MemoryRecordAdapter.source_kind,
                source_id=record_id,
            )

class MemoryGraphStore(AsyncGraphStore):
    """Read-only graph view over authoritative memory links."""

    def __init__(self, repository: MemoryReadPort) -> None:
        self._repository = repository

    def epochs(self) -> Mapping[str, int]:
        return _repository_search_epochs(self._repository)

    def graph_epoch(self) -> int:
        return _repository_search_epochs(self._repository)["graph"]

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
        max_neighbors: int | None = None,
    ) -> list[GraphNeighbor]:
        return await asyncio.to_thread(
            self._neighbors_sync,
            record_id,
            edge_types,
            depth,
            max_neighbors,
        )

    def _neighbors_sync(
        self,
        record_id: str | RecordIdentity,
        edge_types: list[str] | None,
        depth: int,
        max_neighbors: int | None,
    ) -> list[GraphNeighbor]:
        identity = _record_identity(record_id)
        results = self._neighbors_many_sync(
            [identity],
            edge_types=edge_types,
            depth=depth,
        )
        neighbors = next(iter(results.values()), [])
        return neighbors if max_neighbors is None else neighbors[:max_neighbors]

    async def neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int,
    ) -> Mapping[str, Sequence[GraphNeighbor]]:
        return await asyncio.to_thread(
            self._neighbors_many_sync,
            identities,
            edge_types=None,
            depth=depth,
        )

    def _neighbors_many_sync(
        self,
        identities: Sequence[RecordIdentity],
        *,
        edge_types: list[str] | None,
        depth: int,
    ) -> dict[str, list[GraphNeighbor]]:
        seed_identities = list(dict.fromkeys(_record_identity(identity) for identity in identities))
        seed_records = _load_memory_records(
            self._repository,
            [identity.source_id for identity in seed_identities],
            include_superseded=True,
        )
        seeds = [
            (
                MemoryRecordAdapter.identity(
                    seed_records[identity.source_id],
                )
                if identity.workspace_id is None
                and identity.source_id in seed_records
                else identity
            )
            for identity in seed_identities
        ]
        seeds = list(dict.fromkeys(seeds))
        results = {identity.storage_key: [] for identity in seeds}
        if depth < 1 or not seeds:
            return results

        allowed_types = set(edge_types) if edge_types is not None else set(_GRAPH_EDGE_DISCOUNTS)
        frontiers = {
            identity.storage_key: [identity.source_id] for identity in seeds
        }
        visited = {
            identity.storage_key: {identity.source_id} for identity in seeds
        }
        workspace_by_seed = {
            identity.storage_key: identity.workspace_id for identity in seeds
        }

        for distance in range(1, depth + 1):
            source_ids = list(
                dict.fromkeys(
                    source_id
                    for frontier in frontiers.values()
                    for source_id in frontier
                )
            )
            if not source_ids:
                break
            links_by_source = self._get_links_many_sync(source_ids)
            target_ids = list(
                dict.fromkeys(
                    link.target_id
                    for links in links_by_source.values()
                    for link in links
                )
            )
            records_by_id = _load_memory_records(
                self._repository,
                target_ids,
                include_superseded=True,
            )
            next_frontiers = {seed_key: [] for seed_key in frontiers}
            for seed_key, frontier in frontiers.items():
                for source_id in frontier:
                    for link in links_by_source.get(source_id, ()):
                        if link.link_type not in allowed_types:
                            continue
                        if link.target_id in visited[seed_key]:
                            continue
                        discount = _GRAPH_EDGE_DISCOUNTS.get(link.link_type)
                        if discount is None:
                            continue
                        visited[seed_key].add(link.target_id)
                        next_frontiers[seed_key].append(link.target_id)
                        results[seed_key].append(
                            GraphNeighbor(
                                _memory_identity(
                                    link.target_id,
                                    records_by_id.get(link.target_id),
                                    workspace_id=workspace_by_seed[seed_key],
                                ),
                                link.link_type,
                                discount / distance,
                            )
                        )
            frontiers = next_frontiers
            if not any(frontiers.values()):
                break
        return results

    def _get_links_many_sync(
        self,
        memory_ids: Sequence[str],
    ) -> dict[str, Sequence[Any]]:
        for method_name in ("get_links_many", "get_links_by_memory_ids"):
            method = getattr(self._repository, method_name, None)
            if not callable(method):
                continue
            try:
                value = method(list(memory_ids), direction="outgoing")
            except TypeError:
                value = method(list(memory_ids))
            if isinstance(value, Mapping):
                return {
                    memory_id: value.get(memory_id, ())
                    for memory_id in memory_ids
                }
        return {
            memory_id: self._repository.get_links(
                memory_id,
                direction="outgoing",
            )
            for memory_id in memory_ids
        }


class MemoryHydrator:
    """Hydrate authoritative memory records without retrieval side effects."""

    def __init__(
        self,
        repository: MemoryReadPort | MemoryMaintenanceReadPort,
    ) -> None:
        self._repository = repository

    async def hydrate_context(self, memory_id: str) -> MemoryReadContext | None:
        get_memory = getattr(self._repository, "get_memory", None)
        if callable(get_memory):
            typed_get_memory = cast(
                Callable[[str], MemoryRecord | None],
                get_memory,
            )
            record = await asyncio.to_thread(typed_get_memory, memory_id)
            if record is None:
                return None
            return MemoryReadContext(record, {}, [])
        maintenance_repository = cast(MemoryMaintenanceReadPort, self._repository)
        return await asyncio.to_thread(
            maintenance_repository.peek_memory,
            memory_id,
        )

    async def hydrate_record(
        self,
        record_id: RecordIdentity | str,
    ) -> Record | None:
        identity = _record_identity(record_id)
        source_id = identity.source_id
        if not isinstance(source_id, str):
            return None
        if identity.source_kind != MemoryRecordAdapter.source_kind:
            return None
        context = await self.hydrate_context(source_id)
        if context is None:
            return None
        return MemoryRecordAdapter.to_record(context.record, identity=identity)

    async def hydrate_records(
        self,
        identities: Sequence[RecordIdentity],
    ) -> Mapping[str, Record | None]:
        unique_identities = list(dict.fromkeys(identities))
        records_by_id = await asyncio.to_thread(
            _load_memory_records,
            self._repository,
            [identity.source_id for identity in unique_identities],
            include_superseded=True,
        )
        return {
            identity.storage_key: (
                MemoryRecordAdapter.to_record(
                    record,
                    identity=identity,
                )
                if (record := records_by_id.get(identity.source_id)) is not None
                and identity.source_kind == MemoryRecordAdapter.source_kind
                else None
            )
            for identity in unique_identities
        }


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


def _record_identity(record_id: RecordIdentity | str) -> RecordIdentity:
    if isinstance(record_id, RecordIdentity):
        return record_id
    if record_id.startswith("record:"):
        try:
            return RecordIdentity.from_storage_key(record_id)
        except (TypeError, ValueError):
            pass
    return RecordIdentity(None, MemoryRecordAdapter.source_kind, record_id)


def _memory_identity(
    memory_id: str,
    memory: MemoryRecord | None,
    *,
    workspace_id: str | None = None,
) -> RecordIdentity:
    if memory is not None:
        return MemoryRecordAdapter.identity(memory, workspace_id=workspace_id)
    return RecordIdentity(workspace_id, MemoryRecordAdapter.source_kind, memory_id)


def _load_memory_records(
    repository: object | None,
    memory_ids: Sequence[str],
    *,
    status: str | None = None,
    include_superseded: bool = True,
    scalar_fallback: bool = True,
) -> dict[str, MemoryRecord]:
    normalized_ids = list(dict.fromkeys(memory_id for memory_id in memory_ids if memory_id))
    if repository is None or not normalized_ids:
        return {}

    get_searchable_memories = getattr(repository, "get_searchable_memories", None)
    if callable(get_searchable_memories):
        batch_loader = cast(
            Callable[..., Sequence[MemoryRecord]],
            get_searchable_memories,
        )
        try:
            records = batch_loader(
                normalized_ids,
                status=status,
                include_superseded=include_superseded,
            )
        except TypeError:
            records = batch_loader(normalized_ids)
        return {record.id: record for record in records}

    get_memory = getattr(repository, "get_memory", None)
    records_by_id: dict[str, MemoryRecord] = {}
    if scalar_fallback and callable(get_memory):
        typed_get_memory = cast(Callable[[str], MemoryRecord | None], get_memory)
        for memory_id in normalized_ids:
            record = typed_get_memory(memory_id)
            if record is not None:
                records_by_id[record.id] = record
        return records_by_id

    if not scalar_fallback:
        return records_by_id
    peek_memory = getattr(repository, "peek_memory", None)
    if not callable(peek_memory):
        raise RuntimeError(
            "memory repository must provide get_searchable_memories, get_memory, or peek_memory"
        )
    typed_peek_memory = cast(
        Callable[[str], MemoryReadContext | None],
        peek_memory,
    )
    for memory_id in normalized_ids:
        context = typed_peek_memory(memory_id)
        if context is not None:
            records_by_id[context.record.id] = context.record
    return records_by_id


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


def _repository_search_epochs(repository: object) -> dict[str, int]:
    get_search_epochs = getattr(repository, "get_search_epochs", None)
    if not callable(get_search_epochs):
        raise RuntimeError("memory repository does not expose authoritative search epochs")
    values = get_search_epochs()
    if isinstance(values, Mapping):
        return {
            lane: _epoch_value(values, lane)
            for lane in ("keyword", "vector", "graph")
        }
    epochs = {
        lane: getattr(values, lane, None)
        for lane in ("keyword", "vector", "graph")
    }
    if all(isinstance(value, int) for value in epochs.values()):
        return cast(dict[str, int], epochs)
    raise TypeError("memory repository returned an invalid search epoch snapshot")


def _epoch_value(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int):
        raise TypeError(f"memory repository returned an invalid {key} search epoch")
    return value


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
