from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class ReadCachePort(Protocol):
    def __getattr__(self, name: str) -> Any: ...


class MemorySurfaceTrackerPort(Protocol):
    def touch_last_surfaced(
        self, memory_ids: list[str], surfaced_at: str, *, best_effort: bool
    ) -> None: ...


@dataclass(frozen=True)
class MemoryReadDependencies:
    config: Any = None
    workspace_id: str | None = None
    repository: Any = None
    surface_tracker: MemorySurfaceTrackerPort | None = None
    relational_search: Any = None
    memory_retrieval: Any = None
    read_cache: ReadCachePort | None = None
    vector_store: Any = None
    embedder: Any = None
    embedding_maintenance: Any = None


class MemoryReadContext(Protocol):
    config: Any
    workspace_id: str | None
    db_manager: Any
    repository: Any
    relational_search: Any
    read_cache: ReadCachePort | None
    storage_backend: str | None
    runtime_logs: Any
    retrieval_telemetry: Any
    embedder: Any
    vector_store: Any
    embedding_maintenance: Any


class MemoryMutationDependencies(Protocol):
    config: Any
    journal: Any
    task_queue: Any
    workspace_id: str | None


class MemoryReadPort(Protocol):
    @property
    def workspace_id(self) -> str | None: ...

    @property
    def relational_search(self) -> Any: ...

    @property
    def read_cache(self) -> ReadCachePort | None: ...


class RetrievalTelemetryPort(Protocol):
    def record_search(
        self,
        *,
        caller_kind: str,
        query: str,
        surfaced_memory_ids: list[str],
        duration_ms: float,
    ) -> None: ...

    def record_read(
        self,
        *,
        caller_kind: str,
        memory_id: str,
        duration_ms: float,
    ) -> None: ...
