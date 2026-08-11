from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from mcp_memory.core.ports import MemoryIDResolutionPort, ReadCacheValidationPort


class MemorySearchPort(Protocol):
    def get_health(self) -> Any: ...

    def read_memory(self, memory_id: str) -> Any: ...

    def peek_memory(self, memory_id: str) -> Any: ...

    def search_memories_for_maintenance(
        self,
        query: str,
        workspace_id: str | None = None,
        limit: int = 50,
        *,
        memory_type: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
    ) -> Any: ...

    def resolve_memory_id(self, memory_id: str) -> str | None: ...



CACHE_SCHEMA_VERSION = 1
DEFAULT_SEARCH_POLICY_VERSION = "default"


@dataclass(frozen=True)
class SharedReadCacheSearchRequest:
    query: str
    workspace_id: str | None
    limit: int
    adaptive_limit: bool
    memory_type: str | None
    status: str | None
    include_superseded: bool
    ranking_workspace_id: str | None = None
    policy_version: str = DEFAULT_SEARCH_POLICY_VERSION
    feature_fingerprint: str | None = None

    def normalized_params(self) -> dict[str, Any]:
        params = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "query": self.query,
            "workspace_id": self.workspace_id,
            "limit": self.limit,
            "adaptive_limit": self.adaptive_limit,
            "memory_type": self.memory_type,
            "status": self.status,
            "include_superseded": self.include_superseded,
        }
        if self.ranking_workspace_id is not None:
            params["ranking_workspace_id"] = self.ranking_workspace_id
        if self.policy_version != DEFAULT_SEARCH_POLICY_VERSION:
            params["policy_version"] = self.policy_version
        if self.feature_fingerprint is not None:
            params["feature_fingerprint"] = self.feature_fingerprint
        return params


@dataclass(frozen=True)
class SharedReadCacheReadEntry:
    payload: dict[str, Any]
    validation_token: str | None


@dataclass(frozen=True)
class SharedReadCacheProjectionUpsert:
    memory_id: str
    payload: dict[str, Any]
    validation_token: str | None = None


@dataclass(frozen=True)
class SharedReadCacheProjectionEntry:
    memory_id: str
    payload: dict[str, Any]
    validation_token: str | None
    cached_at: float


@dataclass(frozen=True)
class SharedReadCacheInFlightSearch:
    cache_key: str
    is_leader: bool
    _state: Any


class ReadCachePort(Protocol):
    def load_search_response(self, request: SharedReadCacheSearchRequest) -> dict[str, Any] | None: ...
    def load_fresh_search_response(self, request: SharedReadCacheSearchRequest, *, ttl_seconds: float) -> dict[str, Any] | None: ...
    def store_search_response(self, request: SharedReadCacheSearchRequest, payload: dict[str, Any]) -> None: ...
    def begin_inflight_search(self, request: SharedReadCacheSearchRequest) -> SharedReadCacheInFlightSearch: ...
    def wait_for_inflight_search(self, entry: SharedReadCacheInFlightSearch) -> dict[str, Any]: ...
    def finish_inflight_search(self, entry: SharedReadCacheInFlightSearch, *, payload: dict[str, Any] | None = None, error: Exception | None = None) -> None: ...
    def load_read_response(self, memory_id: str) -> dict[str, Any] | None: ...
    def load_read_entry(self, memory_id: str) -> SharedReadCacheReadEntry | None: ...
    def store_read_response(self, memory_id: str, payload: dict[str, Any], *, validation_token: str | None = None) -> None: ...
    def search_projection_entries(self, request: SharedReadCacheSearchRequest, *, limit: int | None = None) -> list[SharedReadCacheProjectionEntry]: ...
    def load_projection_entries(self, memory_ids: list[str]) -> list[SharedReadCacheProjectionEntry]: ...
    def store_projection_entries(self, entries: list[SharedReadCacheProjectionUpsert]) -> None: ...
    def delete_projection_entries(self, memory_ids: list[str]) -> None: ...
    def increment_metric(self, metric_name: str, *, amount: int = 1) -> None: ...


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
    relational_search: MemorySearchPort | None = None
    memory_retrieval: Any = None
    read_cache: ReadCachePort | None = None
    vector_store: Any = None
    embedder: Any = None
    embedding_maintenance: Any = None
    read_cache_validation: ReadCacheValidationPort | None = None
    memory_id_resolution: MemoryIDResolutionPort | None = None


class MemoryReadContext(Protocol):
    config: Any
    workspace_id: str | None
    db_manager: Any
    repository: Any
    relational_search: MemorySearchPort | None
    read_cache: ReadCachePort | None
    storage_backend: str | None
    runtime_logs: Any
    retrieval_telemetry: Any
    embedder: Any
    vector_store: Any
    embedding_maintenance: Any
    read_cache_validation: ReadCacheValidationPort | None
    memory_id_resolution: MemoryIDResolutionPort | None


class MemoryMutationDependencies(Protocol):
    config: Any
    journal: Any
    repository: Any
    task_queue: Any
    workspace_id: str | None


class MemoryReadPort(Protocol):
    @property
    def workspace_id(self) -> str | None: ...

    @property
    def relational_search(self) -> MemorySearchPort | None: ...

    @property
    def read_cache(self) -> ReadCachePort | None: ...

    @property
    def read_cache_validation(self) -> ReadCacheValidationPort | None: ...

    @property
    def memory_id_resolution(self) -> MemoryIDResolutionPort | None: ...


class RetrievalTelemetryPort(Protocol):
    def record_search(
        self,
        *,
        caller_kind: str,
        query: str,
        surfaced_memory_ids: list[str],
        graph_provenance: Mapping[str, object] | None = None,
        duration_ms: float,
    ) -> None: ...

    def record_read(
        self,
        *,
        caller_kind: str,
        memory_id: str,
        duration_ms: float,
    ) -> None: ...
