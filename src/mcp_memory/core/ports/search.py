"""Provider-neutral capability ports for relational memory search."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

INTERNAL_SEARCH_TOOL_NAME = "internal_search_memory_records"


class SearchHealthSnapshot(Protocol):
    semantic_enabled: bool
    available: bool
    degraded: bool
    fallback_count: int
    rebuild_count: int
    background_repair_enabled: bool
    background_repair_wait_seconds: float
    queued_repair_backlog_count: int
    running_repair_count: int
    oldest_queued_repair_age_seconds: float | None
    repair_wait_count: int
    partial_semantic_search_count: int
    last_partial_semantic_at: str | None
    last_repair_wait_seconds: float
    last_repair_candidate_count: int
    last_repair_pending_count: int
    last_error: str | None
    last_failure_at: str | None
    last_recovery_at: str | None
    last_integrity_check_at: str | None
    integrity_check_error: str | None


@runtime_checkable
class SearchHealthPort(Protocol):
    def get_health(self) -> SearchHealthSnapshot: ...


@runtime_checkable
class StartupHealthPort(Protocol):
    def run_startup_health_check(self) -> SearchHealthSnapshot: ...


@runtime_checkable
class EmbeddingMaintenancePort(Protocol):
    def rebuild_semantic_index(
        self, *, limit: int = 10_000
    ) -> Mapping[str, int | bool | str | None]: ...


@runtime_checkable
class ReadCacheValidationPort(Protocol):
    def get_read_cache_validation_tokens(
        self, memory_ids: list[str]
    ) -> dict[str, str]: ...


@runtime_checkable
class MemoryIDResolutionPort(Protocol):
    def resolve_memory_id(self, memory_id: str) -> str | None: ...


__all__ = [
    "EmbeddingMaintenancePort",
    "INTERNAL_SEARCH_TOOL_NAME",
    "MemoryIDResolutionPort",
    "ReadCacheValidationPort",
    "SearchHealthPort",
    "SearchHealthSnapshot",
    "StartupHealthPort",
]
