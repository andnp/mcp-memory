"""Provider-neutral capability ports for relational memory search."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from mcp_memory.relational.search import SearchHealthStatus


@runtime_checkable
class SearchHealthPort(Protocol):
    def get_health(self) -> SearchHealthStatus: ...


@runtime_checkable
class StartupHealthPort(Protocol):
    def run_startup_health_check(self) -> SearchHealthStatus: ...


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
    "MemoryIDResolutionPort",
    "ReadCacheValidationPort",
    "SearchHealthPort",
    "StartupHealthPort",
]
