"""Narrow structural contracts for daemon lifecycle dependencies."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from mcp_memory.config import IngestSuppressionConfig
from mcp_memory.core.journal import System1Journal
from mcp_memory.core.journal_operations import RecordThoughtWritebackFlushResult
from mcp_memory.core.ports.tasks import TaskQueue
from mcp_memory.sqlite_backup import SQLiteBackupResult
from mcp_memory.storage.shared_read_cache import SharedReadCache

if TYPE_CHECKING:
    from mcp_memory.daemon_models import DaemonMetadata, DaemonRoutes


@runtime_checkable
class EmbeddingModelPort(Protocol):
    def cache_model(self) -> bool: ...


@runtime_checkable
class EmbeddingWarmupPort(Protocol):
    async def __call__(self, embedder: EmbeddingModelPort | None) -> bool: ...


@runtime_checkable
class EmbeddingWarmupCompletionPort(Protocol):
    def __call__(self, task: asyncio.Task[bool]) -> None: ...


@runtime_checkable
class BackupPort(Protocol):
    def __call__(self, db_path: Path, backup_dir: Path, *, max_snapshots: int) -> SQLiteBackupResult: ...


@runtime_checkable
class DashboardBuilderPort(Protocol):
    def __call__(self, static_root: Path) -> None: ...


@dataclass(frozen=True, slots=True)
class WritebackDependencies:
    journal: System1Journal
    task_queue: TaskQueue | None
    writeback_cache: SharedReadCache
    suppression_config: IngestSuppressionConfig | None = None


@runtime_checkable
class WritebackFlushPort(Protocol):
    def __call__(
        self,
        journal: System1Journal,
        *,
        task_queue: TaskQueue | None,
        suppression_config: IngestSuppressionConfig | None,
        writeback_cache: SharedReadCache | None,
        flush_limit: int = 8,
    ) -> RecordThoughtWritebackFlushResult: ...


@runtime_checkable
class HookPersistencePort(Protocol):
    def record_session_start(
        self,
        conversation_id: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, str]: ...

    def record_post_tool_use(self, payload: dict[str, object]) -> dict[str, str]: ...

    def record_session_end(
        self,
        conversation_id: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, str]: ...

    def get_active_client_count(self, *, now: float | None = None) -> int: ...


@runtime_checkable
class HookClientCountPort(Protocol):
    def get_active_client_count(self) -> int: ...


@runtime_checkable
class TransportLifecyclePort(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...


@runtime_checkable
class TransportDiagnosticsPort(Protocol):
    def get_diagnostics_snapshot(self) -> object | None: ...


@runtime_checkable
class DaemonRoutesProvider(Protocol):
    def __call__(self) -> DaemonRoutes: ...


@runtime_checkable
class DaemonMetadataProvider(Protocol):
    def __call__(self) -> DaemonMetadata: ...


@runtime_checkable
class ManagementControllerPort(Protocol):
    @property
    def has_runtime(self) -> bool: ...

    @property
    def client_count(self) -> int: ...

    @property
    def transport_diagnostics(self) -> dict[str, object] | None: ...


__all__ = [
    "BackupPort",
    "DaemonMetadataProvider",
    "DaemonRoutesProvider",
    "DashboardBuilderPort",
    "EmbeddingModelPort",
    "EmbeddingWarmupCompletionPort",
    "EmbeddingWarmupPort",
    "HookClientCountPort",
    "HookPersistencePort",
    "ManagementControllerPort",
    "TransportDiagnosticsPort",
    "TransportLifecyclePort",
    "WritebackDependencies",
    "WritebackFlushPort",
]
