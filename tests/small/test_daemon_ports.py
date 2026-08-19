from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mcp_memory.config import IngestSuppressionConfig
from mcp_memory.core.journal import System1Journal
from mcp_memory.core.journal_operations import RecordThoughtWritebackFlushResult
from mcp_memory.core.ports.tasks import TaskQueue
from mcp_memory.daemon_models import DaemonControllerView, DaemonMetadata
from mcp_memory.daemon_ports import (
    BackupPort,
    DaemonMetadataProvider,
    DashboardBuilderPort,
    EmbeddingModelPort,
    EmbeddingWarmupPort,
    HookPersistencePort,
    ManagementControllerPort,
    TransportDiagnosticsPort,
    TransportLifecyclePort,
    WritebackFlushPort,
)
from mcp_memory.daemon_transport import DaemonTransportDiagnosticsSnapshot
from mcp_memory.sqlite_backup import SQLiteBackupResult
from mcp_memory.storage.shared_read_cache import SharedReadCache

pytestmark = pytest.mark.small


class _Embedding:
    def cache_model(self) -> bool:
        return True


class _HookStore:
    def record_session_start(self, conversation_id: str, payload: dict[str, object] | None = None) -> dict[str, str]:
        del payload
        return {"conversation_id": conversation_id}

    def record_post_tool_use(self, payload: dict[str, object]) -> dict[str, str]:
        del payload
        return {}

    def record_session_end(self, conversation_id: str, payload: dict[str, object] | None = None) -> dict[str, str]:
        del payload
        return {"conversation_id": conversation_id}

    def get_active_client_count(self, *, now: float | None = None) -> int:
        del now
        return 2


class _Transport:
    def __init__(self) -> None:
        self.started = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False

    def get_diagnostics_snapshot(self) -> DaemonTransportDiagnosticsSnapshot:
        return DaemonTransportDiagnosticsSnapshot(current_in_flight_count=1)


@pytest.mark.asyncio
async def test_daemon_lifecycle_ports_accept_small_compatible_fakes(tmp_path: Path) -> None:
    """Small fakes can satisfy each independent lifecycle contract.

    The assertions exercise the operations exposed by the ports, rather than
    depending on a concrete daemon implementation.
    """

    async def warm(embedder: EmbeddingModelPort | None) -> bool:
        return embedder is not None and embedder.cache_model()

    def backup(db_path: Path, backup_dir: Path, *, max_snapshots: int) -> SQLiteBackupResult:
        return SQLiteBackupResult(db_path, (backup_dir / str(max_snapshots),))

    def flush(
        journal: System1Journal | None,
        *,
        task_queue: TaskQueue | None,
        suppression_config: IngestSuppressionConfig | None,
        writeback_cache: SharedReadCache | None,
        flush_limit: int = 8,
    ) -> RecordThoughtWritebackFlushResult:
        del journal, task_queue, suppression_config, writeback_cache, flush_limit
        return RecordThoughtWritebackFlushResult(flushed_count=3)

    def build_dashboard(static_root: Path) -> None:
        static_root.mkdir()

    warm_port: EmbeddingWarmupPort = warm
    backup_port: BackupPort = backup
    flush_port: WritebackFlushPort = flush
    dashboard_port: DashboardBuilderPort = build_dashboard

    assert await warm_port(_Embedding()) is True
    assert backup_port(tmp_path / "memory.sqlite3", tmp_path, max_snapshots=2).pruned_paths
    assert flush_port(None, task_queue=None, suppression_config=None, writeback_cache=None).flushed_count == 3
    dashboard_port(tmp_path / "static")
    assert (tmp_path / "static").is_dir()


def test_daemon_controller_ports_preserve_health_diagnostics() -> None:
    """Controller health combines hook persistence and transport diagnostics."""

    hook_store: HookPersistencePort = _HookStore()
    transport: TransportDiagnosticsPort = _Transport()
    controller = DaemonControllerView(hook_service=hook_store, transport_server=transport)
    controller_port: ManagementControllerPort = controller

    assert controller_port.has_runtime is True
    assert controller_port.client_count == 2
    assert controller_port.transport_diagnostics == {
        "current_in_flight_count": 1,
        "current_constrained_in_flight_count": 0,
        "max_concurrent_requests": 0,
        "request_slots_available": 0,
        "queued_waiter_count": 0,
        "recent_completed_request_count": 0,
        "recent_queue_wait_avg_ms": 0.0,
        "recent_queue_wait_max_ms": 0.0,
        "recent_execution_avg_ms": 0.0,
        "recent_execution_max_ms": 0.0,
        "active_requests": (),
        "recent_requests": (),
    }


def test_daemon_metadata_provider_exposes_current_value() -> None:
    """Metadata providers expose current values without a service bag."""

    metadata = DaemonMetadata(host="127.0.0.1", port=8765, pid=42, started_at=1.0, status="ready")

    def provide_metadata() -> DaemonMetadata:
        return metadata

    metadata_port: DaemonMetadataProvider = provide_metadata

    assert metadata_port() is metadata


def test_transport_lifecycle_port_has_async_start_and_stop() -> None:
    """Transport lifecycle remains separate from diagnostics consumption."""

    async def exercise(transport: TransportLifecyclePort) -> bool:
        await transport.start()
        await transport.stop()
        return True

    assert asyncio.run(exercise(_Transport())) is True
