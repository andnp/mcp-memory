import asyncio
from collections.abc import Callable, Coroutine
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import FastAPI

from mcp_memory.config import Config
from mcp_memory.context import (
    ApplicationContext,
    BackgroundTaskCapabilities,
    ManagementRuntimeCapabilities,
    MemoryReadCapabilities,
    MutationCapabilities,
    ProviderCapabilities,
    TaskRuntimeCapabilities,
)
from mcp_memory.curation_quality_store import SQLiteCurationQualityStore
from mcp_memory.daemon_lifecycle import FilesystemLock
from mcp_memory.daemon_runtime import DaemonRuntimeSession
from mcp_memory.internal_tool_call_tracking import InternalToolCallTracker
from mcp_memory.mcp.runtime import (
    RuntimeBootstrapSpec,
    RuntimeCapabilityBundles,
    RuntimeComposition,
    RuntimeResources,
    WorkspaceRuntimeSpec,
    create_runtime_composition,
)
from mcp_memory.storage.types import StorageBackendResources


class _MemorySearchPort:
    def get_health(self) -> object:
        return object()

    def read_memory(self, memory_id: str) -> object:
        raise NotImplementedError

    def peek_memory(self, memory_id: str) -> object:
        raise NotImplementedError

    def search_memories_for_maintenance(
        self,
        query: str,
        workspace_id: str | None = None,
        limit: int = 50,
        *,
        memory_type: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
    ) -> object:
        raise NotImplementedError

    def resolve_memory_id(self, memory_id: str) -> str | None:
        raise NotImplementedError


class _Closeable:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    def close(self) -> None:
        self.calls.append(self.name)


def _empty_capabilities() -> RuntimeCapabilityBundles:
    return RuntimeCapabilityBundles(
        memory=MemoryReadCapabilities(),
        mutation=MutationCapabilities(),
        provider=ProviderCapabilities(),
        background=BackgroundTaskCapabilities(),
        task=TaskRuntimeCapabilities(
            memory=MemoryReadCapabilities(),
            mutation=MutationCapabilities(),
            provider=ProviderCapabilities(),
        ),
        management=ManagementRuntimeCapabilities(
            memory=MemoryReadCapabilities(),
            mutation=MutationCapabilities(),
            provider=ProviderCapabilities(),
        ),
    )


def test_create_runtime_composition_exposes_grouped_resources(monkeypatch, tmp_path) -> None:
    embedder = object()
    provider_registry = {"test": {"json": object()}}
    search_port = _MemorySearchPort()
    ingress_batch_evidence = object()
    ingress_action_receipts = object()
    source_coverage = object()
    ingress_quality_evidence = object()
    ingress_mutation_transaction = object()
    storage = StorageBackendResources(
        backend="sqlite",
        db_manager=object(),
        journal=object(),
        repository=object(),
        relational_search=search_port,
        read_cache=object(),
        task_queue=object(),
        provider_usage=object(),
        runtime_logs=object(),
        provider_policy_events=object(),
        embedding_integrity_events=object(),
        task_execution_attempts=object(),
        work_items=object(),
        embedding_repair_queue=object(),
        vector_store=object(),
        ingress_batch_evidence=ingress_batch_evidence,
        ingress_action_receipts=ingress_action_receipts,
        source_coverage=source_coverage,
        ingress_quality_evidence=ingress_quality_evidence,
        ingress_mutation_transaction=ingress_mutation_transaction,
    )

    monkeypatch.setattr("mcp_memory.mcp.runtime.build_embedder", lambda config: embedder)
    monkeypatch.setattr(
        "mcp_memory.mcp.runtime.build_storage_runtime_components",
        lambda *args, **kwargs: storage,
    )
    monkeypatch.setattr(
        "mcp_memory.mcp.runtime._build_provider_registry",
        lambda **kwargs: provider_registry,
    )

    composition = create_runtime_composition(
        WorkspaceRuntimeSpec(
            memory_path=tmp_path / "memory",
            config=Config(),
            workspace_id="workspace",
            workspace_root=tmp_path,
            lock_path=tmp_path / "lock",
        )
    )

    resources = composition.resources
    assert resources is not None
    assert resources.storage is storage
    assert composition.context.relational_search is search_port
    assert composition.context.ingress_batch_evidence is ingress_batch_evidence
    assert composition.context.ingress_action_receipts is ingress_action_receipts
    assert composition.context.source_coverage is source_coverage
    assert composition.context.ingress_quality_evidence is ingress_quality_evidence
    assert composition.context.ingress_mutation_transaction is ingress_mutation_transaction
    assert isinstance(composition.context.curation_quality, SQLiteCurationQualityStore)
    assert resources.embedder is embedder
    assert resources.provider_registry is provider_registry
    assert resources.internal_tool_call_tracker is composition.context.internal_tool_call_tracker
    assert composition.daemon.memory is composition.capabilities.memory
    assert composition.daemon.background is composition.capabilities.background
    assert composition.daemon.task is composition.capabilities.task
    assert composition.daemon.resources is resources
    assert composition.daemon.writeback is None


def test_runtime_resources_close_order_is_idempotent_and_skips_missing_close_methods() -> None:
    calls: list[str] = []
    storage = StorageBackendResources(
        backend="sqlite",
        db_manager=_Closeable("db", calls),
        journal=object(),
        repository=_Closeable("repository", calls),
        relational_search=_MemorySearchPort(),
        read_cache=_Closeable("cache", calls),
        task_queue=object(),
        provider_usage=object(),
        runtime_logs=_Closeable("logs", calls),
        provider_policy_events=object(),
        embedding_integrity_events=_Closeable("integrity", calls),
        task_execution_attempts=object(),
        work_items=object(),
        embedding_repair_queue=object(),
        vector_store=object(),
    )
    resources = RuntimeResources(
        storage=storage,
        embedder=None,
        provider_registry={},
        internal_tool_call_tracker=cast(InternalToolCallTracker, object()),
    )

    resources.close()
    resources.close()

    assert calls == ["cache", "logs", "integrity", "repository", "db"]


def test_runtime_resources_do_not_close_aliased_storage_resources_twice() -> None:
    calls: list[str] = []
    shared_logs = _Closeable("shared", calls)
    storage = StorageBackendResources(
        backend="sqlite",
        db_manager=_Closeable("db", calls),
        journal=object(),
        repository=_Closeable("repository", calls),
        relational_search=_MemorySearchPort(),
        read_cache=_Closeable("cache", calls),
        task_queue=object(),
        provider_usage=object(),
        runtime_logs=shared_logs,
        provider_policy_events=object(),
        embedding_integrity_events=shared_logs,
        task_execution_attempts=object(),
        work_items=object(),
        embedding_repair_queue=object(),
        vector_store=object(),
    )
    resources = RuntimeResources(
        storage=storage,
        embedder=None,
        provider_registry={},
        internal_tool_call_tracker=cast(InternalToolCallTracker, object()),
    )

    resources.close()

    assert calls == ["cache", "shared", "repository", "db"]


def test_runtime_composition_closes_lazy_telemetry_and_wraps_legacy_context() -> None:
    calls: list[str] = []
    context = ApplicationContext(
        storage_backend="sqlite",
        db_manager=_Closeable("db", calls),
        read_cache=_Closeable("cache", calls),
        runtime_logs=_Closeable("logs", calls),
        retrieval_telemetry=_Closeable("telemetry", calls),
        embedding_integrity_events=_Closeable("integrity", calls),
        repository=_Closeable("repository", calls),
        ingress_batch_evidence=object(),
        ingress_action_receipts=object(),
        source_coverage=object(),
        ingress_mutation_transaction=object(),
    )
    composition = RuntimeComposition(context=context, capabilities=_empty_capabilities())

    resources = composition.resources
    assert resources is not None
    assert composition.daemon.memory is composition.capabilities.memory
    assert composition.daemon.background is composition.capabilities.background
    assert composition.daemon.task is composition.capabilities.task
    assert composition.daemon.resources is resources
    assert composition.daemon.writeback is None
    assert resources.storage.ingress_batch_evidence is context.ingress_batch_evidence
    assert resources.storage.ingress_action_receipts is context.ingress_action_receipts
    assert resources.storage.source_coverage is context.source_coverage
    assert resources.storage.ingress_mutation_transaction is context.ingress_mutation_transaction

    context.close = lambda: pytest.fail("composition must not delegate to ApplicationContext.close")
    composition.close()
    composition.close()

    assert calls == ["telemetry", "cache", "logs", "integrity", "repository", "db"]


def test_application_context_close_keeps_legacy_resource_order() -> None:
    calls: list[str] = []
    context = ApplicationContext(
        read_cache=_Closeable("cache", calls),
        retrieval_telemetry=_Closeable("telemetry", calls),
        runtime_logs=_Closeable("logs", calls),
        embedding_integrity_events=_Closeable("integrity", calls),
        repository=_Closeable("repository", calls),
        db_manager=_Closeable("db", calls),
    )

    context.close()

    assert calls == ["cache", "telemetry", "logs", "integrity", "repository", "db"]


@pytest.mark.asyncio
async def test_daemon_stop_closes_composition_once(monkeypatch, tmp_path) -> None:
    close_calls: list[str] = []

    class _Composition:
        context = SimpleNamespace(close=lambda: pytest.fail("legacy context close must not be used"))

        def close(self) -> None:
            close_calls.append("composition")

    class _Lock:
        def release(self) -> None:
            return None

    session = object.__new__(DaemonRuntimeSession)
    session.backup_task = None
    session.writeback_task = None
    session.writeback_executor = None
    session.transport = None
    session.warmup_task = None
    session.worker = None
    session.composition = cast(RuntimeComposition, _Composition())
    session.metadata_path = tmp_path / "metadata"
    session.runtime_lock = cast(FilesystemLock, _Lock())
    monkeypatch.setattr("mcp_memory.daemon_runtime.remove_metadata", lambda *args, **kwargs: None)

    await session.stop()
    await session.stop()

    assert close_calls == ["composition"]


@pytest.mark.asyncio
async def test_daemon_session_uses_typed_bundle_for_startup_and_cleanup(monkeypatch, tmp_path) -> None:
    """The daemon session uses composed ports while preserving lifecycle order."""
    events: list[str] = []

    class _Lock:
        def __init__(self, path) -> None:
            return None

        def acquire(self, *, timeout_seconds: float) -> None:
            events.append("lock.acquire")

        def release(self) -> None:
            events.append("lock.release")

    class _Worker:
        async def start(self) -> None:
            events.append("worker.start")

        async def stop(self, grace_seconds: float) -> None:
            events.append("worker.stop")

    class _Transport:
        def __init__(self, **kwargs) -> None:
            events.append("transport.construct")

        async def start(self) -> None:
            events.append("transport.start")

        async def stop(self) -> None:
            events.append("transport.stop")

    class _HookService:
        def __init__(self, db_manager, workspace_id) -> None:
            events.append("hook.construct")

        def get_active_client_count(self, *, now=None) -> int:
            return 0

    class _ManagementService:
        capabilities = object()
        federation_source = None
        dashboard_static_root = tmp_path

        def __init__(self, capabilities, *, controller) -> None:
            events.append("management.construct")

    async def _pending_payload() -> dict[str, object]:
        await asyncio.Event().wait()
        return {}

    async def _pending_bool() -> bool:
        await asyncio.Event().wait()
        return False

    def _warmup(daemon) -> Coroutine[object, object, bool]:
        events.append("warmup.create")
        return _pending_bool()

    def _backup(daemon):
        events.append("backup.create")
        return _pending_payload()

    class _Composition:
        context = object()
        capabilities = SimpleNamespace(management=object())
        daemon = SimpleNamespace(
            memory=SimpleNamespace(db_manager=object()),
            background=object(),
            task=object(),
        )

        def close(self) -> None:
            events.append("composition.close")

    spec = SimpleNamespace(
        lock_path=tmp_path / "daemon.lock",
        config=SimpleNamespace(daemon=SimpleNamespace(shutdown_grace_seconds=0.2)),
    )
    app = FastAPI()
    monkeypatch.setattr("mcp_memory.daemon_runtime.FilesystemLock", _Lock)
    monkeypatch.setattr("mcp_memory.daemon_runtime.resolve_daemon_metadata_path", lambda scope: tmp_path / "metadata")
    monkeypatch.setattr("mcp_memory.daemon_runtime.resolve_daemon_socket_path", lambda: tmp_path / "daemon.sock")
    monkeypatch.setattr(
        "mcp_memory.daemon_runtime.bootstrap_background_tasks",
        lambda capabilities: events.append("background.bootstrap"),
    )
    monkeypatch.setattr("mcp_memory.daemon_runtime.build_runtime_task_worker", lambda capabilities: _Worker())
    monkeypatch.setattr("mcp_memory.daemon_runtime.run_periodic_backup_loop", _backup)
    monkeypatch.setattr("mcp_memory.daemon_runtime.HookReminderService", _HookService)
    monkeypatch.setattr("mcp_memory.daemon_runtime.DaemonZmqServer", _Transport)
    monkeypatch.setattr("mcp_memory.daemon_runtime.ManagementService", _ManagementService)
    monkeypatch.setattr(
        "mcp_memory.daemon_runtime.resolve_record_thought_writeback_flush_context",
        lambda daemon: None,
    )
    monkeypatch.setattr(
        "mcp_memory.daemon_runtime.write_metadata",
        lambda path, metadata: events.append("metadata.write"),
    )
    monkeypatch.setattr(
        "mcp_memory.daemon_runtime.remove_metadata",
        lambda path, expected_pid: events.append("metadata.remove"),
    )

    session = DaemonRuntimeSession(
        app=app,
        spec=spec,
        host="127.0.0.1",
        port=8123,
        enable_idle_shutdown=False,
        request_scope_context_factory=lambda arguments: cast(ApplicationContext, object()),
        session_start_handler=lambda arguments: _pending_payload(),
        post_tool_use_handler=lambda arguments: _pending_payload(),
        session_end_handler=lambda arguments: _pending_payload(),
        runtime_version=None,
        runtime_factory=cast(
            Callable[[RuntimeBootstrapSpec], RuntimeComposition],
            lambda runtime_spec: _Composition(),
        ),
        embedding_warmup=_warmup,
        dashboard_builder=lambda path: events.append("dashboard.build"),
    )

    await session.start()

    assert events == [
        "lock.acquire",
        "background.bootstrap",
        "warmup.create",
        "backup.create",
        "worker.start",
        "hook.construct",
        "transport.construct",
        "transport.start",
        "management.construct",
        "metadata.write",
        "dashboard.build",
    ]

    await session.stop()

    assert events[-5:] == [
        "transport.stop",
        "worker.stop",
        "composition.close",
        "metadata.remove",
        "lock.release",
    ]
