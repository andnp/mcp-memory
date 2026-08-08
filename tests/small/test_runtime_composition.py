from types import SimpleNamespace
from typing import cast

import pytest

from mcp_memory.config import Config
from mcp_memory.context import (
    BackgroundTaskCapabilities,
    ManagementRuntimeCapabilities,
    MemoryReadCapabilities,
    MutationCapabilities,
    ProviderCapabilities,
    TaskRuntimeCapabilities,
    ApplicationContext,
)
from mcp_memory.daemon_runtime import DaemonRuntimeSession
from mcp_memory.daemon_lifecycle import FilesystemLock
from mcp_memory.internal_tool_call_tracking import InternalToolCallTracker
from mcp_memory.mcp.runtime import (
    RuntimeCapabilityBundles,
    RuntimeComposition,
    RuntimeResources,
    WorkspaceRuntimeSpec,
    create_runtime_composition,
)
from mcp_memory.curation_quality_store import SQLiteCurationQualityStore
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
    assert isinstance(composition.context.curation_quality, SQLiteCurationQualityStore)
    assert resources.embedder is embedder
    assert resources.provider_registry is provider_registry
    assert resources.internal_tool_call_tracker is composition.context.internal_tool_call_tracker


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
    )
    composition = RuntimeComposition(context=context, capabilities=_empty_capabilities())

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
