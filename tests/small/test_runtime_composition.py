from types import SimpleNamespace

from mcp_memory.config import Config
from mcp_memory.mcp.runtime import (
    WorkspaceRuntimeSpec,
    create_runtime_composition,
)
from mcp_memory.storage.types import StorageBackendResources


def test_create_runtime_composition_exposes_grouped_resources(monkeypatch, tmp_path) -> None:
    embedder = object()
    provider_registry = {"test": {"json": object()}}
    storage = StorageBackendResources(
        backend="sqlite",
        db_manager=object(),
        journal=object(),
        repository=object(),
        relational_search=SimpleNamespace(get_health=lambda: object()),
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

    assert composition.resources.storage is storage
    assert composition.resources.embedder is embedder
    assert composition.resources.provider_registry is provider_registry
    assert composition.resources.internal_tool_call_tracker is composition.context.internal_tool_call_tracker
