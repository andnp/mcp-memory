from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import TypeAlias

from mcp_memory.config import (
    GLOBAL_DAEMON_IDENTITY,
    Config,
    ensure_default_config_exists,
    load_config,
    resolve_daemon_lock_path,
    resolve_memory_path,
    resolve_workspace_id,
    resolve_workspace_root,
)
from mcp_memory.context import ApplicationContext
from mcp_memory.internal_tool_call_tracking import InternalToolCallTracker
from mcp_memory.embeddings import build_embedder
from mcp_memory.core.providers.instrumented import InstrumentedAIProvider
from mcp_memory.core.providers import build_agentic_ai_provider
from mcp_memory.core.providers import build_json_ai_provider
from mcp_memory.core.storage import ensure_memory_dirs
from mcp_memory.storage.factory import StorageBackendResources, build_storage_runtime_components
from mcp_memory.storage.types import StorageBootstrapSpec


MCPRuntime = ApplicationContext


@dataclass(frozen=True)
class WorkspaceRuntimeSpec:
    memory_path: Path
    config: Config
    workspace_id: str
    workspace_root: Path
    lock_path: Path


@dataclass(frozen=True)
class GlobalDaemonBootstrapSpec:
    memory_path: Path
    config: Config
    workspace_root: Path
    lock_path: Path


RuntimeBootstrapSpec: TypeAlias = WorkspaceRuntimeSpec | GlobalDaemonBootstrapSpec


def resolve_workspace_runtime_spec(
    workspace_root_override: str | None = None,
    cwd: Path | None = None,
) -> WorkspaceRuntimeSpec:
    config, memory_path, workspace_root, lock_path = _resolve_runtime_bootstrap_fields(
        workspace_root_override,
        cwd,
    )
    runtime_cwd = cwd if cwd is not None else Path.cwd()
    workspace_id = resolve_workspace_id(runtime_cwd, workspace_root_override)
    return WorkspaceRuntimeSpec(
        config=config,
        workspace_id=workspace_id,
        workspace_root=workspace_root,
        memory_path=memory_path,
        lock_path=lock_path,
    )


def resolve_global_daemon_bootstrap_spec(
    workspace_root_override: str | None = None,
    cwd: Path | None = None,
) -> GlobalDaemonBootstrapSpec:
    config, memory_path, workspace_root, lock_path = _resolve_runtime_bootstrap_fields(
        workspace_root_override,
        cwd,
    )
    return GlobalDaemonBootstrapSpec(
        config=config,
        workspace_root=workspace_root,
        memory_path=memory_path,
        lock_path=lock_path,
    )


def _resolve_runtime_bootstrap_fields(
    workspace_root_override: str | None = None,
    cwd: Path | None = None,
) -> tuple[Config, Path, Path, Path]:
    ensure_default_config_exists()
    config = load_config()
    runtime_cwd = cwd if cwd is not None else Path.cwd()
    workspace_root = resolve_workspace_root(runtime_cwd, workspace_root_override)
    memory_path = resolve_memory_path(config)
    ensure_memory_dirs(memory_path)
    return (
        config,
        memory_path,
        workspace_root,
        resolve_daemon_lock_path(GLOBAL_DAEMON_IDENTITY),
    )


def create_runtime_from_spec(
    spec: RuntimeBootstrapSpec,
    *,
    enable_background_repair_queue: bool = False,
) -> ApplicationContext:
    workspace_id = _workspace_id_for_runtime(spec)
    embedder = build_embedder(spec.config.embeddings)
    internal_tool_call_tracker = InternalToolCallTracker()
    storage = build_storage_runtime_components(
        StorageBootstrapSpec(
            memory_path=spec.memory_path,
            config=spec.config,
            workspace_id=workspace_id,
        ),
        embedder=embedder,
        enable_background_repair_queue=enable_background_repair_queue,
    )
    provider_registry = _build_provider_registry(
        config=spec.config,
        workspace_root=spec.workspace_root,
        workspace_id=workspace_id,
        storage=storage,
    )
    default_profile_key = next(
        (k for k in spec.config.provider_routing.default_json_route if k in provider_registry),
        next(iter(provider_registry), None),
    )
    default_bundle = {} if default_profile_key is None else provider_registry.get(default_profile_key, {})
    ai_json_provider = default_bundle.get("json")
    ai_agent_provider = default_bundle.get("agentic")
    return ApplicationContext(
        config=spec.config,
        workspace_id=workspace_id,
        workspace_root=spec.workspace_root,
        memory_path=spec.memory_path,
        storage_backend=storage.backend,
        db_manager=storage.db_manager,
        journal=storage.journal,
        repository=storage.repository,
        relational_search=storage.relational_search,
        read_cache=storage.read_cache,
        task_queue=storage.task_queue,
        ai_json_provider=ai_json_provider,
        ai_agent_provider=ai_agent_provider,
        ai_provider=ai_json_provider,
        ai_provider_registry=provider_registry,
        provider_usage=storage.provider_usage,
        runtime_logs=storage.runtime_logs,
        provider_policy_events=storage.provider_policy_events,
        embedding_integrity_events=storage.embedding_integrity_events,
        task_execution_attempts=storage.task_execution_attempts,
        work_items=storage.work_items,
        embedding_repair_queue=storage.embedding_repair_queue,
        embedder=embedder,
        vector_store=storage.vector_store,
        search_health=storage.relational_search.get_health(),
        internal_tool_call_tracker=internal_tool_call_tracker,
    )


def create_runtime(
    workspace_root_override: str | None = None,
    cwd: Path | None = None,
) -> ApplicationContext:
    spec = resolve_workspace_runtime_spec(workspace_root_override, cwd)
    return create_runtime_from_spec(spec)


def _build_provider_registry(
    *,
    config: Config,
    workspace_root: Path,
    workspace_id: str | None,
    storage: StorageBackendResources,
) -> dict[str, dict[str, object]]:
    if _runtime_providers_disabled_for_tests():
        return {}

    registry: dict[str, dict[str, object]] = {}
    usage_repository = storage.provider_usage
    task_execution_attempts = storage.task_execution_attempts
    provider_task_queue = _provider_task_queue_capability(storage.task_queue)

    def add_profile(profile_key: str, ai_config) -> None:
        budget_limit = config.provider_routing.profile_daily_call_limits.get(profile_key)
        bundle: dict[str, object] = {}
        json_provider = build_json_ai_provider(ai_config, config, workspace_root)
        if json_provider is not None and _provider_command_available(json_provider):
            bundle["json"] = InstrumentedAIProvider(
                json_provider,
                usage_repository=usage_repository,
                provider_key=profile_key,
                provider_name=getattr(json_provider, "provider_name", profile_key),
                model_name=ai_config.model,
                workspace_id=workspace_id,
                task_queue=provider_task_queue,
                task_execution_attempts=task_execution_attempts,
                budget_key=profile_key,
                daily_call_limit=budget_limit,
                model_burst_call_limit=config.provider_routing.model_burst_call_limit,
                model_burst_window_seconds=config.provider_routing.model_burst_window_seconds,
            )
        agentic_provider = build_agentic_ai_provider(ai_config, config, workspace_root)
        if agentic_provider is not None and _provider_command_available(agentic_provider):
            bundle["agentic"] = InstrumentedAIProvider(
                agentic_provider,
                usage_repository=usage_repository,
                provider_key=f"{profile_key}:agentic",
                provider_name=getattr(agentic_provider, "provider_name", f"{profile_key}:agentic"),
                model_name=ai_config.model,
                workspace_id=workspace_id,
                task_queue=provider_task_queue,
                task_execution_attempts=task_execution_attempts,
                budget_key=profile_key,
                daily_call_limit=budget_limit,
                model_burst_call_limit=config.provider_routing.model_burst_call_limit,
                model_burst_window_seconds=config.provider_routing.model_burst_window_seconds,
            )
        if bundle:
            registry[profile_key] = bundle

    for profile_key, ai_config in config.provider_routing.profiles.items():
        add_profile(profile_key, ai_config)

    return registry


def _workspace_id_for_runtime(spec: RuntimeBootstrapSpec) -> str | None:
    if isinstance(spec, WorkspaceRuntimeSpec):
        return spec.workspace_id
    return None


def _provider_command_available(provider: object) -> bool:
    command = getattr(provider, "_command", None)
    if not isinstance(command, str) or not command.strip():
        return True
    resolved = shutil.which(command)
    if resolved is not None:
        return True
    candidate = Path(command).expanduser()
    return candidate.exists()


def _provider_task_queue_capability(task_queue: object) -> object | None:
    if task_queue is None:
        return None
    try:
        touch_running_task = getattr(task_queue, "touch_running_task", None)
    except NotImplementedError:
        return None
    if not callable(touch_running_task):
        return None
    return task_queue


def _runtime_providers_disabled_for_tests() -> bool:
    return os.environ.get("MCP_MEMORY_TEST_MODE") == "1"
