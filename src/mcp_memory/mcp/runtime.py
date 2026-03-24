from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil

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
from mcp_memory.embeddings import SQLiteVectorStore, build_embedder
from mcp_memory.core.journal import System1Journal
from mcp_memory.core.providers.instrumented import InstrumentedAIProvider
from mcp_memory.core.providers import build_agentic_ai_provider
from mcp_memory.core.providers import build_json_ai_provider
from mcp_memory.provider_usage_store import ProviderUsageRepository
from mcp_memory.provider_policy_event_store import ProviderPolicyEventRepository
from mcp_memory.work_item_store import SQLiteWorkItemRepository
from mcp_memory.core.tasks import SQLiteTaskQueue
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.relational.search import RelationalMemorySearchService
from mcp_memory.core.storage import ensure_memory_dirs
from mcp_memory.utils.db import DatabaseManager


MCPRuntime = ApplicationContext


@dataclass(frozen=True)
class RuntimeSpec:
    memory_path: Path
    config: Config
    workspace_id: str
    workspace_root: Path
    lock_path: Path


def resolve_runtime_spec(
    workspace_root_override: str | None = None,
    cwd: Path | None = None,
) -> RuntimeSpec:
    ensure_default_config_exists()
    config = load_config()
    runtime_cwd = cwd if cwd is not None else Path.cwd()
    workspace_root = resolve_workspace_root(runtime_cwd, workspace_root_override)
    workspace_id = resolve_workspace_id(runtime_cwd, workspace_root_override)
    memory_path = resolve_memory_path(config)
    ensure_memory_dirs(memory_path)
    return RuntimeSpec(
        config=config,
        workspace_id=workspace_id,
        workspace_root=workspace_root,
        memory_path=memory_path,
        lock_path=resolve_daemon_lock_path(GLOBAL_DAEMON_IDENTITY),
    )


def create_runtime_from_spec(spec: RuntimeSpec, *, enable_background_repair_queue: bool = False) -> ApplicationContext:
    db_manager = DatabaseManager(spec.memory_path / "indices" / "memory.db")
    journal = System1Journal(db_manager)
    repository = RelationalMemoryRepository(db_manager)
    embedder = build_embedder(spec.config.embeddings)
    vector_store = SQLiteVectorStore(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    work_items = SQLiteWorkItemRepository(db_manager)
    relational_search = RelationalMemorySearchService(
        repository,
        spec.config,
        embedder=embedder,
        vector_store=vector_store,
        db_manager=db_manager,
        task_queue=task_queue,
        work_items=work_items,
        background_repair_wait_seconds=5.0 if enable_background_repair_queue else 0.0,
    )
    relational_search.run_startup_health_check()
    provider_registry = _build_provider_registry(spec=spec, db_manager=db_manager, task_queue=task_queue)
    default_profile_key = _default_profile_key(spec.config)
    default_bundle = {} if default_profile_key is None else provider_registry.get(default_profile_key, {})
    ai_json_provider = default_bundle.get("json")
    ai_agent_provider = default_bundle.get("agentic")
    return ApplicationContext(
        config=spec.config,
        workspace_id=spec.workspace_id,
        workspace_root=spec.workspace_root,
        memory_path=spec.memory_path,
        db_manager=db_manager,
        journal=journal,
        repository=repository,
        relational_search=relational_search,
        task_queue=task_queue,
        ai_json_provider=ai_json_provider,
        ai_agent_provider=ai_agent_provider,
        ai_provider=ai_json_provider,
        ai_provider_registry=provider_registry,
        provider_policy_events=ProviderPolicyEventRepository(db_manager, workspace_id=spec.workspace_id),
        work_items=work_items,
        embedder=embedder,
        vector_store=vector_store,
        search_health=relational_search.get_health(),
    )


def create_runtime(
    workspace_root_override: str | None = None,
    cwd: Path | None = None,
) -> ApplicationContext:
    spec = resolve_runtime_spec(workspace_root_override, cwd)
    return create_runtime_from_spec(spec)


def _default_profile_key(config: Config) -> str | None:
    if config.ai.provider == "none":
        return None
    return config.ai.provider


def _build_provider_registry(*, spec: RuntimeSpec, db_manager: DatabaseManager, task_queue: SQLiteTaskQueue) -> dict[str, dict[str, object]]:
    if _runtime_providers_disabled_for_tests():
        return {}

    registry: dict[str, dict[str, object]] = {}
    usage_repository = ProviderUsageRepository(db_manager, workspace_id=spec.workspace_id)

    def add_profile(profile_key: str, ai_config) -> None:
        budget_limit = spec.config.provider_routing.profile_daily_call_limits.get(profile_key)
        bundle: dict[str, object] = {}
        json_provider = build_json_ai_provider(ai_config, spec.config, spec.workspace_root)
        if json_provider is not None and _provider_command_available(json_provider):
            bundle["json"] = InstrumentedAIProvider(
                json_provider,
                usage_repository=usage_repository,
                provider_key=profile_key,
                provider_name=getattr(json_provider, "provider_name", profile_key),
                model_name=ai_config.model,
                workspace_id=spec.workspace_id,
                task_queue=task_queue,
                budget_key=profile_key,
                daily_call_limit=budget_limit,
                model_burst_call_limit=spec.config.provider_routing.model_burst_call_limit,
                model_burst_window_seconds=spec.config.provider_routing.model_burst_window_seconds,
            )
        agentic_provider = build_agentic_ai_provider(ai_config, spec.config, spec.workspace_root)
        if agentic_provider is not None and _provider_command_available(agentic_provider):
            bundle["agentic"] = InstrumentedAIProvider(
                agentic_provider,
                usage_repository=usage_repository,
                provider_key=f"{profile_key}:agentic",
                provider_name=getattr(agentic_provider, "provider_name", f"{profile_key}:agentic"),
                model_name=ai_config.model,
                workspace_id=spec.workspace_id,
                task_queue=task_queue,
                budget_key=profile_key,
                daily_call_limit=budget_limit,
                model_burst_call_limit=spec.config.provider_routing.model_burst_call_limit,
                model_burst_window_seconds=spec.config.provider_routing.model_burst_window_seconds,
            )
        if bundle:
            registry[profile_key] = bundle

    default_profile_key = _default_profile_key(spec.config)
    if default_profile_key is not None:
        add_profile(default_profile_key, spec.config.ai)

    for profile_key, ai_config in spec.config.provider_routing.profiles.items():
        add_profile(profile_key, ai_config)

    return registry


def _provider_command_available(provider: object) -> bool:
    command = getattr(provider, "_command", None)
    if not isinstance(command, str) or not command.strip():
        return True
    resolved = shutil.which(command)
    if resolved is not None:
        return True
    candidate = Path(command).expanduser()
    return candidate.exists()


def _runtime_providers_disabled_for_tests() -> bool:
    return os.environ.get("MCP_MEMORY_TEST_MODE") == "1"
