from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
from mcp_memory.core.providers import build_agentic_ai_provider_from_config
from mcp_memory.core.providers import build_json_ai_provider_from_config
from mcp_memory.provider_usage_store import ProviderUsageRepository
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


def create_runtime_from_spec(spec: RuntimeSpec) -> ApplicationContext:
    db_manager = DatabaseManager(spec.memory_path / "indices" / "memory.db")
    journal = System1Journal(db_manager)
    repository = RelationalMemoryRepository(db_manager)
    embedder = build_embedder(spec.config.embeddings)
    vector_store = SQLiteVectorStore(db_manager)
    relational_search = RelationalMemorySearchService(
        repository,
        spec.config,
        embedder=embedder,
        vector_store=vector_store,
        db_manager=db_manager,
    )
    relational_search.run_startup_health_check()
    task_queue = SQLiteTaskQueue(db_manager)
    ai_json_provider = build_json_ai_provider_from_config(spec.config, spec.workspace_root)
    if ai_json_provider is not None:
        ai_json_provider = InstrumentedAIProvider(
            ai_json_provider,
            usage_repository=ProviderUsageRepository(db_manager, workspace_id=spec.workspace_id),
            provider_key=spec.config.ai.provider,
            provider_name=getattr(ai_json_provider, "provider_name", spec.config.ai.provider),
            model_name=spec.config.ai.model,
            workspace_id=spec.workspace_id,
            task_queue=task_queue,
        )
    ai_agent_provider = build_agentic_ai_provider_from_config(spec.config, spec.workspace_root)
    if ai_agent_provider is not None:
        ai_agent_provider = InstrumentedAIProvider(
            ai_agent_provider,
            usage_repository=ProviderUsageRepository(db_manager, workspace_id=spec.workspace_id),
            provider_key=f"{spec.config.ai.provider}:agentic",
            provider_name=getattr(ai_agent_provider, "provider_name", f"{spec.config.ai.provider}:agentic"),
            model_name=spec.config.ai.model,
            workspace_id=spec.workspace_id,
            task_queue=task_queue,
        )
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