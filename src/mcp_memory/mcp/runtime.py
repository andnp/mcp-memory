from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mcp_memory.config import (
    Config,
    load_config,
    resolve_memory_path,
    resolve_workspace_id,
    resolve_workspace_lock_path,
    resolve_workspace_root,
)
from mcp_memory.context import ApplicationContext
from mcp_memory.core.journal import System1Journal
from mcp_memory.core.providers import build_ai_provider
from mcp_memory.core.tasks import SQLiteTaskQueue
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.relational.search import RelationalMemorySearchService
from mcp_memory.core.storage import ensure_memory_dirs
from mcp_memory.utils.db import DatabaseManager


MCPRuntime = ApplicationContext


@dataclass(frozen=True)
class RuntimeSpec:
    memory_path: Path
    config: Config | None = None
    workspace_id: str | None = None
    project_name: str | None = None
    workspace_root: Path | None = None
    lock_path: Path | None = None

    def __post_init__(self) -> None:
        workspace_id = self.workspace_id or self.project_name
        if workspace_id is None:
            raise ValueError("RuntimeSpec requires workspace_id or project_name")

        object.__setattr__(self, "workspace_id", workspace_id)
        object.__setattr__(self, "project_name", workspace_id)

        workspace_root = self.workspace_root or Path.cwd()
        object.__setattr__(self, "workspace_root", workspace_root)

        if self.lock_path is None:
            object.__setattr__(
                self,
                "lock_path",
                resolve_workspace_lock_path(self.memory_path, workspace_id),
            )


def resolve_runtime_spec(
    project_override: str | None = None,
    cwd: Path | None = None,
) -> RuntimeSpec:
    config = load_config()
    runtime_cwd = cwd if cwd is not None else Path.cwd()
    workspace_root = resolve_workspace_root(runtime_cwd, project_override)
    workspace_id = resolve_workspace_id(runtime_cwd, project_override)
    config.detected_project = workspace_id
    memory_path = resolve_memory_path(config, workspace_id, config.projects)
    ensure_memory_dirs(memory_path)
    return RuntimeSpec(
        config=config,
        workspace_id=workspace_id,
        workspace_root=workspace_root,
        memory_path=memory_path,
        lock_path=resolve_workspace_lock_path(memory_path, workspace_id),
    )


def create_runtime_from_spec(spec: RuntimeSpec) -> ApplicationContext:
    assert spec.config is not None
    db_manager = DatabaseManager(spec.memory_path / "indices" / "memory.db")
    journal = System1Journal(db_manager)
    repository = RelationalMemoryRepository(db_manager)
    relational_search = RelationalMemorySearchService(repository, spec.config)
    task_queue = SQLiteTaskQueue(db_manager)
    ai_provider = build_ai_provider(spec.config.ai)
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
        ai_provider=ai_provider,
    )


def create_runtime(
    project_override: str | None = None,
    cwd: Path | None = None,
) -> ApplicationContext:
    spec = resolve_runtime_spec(project_override, cwd)
    return create_runtime_from_spec(spec)