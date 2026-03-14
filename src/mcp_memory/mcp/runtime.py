from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mcp_memory.config import (
    Config,
    detect_project,
    load_config,
    resolve_memory_path,
)
from mcp_memory.context import ApplicationContext
from mcp_memory.core.journal import System1Journal
from mcp_memory.core.relational_search import RelationalMemorySearchService
from mcp_memory.core.repository import RelationalMemoryRepository
from mcp_memory.core.tasks import SQLiteTaskQueue
from mcp_memory.core.storage import ensure_memory_dirs
from mcp_memory.utils.db import DatabaseManager


MCPRuntime = ApplicationContext


@dataclass(frozen=True)
class RuntimeSpec:
    config: Config
    project_name: str | None
    memory_path: Path


def resolve_runtime_spec(
    project_override: str | None = None,
    cwd: Path | None = None,
) -> RuntimeSpec:
    config = load_config()
    project_name = detect_project(
        cwd=cwd,
        projects=config.projects,
        project_override=project_override,
    )
    config.detected_project = project_name
    memory_path = resolve_memory_path(config, project_name, config.projects)
    ensure_memory_dirs(memory_path)
    return RuntimeSpec(
        config=config,
        project_name=project_name,
        memory_path=memory_path,
    )


def create_runtime_from_spec(spec: RuntimeSpec) -> ApplicationContext:
    db_manager = DatabaseManager(spec.memory_path / "indices" / "memory.db")
    journal = System1Journal(db_manager)
    repository = RelationalMemoryRepository(db_manager)
    relational_search = RelationalMemorySearchService(repository, spec.config)
    task_queue = SQLiteTaskQueue(db_manager)
    return ApplicationContext(
        config=spec.config,
        project_name=spec.project_name,
        memory_path=spec.memory_path,
        db_manager=db_manager,
        journal=journal,
        repository=repository,
        relational_search=relational_search,
        task_queue=task_queue,
    )


def create_runtime(
    project_override: str | None = None,
    cwd: Path | None = None,
) -> ApplicationContext:
    spec = resolve_runtime_spec(project_override, cwd)
    return create_runtime_from_spec(spec)