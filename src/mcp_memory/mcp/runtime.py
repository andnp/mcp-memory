from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mcp_memory.config import (
    Config,
    detect_project,
    load_config,
    resolve_memory_path,
)
from mcp_memory.core.journal import System1Journal
from mcp_memory.core.repository import RelationalMemoryRepository
from mcp_memory.core.storage import ensure_memory_dirs
from mcp_memory.utils.db import DatabaseManager


@dataclass
class MCPRuntime:
    config: Config
    project_name: str | None
    memory_path: Path
    db_manager: DatabaseManager
    journal: System1Journal
    repository: RelationalMemoryRepository

    def close(self) -> None:
        self.db_manager.close()


def create_runtime(
    project_override: str | None = None,
    cwd: Path | None = None,
) -> MCPRuntime:
    config = load_config()
    project_name = detect_project(
        cwd=cwd,
        projects=config.projects,
        project_override=project_override,
    )
    config.detected_project = project_name
    memory_path = resolve_memory_path(config, project_name, config.projects)
    ensure_memory_dirs(memory_path)
    db_manager = DatabaseManager(memory_path / "indices" / "memory.db")
    journal = System1Journal(db_manager)
    repository = RelationalMemoryRepository(db_manager)
    return MCPRuntime(
        config=config,
        project_name=project_name,
        memory_path=memory_path,
        db_manager=db_manager,
        journal=journal,
        repository=repository,
    )