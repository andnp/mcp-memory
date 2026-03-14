from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_memory.config import Config
from mcp_memory.core.journal import System1Journal
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.utils.db import DatabaseManager


@dataclass
class ApplicationContext:
    config: Config | None = None
    project_name: str | None = None
    memory_path: Path | None = None
    db_manager: DatabaseManager | None = None
    journal: System1Journal | None = None
    repository: RelationalMemoryRepository | None = None
    relational_search: Any = None
    task_queue: Any = None
    memory_manager: Any = None
    memory_search: Any = None

    def close(self) -> None:
        if self.db_manager is not None:
            self.db_manager.close()