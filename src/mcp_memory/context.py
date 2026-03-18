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
    workspace_id: str | None = None
    workspace_root: Path | None = None
    memory_path: Path | None = None
    db_manager: DatabaseManager | None = None
    journal: System1Journal | None = None
    repository: RelationalMemoryRepository | None = None
    relational_search: Any = None
    task_queue: Any = None
    ai_json_provider: Any = None
    ai_agent_provider: Any = None
    ai_provider: Any = None
    ai_provider_registry: dict[str, Any] | None = None
    embedder: Any = None
    vector_store: Any = None
    search_health: Any = None

    def close(self) -> None:
        if self.db_manager is not None:
            self.db_manager.close()
