from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_memory.config import Config


@dataclass
class ApplicationContext:
    config: Config | None = None
    workspace_id: str | None = None
    workspace_root: Path | None = None
    memory_path: Path | None = None
    storage_backend: str | None = None
    db_manager: Any = None
    journal: Any = None
    repository: Any = None
    relational_search: Any = None
    task_queue: Any = None
    ai_json_provider: Any = None
    ai_agent_provider: Any = None
    ai_provider: Any = None
    ai_provider_registry: dict[str, Any] | None = None
    provider_policy_events: Any = None
    task_execution_attempts: Any = None
    work_items: Any = None
    embedding_repair_queue: Any = None
    embedder: Any = None
    vector_store: Any = None
    search_health: Any = None

    def close(self) -> None:
        if self.db_manager is None:
            return
        close_method = getattr(self.db_manager, "close", None)
        if callable(close_method):
            close_method()
