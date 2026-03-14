from __future__ import annotations

from pathlib import Path

from mcp_memory.config import Config
from mcp_memory.core.journal import System1Journal
from mcp_memory.utils.db import DatabaseManager


class GetMemoryStatsOperation:
    def __init__(
        self,
        db_manager: DatabaseManager,
        journal: System1Journal | None,
        config: Config | None,
        workspace_id: str | None,
        workspace_root: Path | None,
        memory_path: Path | None,
    ) -> None:
        self._db_manager = db_manager
        self._journal = journal
        self._config = config
        self._workspace_id = workspace_id
        self._workspace_root = workspace_root
        self._memory_path = memory_path

    def execute(self) -> dict:
        conn = self._db_manager.get_connection()
        document_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        relational_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        return {
            "status": "ok",
            "workspace_id": self._workspace_id,
            "workspace_root": str(self._workspace_root) if self._workspace_root is not None else None,
            "ai_provider": self._config.ai.provider if self._config is not None else None,
            "ai_model": self._config.ai.model if self._config is not None else None,
            "memory_path": str(self._memory_path) if self._memory_path is not None else None,
            "documents": document_count,
            "relational_memories": relational_count,
            "journal": self._journal.count_by_status() if self._journal is not None else {},
        }