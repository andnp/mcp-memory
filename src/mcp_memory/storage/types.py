from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_memory.config import Config


class PostgresBackendNotImplementedError(NotImplementedError):
    pass


@dataclass(frozen=True)
class StorageBackendResources:
    backend: str
    db_manager: Any
    journal: Any
    repository: Any
    relational_search: Any
    read_cache: Any
    task_queue: Any
    provider_usage: Any
    runtime_logs: Any
    provider_policy_events: Any
    embedding_integrity_events: Any
    task_execution_attempts: Any
    work_items: Any
    embedding_repair_queue: Any
    vector_store: Any
    mutation_history: Any = None
    curation: Any = None
    curation_action_store: Any = None
    embedding_maintenance: Any = None


@dataclass(frozen=True)
class StorageBootstrapSpec:
    memory_path: Path
    config: Config
    workspace_id: str | None
