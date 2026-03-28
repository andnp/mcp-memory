from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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
    task_queue: Any
    provider_usage: Any
    runtime_logs: Any
    provider_policy_events: Any
    task_execution_attempts: Any
    work_items: Any
    embedding_repair_queue: Any
    vector_store: Any


class RuntimeSpecLike(Protocol):
    @property
    def memory_path(self) -> Path: ...

    @property
    def config(self) -> Config: ...

    @property
    def workspace_id(self) -> str: ...