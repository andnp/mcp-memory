from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp_memory.core.journal import System1Journal
from mcp_memory.core.tasks import SQLiteTaskQueue
from mcp_memory.embedding_repair_store import SQLiteEmbeddingRepairQueue
from mcp_memory.embeddings import SQLiteVectorStore
from mcp_memory.provider_policy_event_store import ProviderPolicyEventRepository
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.relational.search import RelationalMemorySearchService
from mcp_memory.task_execution_store import TaskExecutionAttemptRepository
from mcp_memory.utils.db import DatabaseManager
from mcp_memory.work_item_store import SQLiteWorkItemRepository


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
    provider_policy_events: Any
    task_execution_attempts: Any
    work_items: Any
    embedding_repair_queue: Any
    vector_store: Any


def build_storage_runtime_components(
    spec: Any,
    *,
    embedder: Any,
    enable_background_repair_queue: bool,
) -> StorageBackendResources:
    if spec.config.storage.backend == "postgres":
        raise PostgresBackendNotImplementedError(
            "storage backend 'postgres' is not implemented yet; land the Postgres schema and repositories before enabling it"
        )

    db_manager = DatabaseManager(spec.memory_path / "indices" / "memory.db")
    journal = System1Journal(db_manager)
    repository = RelationalMemoryRepository(db_manager)
    vector_store = SQLiteVectorStore(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    task_execution_attempts = TaskExecutionAttemptRepository(db_manager, workspace_id=spec.workspace_id)
    work_items = SQLiteWorkItemRepository(db_manager)
    embedding_repair_queue = SQLiteEmbeddingRepairQueue(db_manager)
    relational_search = RelationalMemorySearchService(
        repository,
        spec.config,
        embedder=embedder,
        vector_store=vector_store,
        db_manager=db_manager,
        task_queue=task_queue,
        work_items=work_items,
        embedding_repair_queue=embedding_repair_queue,
        background_repair_wait_seconds=5.0 if enable_background_repair_queue else 0.0,
    )
    relational_search.run_startup_health_check()
    return StorageBackendResources(
        backend="sqlite",
        db_manager=db_manager,
        journal=journal,
        repository=repository,
        relational_search=relational_search,
        task_queue=task_queue,
        provider_policy_events=ProviderPolicyEventRepository(db_manager, workspace_id=spec.workspace_id),
        task_execution_attempts=task_execution_attempts,
        work_items=work_items,
        embedding_repair_queue=embedding_repair_queue,
        vector_store=vector_store,
    )