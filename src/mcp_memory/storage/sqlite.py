from __future__ import annotations

from typing import Any

from mcp_memory.core.journal import System1Journal
from mcp_memory.embedding_integrity_event_store import EmbeddingIntegrityEventRepository
from mcp_memory.core.tasks import SQLiteTaskQueue
from mcp_memory.embedding_repair_store import SQLiteEmbeddingRepairQueue
from mcp_memory.embeddings import SQLiteVectorStore
from mcp_memory.provider_usage_store import ProviderUsageRepository
from mcp_memory.provider_policy_event_store import ProviderPolicyEventRepository
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.relational.search import RelationalMemorySearchService
from mcp_memory.runtime_log_store import RuntimeLogRepository
from mcp_memory.mutation_history_store import SQLiteMutationHistoryStore
from mcp_memory.storage.types import StorageBackendResources, StorageBootstrapSpec
from mcp_memory.task_execution_store import TaskExecutionAttemptRepository
from mcp_memory.utils.db import DatabaseManager
from mcp_memory.work_item_store import SQLiteWorkItemRepository


def build_sqlite_runtime_components(
    spec: StorageBootstrapSpec,
    *,
    embedder: Any,
    enable_background_repair_queue: bool,
) -> StorageBackendResources:
    db_manager = DatabaseManager(spec.memory_path / "indices" / "memory.db")
    journal = System1Journal(db_manager)
    repository = RelationalMemoryRepository(db_manager)
    vector_store = SQLiteVectorStore(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    provider_usage = ProviderUsageRepository(db_manager, workspace_id=spec.workspace_id)
    runtime_logs = RuntimeLogRepository(
        db_manager,
        workspace_id=spec.workspace_id,
        config=spec.config.logging,
    )
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
        read_cache=None,
        task_queue=task_queue,
        provider_usage=provider_usage,
        runtime_logs=runtime_logs,
        provider_policy_events=ProviderPolicyEventRepository(db_manager, workspace_id=spec.workspace_id),
        embedding_integrity_events=EmbeddingIntegrityEventRepository(db_manager, workspace_id=None),
        task_execution_attempts=task_execution_attempts,
        work_items=work_items,
        embedding_repair_queue=embedding_repair_queue,
        vector_store=vector_store,
        mutation_history=SQLiteMutationHistoryStore(db_manager),
    )
