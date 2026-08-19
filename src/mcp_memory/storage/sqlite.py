from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

from searchkernel.runtime import QueryEmbeddingCache

from mcp_memory.application.memory_embedding_maintenance import (
    MemoryEmbeddingMaintenance,
)
from mcp_memory.core.journal import System1Journal
from mcp_memory.core.ports.embedding_maintenance import EmbeddingDatabaseHealthError
from mcp_memory.curation_action_store import SQLiteCurationActionStore
from mcp_memory.curation_store import SQLiteCurationStore
from mcp_memory.embedding_integrity_event_store import EmbeddingIntegrityEventRepository
from mcp_memory.embedding_repair_store import SQLiteEmbeddingRepairQueue
from mcp_memory.embeddings import SQLiteVectorStore
from mcp_memory.mutation_history_store import SQLiteMutationHistoryStore
from mcp_memory.provider_policy_event_store import ProviderPolicyEventRepository
from mcp_memory.provider_usage_store import ProviderUsageRepository
from mcp_memory.relational.repository import SQLiteRelationalMemoryRepository
from mcp_memory.relational.search import RelationalMemorySearchService
from mcp_memory.runtime_log_store import RuntimeLogRepository
from mcp_memory.storage.direct_mutation_evidence_store import SQLiteDirectMutationEvidenceStore
from mcp_memory.storage.ingress_evidence_store import (
    SQLiteIngressActionReceiptStore,
    SQLiteIngressBatchEvidenceStore,
    SQLiteSourceCoverageStore,
)
from mcp_memory.storage.ingress_mutation_transaction import SQLiteIngressMutationStore
from mcp_memory.storage.ingress_quality_store import SQLiteIngressQualityEvidenceStore
from mcp_memory.storage.sqlite_maintenance_housekeeping import SQLiteMaintenanceHousekeeping
from mcp_memory.storage.sqlite_task_queue import SQLiteTaskQueue
from mcp_memory.storage.sqlite_work_item_store import SQLiteWorkItemRepository
from mcp_memory.storage.types import StorageBackendResources, StorageBootstrapSpec
from mcp_memory.task_execution_store import TaskExecutionAttemptRepository
from mcp_memory.utils.db import DatabaseManager


class _SQLiteEmbeddingDatabaseHealth:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db_manager = db_manager

    def run_integrity_check(self, operation: Callable[[], None]) -> None:
        try:
            quick_check = self._db_manager.get_connection().execute("PRAGMA quick_check").fetchone()
            if quick_check is not None and str(quick_check[0]).lower() != "ok":
                raise EmbeddingDatabaseHealthError(f"quick_check_failed:{quick_check[0]}")
            operation()
        except sqlite3.Error as exc:
            raise EmbeddingDatabaseHealthError(str(exc)) from exc

    def retry_after_reopen(self, operation: Callable[[], None]) -> bool:
        try:
            self._db_manager.close()
            self.run_integrity_check(operation)
        except (EmbeddingDatabaseHealthError, OSError, ValueError):
            return False
        return True


def build_sqlite_runtime_components(
    spec: StorageBootstrapSpec,
    *,
    embedder: Any,
    enable_background_repair_queue: bool,
) -> StorageBackendResources:
    db_manager = DatabaseManager(spec.memory_path / "indices" / "memory.db")
    housekeeping = SQLiteMaintenanceHousekeeping(db_manager)
    journal = System1Journal(db_manager)
    repository = SQLiteRelationalMemoryRepository(db_manager)
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
    database_health = _SQLiteEmbeddingDatabaseHealth(db_manager)
    query_embedding_cache = QueryEmbeddingCache()
    embedding_maintenance = MemoryEmbeddingMaintenance(
        repository,
        spec.config,
        embedder=embedder,
        vector_store=vector_store,
        database_health=database_health,
        task_queue=task_queue,
        work_items=work_items,
        embedding_repair_queue=embedding_repair_queue,
        background_repair_wait_seconds=5.0 if enable_background_repair_queue else 0.0,
        embedding_cache_path=spec.memory_path / "cache" / "embedding-cache.sqlite3",
    )
    relational_search = RelationalMemorySearchService(
        repository,
        spec.config,
        embedder=embedder,
        vector_store=vector_store,
        database_health=database_health,
        task_queue=task_queue,
        work_items=work_items,
        embedding_repair_queue=embedding_repair_queue,
        embedding_maintenance=embedding_maintenance,
        query_embedding_cache=query_embedding_cache,
        background_repair_wait_seconds=5.0 if enable_background_repair_queue else 0.0,
    )
    embedding_maintenance.run_startup_health_check()
    return StorageBackendResources(
        backend="sqlite",
        db_manager=db_manager,
        journal=journal,
        repository=repository,
        relational_search=relational_search,
        embedding_maintenance=embedding_maintenance,
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
        curation=SQLiteCurationStore(db_manager),
        curation_action_store=SQLiteCurationActionStore(db_manager),
        direct_mutation_evidence=SQLiteDirectMutationEvidenceStore(db_manager),
        ingress_batch_evidence=SQLiteIngressBatchEvidenceStore(db_manager),
        ingress_action_receipts=SQLiteIngressActionReceiptStore(db_manager),
        source_coverage=SQLiteSourceCoverageStore(db_manager),
        ingress_quality_evidence=SQLiteIngressQualityEvidenceStore(db_manager),
        ingress_mutation_transaction=SQLiteIngressMutationStore(db_manager),
        search_health=relational_search,
        startup_health=relational_search,
        read_cache_validation=relational_search,
        memory_id_resolution=relational_search,
        housekeeping=housekeeping,
    )
