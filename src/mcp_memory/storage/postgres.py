from __future__ import annotations

import logging
from typing import Any

import psycopg

from mcp_memory.config import PostgresStorageConfig
from mcp_memory.storage.postgres_embedding_integrity_event_store import PostgresEmbeddingIntegrityEventRepository
from mcp_memory.storage.postgres_embedding_repair_store import PostgresEmbeddingRepairQueue
from mcp_memory.storage.postgres_journal import PostgresSystem1Journal
from mcp_memory.storage.postgres_provider_policy_event_store import PostgresProviderPolicyEventRepository
from mcp_memory.storage.postgres_provider_usage_store import PostgresProviderUsageRepository
from mcp_memory.storage.postgres_task_queue import PostgresTaskQueue
from mcp_memory.storage.postgres_vector_store import PostgresVectorStore
from mcp_memory.storage.postgres_work_item_store import PostgresWorkItemRepository
from mcp_memory.relational.search import RelationalMemorySearchService
from mcp_memory.application.memory_embedding_maintenance import MemoryEmbeddingMaintenance
from mcp_memory.storage.bootstrap import StorageBootstrapState
from mcp_memory.storage.postgres_connection import (
    PostgresConnectionManager,
)
from mcp_memory.storage.postgres_repository import PostgresRelationalMemoryRepository
from mcp_memory.storage.postgres_runtime_log_store import PostgresRuntimeLogRepository
from mcp_memory.storage.postgres_task_execution_store import PostgresTaskExecutionAttemptRepository
from mcp_memory.storage.postgres_migrations import apply_postgres_migrations
from mcp_memory.storage.postgres_curation_store import PostgresCurationStore
from mcp_memory.storage.postgres_curation_action_store import PostgresCurationActionStore
from mcp_memory.storage.postgres_mutation_history_store import PostgresMutationHistoryStore
from mcp_memory.storage.shared_read_cache import SharedReadCache
from mcp_memory.storage.session import CursorLike
from mcp_memory.storage.types import PostgresBackendNotImplementedError, StorageBackendResources, StorageBootstrapSpec


logger = logging.getLogger(__name__)


class UnsupportedPostgresRuntimeComponent:
    def __init__(self, feature_name: str) -> None:
        self._feature_name = feature_name

    def __getattr__(self, name: str) -> Any:
        raise PostgresBackendNotImplementedError(
            f"storage backend 'postgres' does not yet support {self._feature_name}; attempted to access {name}"
        )


def inspect_postgres_bootstrap_state(config: PostgresStorageConfig) -> StorageBootstrapState:
    state: StorageBootstrapState | None = None
    with PostgresConnectionManager(config) as manager:
        with manager.open_connection() as connection:
            with connection.cursor() as cursor:
                state = _inspect_postgres_bootstrap_state_on_cursor(cursor)
    assert state is not None
    return state


def ensure_postgres_schema(
    config: PostgresStorageConfig,
    *,
    connect_timeout_seconds: float | None = None,
) -> StorageBootstrapState:
    state: StorageBootstrapState | None = None
    with PostgresConnectionManager(config) as manager:
        with manager.open_connection(timeout=connect_timeout_seconds) as connection:
            with connection.cursor() as cursor:
                current_state = _inspect_postgres_bootstrap_state_on_cursor(cursor)
                apply_postgres_migrations(
                    cursor,
                    current_version=current_state.schema_version if current_state.schema_metadata_present else None,
                )
                connection.commit()
                state = _inspect_postgres_bootstrap_state_on_cursor(cursor)
    assert state is not None
    return state


def _inspect_postgres_bootstrap_state_on_cursor(cursor: CursorLike) -> StorageBootstrapState:
    cursor.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = current_schema()
              AND table_name = %s
        )
        """,
        ("schema_metadata",),
    )
    table_row = cursor.fetchone()
    schema_metadata_present = bool(table_row[0]) if table_row is not None else False
    schema_version: int | None = None
    if schema_metadata_present:
        cursor.execute(
            "SELECT value FROM schema_metadata WHERE key = %s",
            ("schema_version",),
        )
        version_row = cursor.fetchone()
        if version_row is not None and version_row[0] is not None:
            raw_version = version_row[0]
            if isinstance(raw_version, str | int):
                schema_version = int(raw_version)
            else:
                raise TypeError("schema_metadata schema_version must be stored as text or integer")
    return StorageBootstrapState(
        backend="postgres",
        schema_metadata_present=schema_metadata_present,
        schema_version=schema_version,
    )


_STARTUP_SCHEMA_BOOTSTRAP_CONNECT_TIMEOUT_SECONDS = 5.0


def _ensure_postgres_schema_tolerating_outage(
    config: PostgresStorageConfig,
    *,
    tolerate_outage: bool,
) -> StorageBootstrapState | None:
    try:
        # Daemon startup only needs this connection attempt to prove Postgres
        # is reachable before proceeding; the pool's 30s default getconn
        # timeout meant every startup during an outage blocked for 30s before
        # failing (or degrading), making retry/respawn cycles far more
        # expensive than they need to be.
        return ensure_postgres_schema(
            config,
            connect_timeout_seconds=_STARTUP_SCHEMA_BOOTSTRAP_CONNECT_TIMEOUT_SECONDS,
        )
    except psycopg.OperationalError:
        if not tolerate_outage:
            raise
        logger.warning(
            "Postgres unreachable during schema bootstrap; starting daemon in degraded mode. "
            "Writes will queue to the local writeback cache until connectivity returns.",
            exc_info=True,
        )
        return None


def build_postgres_runtime_components(
    spec: StorageBootstrapSpec,
    *,
    embedder: Any,
    enable_background_repair_queue: bool,
) -> StorageBackendResources:
    cache_config = spec.config.storage.cache
    tolerate_outage = cache_config.enabled and cache_config.mode == "writeback"
    bootstrap_state = _ensure_postgres_schema_tolerating_outage(
        spec.config.storage.postgres,
        tolerate_outage=tolerate_outage,
    )
    connection_manager = PostgresConnectionManager(spec.config.storage.postgres)
    repository = PostgresRelationalMemoryRepository(connection_manager)
    journal = PostgresSystem1Journal(connection_manager)
    task_queue = PostgresTaskQueue(connection_manager)
    provider_policy_events = PostgresProviderPolicyEventRepository(connection_manager, workspace_id=spec.workspace_id)
    embedding_integrity_events = PostgresEmbeddingIntegrityEventRepository(connection_manager, workspace_id=None)
    provider_usage = PostgresProviderUsageRepository(connection_manager, workspace_id=spec.workspace_id)
    runtime_logs = PostgresRuntimeLogRepository(
        connection_manager,
        workspace_id=spec.workspace_id,
        config=spec.config.logging,
    )
    task_execution_attempts = PostgresTaskExecutionAttemptRepository(connection_manager, workspace_id=spec.workspace_id)
    work_items = PostgresWorkItemRepository(connection_manager)
    embedding_repair_queue = PostgresEmbeddingRepairQueue(connection_manager)
    mutation_history = PostgresMutationHistoryStore(connection_manager)
    curation = PostgresCurationStore(connection_manager)
    curation_action_store = PostgresCurationActionStore(connection_manager)
    vector_store = PostgresVectorStore(
        connection_manager,
        event_repository=embedding_integrity_events,
    )
    embedding_maintenance = MemoryEmbeddingMaintenance(
        repository,
        spec.config,
        embedder=embedder,
        vector_store=vector_store,
        db_manager=connection_manager,
        task_queue=task_queue if enable_background_repair_queue else None,
        work_items=work_items,
        embedding_repair_queue=embedding_repair_queue,
        embedding_cache_path=spec.memory_path / "cache" / "embedding-cache.sqlite3",
    )
    relational_search = RelationalMemorySearchService(
        repository,
        spec.config,
        embedder=embedder,
        vector_store=vector_store,
        task_queue=task_queue if enable_background_repair_queue else None,
        work_items=work_items,
        embedding_repair_queue=embedding_repair_queue,
        embedding_maintenance=embedding_maintenance,
    )
    read_cache = None
    if spec.config.storage.cache.enabled and spec.config.storage.cache.mode in {"readonly", "writeback"}:
        read_cache = SharedReadCache(spec.memory_path / "cache" / "shared_read_cache.sqlite3")
    if bootstrap_state is not None and (
        not bootstrap_state.schema_metadata_present or bootstrap_state.schema_version is None
    ):
        raise PostgresBackendNotImplementedError(
            "storage backend 'postgres' schema bootstrap did not complete successfully"
        )
    return StorageBackendResources(
        backend="postgres",
        db_manager=connection_manager,
        journal=journal,
        repository=repository,
        relational_search=relational_search,
        embedding_maintenance=embedding_maintenance,
        read_cache=read_cache,
        task_queue=task_queue,
        provider_usage=provider_usage,
        runtime_logs=runtime_logs,
        provider_policy_events=provider_policy_events,
        embedding_integrity_events=embedding_integrity_events,
        task_execution_attempts=task_execution_attempts,
        work_items=work_items,
        embedding_repair_queue=embedding_repair_queue,
        vector_store=vector_store,
        mutation_history=mutation_history,
        curation=curation,
        curation_action_store=curation_action_store,
    )
