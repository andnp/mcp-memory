from __future__ import annotations

from dataclasses import replace
from typing import Any

from mcp_memory.config import PostgresStorageConfig
from mcp_memory.relational.search import SearchHealthStatus
from mcp_memory.storage.bootstrap import StorageBootstrapState
from mcp_memory.storage.postgres_connection import (
    PostgresConnectionManager,
)
from mcp_memory.storage.postgres_repository import PostgresRelationalMemoryRepository
from mcp_memory.storage.postgres_migrations import apply_postgres_migrations
from mcp_memory.storage.session import CursorLike
from mcp_memory.storage.types import PostgresBackendNotImplementedError, RuntimeSpecLike, StorageBackendResources


class UnsupportedPostgresRuntimeComponent:
    def __init__(self, feature_name: str) -> None:
        self._feature_name = feature_name

    def __getattr__(self, name: str) -> Any:
        raise PostgresBackendNotImplementedError(
            f"storage backend 'postgres' does not yet support {self._feature_name}; attempted to access {name}"
        )


class UnsupportedPostgresSearchService(UnsupportedPostgresRuntimeComponent):
    def __init__(self) -> None:
        super().__init__("semantic search")
        self._health = SearchHealthStatus(
            semantic_enabled=False,
            available=False,
            last_error="storage backend 'postgres' does not yet support semantic search",
        )

    def get_health(self) -> SearchHealthStatus:
        return replace(self._health)

    def run_startup_health_check(self) -> SearchHealthStatus:
        return self.get_health()

    def search_memories(self, *args: object, **kwargs: object) -> list[object]:
        del args
        del kwargs
        raise PostgresBackendNotImplementedError(
            "storage backend 'postgres' does not yet support semantic search"
        )

    def read_memory(self, *args: object, **kwargs: object) -> object:
        del args
        del kwargs
        raise PostgresBackendNotImplementedError(
            "storage backend 'postgres' does not yet support search read flows"
        )

    def rebuild_semantic_index(self, *args: object, **kwargs: object) -> dict[str, object]:
        del args
        del kwargs
        raise PostgresBackendNotImplementedError(
            "storage backend 'postgres' does not yet support semantic index rebuilds"
        )


def inspect_postgres_bootstrap_state(config: PostgresStorageConfig) -> StorageBootstrapState:
    state: StorageBootstrapState | None = None
    with PostgresConnectionManager(config) as manager:
        with manager.open_connection() as connection:
            with connection.cursor() as cursor:
                state = _inspect_postgres_bootstrap_state_on_cursor(cursor)
    assert state is not None
    return state


def ensure_postgres_schema(config: PostgresStorageConfig) -> StorageBootstrapState:
    state: StorageBootstrapState | None = None
    with PostgresConnectionManager(config) as manager:
        with manager.open_connection() as connection:
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


def build_postgres_runtime_components(
    spec: RuntimeSpecLike,
    *,
    embedder: Any,
    enable_background_repair_queue: bool,
) -> StorageBackendResources:
    del embedder
    del enable_background_repair_queue
    bootstrap_state = ensure_postgres_schema(spec.config.storage.postgres)
    connection_manager = PostgresConnectionManager(spec.config.storage.postgres)
    repository = PostgresRelationalMemoryRepository(connection_manager)
    unsupported_journal = UnsupportedPostgresRuntimeComponent("system1 journal")
    unsupported_task_queue = UnsupportedPostgresRuntimeComponent("task queue")
    unsupported_provider_policy_events = UnsupportedPostgresRuntimeComponent("provider policy events")
    unsupported_task_execution_attempts = UnsupportedPostgresRuntimeComponent("task execution attempts")
    unsupported_work_items = UnsupportedPostgresRuntimeComponent("work items")
    unsupported_embedding_repairs = UnsupportedPostgresRuntimeComponent("embedding repair queue")
    unsupported_vector_store = UnsupportedPostgresRuntimeComponent("vector store")
    relational_search = UnsupportedPostgresSearchService()
    if not bootstrap_state.schema_metadata_present or bootstrap_state.schema_version is None:
        raise PostgresBackendNotImplementedError(
            "storage backend 'postgres' schema bootstrap did not complete successfully"
        )
    return StorageBackendResources(
        backend="postgres",
        db_manager=connection_manager,
        journal=unsupported_journal,
        repository=repository,
        relational_search=relational_search,
        task_queue=unsupported_task_queue,
        provider_policy_events=unsupported_provider_policy_events,
        task_execution_attempts=unsupported_task_execution_attempts,
        work_items=unsupported_work_items,
        embedding_repair_queue=unsupported_embedding_repairs,
        vector_store=unsupported_vector_store,
    )