from __future__ import annotations

from typing import Any

from mcp_memory.config import PostgresStorageConfig
from mcp_memory.storage.bootstrap import StorageBootstrapState
from mcp_memory.storage.postgres_connection import (
    PostgresConnectionManager,
)
from mcp_memory.storage.postgres_migrations import apply_postgres_migrations
from mcp_memory.storage.session import CursorLike
from mcp_memory.storage.types import PostgresBackendNotImplementedError, RuntimeSpecLike, StorageBackendResources


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
    raise PostgresBackendNotImplementedError(
        "storage backend 'postgres' schema bootstrap completed "
        f"(schema_metadata_present={bootstrap_state.schema_metadata_present}, schema_version={bootstrap_state.schema_version}); "
        "repository wiring is still pending"
    )