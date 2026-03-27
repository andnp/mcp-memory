from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from mcp_memory.storage.bootstrap import StorageBootstrapState
from mcp_memory.storage.types import PostgresBackendNotImplementedError, RuntimeSpecLike, StorageBackendResources


class PostgresDriverMissingError(RuntimeError):
    pass


def load_postgres_driver_modules() -> tuple[ModuleType, ModuleType]:
    try:
        return (
            importlib.import_module("psycopg"),
            importlib.import_module("psycopg_pool"),
        )
    except ModuleNotFoundError as exc:
        raise PostgresDriverMissingError(
            "Postgres backend requires psycopg and psycopg_pool; install project dependencies before enabling storage.backend='postgres'"
        ) from exc


def inspect_postgres_bootstrap_state(dsn: str) -> StorageBootstrapState:
    psycopg, _ = load_postgres_driver_modules()
    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
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
                    schema_version = int(version_row[0])
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
    bootstrap_state = inspect_postgres_bootstrap_state(spec.config.storage.postgres.dsn)
    raise PostgresBackendNotImplementedError(
        "storage backend 'postgres' bootstrap inspection completed "
        f"(schema_metadata_present={bootstrap_state.schema_metadata_present}, schema_version={bootstrap_state.schema_version}); "
        "repository wiring is still pending"
    )