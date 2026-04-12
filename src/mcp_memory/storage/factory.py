from __future__ import annotations

from typing import Any

from mcp_memory.storage.postgres import build_postgres_runtime_components
from mcp_memory.storage.sqlite import build_sqlite_runtime_components
from mcp_memory.storage.types import PostgresBackendNotImplementedError, StorageBackendResources, StorageBootstrapSpec


__all__ = [
    "PostgresBackendNotImplementedError",
    "StorageBackendResources",
    "build_storage_runtime_components",
]


def build_storage_runtime_components(
    spec: StorageBootstrapSpec,
    *,
    embedder: Any,
    enable_background_repair_queue: bool,
) -> StorageBackendResources:
    if spec.config.storage.backend == "postgres":
        return build_postgres_runtime_components(
            spec,
            embedder=embedder,
            enable_background_repair_queue=enable_background_repair_queue,
        )

    return build_sqlite_runtime_components(
        spec,
        embedder=embedder,
        enable_background_repair_queue=enable_background_repair_queue,
    )