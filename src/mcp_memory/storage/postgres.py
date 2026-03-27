from __future__ import annotations

from typing import Any

from mcp_memory.storage.types import PostgresBackendNotImplementedError, RuntimeSpecLike, StorageBackendResources


def build_postgres_runtime_components(
    spec: RuntimeSpecLike,
    *,
    embedder: Any,
    enable_background_repair_queue: bool,
) -> StorageBackendResources:
    del embedder
    del enable_background_repair_queue
    del spec
    raise PostgresBackendNotImplementedError(
        "storage backend 'postgres' is not implemented yet; bootstrap, migrations, and repository wiring are still pending"
    )