from pathlib import Path

import pytest

from mcp_memory.config import Config, PostgresStorageConfig, StorageConfig
from mcp_memory.mcp.runtime import RuntimeSpec
from mcp_memory.storage.factory import PostgresBackendNotImplementedError, build_storage_runtime_components


pytestmark = pytest.mark.small


def test_storage_factory_rejects_postgres_until_backend_is_implemented(tmp_path: Path) -> None:
    config = Config(
        storage=StorageConfig(
            backend="postgres",
            postgres=PostgresStorageConfig(dsn="postgresql://memory@example.invalid/mcp_memory"),
        )
    )
    spec = RuntimeSpec(
        memory_path=tmp_path / "memories",
        config=config,
        workspace_id="workspace-123",
        workspace_root=tmp_path,
        lock_path=tmp_path / "daemon.lock",
    )

    with pytest.raises(PostgresBackendNotImplementedError, match="not implemented yet"):
        build_storage_runtime_components(spec, embedder=None, enable_background_repair_queue=False)