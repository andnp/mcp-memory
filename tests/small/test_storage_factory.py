from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mcp_memory.config import Config, PostgresStorageConfig, StorageConfig
from mcp_memory.mcp.runtime import RuntimeSpec
from mcp_memory.storage.factory import PostgresBackendNotImplementedError, build_storage_runtime_components
from mcp_memory.storage.bootstrap import StorageBootstrapState
from mcp_memory.storage.postgres import PostgresDriverMissingError, inspect_postgres_bootstrap_state, load_postgres_driver_modules


pytestmark = pytest.mark.small


def test_storage_factory_rejects_postgres_until_backend_is_implemented(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    def fake_inspect_postgres_bootstrap_state(dsn: str) -> StorageBootstrapState:
        assert dsn == "postgresql://memory@example.invalid/mcp_memory"
        return StorageBootstrapState(
            backend="postgres",
            schema_metadata_present=False,
            schema_version=None,
        )

    monkeypatch.setattr(
        "mcp_memory.storage.postgres.inspect_postgres_bootstrap_state",
        fake_inspect_postgres_bootstrap_state,
    )

    with pytest.raises(PostgresBackendNotImplementedError, match="bootstrap inspection completed"):
        build_storage_runtime_components(spec, embedder=None, enable_background_repair_queue=False)


def test_load_postgres_driver_modules_raises_helpful_error_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_module(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("mcp_memory.storage.postgres.importlib.import_module", fake_import_module)

    with pytest.raises(PostgresDriverMissingError, match="requires psycopg and psycopg_pool"):
        load_postgres_driver_modules()


def test_inspect_postgres_bootstrap_state_reads_schema_version(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCursor:
        def __init__(self) -> None:
            self._result: tuple[object, ...] | None = None

        def __enter__(self) -> FakeCursor:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def execute(self, query: str, params: tuple[object, ...]) -> None:
            if "information_schema.tables" in query:
                self._result = (True,)
                return
            assert params == ("schema_version",)
            self._result = ("19",)

        def fetchone(self) -> tuple[object, ...] | None:
            return self._result

    class FakeConnection:
        def __enter__(self) -> FakeConnection:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def cursor(self) -> FakeCursor:
            return FakeCursor()

    class FakePsycopgModule:
        @staticmethod
        def connect(dsn: str) -> FakeConnection:
            assert dsn == "postgresql://memory@example.invalid/mcp_memory"
            return FakeConnection()

    def fake_import_module(name: str):
        if name == "psycopg":
            return FakePsycopgModule()
        if name == "psycopg_pool":
            return SimpleNamespace()
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("mcp_memory.storage.postgres.importlib.import_module", fake_import_module)

    state = inspect_postgres_bootstrap_state("postgresql://memory@example.invalid/mcp_memory")

    assert state == StorageBootstrapState(
        backend="postgres",
        schema_metadata_present=True,
        schema_version=19,
    )