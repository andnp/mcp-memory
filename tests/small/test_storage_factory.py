from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import pytest

from mcp_memory.config import Config, PostgresStorageConfig, StorageConfig
from mcp_memory.mcp.runtime import RuntimeSpec, _build_provider_registry
from mcp_memory.relational.search import RelationalMemorySearchService
from mcp_memory.storage.factory import build_storage_runtime_components
from mcp_memory.storage.bootstrap import StorageBootstrapState
from mcp_memory.storage.postgres import ensure_postgres_schema, inspect_postgres_bootstrap_state
from mcp_memory.storage.types import StorageBackendResources


pytestmark = pytest.mark.small


def test_storage_factory_builds_postgres_repository_resources_with_explicit_unsupported_edges(
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

    def fake_ensure_postgres_schema(config: PostgresStorageConfig) -> StorageBootstrapState:
        assert config.dsn == "postgresql://memory@example.invalid/mcp_memory"
        return StorageBootstrapState(
            backend="postgres",
            schema_metadata_present=True,
            schema_version=1,
        )

    monkeypatch.setattr(
        "mcp_memory.storage.postgres.ensure_postgres_schema",
        fake_ensure_postgres_schema,
    )

    storage = build_storage_runtime_components(spec, embedder=None, enable_background_repair_queue=False)

    assert storage.backend == "postgres"
    assert storage.repository is not None
    assert isinstance(storage.relational_search, RelationalMemorySearchService)
    assert storage.relational_search.get_health().available is False
    with pytest.raises(NotImplementedError, match="task queue"):
        storage.task_queue.enqueue_unique("test-task", data={}, workspace_id=None)


def test_inspect_postgres_bootstrap_state_reads_schema_version(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCursor:
        def __init__(self) -> None:
            self.schema_metadata_present = True
            self.schema_version = "19"
            self._result: tuple[object, ...] | None = None

        def __enter__(self) -> FakeCursor:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def execute(self, query: str, params: tuple[object, ...]) -> None:
            if "information_schema.tables" in query:
                self._result = (self.schema_metadata_present,)
                return
            assert params == ("schema_version",)
            self._result = None if self.schema_version is None else (self.schema_version,)

        def fetchone(self) -> tuple[object, ...] | None:
            return self._result

    class FakeConnection:
        cursor_instance = FakeCursor()

        def __enter__(self) -> FakeConnection:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def cursor(self) -> FakeCursor:
            return self.cursor_instance

    class FakeConnectionPool:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def getconn(self) -> FakeConnection:
            return FakeConnection()

        def putconn(self, connection: FakeConnection) -> None:
            del connection

        def close(self) -> None:
            return None

    class FakePsycopgModule:
        @staticmethod
        def connect(dsn: str) -> FakeConnection:
            assert dsn == "postgresql://memory@example.invalid/mcp_memory"
            return FakeConnection()

    def fake_import_module(name: str):
        if name == "psycopg":
            return FakePsycopgModule()
        if name == "psycopg_pool":
            return SimpleNamespace(ConnectionPool=FakeConnectionPool)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("mcp_memory.storage.postgres_connection.importlib.import_module", fake_import_module)

    state = inspect_postgres_bootstrap_state(
        PostgresStorageConfig(dsn="postgresql://memory@example.invalid/mcp_memory")
    )

    assert state == StorageBootstrapState(
        backend="postgres",
        schema_metadata_present=True,
        schema_version=19,
    )


def test_ensure_postgres_schema_bootstraps_missing_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCursor:
        def __init__(self) -> None:
            self.schema_metadata_present = False
            self.schema_version: str | None = None
            self._result: tuple[object, ...] | None = None

        def __enter__(self) -> FakeCursor:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
            normalized = " ".join(query.split())
            if "information_schema.tables" in normalized:
                self._result = (self.schema_metadata_present,)
                return
            if normalized.startswith("SELECT value FROM schema_metadata"):
                self._result = None if self.schema_version is None else (self.schema_version,)
                return
            if normalized.startswith("INSERT INTO schema_metadata"):
                assert params == ("schema_version", "1")
                self.schema_metadata_present = True
                self.schema_version = "1"
                self._result = None
                return
            self._result = None

        def fetchone(self) -> tuple[object, ...] | None:
            return self._result

    class FakeConnection:
        def __init__(self) -> None:
            self.cursor_instance = FakeCursor()

        def __enter__(self) -> FakeConnection:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def cursor(self) -> FakeCursor:
            return self.cursor_instance

        def commit(self) -> None:
            return None

        def rollback(self) -> None:
            return None

    class FakeConnectionPool:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def getconn(self) -> FakeConnection:
            return FakeConnection()

        def putconn(self, connection: FakeConnection) -> None:
            del connection

        def close(self) -> None:
            return None

    class FakePsycopgModule:
        @staticmethod
        def connect(dsn: str) -> FakeConnection:
            assert dsn == "postgresql://memory@example.invalid/mcp_memory"
            return FakeConnection()

    def fake_import_module(name: str):
        if name == "psycopg":
            return FakePsycopgModule()
        if name == "psycopg_pool":
            return SimpleNamespace(ConnectionPool=FakeConnectionPool)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("mcp_memory.storage.postgres_connection.importlib.import_module", fake_import_module)

    state = ensure_postgres_schema(
        PostgresStorageConfig(dsn="postgresql://memory@example.invalid/mcp_memory")
    )

    assert state == StorageBootstrapState(
        backend="postgres",
        schema_metadata_present=True,
        schema_version=1,
    )


def test_build_provider_registry_skips_non_sqlite_backends(tmp_path: Path) -> None:
    spec = RuntimeSpec(
        memory_path=tmp_path / "memories",
        config=Config(
            storage=StorageConfig(
                backend="postgres",
                postgres=PostgresStorageConfig(dsn="postgresql://memory@example.invalid/mcp_memory"),
            )
        ),
        workspace_id="workspace-123",
        workspace_root=tmp_path,
        lock_path=tmp_path / "daemon.lock",
    )
    storage = StorageBackendResources(
        backend="postgres",
        db_manager=object(),
        journal=object(),
        repository=object(),
        relational_search=object(),
        task_queue=object(),
        provider_policy_events=object(),
        task_execution_attempts=object(),
        work_items=object(),
        embedding_repair_queue=object(),
        vector_store=object(),
    )

    registry = _build_provider_registry(spec=spec, storage=storage)

    assert registry == {}