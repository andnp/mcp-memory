from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import pytest

from mcp_memory.config import AIConfig, Config, PostgresStorageConfig, ProviderRoutingConfig, StorageConfig
from mcp_memory.core.providers.instrumented import InstrumentedAIProvider
from mcp_memory.mcp.runtime import RuntimeSpec, _build_provider_registry
from mcp_memory.relational.search import RelationalMemorySearchService
from mcp_memory.storage.factory import build_storage_runtime_components
from mcp_memory.storage.bootstrap import StorageBootstrapState
from mcp_memory.storage.postgres import ensure_postgres_schema, inspect_postgres_bootstrap_state
from mcp_memory.storage.postgres_migrations import POSTGRES_SCHEMA_VERSION
from mcp_memory.storage.shared_read_cache import SharedReadCache
from mcp_memory.storage.postgres_task_execution_store import PostgresTaskExecutionAttemptRepository
from mcp_memory.storage.postgres_task_queue import PostgresTaskQueue
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
            schema_version=5,
        )

    monkeypatch.setattr(
        "mcp_memory.storage.postgres.ensure_postgres_schema",
        fake_ensure_postgres_schema,
    )

    storage = build_storage_runtime_components(spec, embedder=None, enable_background_repair_queue=False)

    assert storage.backend == "postgres"
    assert storage.repository is not None
    assert isinstance(storage.relational_search, RelationalMemorySearchService)
    assert isinstance(storage.task_queue, PostgresTaskQueue)
    assert isinstance(storage.task_execution_attempts, PostgresTaskExecutionAttemptRepository)
    assert storage.relational_search.get_health().available is False
    assert storage.read_cache is None


def test_storage_factory_builds_postgres_shared_read_cache_only_for_enabled_readonly_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(
        storage=StorageConfig(
            backend="postgres",
            postgres=PostgresStorageConfig(dsn="postgresql://memory@example.invalid/mcp_memory"),
        )
    )
    config.storage.cache.enabled = True
    config.storage.cache.mode = "readonly"
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
            schema_version=5,
        )

    monkeypatch.setattr(
        "mcp_memory.storage.postgres.ensure_postgres_schema",
        fake_ensure_postgres_schema,
    )

    storage = build_storage_runtime_components(spec, embedder=None, enable_background_repair_queue=False)

    assert isinstance(storage.read_cache, SharedReadCache)


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
            self.vector_extension_installed = False
            self.embedding_vector_column_present = False
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
                assert params is not None
                assert params[0] == "schema_version"
                assert int(str(params[1])) in range(1, POSTGRES_SCHEMA_VERSION + 1)
                self.schema_metadata_present = True
                self.schema_version = str(params[1])
                self._result = None
                return
            if normalized.startswith("DO $$"):
                if self.vector_extension_installed:
                    self.embedding_vector_column_present = True
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
        schema_version=POSTGRES_SCHEMA_VERSION,
    )


def test_ensure_postgres_schema_adds_optional_vector_column_when_extension_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder: dict[str, FakeCursor] = {}

    class FakeCursor:
        def __init__(self) -> None:
            self.schema_metadata_present = True
            self.schema_version = "8"
            self.embedding_vector_column_present = False
            self.optional_vector_migration_runs = 0
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
                assert params is not None
                self.schema_version = str(params[1])
                self._result = None
                return
            if normalized.startswith("DO $$"):
                self.optional_vector_migration_runs += 1
                self.embedding_vector_column_present = True
                self._result = None
                return
            self._result = None

        def fetchone(self) -> tuple[object, ...] | None:
            return self._result

    class FakeConnection:
        def __init__(self, cursor: FakeCursor) -> None:
            self.cursor_instance = cursor

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
            self.cursor = FakeCursor()
            holder["cursor"] = self.cursor

        def getconn(self) -> FakeConnection:
            return FakeConnection(self.cursor)

        def putconn(self, connection: FakeConnection) -> None:
            del connection

        def close(self) -> None:
            return None

    class FakePsycopgModule:
        @staticmethod
        def connect(dsn: str) -> FakeConnection:
            assert dsn == "postgresql://memory@example.invalid/mcp_memory"
            return FakeConnection(FakeCursor())

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
        schema_version=POSTGRES_SCHEMA_VERSION,
    )
    assert holder["cursor"].optional_vector_migration_runs == 1
    assert holder["cursor"].embedding_vector_column_present is True


def test_ensure_postgres_schema_skips_optional_vector_column_when_extension_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder: dict[str, FakeCursor] = {}

    class FakeCursor:
        def __init__(self) -> None:
            self.schema_metadata_present = True
            self.schema_version = "8"
            self.optional_vector_migration_runs = 0
            self.embedding_vector_column_present = False
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
                assert params is not None
                self.schema_version = str(params[1])
                self._result = None
                return
            if normalized.startswith("DO $$"):
                self.optional_vector_migration_runs += 1
                self._result = None
                return
            self._result = None

        def fetchone(self) -> tuple[object, ...] | None:
            return self._result

    class FakeConnection:
        def __init__(self, cursor: FakeCursor) -> None:
            self.cursor_instance = cursor

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
            self.cursor = FakeCursor()
            holder["cursor"] = self.cursor

        def getconn(self) -> FakeConnection:
            return FakeConnection(self.cursor)

        def putconn(self, connection: FakeConnection) -> None:
            del connection

        def close(self) -> None:
            return None

    class FakePsycopgModule:
        @staticmethod
        def connect(dsn: str) -> FakeConnection:
            assert dsn == "postgresql://memory@example.invalid/mcp_memory"
            return FakeConnection(FakeCursor())

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
        schema_version=POSTGRES_SCHEMA_VERSION,
    )
    assert holder["cursor"].optional_vector_migration_runs == 1
    assert holder["cursor"].embedding_vector_column_present is False


def test_build_provider_registry_uses_backend_capabilities_for_postgres(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MCP_MEMORY_TEST_MODE", raising=False)
    json_provider = SimpleNamespace(provider_name="Gemini CLI")
    agentic_provider = SimpleNamespace(provider_name="Gemini CLI Agent")
    monkeypatch.setattr("mcp_memory.mcp.runtime.build_json_ai_provider", lambda *args, **kwargs: json_provider)
    monkeypatch.setattr("mcp_memory.mcp.runtime.build_agentic_ai_provider", lambda *args, **kwargs: agentic_provider)
    monkeypatch.setattr("mcp_memory.mcp.runtime._provider_command_available", lambda provider: True)

    spec = RuntimeSpec(
        memory_path=tmp_path / "memories",
        config=Config(
            ai=AIConfig(provider="none"),
            provider_routing=ProviderRoutingConfig(
                profiles={
                    "gemini-cheap": AIConfig(
                        provider="gemini-cli",
                        model="gemini-2.5-pro",
                    )
                },
            ),
            storage=StorageConfig(
                backend="postgres",
                postgres=PostgresStorageConfig(dsn="postgresql://memory@example.invalid/mcp_memory"),
            )
        ),
        workspace_id="workspace-123",
        workspace_root=tmp_path,
        lock_path=tmp_path / "daemon.lock",
    )

    class _UnsupportedTaskQueue:
        def __getattr__(self, name: str):
            raise NotImplementedError(name)

    storage = StorageBackendResources(
        backend="postgres",
        db_manager=object(),
        journal=object(),
        repository=object(),
        relational_search=object(),
        read_cache=None,
        task_queue=_UnsupportedTaskQueue(),
        provider_usage=object(),
        runtime_logs=object(),
        provider_policy_events=object(),
        embedding_integrity_events=object(),
        task_execution_attempts=object(),
        work_items=object(),
        embedding_repair_queue=object(),
        vector_store=object(),
    )

    registry = _build_provider_registry(spec=spec, storage=storage)

    assert set(registry) == {"gemini-cheap"}
    assert set(registry["gemini-cheap"]) == {"json", "agentic"}
    json_instrumented = registry["gemini-cheap"]["json"]
    agentic_instrumented = registry["gemini-cheap"]["agentic"]
    assert isinstance(json_instrumented, InstrumentedAIProvider)
    assert isinstance(agentic_instrumented, InstrumentedAIProvider)
    assert json_instrumented._task_queue is None
    assert agentic_instrumented._task_queue is None
    assert json_instrumented._task_execution_attempts is storage.task_execution_attempts
