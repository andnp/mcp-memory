from __future__ import annotations

from types import SimpleNamespace

import pytest

from mcp_memory.config import PostgresStorageConfig
from mcp_memory.storage.postgres_connection import (
    PostgresConnectionManager,
    PostgresDriverMissingError,
    build_postgres_connection_kwargs,
    load_postgres_driver_modules,
)


pytestmark = pytest.mark.small


def test_load_postgres_driver_modules_raises_helpful_error_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_module(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("mcp_memory.storage.postgres_connection.importlib.import_module", fake_import_module)

    with pytest.raises(PostgresDriverMissingError, match="requires psycopg and psycopg_pool"):
        load_postgres_driver_modules()


def test_build_postgres_connection_kwargs_includes_timeouts_and_app_name() -> None:
    config = PostgresStorageConfig(
        dsn="postgresql://memory@example.invalid/mcp_memory",
        statement_timeout_ms=12000,
        lock_timeout_ms=3456,
        application_name="mcp-memory-tests",
    )

    kwargs = build_postgres_connection_kwargs(config)

    assert kwargs == {
        "autocommit": False,
        "application_name": "mcp-memory-tests",
        "options": "-c statement_timeout=12000ms -c lock_timeout=3456ms",
    }


def test_postgres_connection_manager_returns_borrow_to_pool_on_close(monkeypatch: pytest.MonkeyPatch) -> None:
    borrowed_connections: list[object] = []
    returned_connections: list[object] = []
    closed_pools: list[str] = []
    rollbacks: list[str] = []

    class FakeConnection:
        broken = False
        closed = False

        def rollback(self) -> None:
            rollbacks.append("rolled-back")

        def close(self) -> None:
            self.closed = True

    raw_connection = FakeConnection()

    class FakeConnectionPool:
        def __init__(self, **kwargs) -> None:
            assert kwargs["conninfo"] == "postgresql://memory@example.invalid/mcp_memory"
            assert kwargs["min_size"] == 2
            assert kwargs["max_size"] == 5
            assert kwargs["kwargs"] == {
                "autocommit": False,
                "application_name": "mcp-memory",
                "options": "-c statement_timeout=30000ms -c lock_timeout=5000ms",
            }

        def getconn(self) -> object:
            borrowed_connections.append(raw_connection)
            return raw_connection

        def putconn(self, connection: object) -> None:
            returned_connections.append(connection)

        def close(self) -> None:
            closed_pools.append("closed")

    def fake_import_module(name: str):
        if name == "psycopg":
            return SimpleNamespace()
        if name == "psycopg_pool":
            return SimpleNamespace(ConnectionPool=FakeConnectionPool)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("mcp_memory.storage.postgres_connection.importlib.import_module", fake_import_module)

    manager = PostgresConnectionManager(
        PostgresStorageConfig(
            dsn="postgresql://memory@example.invalid/mcp_memory",
            pool_min=2,
            pool_max=5,
        )
    )

    lease = manager.open_connection()
    assert lease.__enter__() is raw_connection
    lease.close()
    lease.close()
    manager.close()

    assert borrowed_connections == [raw_connection]
    assert returned_connections == [raw_connection]
    assert closed_pools == ["closed"]
    assert rollbacks == ["rolled-back"]


def test_postgres_connection_lease_rolls_back_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    returned_connections: list[object] = []
    rollbacks: list[str] = []

    class FakeConnection:
        broken = False
        closed = False

        def commit(self) -> None:
            return None

        def rollback(self) -> None:
            rollbacks.append("rolled-back")

        def close(self) -> None:
            self.closed = True

    raw_connection = FakeConnection()

    class FakeConnectionPool:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def getconn(self) -> FakeConnection:
            return raw_connection

        def putconn(self, connection: object) -> None:
            returned_connections.append(connection)

        def close(self) -> None:
            return None

    def fake_import_module(name: str):
        if name == "psycopg":
            return SimpleNamespace()
        if name == "psycopg_pool":
            return SimpleNamespace(ConnectionPool=FakeConnectionPool)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("mcp_memory.storage.postgres_connection.importlib.import_module", fake_import_module)

    manager = PostgresConnectionManager(PostgresStorageConfig(dsn="postgresql://memory@example.invalid/mcp_memory"))

    with pytest.raises(RuntimeError, match="boom"):
        with manager.open_connection() as _connection:
            raise RuntimeError("boom")

    manager.close()

    assert rollbacks == ["rolled-back"]
    assert returned_connections == [raw_connection]


def test_postgres_connection_lease_resets_transaction_before_context_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    returned_connections: list[object] = []
    rollbacks: list[str] = []

    class FakeConnection:
        broken = False
        closed = False

        def rollback(self) -> None:
            rollbacks.append("rolled-back")

        def close(self) -> None:
            self.closed = True

    raw_connection = FakeConnection()

    class FakeConnectionPool:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def getconn(self) -> FakeConnection:
            return raw_connection

        def putconn(self, connection: object) -> None:
            returned_connections.append(connection)

        def close(self) -> None:
            return None

    def fake_import_module(name: str):
        if name == "psycopg":
            return SimpleNamespace()
        if name == "psycopg_pool":
            return SimpleNamespace(ConnectionPool=FakeConnectionPool)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("mcp_memory.storage.postgres_connection.importlib.import_module", fake_import_module)

    manager = PostgresConnectionManager(PostgresStorageConfig(dsn="postgresql://memory@example.invalid/mcp_memory"))

    with manager.open_connection() as connection:
        assert connection is raw_connection

    manager.close()

    assert rollbacks == ["rolled-back"]
    assert returned_connections == [raw_connection]


def test_postgres_connection_lease_discards_broken_borrow_before_returning_to_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returned_connections: list[object] = []
    closed_connections: list[str] = []
    rollbacks: list[str] = []

    class FakeConnection:
        broken = True
        closed = False

        def rollback(self) -> None:
            rollbacks.append("rolled-back")

        def close(self) -> None:
            closed_connections.append("closed")
            self.closed = True

    raw_connection = FakeConnection()

    class FakeConnectionPool:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def getconn(self) -> FakeConnection:
            return raw_connection

        def putconn(self, connection: object) -> None:
            returned_connections.append(connection)

        def close(self) -> None:
            return None

    def fake_import_module(name: str):
        if name == "psycopg":
            return SimpleNamespace()
        if name == "psycopg_pool":
            return SimpleNamespace(ConnectionPool=FakeConnectionPool)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("mcp_memory.storage.postgres_connection.importlib.import_module", fake_import_module)

    manager = PostgresConnectionManager(PostgresStorageConfig(dsn="postgresql://memory@example.invalid/mcp_memory"))

    with manager.open_connection() as connection:
        assert connection is raw_connection

    manager.close()

    assert rollbacks == []
    assert closed_connections == ["closed"]
    assert returned_connections == [raw_connection]


def test_postgres_connection_lease_discards_connection_when_exception_cleanup_reset_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returned_connections: list[object] = []
    closed_connections: list[str] = []
    rollbacks: list[str] = []

    class FakeConnection:
        broken = False
        closed = False

        def rollback(self) -> None:
            rollbacks.append("rollback-attempted")
            raise RuntimeError("reset failed")

        def close(self) -> None:
            closed_connections.append("closed")
            self.closed = True

    raw_connection = FakeConnection()

    class FakeConnectionPool:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def getconn(self) -> FakeConnection:
            return raw_connection

        def putconn(self, connection: object) -> None:
            returned_connections.append(connection)

        def close(self) -> None:
            return None

    def fake_import_module(name: str):
        if name == "psycopg":
            return SimpleNamespace()
        if name == "psycopg_pool":
            return SimpleNamespace(ConnectionPool=FakeConnectionPool)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("mcp_memory.storage.postgres_connection.importlib.import_module", fake_import_module)

    manager = PostgresConnectionManager(PostgresStorageConfig(dsn="postgresql://memory@example.invalid/mcp_memory"))

    with pytest.raises(RuntimeError, match="boom"):
        with manager.open_connection() as _connection:
            raise RuntimeError("boom")

    manager.close()

    assert rollbacks == ["rollback-attempted"]
    assert closed_connections == ["closed"]
    assert returned_connections == [raw_connection]