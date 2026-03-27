from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType

from mcp_memory.config import PostgresStorageConfig
from mcp_memory.storage.session import ConnectionLease, DbConnectionLike, PoolLike, SessionManager


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


def build_postgres_connection_kwargs(config: PostgresStorageConfig) -> dict[str, object]:
    options = (
        f"-c statement_timeout={config.statement_timeout_ms}ms "
        f"-c lock_timeout={config.lock_timeout_ms}ms"
    )
    return {
        "autocommit": False,
        "application_name": config.application_name,
        "options": options,
    }


@dataclass
class PooledPostgresConnectionLease(ConnectionLease[DbConnectionLike]):
    pool: PoolLike[DbConnectionLike]
    connection: DbConnectionLike
    _closed: bool = False

    def __enter__(self) -> DbConnectionLike:
        return self.connection

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.connection.rollback()
        self.close()
        return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.pool.putconn(self.connection)


class PostgresConnectionManager(SessionManager[DbConnectionLike]):
    def __init__(self, config: PostgresStorageConfig) -> None:
        self._config = config
        self._pool: PoolLike[DbConnectionLike] | None = None

    def __enter__(self) -> PostgresConnectionManager:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def open_connection(self) -> PooledPostgresConnectionLease:
        pool = self._ensure_pool()
        connection = pool.getconn()
        return PooledPostgresConnectionLease(pool=pool, connection=connection)

    def close(self) -> None:
        if self._pool is None:
            return
        self._pool.close()
        self._pool = None

    def _ensure_pool(self) -> PoolLike[DbConnectionLike]:
        if self._pool is not None:
            return self._pool
        _, psycopg_pool = load_postgres_driver_modules()
        pool = psycopg_pool.ConnectionPool(
            conninfo=self._config.dsn,
            min_size=self._config.pool_min,
            max_size=self._config.pool_max,
            kwargs=build_postgres_connection_kwargs(self._config),
            open=True,
        )
        self._pool = pool
        return pool