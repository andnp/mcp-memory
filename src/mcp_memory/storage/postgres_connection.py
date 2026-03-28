from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import cast

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
        self.close()
        return False

    def close(self, *, reset_connection: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        discard_connection = self._should_discard_connection()
        if reset_connection and not discard_connection:
            discard_connection = not self._reset_connection()
        if discard_connection:
            self._discard_connection()
        self.pool.putconn(self.connection)

    def _reset_connection(self) -> bool:
        rollback = getattr(self.connection, "rollback", None)
        if rollback is None:
            return not self._should_discard_connection()
        try:
            rollback()
        except Exception:
            return False
        return not self._should_discard_connection()

    def _should_discard_connection(self) -> bool:
        return bool(getattr(self.connection, "broken", False) or getattr(self.connection, "closed", False))

    def _discard_connection(self) -> None:
        close = getattr(self.connection, "close", None)
        if close is None:
            return
        try:
            close()
        except Exception:
            return


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
        pool = cast(
            PoolLike[DbConnectionLike],
            psycopg_pool.ConnectionPool(
                conninfo=self._config.dsn,
                min_size=self._config.pool_min,
                max_size=self._config.pool_max,
                kwargs=build_postgres_connection_kwargs(self._config),
                open=True,
            ),
        )
        self._pool = pool
        return pool
