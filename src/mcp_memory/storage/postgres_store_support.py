from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg

from mcp_memory.storage.session import DbConnectionLike, SessionManager


@contextmanager
def optional_connection(
    sessions: SessionManager[DbConnectionLike] | None,
) -> Iterator[DbConnectionLike | None]:
    if sessions is None:
        yield None
        return
    with sessions.open_connection() as connection:
        yield connection


@contextmanager
def require_connection(
    sessions: SessionManager[DbConnectionLike] | None,
    *,
    error: str,
) -> Iterator[DbConnectionLike]:
    if sessions is None:
        raise RuntimeError(error)
    try:
        with sessions.open_connection() as connection:
            yield connection
    except psycopg.OperationalError as exc:
        raise RuntimeError(error) from exc
