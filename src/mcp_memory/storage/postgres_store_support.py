from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

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
    with sessions.open_connection() as connection:
        yield connection
