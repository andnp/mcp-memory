from __future__ import annotations

from contextlib import nullcontext
from typing import cast

import psycopg
import pytest

from mcp_memory.storage.postgres_store_support import optional_connection, require_connection
from mcp_memory.storage.session import SessionManager

pytestmark = pytest.mark.small


def test_optional_connection_yields_none_when_sessions_missing() -> None:
    with optional_connection(None) as connection:
        assert connection is None


def test_require_connection_raises_when_sessions_missing() -> None:
    with pytest.raises(RuntimeError, match="journal_unavailable"):
        with require_connection(None, error="journal_unavailable"):
            raise AssertionError("unreachable")


def test_optional_connection_uses_session_manager_when_present() -> None:
    marker = object()

    class FakeSessions:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def close(self) -> None:
            return None

        def open_connection(self):
            return nullcontext(marker)

    with optional_connection(cast(SessionManager, FakeSessions())) as connection:
        assert connection is marker


def test_require_connection_translates_operational_error_to_domain_error() -> None:
    class UnreachableSessions:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def close(self) -> None:
            return None

        def open_connection(self):
            raise psycopg.OperationalError("couldn't get a connection after 30.00 sec")

    with pytest.raises(RuntimeError, match="journal_unavailable") as exc_info:
        with require_connection(cast(SessionManager, UnreachableSessions()), error="journal_unavailable"):
            raise AssertionError("unreachable")

    assert isinstance(exc_info.value.__cause__, psycopg.OperationalError)


def test_require_connection_translates_operational_error_raised_inside_block() -> None:
    class FlakySessions:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def close(self) -> None:
            return None

        def open_connection(self):
            return nullcontext(object())

    with pytest.raises(RuntimeError, match="journal_unavailable"):
        with require_connection(cast(SessionManager, FlakySessions()), error="journal_unavailable"):
            raise psycopg.OperationalError("server closed the connection unexpectedly")
