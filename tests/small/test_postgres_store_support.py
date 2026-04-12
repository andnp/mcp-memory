from __future__ import annotations

from contextlib import nullcontext
from typing import cast

import pytest

from mcp_memory.storage.session import SessionManager
from mcp_memory.storage.postgres_store_support import optional_connection, require_connection


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
