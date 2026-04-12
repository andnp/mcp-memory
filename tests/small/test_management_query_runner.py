from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

from mcp_memory.management.query_runner import ManagementQueryRunner


def test_management_query_runner_uses_sqlite_api_without_query_rewrite() -> None:
    seen: dict[str, object] = {}

    class FakeExecuteResult:
        def fetchall(self):
            return [{"count": 3}]

    class FakeConnection:
        def execute(self, query, params):
            seen["query"] = query
            seen["params"] = params
            return FakeExecuteResult()

    class FakeDbManager:
        def get_connection(self):
            return FakeConnection()

    runner = ManagementQueryRunner(FakeDbManager())
    rows = runner.fetchall("SELECT CHAR(10) FROM tasks WHERE workspace_id = ?", ["workspace-a"])

    assert rows == [{"count": 3}]
    assert seen["query"] == "SELECT CHAR(10) FROM tasks WHERE workspace_id = ?"
    assert seen["params"] == ["workspace-a"]


def test_management_query_runner_adapts_query_for_postgres_cursor_api() -> None:
    seen: dict[str, object] = {}

    class FakeCursor:
        description = [SimpleNamespace(name="count")]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def execute(self, query, params):
            seen["query"] = query
            seen["params"] = params

        def fetchall(self):
            return [(5,)]

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

    class FakeDbManager:
        def open_connection(self):
            return nullcontext(FakeConnection())

    runner = ManagementQueryRunner(FakeDbManager())
    rows = runner.fetchall(
        "SELECT GROUP_CONCAT(DISTINCT workspace_id), CHAR(10) FROM tasks WHERE workspace_id = ?",
        ["workspace-a"],
    )

    assert rows == [{"count": 5}]
    assert seen["query"] == "SELECT STRING_AGG(DISTINCT workspace_id, ','), CHR(10) FROM tasks WHERE workspace_id = %s"
    assert seen["params"] == ("workspace-a",)