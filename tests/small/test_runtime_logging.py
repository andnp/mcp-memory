import logging

import pytest

from mcp_memory.runtime_logging import SQLiteStructuredLogHandler


pytestmark = pytest.mark.small


def test_sqlite_structured_log_handler_writes_runtime_log(db_manager) -> None:
    handler = SQLiteStructuredLogHandler(
        db_manager=db_manager,
        workspace_id="workspace-a",
        source="stdio",
    )

    record = logging.LogRecord(
        name="mcp_memory.server",
        level=logging.INFO,
        pathname=__file__,
        lineno=16,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    handler.emit(record)

    row = db_manager.get_connection().execute(
        "SELECT workspace_id, source, logger_name, level, message FROM runtime_logs"
    ).fetchone()

    assert row is not None
    assert row["workspace_id"] == "workspace-a"
    assert row["source"] == "stdio"
    assert row["logger_name"] == "mcp_memory.server"
    assert row["level"] == "INFO"
    assert row["message"] == "hello world"