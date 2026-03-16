import logging
import time

import pytest

from mcp_memory.config import LoggingConfig
from mcp_memory.core.providers.instrumented import InstrumentedAIProvider
from mcp_memory.provider_usage_store import ProviderUsageRepository
from mcp_memory.runtime_log_store import RuntimeLogRepository
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


def test_runtime_log_repository_prunes_by_age_and_count(db_manager) -> None:
    repository = RuntimeLogRepository(
        db_manager,
        workspace_id="workspace-a",
        config=LoggingConfig(
            max_runtime_logs=2,
            max_log_age_days=1,
            retention_check_interval_seconds=0.01,
        ),
    )

    now = time.time()
    repository.write_log(
        source="daemon",
        logger_name="mcp_memory.old",
        level="INFO",
        message="too old",
        created_at=now - 172800,
        data={},
    )
    repository.write_log(
        source="daemon",
        logger_name="mcp_memory.keep1",
        level="INFO",
        message="keep one",
        created_at=now,
        data={},
    )
    repository.write_log(
        source="daemon",
        logger_name="mcp_memory.keep2",
        level="WARNING",
        message="keep two",
        created_at=now + 1,
        data={},
    )
    repository.write_log(
        source="daemon",
        logger_name="mcp_memory.drop",
        level="ERROR",
        message="drop by count",
        created_at=now + 2,
        data={},
    )

    records = repository.list_logs(limit=10)

    assert [record.message for record in records] == ["drop by count", "keep two"]


def test_runtime_log_repository_queries_are_global_by_default(db_manager) -> None:
    repository_a = RuntimeLogRepository(db_manager, workspace_id="workspace-a")
    repository_b = RuntimeLogRepository(db_manager, workspace_id="workspace-b")

    repository_a.write_log(
        source="daemon",
        logger_name="mcp_memory.a",
        level="INFO",
        message="workspace a",
        created_at=10.0,
        data={},
    )
    repository_b.write_log(
        source="daemon",
        logger_name="mcp_memory.b",
        level="INFO",
        message="workspace b",
        created_at=11.0,
        data={},
    )

    records = repository_a.list_logs(limit=10)

    assert [record.message for record in records] == ["workspace b", "workspace a"]


@pytest.mark.asyncio
async def test_instrumented_provider_records_usage(db_manager) -> None:
    class FakeProvider:
        async def ask(self, prompt: str) -> dict:
            assert prompt == "hello"
            return {"answer": "world"}

    provider = InstrumentedAIProvider(
        FakeProvider(),
        usage_repository=ProviderUsageRepository(db_manager, workspace_id="workspace-a"),
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
    )

    payload = await provider.ask("hello")
    row = db_manager.get_connection().execute(
        "SELECT provider_key, provider_name, model_name, status FROM provider_usage"
    ).fetchone()

    assert payload == {"answer": "world"}
    assert row is not None
    assert row["provider_key"] == "gemini-cli"
    assert row["provider_name"] == "Gemini CLI"
    assert row["model_name"] == "gemini-3-flash-preview"
    assert row["status"] == "success"