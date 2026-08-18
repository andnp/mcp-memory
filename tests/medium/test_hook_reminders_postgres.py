from __future__ import annotations

from collections.abc import Callable, Generator

import pytest

from mcp_memory.hook_reminders import REMINDER_MESSAGE, HookReminderService
from mcp_memory.storage.postgres import ensure_postgres_schema
from mcp_memory.storage.postgres_connection import PostgresConnectionManager
from tests.small.hook_reminder_contract import (
    assert_active_client_count_excludes_stale_unended_sessions,
    assert_active_client_count_is_global_across_workspace_scoped_instances,
    assert_post_tool_use_only_reminds_after_interval,
    assert_session_end_marks_conversation_ended,
)

pytestmark = pytest.mark.medium

ServiceFactory = Callable[[str | None], HookReminderService]


@pytest.fixture
def postgres_hook_reminder_factory(postgres_storage_config) -> Generator[ServiceFactory, None, None]:
    ensure_postgres_schema(postgres_storage_config)

    with PostgresConnectionManager(postgres_storage_config) as manager:
        yield lambda workspace_id: HookReminderService(manager, workspace_id=workspace_id)


def test_postgres_hook_reminders_post_tool_use_only_reminds_every_five_minutes(
    postgres_hook_reminder_factory: ServiceFactory,
) -> None:
    assert_post_tool_use_only_reminds_after_interval(postgres_hook_reminder_factory)


def test_postgres_hook_reminders_session_end_marks_conversation_ended(
    postgres_hook_reminder_factory: ServiceFactory,
) -> None:
    assert_session_end_marks_conversation_ended(postgres_hook_reminder_factory)


def test_postgres_hook_reminders_active_client_count_is_global(
    postgres_hook_reminder_factory: ServiceFactory,
) -> None:
    assert_active_client_count_is_global_across_workspace_scoped_instances(postgres_hook_reminder_factory)


def test_postgres_hook_reminders_active_client_count_excludes_stale_unended_sessions(
    postgres_hook_reminder_factory: ServiceFactory,
) -> None:
    assert_active_client_count_excludes_stale_unended_sessions(postgres_hook_reminder_factory)


def test_postgres_hook_reminders_preserve_null_workspace_for_global_daemon_context(
    postgres_hook_reminder_factory: ServiceFactory,
) -> None:
    service = postgres_hook_reminder_factory(None)

    service.record_session_start("global-conv", {"timestamp": 10.0})
    reminder = service.record_post_tool_use(
        {"sessionId": "global-conv", "tool_name": "read_file", "timestamp": 310.0}
    )

    record = service.get_conversation("global-conv")

    assert reminder["systemMessage"] == REMINDER_MESSAGE
    assert record is not None
    assert record.workspace_id is None
    assert record.last_tool_name == "read_file"