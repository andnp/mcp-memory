from __future__ import annotations

import pytest

from mcp_memory.hook_reminders import HookReminderService
from tests.small.hook_reminder_contract import (
    assert_active_client_count_excludes_stale_unended_sessions,
    assert_active_client_count_is_global_across_workspace_scoped_instances,
    assert_post_tool_use_only_reminds_after_interval,
    assert_session_end_marks_conversation_ended,
)

pytestmark = pytest.mark.small


def test_sqlite_hook_reminders_post_tool_use_only_reminds_every_five_minutes(db_manager) -> None:
    assert_post_tool_use_only_reminds_after_interval(
        lambda workspace_id: HookReminderService(db_manager, workspace_id=workspace_id)
    )


def test_sqlite_hook_reminders_session_end_marks_conversation_ended(db_manager) -> None:
    assert_session_end_marks_conversation_ended(
        lambda workspace_id: HookReminderService(db_manager, workspace_id=workspace_id)
    )


def test_sqlite_hook_reminders_active_client_count_is_global(db_manager) -> None:
    assert_active_client_count_is_global_across_workspace_scoped_instances(
        lambda workspace_id: HookReminderService(db_manager, workspace_id=workspace_id)
    )


def test_sqlite_hook_reminders_active_client_count_excludes_stale_unended_sessions(db_manager) -> None:
    assert_active_client_count_excludes_stale_unended_sessions(
        lambda workspace_id: HookReminderService(db_manager, workspace_id=workspace_id)
    )
