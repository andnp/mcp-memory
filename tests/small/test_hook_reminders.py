from __future__ import annotations

import pytest

from mcp_memory.hook_reminders import HookReminderService, REMINDER_MESSAGE


pytestmark = pytest.mark.small


def test_post_tool_use_only_reminds_every_five_minutes(db_manager) -> None:
    service = HookReminderService(db_manager, workspace_id="workspace-a")

    first = service.record_post_tool_use(
        {"sessionId": "conv-1", "tool_name": "read_file", "timestamp": 100.0}
    )
    second = service.record_post_tool_use(
        {"sessionId": "conv-1", "tool_name": "read_file", "timestamp": 399.0}
    )
    third = service.record_post_tool_use(
        {"sessionId": "conv-1", "tool_name": "read_file", "timestamp": 401.0}
    )

    assert first == {}
    assert second == {}
    assert third["systemMessage"] == REMINDER_MESSAGE


def test_session_end_marks_conversation_ended(db_manager) -> None:
    service = HookReminderService(db_manager, workspace_id="workspace-a")

    service.record_session_start("conv-2", {"timestamp": 50.0})
    service.record_session_end("conv-2", {"timestamp": 75.0})

    record = service.get_conversation("conv-2")
    assert record is not None
    assert record.ended_at == 75.0


def test_active_client_count_is_workspace_scoped(db_manager) -> None:
    workspace_a = HookReminderService(db_manager, workspace_id="workspace-a")
    workspace_b = HookReminderService(db_manager, workspace_id="workspace-b")

    workspace_a.record_session_start("conv-a-1", {"timestamp": 10.0})
    workspace_a.record_session_start("conv-a-2", {"timestamp": 11.0})
    workspace_b.record_session_start("conv-b-1", {"timestamp": 12.0})
    workspace_a.record_session_end("conv-a-1", {"timestamp": 13.0})

    assert workspace_a.get_active_client_count() == 1
    assert workspace_b.get_active_client_count() == 1