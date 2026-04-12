from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from mcp_memory.hook_reminders import HookConversationRecord, REMINDER_MESSAGE


class HookReminderServiceLike(Protocol):
    def record_session_start(
        self,
        conversation_id: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, str]: ...

    def record_post_tool_use(self, payload: dict[str, object]) -> dict[str, str]: ...

    def record_session_end(
        self,
        conversation_id: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, str]: ...

    def get_conversation(self, conversation_id: str) -> HookConversationRecord | None: ...

    def get_active_client_count(self, *, now: float | None = None) -> int: ...


ServiceFactory = Callable[[str | None], HookReminderServiceLike]


def assert_post_tool_use_only_reminds_after_interval(make_service: ServiceFactory) -> None:
    service = make_service("workspace-a")

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


def assert_session_end_marks_conversation_ended(make_service: ServiceFactory) -> None:
    service = make_service("workspace-a")

    service.record_session_start("conv-2", {"timestamp": 50.0})
    service.record_session_end("conv-2", {"timestamp": 75.0})

    record = service.get_conversation("conv-2")
    assert record is not None
    assert record.ended_at == 75.0


def assert_active_client_count_is_global_across_workspace_scoped_instances(
    make_service: ServiceFactory,
) -> None:
    workspace_a = make_service("workspace-a")
    workspace_b = make_service("workspace-b")

    workspace_a.record_session_start("conv-a-1", {"timestamp": 10.0})
    workspace_a.record_session_start("conv-a-2", {"timestamp": 11.0})
    workspace_b.record_session_start("conv-b-1", {"timestamp": 12.0})
    workspace_a.record_session_end("conv-a-1", {"timestamp": 13.0})

    assert workspace_a.get_active_client_count(now=15.0) == 2
    assert workspace_b.get_active_client_count(now=15.0) == 2


def assert_active_client_count_excludes_stale_unended_sessions(make_service: ServiceFactory) -> None:
    service = make_service("workspace-a")

    service.record_session_start("stale-conv", {"timestamp": 10.0})
    service.record_session_start("fresh-conv", {"timestamp": 4000.0})

    assert service.get_active_client_count(now=4000.0) == 1