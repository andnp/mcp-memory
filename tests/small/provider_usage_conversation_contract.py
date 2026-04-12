from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from mcp_memory.provider_usage_store import AIConversationRecord


class ProviderUsageConversationRepositoryLike(Protocol):
    def record_conversation(
        self,
        *,
        request_id: str,
        attempt: int,
        task_name: str | None,
        task_id: str | None,
        provider_key: str,
        provider_name: str,
        model_name: str,
        subprocess_pid: int | None,
        prompt_text: str,
        response_text: str,
        parsed: dict | None,
        status: str,
        error_text: str | None,
        reason_category: str | None = None,
        reason_code: str | None = None,
        retry_delay_seconds: float | None = None,
        started_at: float,
        completed_at: float,
    ) -> None: ...

    def finalize_running_conversation(
        self,
        *,
        request_id: str,
        status: str,
        error_text: str | None,
        reason_category: str | None = None,
        reason_code: str | None = None,
        retry_delay_seconds: float | None = None,
        completed_at: float | None = None,
        response_text: str | object,
        parsed: dict | None | object,
    ) -> int: ...

    def get_conversation(self, request_id: str) -> list[AIConversationRecord]: ...


RepositoryFactory = Callable[[], ProviderUsageConversationRepositoryLike]


def assert_preserves_first_terminal_conversation_finalization(make_repository: RepositoryFactory) -> None:
    repository = make_repository()

    repository.record_conversation(
        request_id="req-late-terminal-finalize",
        attempt=1,
        task_name="ingest-system1",
        task_id="task-late-terminal-finalize",
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-2.5-pro",
        subprocess_pid=9999,
        prompt_text="ingest pending thoughts",
        response_text="",
        parsed=None,
        status="running",
        error_text=None,
        started_at=10.0,
        completed_at=10.0,
    )

    first_finalize = repository.finalize_running_conversation(
        request_id="req-late-terminal-finalize",
        status="error",
        error_text="Provider subprocess 9999 exited unexpectedly",
        reason_category="recovery",
        reason_code="provider_subprocess_exited_retry",
        retry_delay_seconds=45.0,
        completed_at=12.0,
        response_text="partial response preserved",
        parsed={"outcome": "retry"},
    )

    late_conflicting_finalize = repository.finalize_running_conversation(
        request_id="req-late-terminal-finalize",
        status="cancelled",
        error_text="Command cancelled",
        reason_category="cancellation",
        reason_code="provider_cancelled",
        completed_at=20.0,
        response_text="too late to win",
        parsed={"outcome": "cancelled"},
    )

    conversation = repository.get_conversation("req-late-terminal-finalize")[0]

    assert first_finalize == 1
    assert late_conflicting_finalize == 0
    assert conversation.status == "error"
    assert conversation.completed_at == 12.0
    assert conversation.duration_seconds == 2.0
    assert conversation.error_text == "Provider subprocess 9999 exited unexpectedly"
    assert conversation.reason_category == "recovery"
    assert conversation.reason_code == "provider_subprocess_exited_retry"
    assert conversation.retry_delay_seconds == 45.0
    assert conversation.response_text == "partial response preserved"
    assert conversation.parsed == {"outcome": "retry"}