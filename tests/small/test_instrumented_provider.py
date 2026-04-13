from __future__ import annotations

import pytest

from mcp_memory.core.providers.interfaces import ProviderAttemptFinishedEvent
from mcp_memory.core.providers.interfaces import ProviderAttemptHeartbeatEvent
from mcp_memory.core.providers.interfaces import ProviderAttemptStartedEvent
from mcp_memory.core.providers.interfaces import ProviderObserverEvent
from mcp_memory.core.providers.instrumented import InstrumentedAIProvider
from mcp_memory.provider_usage_store import ProviderUsageRepository
from mcp_memory.task_execution_store import TaskExecutionAttemptRepository


pytestmark = pytest.mark.small


@pytest.mark.asyncio
async def test_instrumented_provider_ignores_unsupported_typed_observer_events(db_manager) -> None:
    class _Provider:
        def __init__(self, observer=None) -> None:
            self._observer = observer

        def with_observer(self, observer):
            return _Provider(observer)

        async def ask(self, prompt: str) -> dict[str, object]:
            assert self._observer is not None
            self._observer(ProviderObserverEvent(attempt=1, prompt=prompt, subprocess_pid=1234))
            return {"ok": True}

    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    provider = InstrumentedAIProvider(
        _Provider(),
        usage_repository=repository,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
    ).with_usage_context(
        task_name="memory-curator",
        task_id="unsupported-typed-observer-task",
        workspace_id="workspace-a",
    )

    result = await provider.ask("unsupported typed observer event")
    usage_row = db_manager.get_connection().execute(
        "SELECT status, subprocess_pid FROM provider_usage WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        ("unsupported-typed-observer-task",),
    ).fetchone()

    assert result == {"ok": True}
    assert usage_row is not None
    assert usage_row["status"] == "success"
    assert usage_row["subprocess_pid"] is None
    assert repository.list_conversations(task_name="memory-curator", limit=10) == []


@pytest.mark.asyncio
async def test_instrumented_provider_records_attempts_from_typed_observer_events(db_manager) -> None:
    class _Provider:
        def __init__(self, observer=None) -> None:
            self._observer = observer

        def with_observer(self, observer):
            return _Provider(observer)

        async def ask(self, prompt: str) -> dict[str, object]:
            assert self._observer is not None
            self._observer(
                ProviderAttemptStartedEvent(
                    attempt=1,
                    prompt=prompt,
                    subprocess_pid=8181,
                    started_at=10.0,
                )
            )
            self._observer(
                ProviderAttemptHeartbeatEvent(
                    attempt=1,
                    prompt=prompt,
                    subprocess_pid=8181,
                    started_at=10.0,
                    heartbeat_at=11.5,
                    elapsed_seconds=1.5,
                )
            )
            self._observer(
                ProviderAttemptFinishedEvent(
                    attempt=1,
                    prompt=prompt,
                    subprocess_pid=8181,
                    started_at=10.0,
                    completed_at=12.0,
                    duration_seconds=2.0,
                    status="success",
                    returncode=0,
                    raw_text='{"ok": true}',
                    parsed={"ok": True},
                    error_text=None,
                )
            )
            return {"ok": True}

    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    attempt_repository = TaskExecutionAttemptRepository(db_manager, workspace_id="workspace-a")
    provider = InstrumentedAIProvider(
        _Provider(),
        usage_repository=repository,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
        task_execution_attempts=attempt_repository,
    ).with_usage_context(
        task_name="memory-curator",
        task_id="typed-observer-task",
        execution_epoch=7,
        workspace_id="workspace-a",
    )

    result = await provider.ask("typed observer event")
    attempt = attempt_repository.get_attempt(task_id="typed-observer-task", execution_epoch=7)
    usage_row = db_manager.get_connection().execute(
        "SELECT status, subprocess_pid FROM provider_usage WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        ("typed-observer-task",),
    ).fetchone()
    conversation = repository.list_conversations(task_name="memory-curator", limit=1)[0]

    assert result == {"ok": True}
    assert usage_row is not None
    assert usage_row["status"] == "success"
    assert usage_row["subprocess_pid"] == 8181
    assert conversation.task_id == "typed-observer-task"
    assert conversation.subprocess_pid == 8181
    assert conversation.status == "success"
    assert conversation.response_text == '{"ok": true}'
    assert attempt.subprocess_pid == 8181
    assert attempt.status == "success"
    assert attempt.started_at == pytest.approx(10.0)
    assert attempt.last_heartbeat_at == pytest.approx(12.0)
    assert attempt.completed_at == pytest.approx(12.0)