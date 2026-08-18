from __future__ import annotations

import pytest

from mcp_memory.core.provider_admission import ProviderAdmissionDecision
from mcp_memory.core.providers.instrumented import InstrumentedAIProvider
from mcp_memory.core.providers.interfaces import (
    AgenticRunResult,
    ProviderAttemptFinishedEvent,
    ProviderAttemptHeartbeatEvent,
    ProviderAttemptStartedEvent,
    ProviderBudgetExceeded,
    ProviderObserverEvent,
)
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
        "SELECT execution_epoch, attempt, attempt_identity, status, subprocess_pid FROM provider_usage WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        ("typed-observer-task",),
    ).fetchone()
    conversation = repository.list_conversations(task_name="memory-curator", limit=1)[0]

    assert result == {"ok": True}
    assert usage_row is not None
    assert usage_row["execution_epoch"] == 7
    assert usage_row["attempt"] == 1
    assert usage_row["attempt_identity"].startswith("typed-observer-task:7:")
    assert usage_row["attempt_identity"].endswith(":1")
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


@pytest.mark.asyncio
async def test_instrumented_provider_propagates_retry_attempt_identities(db_manager, monkeypatch) -> None:
    """Each observed provider retry keeps the run epoch and deterministic identity."""
    class _Provider:
        def __init__(self, observer=None) -> None:
            self._observer = observer

        def with_observer(self, observer):
            return _Provider(observer)

        async def ask(self, prompt: str) -> dict[str, object]:
            assert self._observer is not None
            for attempt in (1, 2):
                self._observer(
                    ProviderAttemptStartedEvent(
                        attempt=attempt,
                        prompt=prompt,
                        subprocess_pid=8000 + attempt,
                        started_at=float(attempt),
                    )
                )
                self._observer(
                    ProviderAttemptFinishedEvent(
                        attempt=attempt,
                        prompt=prompt,
                        subprocess_pid=8000 + attempt,
                        started_at=float(attempt),
                        completed_at=float(attempt) + 0.5,
                        duration_seconds=0.5,
                        status="error" if attempt == 1 else "success",
                        raw_text="retry" if attempt == 1 else '{"ok": true}',
                        parsed=None if attempt == 1 else {"ok": True},
                    )
                )
            return {"ok": True}

    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    conversations: list[dict[str, object]] = []

    def record_conversation(**kwargs: object) -> None:
        conversations.append(kwargs)

    monkeypatch.setattr(repository, "record_conversation", record_conversation)
    provider = InstrumentedAIProvider(
        _Provider(),
        usage_repository=repository,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
    ).with_usage_context(
        task_name="memory-curator",
        task_id="retry-attempt-task",
        execution_epoch=11,
        workspace_id="workspace-a",
    )

    await provider.ask("retry this")

    request_ids = {str(event["request_id"]) for event in conversations}
    assert len(request_ids) == 1
    request_id = request_ids.pop()
    assert [event["attempt"] for event in conversations] == [1, 1, 2, 2]
    assert {
        event["attempt_identity"]
        for event in conversations
    } == {
        f"retry-attempt-task:11:{request_id}:1",
        f"retry-attempt-task:11:{request_id}:2",
    }


@pytest.mark.asyncio
async def test_instrumented_provider_admission_skip_has_no_call_identity(db_manager) -> None:
    """Admission skips retain the run epoch without claiming a provider call."""
    class _Provider:
        async def ask(self, prompt: str) -> dict[str, object]:
            raise AssertionError("admission should prevent provider execution")

    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    provider = InstrumentedAIProvider(
        _Provider(),
        usage_repository=repository,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
        daily_call_limit=0,
    ).with_usage_context(
        task_name="memory-curator",
        task_id="admission-skip-task",
        execution_epoch=13,
        workspace_id="workspace-a",
    )

    with pytest.raises(ProviderBudgetExceeded):
        await provider.ask("do not call")

    row = db_manager.get_connection().execute(
        "SELECT execution_epoch, request_id, attempt, attempt_identity, status "
        "FROM provider_usage WHERE task_id = ?",
        ("admission-skip-task",),
    ).fetchone()
    assert tuple(row) == (13, None, None, None, "skipped")


@pytest.mark.asyncio
async def test_instrumented_provider_exposes_authoritative_json_call_telemetry(db_manager) -> None:
    class _Provider:
        def __init__(self, observer=None) -> None:
            self._observer = observer

        def with_observer(self, observer):
            return _Provider(observer)

        async def ask_json(self, prompt: str) -> dict[str, object]:
            assert self._observer is not None
            self._observer(
                ProviderAttemptStartedEvent(attempt=1, prompt=prompt, subprocess_pid=5150, started_at=10.0)
            )
            self._observer(
                ProviderAttemptFinishedEvent(
                    attempt=1,
                    prompt=prompt,
                    subprocess_pid=5150,
                    started_at=10.0,
                    completed_at=11.0,
                    duration_seconds=1.0,
                    status="success",
                    raw_text='{"ok": true}',
                    parsed={"ok": True},
                )
            )
            return {"ok": True}

    provider = InstrumentedAIProvider(
        _Provider(),
        usage_repository=ProviderUsageRepository(db_manager, workspace_id="workspace-a"),
        provider_key="copilot-strong",
        provider_name="Copilot SDK",
        model_name="gpt-5.4-mini",
    )

    result = await provider.ask_json_with_telemetry("record this call")

    assert result.response == {"ok": True}
    assert result.request_id
    assert result.attempt == 1
    assert result.started_at == pytest.approx(10.0)
    assert result.completed_at == pytest.approx(11.0)
    assert result.provider_call is True


def test_instrumented_provider_admission_skip_keeps_epoch_without_request_id(db_manager) -> None:
    """Policy admission skips remain task-attributed while lacking call identity."""
    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    provider = InstrumentedAIProvider(
        object(),
        usage_repository=repository,
        provider_key="copilot-strong",
        provider_name="Copilot SDK",
        model_name="gpt-5.4-mini",
    ).with_usage_context(
        task_name="memory-curator",
        task_id="skipped-curator-task",
        execution_epoch=3,
        workspace_id="workspace-a",
    )

    provider.record_admission_skip(
        ProviderAdmissionDecision(
            allowed=False,
            reason="provider_rate_limited",
            reason_category="admission",
            error_text="temporarily unavailable",
        ),
        now=20.0,
    )

    row = db_manager.get_connection().execute(
        "SELECT task_id, execution_epoch, request_id, attempt, attempt_identity, status "
        "FROM provider_usage WHERE task_id = ?",
        ("skipped-curator-task",),
    ).fetchone()
    assert tuple(row) == ("skipped-curator-task", 3, None, None, None, "skipped")


@pytest.mark.asyncio
async def test_instrumented_provider_usage_identity_changes_with_restart_epoch(db_manager) -> None:
    """Provider usage identities distinguish retry calls from restarted epochs."""
    class _Provider:
        async def ask(self, prompt: str) -> dict[str, object]:
            return {"prompt": prompt}

    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    base = InstrumentedAIProvider(
        _Provider(),
        usage_repository=repository,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
    )
    await base.with_usage_context(
        task_name="memory-curator",
        task_id="restartable-curator-task",
        execution_epoch=4,
        workspace_id="workspace-a",
    ).ask("first epoch")
    await base.with_usage_context(
        task_name="memory-curator",
        task_id="restartable-curator-task",
        execution_epoch=5,
        workspace_id="workspace-a",
    ).ask("restarted epoch")

    rows = db_manager.get_connection().execute(
        "SELECT execution_epoch, attempt_identity FROM provider_usage "
        "WHERE task_id = ? ORDER BY execution_epoch",
        ("restartable-curator-task",),
    ).fetchall()
    assert [row["execution_epoch"] for row in rows] == [4, 5]
    assert rows[0]["attempt_identity"].startswith("restartable-curator-task:4:")
    assert rows[1]["attempt_identity"].startswith("restartable-curator-task:5:")


@pytest.mark.asyncio
async def test_instrumented_agentic_session_correlates_observer_and_usage_request(db_manager) -> None:
    """Persistent agent turns share request identity across observer and usage rows."""
    class _Session:
        def __init__(self, observer) -> None:
            self._observer = observer

        async def run_agent(self, prompt: str) -> AgenticRunResult:
            self._observer(ProviderAttemptStartedEvent(attempt=1, prompt=prompt, subprocess_pid=777, started_at=1.0))
            return AgenticRunResult(status="success", summary="done")

        async def close(self) -> None:
            return None

    class _Provider:
        def __init__(self, observer=None) -> None:
            self._observer = observer

        def with_observer(self, observer):
            return _Provider(observer)

        async def open_agent_session(self, *, allowed_tool_names=None, tools=None):
            del allowed_tool_names, tools
            return _Session(self._observer)

    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    provider = InstrumentedAIProvider(
        _Provider(),
        usage_repository=repository,
        provider_key="copilot-sdk",
        provider_name="Copilot SDK",
        model_name="gpt-5.4-mini",
    ).with_usage_context(
        task_name="memory-curator",
        task_id="agentic-session-task",
        execution_epoch=8,
        workspace_id="workspace-a",
    )

    session = await provider.open_agent_session()
    await session.run_agent("inspect")
    await session.close()

    usage = db_manager.get_connection().execute(
        "SELECT request_id, attempt_identity FROM provider_usage WHERE task_id = ?",
        ("agentic-session-task",),
    ).fetchone()
    conversation = repository.get_conversation(usage["request_id"])[0]
    assert usage["attempt_identity"].startswith("agentic-session-task:8:")
    assert conversation.task_id == "agentic-session-task"


@pytest.mark.asyncio
async def test_instrumented_provider_propagates_packet_id_through_agentic_wrappers(db_manager) -> None:
    """Keep curator packet attribution through provider wrapper factories and storage rows."""
    class _Session:
        def __init__(self, observer) -> None:
            self._observer = observer

        async def run_agent(self, prompt: str) -> AgenticRunResult:
            self._observer(
                ProviderAttemptStartedEvent(
                    attempt=1,
                    prompt=prompt,
                    subprocess_pid=901,
                    started_at=1.0,
                )
            )
            self._observer(
                ProviderAttemptFinishedEvent(
                    attempt=1,
                    prompt=prompt,
                    subprocess_pid=901,
                    started_at=1.0,
                    completed_at=2.0,
                    duration_seconds=1.0,
                    status="success",
                    raw_text="done",
                    parsed={"summary": "done"},
                )
            )
            return AgenticRunResult(status="success", summary="done", raw_text="done")

        async def close(self) -> None:
            return None

    class _Provider:
        def __init__(self, observer=None) -> None:
            self._observer = observer
            self.allowed_tool_names: tuple[str, ...] | None = None

        def with_allowed_tool_names(self, allowed_tool_names: tuple[str, ...]):
            provider = _Provider(self._observer)
            provider.allowed_tool_names = allowed_tool_names
            return provider

        def with_observer(self, observer):
            return _Provider(observer)

        async def open_agent_session(self, *, allowed_tool_names=None, tools=None):
            del allowed_tool_names, tools
            return _Session(self._observer)

    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    provider = InstrumentedAIProvider(
        _Provider(),
        usage_repository=repository,
        provider_key="copilot-sdk",
        provider_name="Copilot SDK",
        model_name="gpt-5.4-mini",
    ).with_usage_context(
        task_name="memory-curator",
        task_id="packet-attributed-task",
        execution_epoch=9,
        workspace_id="workspace-a",
        curation_packet_id="packet-123",
    ).with_route_context(
        provider_profile="copilot-strong",
        route_available=True,
    ).with_allowed_tool_names(("internal_update_memory_record",))

    session = await provider.open_agent_session()
    await session.run_agent("inspect packet")
    await session.close()

    usage = db_manager.get_connection().execute(
        "SELECT request_id, curation_packet_id FROM provider_usage WHERE task_id = ?",
        ("packet-attributed-task",),
    ).fetchone()
    conversation = db_manager.get_connection().execute(
        "SELECT curation_packet_id FROM ai_conversations WHERE task_id = ?",
        ("packet-attributed-task",),
    ).fetchone()

    assert usage["curation_packet_id"] == "packet-123"
    assert conversation["curation_packet_id"] == "packet-123"


@pytest.mark.asyncio
async def test_instrumented_provider_leaves_packet_id_null_for_ordinary_calls(db_manager) -> None:
    """Keep ordinary provider usage and conversation rows unattributed to curator packets."""
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
                    subprocess_pid=902,
                    started_at=3.0,
                )
            )
            self._observer(
                ProviderAttemptFinishedEvent(
                    attempt=1,
                    prompt=prompt,
                    subprocess_pid=902,
                    started_at=3.0,
                    completed_at=4.0,
                    duration_seconds=1.0,
                    status="success",
                    raw_text='{"ok": true}',
                    parsed={"ok": True},
                )
            )
            return {"ok": True}

    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    provider = InstrumentedAIProvider(
        _Provider(),
        usage_repository=repository,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
    ).with_usage_context(
        task_name="memory-ingest",
        task_id="ordinary-provider-task",
        workspace_id="workspace-a",
    )

    await provider.ask("ordinary request")

    usage = db_manager.get_connection().execute(
        "SELECT curation_packet_id FROM provider_usage WHERE task_id = ?",
        ("ordinary-provider-task",),
    ).fetchone()
    conversation = db_manager.get_connection().execute(
        "SELECT curation_packet_id FROM ai_conversations WHERE task_id = ?",
        ("ordinary-provider-task",),
    ).fetchone()

    assert usage["curation_packet_id"] is None
    assert conversation["curation_packet_id"] is None
