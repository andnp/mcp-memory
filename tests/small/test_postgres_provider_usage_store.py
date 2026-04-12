from __future__ import annotations

from collections.abc import Sequence
import json

import pytest

from mcp_memory.storage.postgres_provider_usage_store import PostgresProviderUsageRepository
from tests.small.provider_usage_conversation_contract import (
    assert_preserves_first_terminal_conversation_finalization,
)


pytestmark = pytest.mark.small


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Expected int-compatible value, got {type(value)!r}")


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected float-compatible value, got {type(value)!r}")


class FakeProviderUsageState:
    def __init__(self) -> None:
        self.provider_usage: list[dict[str, object]] = []
        self.ai_conversations: list[dict[str, object]] = []
        self.provider_admission_state: dict[tuple[str, str], dict[str, object]] = {}
        self.next_conversation_id = 1


class FakeCursor:
    def __init__(self, state: FakeProviderUsageState) -> None:
        self._state = state
        self._result: list[tuple[object, ...]] = []
        self.rowcount = 0

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        normalized = " ".join(query.split())
        arguments = tuple(() if params is None else params)
        self.rowcount = 0
        if normalized.startswith("INSERT INTO provider_usage"):
            self._state.provider_usage.append(
                {
                    "workspace_id": arguments[0],
                    "task_name": arguments[1],
                    "task_id": arguments[2],
                    "request_id": arguments[3],
                    "subprocess_pid": arguments[4],
                    "provider_key": arguments[5],
                    "provider_name": arguments[6],
                    "model_name": arguments[7],
                    "status": arguments[8],
                    "duration_seconds": arguments[9],
                    "created_at": arguments[10],
                    "error_text": arguments[11],
                    "reason_category": arguments[12],
                    "reason_code": arguments[13],
                    "retry_delay_seconds": arguments[14],
                }
            )
            return
        if normalized.startswith("INSERT INTO ai_conversations"):
            request_id = str(arguments[0])
            attempt = _as_int(arguments[1])
            existing = next(
                (row for row in self._state.ai_conversations if str(row["request_id"]) == request_id and _as_int(row["attempt"]) == attempt),
                None,
            )
            payload = {
                "request_id": request_id,
                "attempt": attempt,
                "workspace_id": arguments[2],
                "task_name": arguments[3],
                "task_id": arguments[4],
                "provider_key": arguments[5],
                "provider_name": arguments[6],
                "model_name": arguments[7],
                "subprocess_pid": arguments[8],
                "prompt_text": arguments[9],
                "response_text": arguments[10],
                "parsed_json": arguments[11],
                "status": arguments[12],
                "error_text": arguments[13],
                "reason_category": arguments[14],
                "reason_code": arguments[15],
                "retry_delay_seconds": arguments[16],
                "started_at": arguments[17],
                "completed_at": arguments[18],
                "duration_seconds": arguments[19],
            }
            if existing is None:
                payload["id"] = self._state.next_conversation_id
                self._state.next_conversation_id += 1
                self._state.ai_conversations.append(payload)
            else:
                existing.update(payload)
            return
        if normalized.startswith("SELECT id, request_id, attempt, workspace_id, task_name, task_id, provider_key, provider_name, model_name, subprocess_pid, prompt_text, response_text, parsed_json, status, error_text, reason_category, reason_code, retry_delay_seconds, started_at, completed_at, duration_seconds FROM ai_conversations"):
            rows = list(self._state.ai_conversations)
            clauses = normalized.split(" WHERE ", 1)[1].split(" ORDER BY ", 1)[0] if " WHERE " in normalized else ""
            params_list = list(arguments)
            limit = _as_int(params_list.pop())
            rows = self._filter_conversations(rows, clauses, params_list)
            rows.sort(key=lambda row: (_as_float(row["completed_at"]), _as_int(row["id"])), reverse=True)
            self._result = [self._conversation_row(row) for row in rows[:limit]]
            return
        if normalized.startswith("UPDATE ai_conversations SET completed_at = %s"):
            request_id = str(arguments[4])
            count = 0
            for row in self._state.ai_conversations:
                if str(row["request_id"]) != request_id or str(row["status"]) != "running":
                    continue
                row["completed_at"] = arguments[0]
                row["duration_seconds"] = max(_as_float(arguments[1]) - _as_float(row["started_at"]), 0.0)
                if not bool(arguments[2]):
                    row["subprocess_pid"] = arguments[3]
                count += 1
            self.rowcount = count
            return
        if normalized.startswith("UPDATE ai_conversations SET status = %s"):
            count = 0
            if "WHERE request_id = %s AND status = 'running'" in normalized:
                match_key = "request_id"
                match_value = str(arguments[11])
            elif "WHERE task_id = %s AND status = 'running'" in normalized:
                match_key = "task_id"
                match_value = str(arguments[11])
            else:
                raise AssertionError(f"Unhandled finalize query: {normalized}")
            for row in self._state.ai_conversations:
                if str(row.get(match_key)) != match_value or str(row["status"]) != "running":
                    continue
                row["status"] = arguments[0]
                if arguments[1] is not None:
                    row["error_text"] = arguments[1]
                if arguments[2] is not None:
                    row["reason_category"] = arguments[2]
                if arguments[3] is not None:
                    row["reason_code"] = arguments[3]
                if arguments[4] is not None:
                    row["retry_delay_seconds"] = arguments[4]
                row["completed_at"] = arguments[5]
                row["duration_seconds"] = max(_as_float(arguments[6]) - _as_float(row["started_at"]), 0.0)
                if not bool(arguments[7]):
                    row["response_text"] = arguments[8]
                if not bool(arguments[9]):
                    row["parsed_json"] = arguments[10]
                count += 1
            self.rowcount = count
            return
        if normalized.startswith("SELECT task_name, provider_key, provider_name, model_name, status, duration_seconds, created_at, reason_code FROM provider_usage"):
            rows = list(self._state.provider_usage)
            if " WHERE workspace_id = %s" in normalized:
                workspace_id = arguments[0]
                rows = [row for row in rows if row["workspace_id"] == workspace_id]
            self._result = [
                (
                    row["task_name"],
                    row["provider_key"],
                    row["provider_name"],
                    row["model_name"],
                    row["status"],
                    row["duration_seconds"],
                    row["created_at"],
                    row["reason_code"],
                )
                for row in rows
            ]
            return
        if normalized.startswith("INSERT INTO provider_admission_state"):
            self._state.provider_admission_state[(str(arguments[0]), str(arguments[1]))] = {
                "provider_key": arguments[0],
                "model_name": arguments[1],
                "reason_category": arguments[2],
                "reason_code": arguments[3],
                "error_text": arguments[4],
                "retry_delay_seconds": arguments[5],
                "active_until": arguments[6],
                "updated_at": arguments[7],
            }
            return
        if normalized == "DELETE FROM provider_admission_state WHERE provider_key = %s AND model_name = %s":
            self._state.provider_admission_state.pop((str(arguments[0]), str(arguments[1])), None)
            return
        if normalized.startswith("SELECT provider_key, model_name, reason_category, reason_code, error_text, retry_delay_seconds, active_until, updated_at FROM provider_admission_state WHERE active_until > %s"):
            rows = list(self._state.provider_admission_state.values())
            params_list = list(arguments)
            current_time = _as_float(params_list.pop(0))
            rows = [row for row in rows if _as_float(row["active_until"]) > current_time]
            clauses = normalized.split(" WHERE ", 1)[1].split(" ORDER BY ", 1)[0].replace("active_until > %s AND ", "")
            if clauses:
                for clause in clauses.split(" AND "):
                    if clause == "provider_key = %s":
                        value = str(params_list.pop(0))
                        rows = [row for row in rows if str(row["provider_key"]) == value]
                    elif clause == "model_name = %s":
                        value = str(params_list.pop(0))
                        rows = [row for row in rows if str(row["model_name"]) == value]
            rows.sort(key=lambda row: (_as_float(row["active_until"]), _as_float(row["updated_at"])), reverse=True)
            self._result = [
                (
                    row["provider_key"],
                    row["model_name"],
                    row["reason_category"],
                    row["reason_code"],
                    row["error_text"],
                    row["retry_delay_seconds"],
                    row["active_until"],
                    row["updated_at"],
                )
                for row in rows
            ]
            return
        if normalized.startswith("SELECT COUNT(*) FROM provider_usage WHERE provider_key IN ("):
            rows = [row for row in self._state.provider_usage if str(row["status"]) != "skipped"]
            placeholder_segment = normalized.split("provider_key IN (", 1)[1].split(")", 1)[0]
            provider_key_count = placeholder_segment.count("%s")
            provider_keys = {str(value) for value in arguments[:provider_key_count]}
            cutoff = _as_float(arguments[provider_key_count])
            rows = [row for row in rows if str(row["provider_key"]) in provider_keys and _as_float(row["created_at"]) >= cutoff]
            if normalized.endswith("AND workspace_id = %s"):
                workspace_id = arguments[provider_key_count + 1]
                rows = [row for row in rows if row["workspace_id"] == workspace_id]
            self._result = [(len(rows),)]
            return
        if normalized.startswith("SELECT COUNT(*) FROM provider_usage WHERE model_name IN ("):
            rows = [row for row in self._state.provider_usage if str(row["status"]) != "skipped"]
            placeholder_segment = normalized.split("model_name IN (", 1)[1].split(")", 1)[0]
            model_count = placeholder_segment.count("%s")
            model_names = {str(value) for value in arguments[:model_count]}
            cutoff = _as_float(arguments[model_count])
            rows = [row for row in rows if str(row["model_name"]) in model_names and _as_float(row["created_at"]) >= cutoff]
            if normalized.endswith("AND workspace_id = %s"):
                workspace_id = arguments[model_count + 1]
                rows = [row for row in rows if row["workspace_id"] == workspace_id]
            self._result = [(len(rows),)]
            return
        raise AssertionError(f"Unhandled query: {normalized}")

    def executemany(self, query: str, rows: Sequence[tuple[object, ...]]) -> None:
        for row in rows:
            self.execute(query, tuple(row))

    def fetchone(self) -> tuple[object, ...] | None:
        return None if not self._result else self._result[0]

    def fetchall(self) -> list[tuple[object, ...]]:
        return list(self._result)

    def _filter_conversations(self, rows: list[dict[str, object]], clauses: str, params: list[object]) -> list[dict[str, object]]:
        filtered = rows
        if not clauses:
            return filtered
        for clause in clauses.split(" AND "):
            if clause == "workspace_id = %s":
                value = params.pop(0)
                filtered = [row for row in filtered if row["workspace_id"] == value]
            elif clause == "request_id = %s":
                value = str(params.pop(0))
                filtered = [row for row in filtered if str(row["request_id"]) == value]
            elif clause == "task_name = %s":
                value = params.pop(0)
                filtered = [row for row in filtered if row["task_name"] == value]
            elif clause == "status = %s":
                value = str(params.pop(0))
                filtered = [row for row in filtered if str(row["status"]) == value]
        return filtered

    def _conversation_row(self, row: dict[str, object]) -> tuple[object, ...]:
        return (
            row["id"],
            row["request_id"],
            row["attempt"],
            row["workspace_id"],
            row["task_name"],
            row["task_id"],
            row["provider_key"],
            row["provider_name"],
            row["model_name"],
            row["subprocess_pid"],
            row["prompt_text"],
            row["response_text"],
            row["parsed_json"],
            row["status"],
            row["error_text"],
            row["reason_category"],
            row["reason_code"],
            row["retry_delay_seconds"],
            row["started_at"],
            row["completed_at"],
            row["duration_seconds"],
        )


class FakeConnection:
    def __init__(self, state: FakeProviderUsageState) -> None:
        self._state = state

    def cursor(self) -> FakeCursor:
        return FakeCursor(self._state)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class FakeLease:
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection

    def __enter__(self) -> FakeConnection:
        return self._connection

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self._connection.rollback()
        return False

    def close(self) -> None:
        return None


class FakeSessionManager:
    def __init__(self) -> None:
        self.state = FakeProviderUsageState()

    def __enter__(self) -> FakeSessionManager:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def open_connection(self) -> FakeLease:
        return FakeLease(FakeConnection(self.state))

    def close(self) -> None:
        return None


def test_postgres_provider_usage_repository_summarizes_usage_and_counts_calls() -> None:
    session_manager = FakeSessionManager()
    workspace_a = PostgresProviderUsageRepository(session_manager, workspace_id="workspace-a")
    workspace_b = PostgresProviderUsageRepository(session_manager, workspace_id="workspace-b")

    workspace_a.record_call(
        task_name="memory-curator",
        task_id="task-a",
        request_id="req-1",
        subprocess_pid=123,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-2.5-pro",
        status="success",
        duration_seconds=1.5,
        created_at=9_900.0,
        error_text=None,
    )
    workspace_a.record_call(
        task_name="memory-curator",
        task_id="task-a",
        request_id="req-2",
        subprocess_pid=123,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-2.5-pro",
        status="error",
        duration_seconds=2.5,
        created_at=9_950.0,
        error_text="boom",
        reason_category="provider",
        reason_code="rate_limited",
        retry_delay_seconds=30.0,
    )
    workspace_a.record_call(
        task_name="memory-curator",
        task_id="task-a",
        request_id="req-3",
        subprocess_pid=123,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-2.5-pro",
        status="skipped",
        duration_seconds=0.0,
        created_at=9_960.0,
        error_text=None,
        reason_category="provider",
        reason_code="provider_disabled",
    )
    workspace_b.record_call(
        task_name="memory-curator",
        task_id="task-b",
        request_id="req-4",
        subprocess_pid=456,
        provider_key="claude-code",
        provider_name="Claude Code",
        model_name="sonnet",
        status="success",
        duration_seconds=4.0,
        created_at=9_970.0,
        error_text=None,
    )
    workspace_a.upsert_admission_state(
        provider_key="gemini-cli",
        model_name="gemini-2.5-pro",
        reason_category="provider",
        reason_code="rate_limited",
        error_text="burst exceeded",
        retry_delay_seconds=45.0,
        active_until=10_050.0,
        updated_at=10_000.0,
    )

    summaries = workspace_a.summarize_usage(workspace_id="workspace-a", now=10_000.0)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.calls_last_hour == 2
    assert summary.failures_last_day == 1
    assert summary.skips_last_day == 1
    assert summary.avg_duration_last_day == pytest.approx(2.0)
    assert summary.top_failure_reason_last_day == "rate_limited"
    assert summary.top_skip_reason_last_day == "provider_disabled"
    assert summary.active_admission_reason == "rate_limited"
    assert summary.active_retry_delay_seconds == pytest.approx(50.0)
    assert workspace_a.count_recent_calls(provider_keys=["gemini-cli"], now=10_000.0, window_seconds=200.0) == 2
    assert workspace_a.count_recent_model_calls(model_names=["gemini-2.5-pro"], now=10_000.0, window_seconds=200.0) == 2


def test_postgres_provider_usage_repository_tracks_conversation_lifecycle() -> None:
    session_manager = FakeSessionManager()
    repository = PostgresProviderUsageRepository(session_manager, workspace_id="workspace-a")

    repository.record_conversation(
        request_id="req-running",
        attempt=1,
        task_name="memory-curator",
        task_id="task-1",
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-2.5-pro",
        subprocess_pid=111,
        prompt_text="hello",
        response_text="",
        parsed=None,
        status="running",
        error_text=None,
        started_at=100.0,
        completed_at=100.0,
    )

    touched = repository.touch_running_conversation(request_id="req-running", completed_at=102.0, subprocess_pid=222)
    finalized = repository.finalize_running_conversation(
        request_id="req-running",
        status="success",
        error_text=None,
        completed_at=105.0,
        response_text="world",
        parsed={"answer": 42},
    )
    duplicate_finalize = repository.finalize_running_conversation(
        request_id="req-running",
        status="error",
        error_text="too late",
        completed_at=120.0,
    )
    conversation = repository.get_conversation("req-running")[0]

    assert touched == 1
    assert finalized == 1
    assert duplicate_finalize == 0
    assert conversation.status == "success"
    assert conversation.subprocess_pid == 222
    assert conversation.response_text == "world"
    assert conversation.parsed == {"answer": 42}
    assert conversation.duration_seconds == pytest.approx(5.0)


def test_postgres_provider_usage_repository_preserves_first_terminal_conversation_finalization() -> None:
    assert_preserves_first_terminal_conversation_finalization(
        lambda: PostgresProviderUsageRepository(FakeSessionManager(), workspace_id="workspace-a")
    )


def test_postgres_provider_usage_repository_reconciles_running_task_conversations_and_filters_active_states() -> None:
    session_manager = FakeSessionManager()
    repository = PostgresProviderUsageRepository(session_manager, workspace_id="workspace-a")

    repository.record_conversation(
        request_id="req-1",
        attempt=1,
        task_name="memory-curator",
        task_id="task-stuck",
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-2.5-pro",
        subprocess_pid=111,
        prompt_text="hello",
        response_text="",
        parsed=None,
        status="running",
        error_text=None,
        started_at=100.0,
        completed_at=100.0,
    )
    repository.record_conversation(
        request_id="req-2",
        attempt=2,
        task_name="memory-curator",
        task_id="task-stuck",
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-2.5-pro",
        subprocess_pid=222,
        prompt_text="retry",
        response_text="",
        parsed=None,
        status="running",
        error_text=None,
        started_at=101.0,
        completed_at=101.0,
    )
    repository.upsert_admission_state(
        provider_key="gemini-cli",
        model_name="gemini-2.5-pro",
        reason_category="provider",
        reason_code="rate_limited",
        error_text="wait",
        retry_delay_seconds=60.0,
        active_until=200.0,
        updated_at=150.0,
    )
    repository.upsert_admission_state(
        provider_key="claude-code",
        model_name="sonnet",
        reason_category="provider",
        reason_code="disabled",
        error_text=None,
        retry_delay_seconds=None,
        active_until=90.0,
        updated_at=80.0,
    )

    reconciled = repository.reconcile_running_task_conversations(
        task_id="task-stuck",
        status="error",
        error_text="worker vanished",
        reason_category="runtime",
        reason_code="worker_missing",
        retry_delay_seconds=15.0,
        completed_at=110.0,
    )
    conversations = repository.list_conversations(task_name="memory-curator", status="error", limit=10)
    active_states = repository.list_active_admission_states(now=100.0)
    active_state = repository.get_active_admission_state(provider_key="gemini-cli", model_name="gemini-2.5-pro", now=100.0)
    repository.clear_admission_state(provider_key="gemini-cli", model_name="gemini-2.5-pro")

    assert reconciled == 2
    assert [conversation.request_id for conversation in conversations] == ["req-2", "req-1"]
    assert all(conversation.reason_code == "worker_missing" for conversation in conversations)
    assert len(active_states) == 1
    assert active_state is not None
    assert active_state.reason_code == "rate_limited"
    assert repository.get_active_admission_state(provider_key="gemini-cli", model_name="gemini-2.5-pro", now=100.0) is None
    stored_payload = session_manager.state.ai_conversations[0]["parsed_json"]
    assert stored_payload is None or isinstance(stored_payload, str)
    if isinstance(stored_payload, str):
        assert isinstance(json.loads(stored_payload), dict)
