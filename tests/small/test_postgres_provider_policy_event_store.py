from __future__ import annotations

import json
from typing import Any, cast

import pytest

from mcp_memory.storage.postgres_provider_policy_event_store import PostgresProviderPolicyEventRepository


pytestmark = pytest.mark.small


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected float-compatible value, got {type(value)!r}")


class FakeProviderPolicyEventState:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []


class FakeCursor:
    def __init__(self, state: FakeProviderPolicyEventState) -> None:
        self._state = state

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        normalized = " ".join(query.split())
        arguments = tuple(() if params is None else params)
        if normalized.startswith("INSERT INTO provider_policy_events"):
            self._state.events.append(
                {
                    "workspace_id": arguments[0],
                    "task_name": arguments[1],
                    "task_id": arguments[2],
                    "event_kind": arguments[3],
                    "warning_kind": arguments[4],
                    "provider_key": arguments[5],
                    "provider_name": arguments[6],
                    "model_name": arguments[7],
                    "route_key": arguments[8],
                    "candidate_routes_json": arguments[9],
                    "reason_category": arguments[10],
                    "reason_code": arguments[11],
                    "retry_delay_seconds": arguments[12],
                    "warning_suppressed": arguments[13],
                    "created_at": arguments[14],
                }
            )
            return
        raise AssertionError(f"Unhandled query: {normalized}")


class FakeConnection:
    def __init__(self, state: FakeProviderPolicyEventState) -> None:
        self._state = state
        self.commits = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self._state)

    def commit(self) -> None:
        self.commits += 1

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
        self.state = FakeProviderPolicyEventState()
        self.connection = FakeConnection(self.state)

    def __enter__(self) -> FakeSessionManager:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def open_connection(self) -> FakeLease:
        return FakeLease(self.connection)

    def close(self) -> None:
        return None


def test_postgres_provider_policy_event_repository_records_structured_events() -> None:
    session_manager = FakeSessionManager()
    repository = PostgresProviderPolicyEventRepository(cast(Any, session_manager), workspace_id="workspace-a")

    repository.record_event(
        task_name="graph-linker",
        task_id="task-1",
        event_kind="route_exhausted",
        warning_kind="routes_exhausted",
        provider_key="gemini-cheap",
        provider_name="Gemini CLI",
        model_name="gemini-2.5-pro",
        route_key="gemini-cheap",
        candidate_routes=["gemini-cheap", "copilot-mini"],
        reason_category="provider",
        reason_code="rate_limited",
        retry_delay_seconds=30.0,
        warning_suppressed=True,
        created_at=1234.5,
    )

    assert session_manager.connection.commits == 1
    assert len(session_manager.state.events) == 1
    stored = session_manager.state.events[0]
    assert stored["workspace_id"] == "workspace-a"
    assert stored["event_kind"] == "route_exhausted"
    assert stored["warning_suppressed"] is True
    assert _as_float(stored["created_at"]) == pytest.approx(1234.5)
    assert json.loads(str(stored["candidate_routes_json"])) == ["gemini-cheap", "copilot-mini"]
