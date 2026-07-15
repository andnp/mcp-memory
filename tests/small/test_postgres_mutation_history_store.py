from __future__ import annotations

import sqlite3
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from mcp_memory.mutation_history import (
    LinkRevision,
    MutationActorKind,
    MutationEvent,
    MutationEventStatus,
    Protection,
    ProtectionMode,
    RecordRevision,
    RestoreRequest,
    RestoreResult,
    RestoreResultStatus,
    RevisionRole,
)
from mcp_memory.storage.postgres_migrations import POSTGRES_MIGRATIONS, POSTGRES_SCHEMA_VERSION
from mcp_memory.storage.postgres_mutation_history_store import PostgresMutationHistoryStore
from mcp_memory.mutation_history_store import MutationHistoryTerminalizationError


pytestmark = pytest.mark.small

MEMORY_ID = UUID("00000000-0000-0000-0000-000000000001")
OTHER_MEMORY_ID = UUID("00000000-0000-0000-0000-000000000002")


class FakePostgresCursor:
    def __init__(self, connection: FakePostgresConnection) -> None:
        self._connection = connection
        self._cursor = connection._database.cursor()

    def __enter__(self) -> FakePostgresCursor:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._cursor.close()
        return False

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        self._cursor.execute(_sqlite_query(query), tuple(() if params is None else params))

    def executemany(self, query: str, rows: list[tuple[object, ...]]) -> None:
        self._cursor.executemany(_sqlite_query(query), rows)

    def fetchone(self) -> tuple[object, ...] | None:
        return self._cursor.fetchone()

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._cursor.fetchall()


class FakePostgresConnection:
    def __init__(self) -> None:
        self._database = sqlite3.connect(":memory:")
        self._database.execute("PRAGMA foreign_keys = ON")
        self._database.executescript(
            """
            CREATE TABLE memory_mutation_events (
                id TEXT PRIMARY KEY, operation TEXT NOT NULL, actor_kind TEXT NOT NULL,
                actor_id TEXT, family TEXT, task_id TEXT, curation_run_id TEXT,
                plan_id TEXT, action_id TEXT, provider_id TEXT, reason_code TEXT,
                rationale TEXT, policy_version TEXT, schema_version INTEGER NOT NULL,
                status TEXT NOT NULL, restores_event_id TEXT REFERENCES memory_mutation_events(id),
                idempotency_key TEXT, created_at TEXT NOT NULL, terminalized_at TEXT
            );
            CREATE UNIQUE INDEX uq_events_action ON memory_mutation_events(curation_run_id, action_id);
            CREATE UNIQUE INDEX uq_events_idempotency ON memory_mutation_events(idempotency_key);
            CREATE TABLE memory_record_revisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL,
                memory_id TEXT NOT NULL, role TEXT NOT NULL, before_exists BOOLEAN NOT NULL,
                before_snapshot TEXT, after_exists BOOLEAN NOT NULL, after_snapshot TEXT,
                before_token TEXT, after_token TEXT,
                UNIQUE(event_id, memory_id, role),
                FOREIGN KEY(event_id) REFERENCES memory_mutation_events(id)
            );
            CREATE TABLE memory_link_revisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL,
                source_id TEXT NOT NULL, target_id TEXT NOT NULL, link_type TEXT NOT NULL,
                context TEXT, before_exists BOOLEAN NOT NULL, after_exists BOOLEAN NOT NULL,
                UNIQUE(event_id, source_id, target_id, link_type),
                FOREIGN KEY(event_id) REFERENCES memory_mutation_events(id)
            );
            CREATE TABLE memory_protections (
                memory_id TEXT NOT NULL, mode TEXT NOT NULL, reason TEXT NOT NULL,
                actor_id TEXT, created_at TEXT NOT NULL, expires_at TEXT,
                PRIMARY KEY(memory_id, mode)
            );
            CREATE TABLE memory_restore_requests (
                id TEXT PRIMARY KEY, target_event_id TEXT NOT NULL, scope TEXT NOT NULL,
                expected_record_tokens TEXT NOT NULL, expected_link_tokens TEXT NOT NULL,
                actor_id TEXT, reason TEXT NOT NULL, idempotency_key TEXT NOT NULL UNIQUE,
                confirmation BOOLEAN NOT NULL, status TEXT NOT NULL, event_id TEXT,
                conflict_reason TEXT, conflict_details TEXT NOT NULL, created_at TEXT NOT NULL,
                terminalized_at TEXT, FOREIGN KEY(target_event_id) REFERENCES memory_mutation_events(id)
            );
            """
        )

    def cursor(self) -> FakePostgresCursor:
        return FakePostgresCursor(self)

    def commit(self) -> None:
        self._database.commit()

    def rollback(self) -> None:
        self._database.rollback()


class FakePostgresLease:
    def __init__(self, connection: FakePostgresConnection) -> None:
        self._connection = connection

    def __enter__(self) -> FakePostgresConnection:
        return self._connection

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class FakePostgresSessionManager:
    def __init__(self) -> None:
        self.connection = FakePostgresConnection()

    def open_connection(self) -> FakePostgresLease:
        return FakePostgresLease(self.connection)


def _sqlite_query(query: str) -> str:
    return query.replace("'{}'::jsonb", "'{}'").replace("%s::jsonb", "?").replace("%s", "?")


def _event(*, run_id: UUID | None = None, action_id: UUID | None = None) -> MutationEvent:
    return MutationEvent(
        id=uuid4(),
        operation="normalize_memory",
        actor_kind=MutationActorKind.MAINTENANCE,
        curation_run_id=run_id,
        action_id=action_id,
        status=MutationEventStatus.APPLIED,
    )


def test_postgres_history_store_round_trips_history_protection_and_idempotency() -> None:
    store = PostgresMutationHistoryStore(cast(Any, FakePostgresSessionManager()))
    event = store.append_event(_event(run_id=uuid4(), action_id=uuid4()))
    store.append_record_revisions(
        [
            RecordRevision(
                event_id=event.id,
                memory_id=MEMORY_ID,
                role=RevisionRole.TARGET,
                before_exists=True,
                before_snapshot={"content": "before"},
                after_exists=True,
                after_snapshot={"content": "after"},
                before_token="before-token",
                after_token="after-token",
            )
        ]
    )
    store.append_link_revisions(
        [
            LinkRevision(
                event_id=event.id,
                source_id=MEMORY_ID,
                target_id=OTHER_MEMORY_ID,
                link_type="supports",
                context="context",
                before_exists=False,
                after_exists=True,
            )
        ]
    )
    protection = store.set_protection(
        Protection(memory_id=MEMORY_ID, mode=ProtectionMode.PINNED_ACTIVE, reason="important")
    )

    assert store.get_record_revisions(event.id)[0].after_snapshot == {"content": "after"}
    assert store.get_link_revisions(event.id)[0].after_exists is True
    assert store.list_events(memory_id=MEMORY_ID) == [event]
    assert store.get_protections(MEMORY_ID) == [protection]

    replay = store.append_event(event.model_copy(update={"id": uuid4()}))
    assert replay.id == event.id
    request = RestoreRequest(target_event_id=event.id, reason="undo", idempotency_key="restore-1")
    first_restore = store.request_restore(request)
    replay_restore = store.request_restore(request)
    assert first_restore.status is RestoreResultStatus.APPLIED
    assert replay_restore.status is RestoreResultStatus.ALREADY_APPLIED
    assert replay_restore.request_id == first_restore.request_id


def test_postgres_history_store_first_terminal_write_wins() -> None:
    store = PostgresMutationHistoryStore(cast(Any, FakePostgresSessionManager()))
    event = store.append_event(_event())

    terminal = store.terminalize_event(event.id, MutationEventStatus.FAILED)
    assert store.terminalize_event(event.id, MutationEventStatus.FAILED) == terminal
    with pytest.raises(MutationHistoryTerminalizationError):
        store.terminalize_event(event.id, MutationEventStatus.STALE)

    request = store.request_restore(
        RestoreRequest(target_event_id=event.id, reason="undo", idempotency_key="restore-2")
    )
    assert request.request_id is not None
    request_id = request.request_id
    result = store.terminalize_restore_request(
        request_id,
        RestoreResult(status=RestoreResultStatus.CONFLICT, target_event_id=event.id, conflict_reason="changed"),
    )
    assert result.status is RestoreResultStatus.CONFLICT
    with pytest.raises(MutationHistoryTerminalizationError):
        store.terminalize_restore_request(
            request_id,
            RestoreResult(status=RestoreResultStatus.APPLIED, target_event_id=event.id),
        )


def test_postgres_mutation_history_migration_is_latest_and_additive() -> None:
    migration = POSTGRES_MIGRATIONS[-1]
    assert migration.version == POSTGRES_SCHEMA_VERSION == 11
    assert migration.name == "add_mutation_history_and_protections"
    statements = " ".join(migration.statements)
    for table in (
        "memory_mutation_events",
        "memory_record_revisions",
        "memory_link_revisions",
        "memory_protections",
        "memory_restore_requests",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in statements
