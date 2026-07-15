import sqlite3
from pathlib import Path
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
from mcp_memory.mutation_history_store import (
    MutationHistoryTerminalizationError,
    SQLiteMutationHistoryStore,
)
from mcp_memory.utils.db import DatabaseManager, SCHEMA_VERSION
from mcp_memory.utils.db_schema import (
    apply_legacy_additive_migrations,
    create_current_schema,
    finalize_schema_setup,
)


pytestmark = pytest.mark.small

MEMORY_ID = UUID("00000000-0000-0000-0000-000000000001")
OTHER_MEMORY_ID = UUID("00000000-0000-0000-0000-000000000002")


def _event(*, run_id: UUID | None = None, action_id: UUID | None = None, status: MutationEventStatus = MutationEventStatus.APPLIED) -> MutationEvent:
    return MutationEvent(
        id=uuid4(),
        operation="normalize_memory",
        actor_kind=MutationActorKind.MAINTENANCE,
        curation_run_id=run_id,
        action_id=action_id,
        status=status,
    )


def test_fresh_schema_contains_additive_history_tables_and_indexes(tmp_path: Path) -> None:
    path = tmp_path / "fresh.db"
    conn = sqlite3.connect(path)
    try:
        create_current_schema(conn)
        finalize_schema_setup(conn)

        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        assert {
            "memory_mutation_events",
            "memory_record_revisions",
            "memory_link_revisions",
            "memory_protections",
            "memory_restore_requests",
        } <= tables
        indexes = {
            row[1]
            for row in conn.execute("PRAGMA index_list(memory_mutation_events)").fetchall()
        }
        assert "uq_memory_mutation_events_action" in indexes
        assert conn.execute(
            "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
        ).fetchone() == (str(SCHEMA_VERSION),)
    finally:
        conn.close()


def test_legacy_schema_migration_adds_history_tables_without_requiring_data(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    try:
        create_current_schema(conn)
        conn.executescript(
            """
            DROP TABLE memory_restore_requests;
            DROP TABLE memory_protections;
            DROP TABLE memory_link_revisions;
            DROP TABLE memory_record_revisions;
            DROP TABLE memory_mutation_events;
            """
        )
        apply_legacy_additive_migrations(conn)
        finalize_schema_setup(conn)
        assert conn.execute(
            "SELECT COUNT(*) FROM memory_mutation_events"
        ).fetchone() == (0,)
    finally:
        conn.close()


def test_store_round_trips_event_revisions_and_bounded_memory_history(db_manager: DatabaseManager) -> None:
    store = SQLiteMutationHistoryStore(db_manager)
    run_id = uuid4()
    action_id = uuid4()
    event = store.append_event(_event(run_id=run_id, action_id=action_id))
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

    assert store.get_event_by_action(run_id, action_id) == event
    assert store.get_record_revisions(event.id)[0].after_snapshot == {"content": "after"}
    assert store.get_link_revisions(event.id)[0].after_exists
    assert store.list_events(memory_id=MEMORY_ID, limit=10) == [event]
    with pytest.raises(ValueError):
        store.list_events(limit=0)


def test_action_event_and_restore_request_replays_are_idempotent(db_manager: DatabaseManager) -> None:
    store = SQLiteMutationHistoryStore(db_manager)
    run_id = uuid4()
    action_id = uuid4()
    event = _event(run_id=run_id, action_id=action_id)

    assert store.append_event(event).id == event.id
    assert store.append_event(event.model_copy(update={"id": uuid4()})).id == event.id

    request = RestoreRequest(
        target_event_id=event.id,
        reason="undo test",
        idempotency_key="restore-1",
    )
    first = store.request_restore(request)
    replay = store.request_restore(request)
    assert first.request_id is not None
    assert first.status is RestoreResultStatus.APPLIED
    assert replay.status is RestoreResultStatus.ALREADY_APPLIED
    assert replay.request_id == first.request_id


def test_event_terminalization_is_first_write_wins(db_manager: DatabaseManager) -> None:
    store = SQLiteMutationHistoryStore(db_manager)
    event = store.append_event(_event(status=MutationEventStatus.APPLIED))

    terminal = store.terminalize_event(event.id, MutationEventStatus.FAILED)
    assert terminal.status is MutationEventStatus.FAILED
    assert store.terminalize_event(event.id, MutationEventStatus.FAILED) == terminal
    with pytest.raises(MutationHistoryTerminalizationError):
        store.terminalize_event(event.id, MutationEventStatus.STALE)
    stored = store.get_event(event.id)
    assert stored is not None
    assert stored.status is MutationEventStatus.FAILED


def test_protection_crud_and_restore_terminalization_are_conditional(db_manager: DatabaseManager) -> None:
    store = SQLiteMutationHistoryStore(db_manager)
    event = store.append_event(_event())
    protection = store.set_protection(
        Protection(
            memory_id=MEMORY_ID,
            mode=ProtectionMode.PINNED_ACTIVE,
            reason="important",
        )
    )
    assert store.get_protections(MEMORY_ID) == [protection]
    store.remove_protection(MEMORY_ID, ProtectionMode.PINNED_ACTIVE)
    assert store.get_protections(MEMORY_ID) == []

    request_result = store.request_restore(
        RestoreRequest(target_event_id=event.id, reason="undo", idempotency_key="restore-2")
    )
    conflict = RestoreResult(
        status=RestoreResultStatus.CONFLICT,
        target_event_id=event.id,
        conflict_reason="changed",
    )
    assert request_result.request_id is not None
    request_id = request_result.request_id
    terminal = store.terminalize_restore_request(request_id, conflict)
    assert terminal.status is RestoreResultStatus.CONFLICT
    with pytest.raises(MutationHistoryTerminalizationError):
        store.terminalize_restore_request(
            request_id,
            RestoreResult(status=RestoreResultStatus.APPLIED, target_event_id=event.id),
        )
