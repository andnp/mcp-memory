from datetime import datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

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
    RestoreScope,
    RevisionRole,
    semantic_snapshot,
    semantic_snapshot_token,
)

MEMORY_ID = UUID("00000000-0000-0000-0000-000000000001")
EVENT_ID = UUID("00000000-0000-0000-0000-000000000002")


def test_mutation_history_models_cover_event_revisions_protection_and_restore() -> None:
    event = MutationEvent(
        id=EVENT_ID,
        operation="rewrite_memory",
        actor_kind=MutationActorKind.MAINTENANCE,
        status=MutationEventStatus.APPLIED,
    )
    revision = RecordRevision(
        event_id=event.id,
        memory_id=MEMORY_ID,
        role=RevisionRole.TARGET,
        before_exists=True,
        after_exists=True,
        before_snapshot={"content": "before"},
        after_snapshot={"content": "after"},
    )
    link = LinkRevision(
        event_id=event.id,
        source_id=MEMORY_ID,
        target_id=EVENT_ID,
        link_type="supports",
        before_exists=False,
        after_exists=True,
    )
    protection = Protection(memory_id=MEMORY_ID, mode=ProtectionMode.PINNED_ACTIVE, reason="important")
    request = RestoreRequest(
        target_event_id=event.id,
        scope=RestoreScope.ALL,
        reason="undo test",
        idempotency_key="restore-1",
    )
    result = RestoreResult(status=RestoreResultStatus.APPLIED, target_event_id=event.id, event_id=EVENT_ID)

    assert event.status is MutationEventStatus.APPLIED
    assert revision.after_snapshot == {"content": "after"}
    assert link.after_exists and protection.mode is ProtectionMode.PINNED_ACTIVE
    assert request.scope is RestoreScope.ALL and result.status is RestoreResultStatus.APPLIED


def test_semantic_snapshot_excludes_volatile_access_telemetry() -> None:
    record = {
        "id": MEMORY_ID,
        "title": "Title",
        "content": "Content",
        "summary": None,
        "type": "observation",
        "status": "active",
        "tags": ["b", "a"],
        "workspace_ids": [],
        "read_count": 1,
        "access_score": 0.1,
        "last_accessed_at": datetime(2026, 1, 1),
        "last_surfaced_at": "before",
    }
    changed = {**record, "read_count": 99, "access_score": 10, "last_surfaced_at": "after"}

    assert semantic_snapshot(record) == semantic_snapshot(changed)
    assert semantic_snapshot_token(record) == semantic_snapshot_token(changed)
    assert semantic_snapshot_token(record) != semantic_snapshot_token({**record, "content": "changed"})


def test_domain_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        MutationEvent.model_validate(
            {
                "id": EVENT_ID,
                "operation": "archive_memory",
                "actor_kind": "system",
                "status": "applied",
                "database_connection": object(),
            }
        )
