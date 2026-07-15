from uuid import UUID

from mcp_memory.core.mutation_restore import (
    InverseDescription,
    RestoreConflict,
    RestoreConflictCode,
    build_inverse,
)
from mcp_memory.mutation_history import (
    LinkRevision,
    MutationActorKind,
    MutationEvent,
    MutationEventStatus,
    RecordRevision,
    RevisionRole,
)

MEMORY_ID = UUID("00000000-0000-0000-0000-000000000001")
OTHER_ID = UUID("00000000-0000-0000-0000-000000000002")
EVENT_ID = UUID("00000000-0000-0000-0000-000000000003")


def _event(operation: str) -> MutationEvent:
    return MutationEvent(id=EVENT_ID, operation=operation, actor_kind=MutationActorKind.MAINTENANCE, status=MutationEventStatus.APPLIED)


def _snapshot(content: str, *, telemetry: int = 0) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record": {
            "id": MEMORY_ID,
            "title": "Title",
            "content": content,
            "summary": None,
            "type": "observation",
            "status": "active",
            "tags": ["tag"],
            "workspace_ids": [],
            "lineage": {},
            "mutation_metadata": {},
            "read_count": telemetry,
        },
    }


def test_normalize_inverse_preserves_semantics_and_excludes_telemetry() -> None:
    before = _snapshot("before", telemetry=1)
    after = _snapshot("after", telemetry=9)
    revision = RecordRevision(
        event_id=EVENT_ID,
        memory_id=MEMORY_ID,
        role=RevisionRole.TARGET,
        before_exists=True,
        before_snapshot=before,
        after_exists=True,
        after_snapshot=after,
        before_token="before-token",
        after_token="after-token",
    )
    result = build_inverse(_event("normalize_memory"), [revision])
    assert isinstance(result, InverseDescription)
    assert result.record_changes[0].snapshot != revision.before_snapshot
    snapshot = result.record_changes[0].snapshot
    assert snapshot is not None
    assert "read_count" not in snapshot["record"]
    assert result.record_changes[0].expected_current_token == "after-token"


def test_link_inverse_reverses_transition_without_execution() -> None:
    revision = LinkRevision(
        event_id=EVENT_ID, source_id=MEMORY_ID, target_id=OTHER_ID, link_type="supports", context="why", before_exists=False, after_exists=True
    )
    result = build_inverse(_event("create_link"), link_revisions=[revision])
    assert isinstance(result, InverseDescription)
    assert result.inverse_operation == "remove_link"
    assert result.link_changes[0].exists is False


def test_malformed_and_unsupported_history_fail_closed() -> None:
    malformed = build_inverse(_event("normalize_memory"), [])
    unsupported = build_inverse(_event("merge_memories"))
    assert isinstance(malformed, RestoreConflict) and malformed.code is RestoreConflictCode.INCOMPLETE_HISTORY
    assert isinstance(unsupported, RestoreConflict) and unsupported.code is RestoreConflictCode.UNSUPPORTED_OPERATION


def test_rewrite_and_remove_link_restore_remain_unsupported() -> None:
    assert isinstance(build_inverse(_event("rewrite_memory")), RestoreConflict)
    assert isinstance(build_inverse(_event("remove_link")), RestoreConflict)
