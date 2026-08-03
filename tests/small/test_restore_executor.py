from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from mcp_memory.core.curation_executor import CurationExecutor
from mcp_memory.core.curation_identity import link_token, record_token
from mcp_memory.core.curation_models import ActionPreconditions, CreateLinkAction, EvidenceRef, LinkAssertion, NormalizeMemoryAction
from mcp_memory.core.mutation_restore import RestoreExecutor
from mcp_memory.curation_action_store import SQLiteCurationActionStore
from mcp_memory.curation_store import CurationRun, CurationRunState, SQLiteCurationStore
from mcp_memory.mutation_history import (
    MutationActorKind,
    RestoreRequest,
    RestoreResultStatus,
    RestoreScope,
)
from mcp_memory.mutation_history_store import SQLiteMutationHistoryStore
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.utils.db import DatabaseManager


pytestmark = pytest.mark.small


def _seed(db_manager: DatabaseManager) -> tuple[RelationalMemoryRepository, CurationRun, UUID]:
    repository = RelationalMemoryRepository(db_manager)
    memory_id = uuid4()
    repository.create_memory(
        "Original title",
        "Content that must survive restore.",
        ["workspace"],
        memory_id=str(memory_id),
        summary="Original summary",
        tags=["old"],
        memory_type="observation",
    )
    run = CurationRun(
        run_id=uuid4(),
        frontier_key="restore-test",
        context_fingerprint="restore-context",
        state=CurationRunState.EXECUTING,
    )
    SQLiteCurationStore(db_manager).create_run(run)
    return repository, run, memory_id


def _normalize(memory_id: UUID, token: str) -> NormalizeMemoryAction:
    return NormalizeMemoryAction(
        action_id=uuid4(),
        target_id=memory_id,
        confidence=1,
        rationale="specific normalization",
        title="Changed title",
        preconditions=ActionPreconditions(record_tokens={memory_id: token}),
    )


def test_normalize_restore_preserves_telemetry_creates_event_and_replays(
    db_manager: DatabaseManager,
) -> None:
    repository, run, memory_id = _seed(db_manager)
    before = repository.get_memory(str(memory_id))
    assert before is not None
    action = _normalize(memory_id, record_token(before))
    original = CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_normalize(
        action,
        run_id=run.run_id,
        memory_type=before.type,
    )
    changed = repository.get_memory(str(memory_id))
    assert changed is not None
    current = db_manager.get_connection()
    current.execute(
        "UPDATE memories SET read_count = 7, access_score = 2.5, last_accessed_at = ?, last_surfaced_at = ? WHERE id = ?",
        ("access-time", "surface-time", str(memory_id)),
    )
    current.commit()

    request = RestoreRequest(
        target_event_id=original.mutation_event_id,  # type: ignore[arg-type]
        expected_record_tokens={memory_id: record_token(changed)},
        reason="restore the accepted normalization",
        idempotency_key="restore-normalize-1",
    )
    history = SQLiteMutationHistoryStore(db_manager)
    curation = SQLiteCurationStore(db_manager)
    restored = RestoreExecutor(
        SQLiteCurationActionStore(db_manager),
        history,
        curation_store=curation,
    ).execute(request)

    assert restored.status is RestoreResultStatus.APPLIED
    assert restored.event_id is not None
    restored_record = repository.get_memory(str(memory_id))
    assert restored_record is not None
    assert restored_record.title == before.title
    assert restored_record.read_count == 7
    assert restored_record.access_score == 2.5
    assert restored_record.last_accessed_at == "access-time"
    assert restored_record.last_surfaced_at == "surface-time"
    restore_event = history.get_event(restored.event_id)
    assert restore_event is not None
    assert restore_event.actor_kind is MutationActorKind.RESTORE
    assert restore_event.restores_event_id == original.mutation_event_id

    replay = RestoreExecutor(SQLiteCurationActionStore(db_manager), history, curation_store=curation).execute(request)
    assert replay.status is RestoreResultStatus.ALREADY_APPLIED
    assert replay.event_id == restored.event_id
    assert len(history.list_events()) == 2


def test_normalize_restore_preserves_lineage_and_mutation_metadata(db_manager: DatabaseManager) -> None:
    repository, run, memory_id = _seed(db_manager)
    durable_metadata: dict[str, object] = {
        "lineage": {"parent_id": "parent", "split_index": 1},
        "mutation_metadata": {"source": "import", "revision": 3},
    }
    repository.update_memory(str(memory_id), metadata=durable_metadata)
    before = repository.get_memory(str(memory_id))
    assert before is not None
    original = CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_normalize(
        _normalize(memory_id, record_token(before)),
        run_id=run.run_id,
        memory_type=before.type,
    )
    changed = repository.get_memory(str(memory_id))
    assert changed is not None
    result = RestoreExecutor(
        SQLiteCurationActionStore(db_manager),
        SQLiteMutationHistoryStore(db_manager),
        curation_store=SQLiteCurationStore(db_manager),
    ).execute(
        RestoreRequest(
            target_event_id=original.mutation_event_id,  # type: ignore[arg-type]
            expected_record_tokens={memory_id: record_token(changed)},
            reason="restore durable metadata",
            idempotency_key="restore-normalize-metadata-1",
        )
    )

    assert result.status is RestoreResultStatus.APPLIED
    restored = repository.get_memory(str(memory_id))
    assert restored is not None
    assert restored.metadata == durable_metadata


def test_restore_of_later_edited_normalization_is_a_typed_conflict(db_manager: DatabaseManager) -> None:
    repository, run, memory_id = _seed(db_manager)
    before = repository.get_memory(str(memory_id))
    assert before is not None
    original = CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_normalize(
        _normalize(memory_id, record_token(before)),
        run_id=run.run_id,
        memory_type=before.type,
    )
    changed = repository.get_memory(str(memory_id))
    assert changed is not None
    repository.update_memory(str(memory_id), title="Later human edit")
    request = RestoreRequest(
        target_event_id=original.mutation_event_id,  # type: ignore[arg-type]
        expected_record_tokens={memory_id: record_token(changed)},
        reason="must not overwrite a later edit",
        idempotency_key="restore-conflict-1",
    )

    result = RestoreExecutor(
        SQLiteCurationActionStore(db_manager),
        SQLiteMutationHistoryStore(db_manager),
        curation_store=SQLiteCurationStore(db_manager),
    ).execute(request)

    assert result.status is RestoreResultStatus.CONFLICT
    assert result.conflict_details["code"] == "stale_state"
    assert repository.get_memory(str(memory_id)).title == "Later human edit"  # type: ignore[union-attr]


def test_create_link_restore_removes_only_the_exact_edge(db_manager: DatabaseManager) -> None:
    repository, run, source_id = _seed(db_manager)
    target_id = uuid4()
    repository.create_memory("Target", "Target content", ["workspace"], memory_id=str(target_id), memory_type="fact")
    source = repository.get_memory(str(source_id))
    target = repository.get_memory(str(target_id))
    assert source is not None and target is not None
    edge = LinkAssertion(
        source_id=source_id,
        target_id=target_id,
        link_type="SUPPORTS",
        context="exact restore edge",
    )
    absent = LinkAssertion(source_id=source_id, target_id=target_id, link_type=edge.link_type)
    action = CreateLinkAction(
        action_id=uuid4(),
        source_id=source_id,
        target_id=target_id,
        confidence=1,
        rationale="exact relationship evidence",
        evidence=[EvidenceRef(link=edge)],
        preconditions=ActionPreconditions(
            record_tokens={source_id: record_token(source), target_id: record_token(target)},
            absent_links=[absent],
        ),
        link_type=edge.link_type,
        context=edge.context,
    )
    original = CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_create_link(
        action,
        run_id=run.run_id,
        source_type=source.type,
        target_type=target.type,
    )
    request = RestoreRequest(
        target_event_id=original.mutation_event_id,  # type: ignore[arg-type]
        scope=RestoreScope.LINKS,
        expected_record_tokens={source_id: record_token(source), target_id: record_token(target)},
        expected_link_tokens={
            f"{source_id}:{target_id}:SUPPORTS": link_token(
                source_id,
                target_id,
                "SUPPORTS",
                edge.context,
            )
        },
        reason="remove the accepted edge",
        idempotency_key="restore-link-1",
    )
    history = SQLiteMutationHistoryStore(db_manager)
    result = RestoreExecutor(
        SQLiteCurationActionStore(db_manager),
        history,
        curation_store=SQLiteCurationStore(db_manager),
    ).execute(request)

    assert result.status is RestoreResultStatus.APPLIED
    assert repository.get_links(str(source_id)) == []
    assert result.event_id is not None
    assert history.get_event(result.event_id).restores_event_id == original.mutation_event_id  # type: ignore[union-attr]
