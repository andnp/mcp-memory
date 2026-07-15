from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from mcp_memory.core.curation_executor import CurationExecutor, CurationPolicyRejection
from mcp_memory.core.curation_identity import canonical_token, record_snapshot, record_token
from mcp_memory.core.curation_models import ActionPreconditions, NormalizeMemoryAction
from mcp_memory.curation_action_store import (
    CurationActionFatalError,
    CurationTransaction,
    MutationResult,
    SQLiteCurationActionStore,
)
from mcp_memory.curation_store import CurationActionReceipt, CurationReceiptState, CurationRun, CurationRunState, SQLiteCurationStore
from mcp_memory.mutation_history import ProtectionMode
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.utils.db import DatabaseManager


pytestmark = pytest.mark.small


def _seed(db_manager: DatabaseManager) -> tuple[RelationalMemoryRepository, CurationRun, UUID]:
    repository = RelationalMemoryRepository(db_manager)
    memory_id = uuid4()
    repository.create_memory(
        "Original title",
        "Immutable content with exact spacing.",
        ["workspace"],
        memory_id=str(memory_id),
        summary="Original summary",
        tags=["old"],
        memory_type="observation",
    )
    run = CurationRun(
        run_id=uuid4(),
        frontier_key="test-frontier",
        context_fingerprint="test-context",
        state=CurationRunState.EXECUTING,
    )
    SQLiteCurationStore(db_manager).create_run(run)
    return repository, run, memory_id


def _action(
    memory_id: UUID,
    token: str,
    *,
    title: str | None = None,
    summary: str | None = None,
    tags: list[str] | None = None,
) -> NormalizeMemoryAction:
    return NormalizeMemoryAction(
        action_id=uuid4(),
        target_id=memory_id,
        confidence=1,
        rationale="make the metadata specific",
        preconditions=ActionPreconditions(record_tokens={memory_id: token}),
        title=title,
        summary=summary,
        tags=tags,
    )


def test_normalize_changes_only_metadata_and_records_history(db_manager: DatabaseManager) -> None:
    repository, run, memory_id = _seed(db_manager)
    before = repository.get_memory(str(memory_id))
    assert before is not None

    receipt = CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_normalize(
        _action(memory_id, record_token(before), title="Specific title", summary="A concrete result", tags=["new"]),
        run_id=run.run_id,
        memory_type=before.type,
    )

    after = repository.get_memory(str(memory_id))
    assert after is not None
    assert after.content == before.content
    assert after.title == "Specific title"
    assert after.summary == "A concrete result"
    assert after.tags == ["new"]
    assert receipt.status.value == "applied_unverified"
    assert receipt.after_token is not None
    assert receipt.after_token == canonical_token(
        {"targets": [str(memory_id)], "records": {str(memory_id): record_snapshot(after)}}
    )
    assert receipt.mutation_event_id is not None

    connection = db_manager.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM memory_mutation_events").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM memory_record_revisions").fetchone()[0] == 1
    revision = connection.execute(
        "SELECT before_snapshot, after_snapshot FROM memory_record_revisions"
    ).fetchone()
    assert revision is not None
    assert '"content": "Immutable content with exact spacing."' in revision[0]
    assert '"content": "Immutable content with exact spacing."' in revision[1]


@pytest.mark.parametrize("summary", ["Covers this memory.", "Added this memory."])
def test_normalize_rejects_generic_summary_without_writes(db_manager: DatabaseManager, summary: str) -> None:
    repository, run, memory_id = _seed(db_manager)
    before = repository.get_memory(str(memory_id))
    assert before is not None

    with pytest.raises(CurationPolicyRejection) as error:
        CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_normalize(
            _action(memory_id, record_token(before), summary=summary),
            run_id=run.run_id,
            memory_type=before.type,
        )

    assert "generic_summary" in str(error.value)
    assert repository.get_memory(str(memory_id)) == before
    assert db_manager.get_connection().execute("SELECT COUNT(*) FROM memory_mutation_events").fetchone()[0] == 0


def test_normalize_rejects_protection_before_transaction(db_manager: DatabaseManager) -> None:
    repository, run, memory_id = _seed(db_manager)
    before = repository.get_memory(str(memory_id))
    assert before is not None

    with pytest.raises(CurationPolicyRejection):
        CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_normalize(
            _action(memory_id, record_token(before), title="blocked"),
            run_id=run.run_id,
            memory_type=before.type,
            protections=[ProtectionMode.MANUAL_REVIEW_REQUIRED],
        )

    assert repository.get_memory(str(memory_id)) == before


def test_normalize_replay_returns_compact_receipt_without_second_mutation(db_manager: DatabaseManager) -> None:
    repository, run, memory_id = _seed(db_manager)
    before = repository.get_memory(str(memory_id))
    assert before is not None
    action = _action(memory_id, record_token(before), title="Specific title", tags=["new"])
    executor = CurationExecutor(SQLiteCurationActionStore(db_manager))

    first = executor.execute_normalize(action, run_id=run.run_id, memory_type=before.type)
    replay = executor.execute_normalize(action, run_id=run.run_id, memory_type=before.type)

    assert replay == first
    assert repository.get_memory(str(memory_id)).content == before.content  # type: ignore[union-attr]
    connection = db_manager.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM memory_mutation_events").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM memory_record_revisions").fetchone()[0] == 1


class _FakeTransaction:
    def __init__(self) -> None:
        self.changes: dict[str, object] | None = None

    def update_memory(self, _memory_id: str, **changes: object) -> object:
        self.changes = changes
        return object()


class _FakeActionStore:
    def __init__(self) -> None:
        self.transaction = _FakeTransaction()
        self.arguments: dict[str, object] | None = None

    def execute_action(
        self,
        *,
        run_id: UUID,
        action_id: UUID,
        target_ids: Sequence[str],
        expected_tokens: Mapping[str, str],
        apply: Callable[[CurationTransaction], MutationResult],
        preconditions: Any | None = None,
    ) -> CurationActionReceipt:
        arguments: dict[str, object] = {
            "run_id": run_id,
            "action_id": action_id,
            "target_ids": target_ids,
            "expected_tokens": expected_tokens,
            "apply": apply,
            "preconditions": preconditions,
        }
        self.arguments = arguments
        apply(cast(CurationTransaction, self.transaction))
        return CurationActionReceipt(
            run_id=run_id,
            action_id=action_id,
            operation="normalize_memory",
            affected_ids=[UUID(target_ids[0])],
            status=CurationReceiptState.APPLIED_UNVERIFIED,
        )


def test_normalize_executor_is_backend_neutral_and_never_passes_content() -> None:
    memory_id = uuid4()
    token = "v1:before"
    action = _action(memory_id, token, title="title", summary="summary", tags=["tag"])
    store = _FakeActionStore()

    CurationExecutor(store).execute_normalize(action, run_id=uuid4(), memory_type="observation")

    assert store.transaction.changes is not None
    assert store.transaction.changes == {"title": "title", "summary": "summary", "tags": ["tag"]}
    assert store.arguments is not None
    assert store.arguments["expected_tokens"] == {str(memory_id): token}
    assert "content" not in store.transaction.changes


def test_normalize_requires_expected_record_token(db_manager: DatabaseManager) -> None:
    _repository, run, memory_id = _seed(db_manager)
    action = NormalizeMemoryAction(
        action_id=uuid4(), target_id=memory_id, confidence=1, rationale="missing precondition"
    )

    with pytest.raises(CurationActionFatalError, match="expected record token"):
        CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_normalize(
            action, run_id=run.run_id, memory_type="observation"
        )
