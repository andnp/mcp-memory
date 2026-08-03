from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from mcp_memory.core.curation_executor import CurationExecutor, CurationPolicyRejection
from mcp_memory.core.curation_identity import (
    canonical_token,
    record_snapshot,
    record_token,
)
from mcp_memory.core.curation_models import (
    ActionPreconditions,
    AbsentLinkAssertion,
    ArchiveMemoryAction,
    ClaimManifest,
    ClaimMapping,
    CreateLinkAction,
    EvidenceRef,
    LinkAssertion,
    MergeMemoriesAction,
    NormalizeMemoryAction,
    RemoveLinkAction,
    RewriteMemoryAction,
    SplitMemoryAction,
)
from mcp_memory.curation_action_store import (
    CurationActionFatalError,
    CurationActionStaleError,
    CurationTransaction,
    MutationResult,
    SQLiteCurationActionStore,
)
from mcp_memory.curation_store import (
    CurationActionReceipt,
    CurationReceiptState,
    CurationRun,
    CurationRunState,
    SQLiteCurationStore,
)
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


def _seed_link_endpoints(db_manager: DatabaseManager) -> tuple[RelationalMemoryRepository, CurationRun, UUID, UUID]:
    repository, run, source_id = _seed(db_manager)
    target_id = uuid4()
    repository.create_memory(
        "Target title",
        "Target content.",
        ["workspace"],
        memory_id=str(target_id),
        summary="Target summary",
        tags=["target"],
        memory_type="fact",
    )
    return repository, run, source_id, target_id


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


def test_normalize_rejects_empty_action_without_timestamp_history_or_receipt(db_manager: DatabaseManager) -> None:
    repository, run, memory_id = _seed(db_manager)
    before = repository.get_memory(str(memory_id))
    assert before is not None
    action = NormalizeMemoryAction.model_construct(
        action_id=uuid4(),
        target_id=memory_id,
        confidence=1,
        rationale="empty",
        preconditions=ActionPreconditions(record_tokens={memory_id: record_token(before)}),
    )

    with pytest.raises(CurationPolicyRejection, match="empty_normalize"):
        CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_normalize(
            action, run_id=run.run_id, memory_type=before.type
        )

    assert repository.get_memory(str(memory_id)) == before
    connection = db_manager.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM memory_mutation_events").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM memory_record_revisions").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM curation_action_receipts").fetchone()[0] == 0


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
        operation: str | None = None,
        payload: Any | None = None,
    ) -> CurationActionReceipt:
        arguments: dict[str, object] = {
            "run_id": run_id,
            "action_id": action_id,
            "target_ids": target_ids,
            "expected_tokens": expected_tokens,
            "apply": apply,
            "preconditions": preconditions,
            "operation": operation,
            "payload": payload,
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
        action_id=uuid4(), target_id=memory_id, confidence=1, rationale="missing precondition", title="title"
    )

    with pytest.raises(CurationActionFatalError, match="expected record token"):
        CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_normalize(
            action, run_id=run.run_id, memory_type="observation"
        )


def _link_action(
    source_id: UUID,
    target_id: UUID,
    source_token: str,
    target_token: str,
    *,
    rationale: str = "the source explicitly supports the target",
    link_type: str = "SUPPORTS",
    context: str = "The source records the target as supporting evidence.",
) -> CreateLinkAction:
    assertion = AbsentLinkAssertion(source_id=source_id, target_id=target_id, link_type=link_type)
    evidence = LinkAssertion(source_id=source_id, target_id=target_id, link_type=link_type, context=context)
    return CreateLinkAction(
        action_id=uuid4(),
        source_id=source_id,
        target_id=target_id,
        confidence=1,
        rationale=rationale,
        evidence=[EvidenceRef(link=evidence)],
        preconditions=ActionPreconditions(
            record_tokens={source_id: source_token, target_id: target_token},
            absent_links=[assertion],
        ),
        link_type=link_type,
        context=context,
    )


def _claim_manifest(*outputs: tuple[str, list[UUID]]) -> ClaimManifest:
    return ClaimManifest(
        preserved_claims=["claim"],
        transformed_claims=[output for output, _ in outputs],
        source_mapping=[ClaimMapping(output=output, source_memory_ids=source_ids) for output, source_ids in outputs],
    )


def _rewrite_action(memory_id: UUID, token: str, *, content: str = "Rewritten content.") -> RewriteMemoryAction:
    return RewriteMemoryAction(
        action_id=uuid4(),
        target_id=memory_id,
        confidence=1,
        rationale="rewrite the record",
        content=content,
        claim_manifest=_claim_manifest(("rewritten claim", [memory_id])),
        evidence=[EvidenceRef(memory_id=memory_id)],
        preconditions=ActionPreconditions(record_tokens={memory_id: token}),
    )


def _remove_link_action(source_id: UUID, target_id: UUID, source_token: str, target_token: str) -> RemoveLinkAction:
    assertion = LinkAssertion(source_id=source_id, target_id=target_id, link_type="DEPENDS_ON")
    return RemoveLinkAction(
        action_id=uuid4(),
        source_id=source_id,
        target_id=target_id,
        confidence=1,
        rationale="remove the link",
        evidence=[EvidenceRef(link=assertion)],
        preconditions=ActionPreconditions(
            record_tokens={source_id: source_token, target_id: target_token},
            required_links=[assertion],
        ),
        link_type="DEPENDS_ON",
    )


def _merge_action(canonical_id: UUID, source_id: UUID, canonical_token: str, source_token: str) -> MergeMemoriesAction:
    return MergeMemoriesAction(
        action_id=uuid4(),
        canonical_id=canonical_id,
        source_ids=[source_id],
        confidence=1,
        rationale="merge the records",
        content="Merged content.",
        claim_manifest=_claim_manifest(("merged claim", [canonical_id, source_id])),
        evidence=[EvidenceRef(memory_id=canonical_id), EvidenceRef(memory_id=source_id)],
        preconditions=ActionPreconditions(record_tokens={canonical_id: canonical_token, source_id: source_token}),
    )


def _split_action(memory_id: UUID, token: str) -> SplitMemoryAction:
    outputs = [("child one", [memory_id]), ("child two", [memory_id])]
    return SplitMemoryAction(
        action_id=uuid4(),
        target_id=memory_id,
        confidence=1,
        rationale="split the record",
        children=[ClaimMapping(output=output, source_memory_ids=source_ids) for output, source_ids in outputs],
        claim_manifest=_claim_manifest(*outputs),
        evidence=[EvidenceRef(memory_id=memory_id)],
        preconditions=ActionPreconditions(record_tokens={memory_id: token}),
    )


def _archive_action(memory_id: UUID, token: str) -> ArchiveMemoryAction:
    return ArchiveMemoryAction(
        action_id=uuid4(),
        target_id=memory_id,
        confidence=1,
        rationale="archive the record",
        claim_manifest=ClaimManifest(preserved_claims=["claim"]),
        evidence=[EvidenceRef(memory_id=memory_id)],
        preconditions=ActionPreconditions(record_tokens={memory_id: token}),
    )


def test_create_link_records_edge_history_and_compact_receipt(db_manager: DatabaseManager) -> None:
    repository, run, source_id, target_id = _seed_link_endpoints(db_manager)
    source = repository.get_memory(str(source_id))
    target = repository.get_memory(str(target_id))
    assert source is not None and target is not None

    receipt = CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_create_link(
        _link_action(source_id, target_id, record_token(source), record_token(target)),
        run_id=run.run_id,
        source_type=source.type,
        target_type=target.type,
    )

    links = repository.get_links(str(source_id))
    assert [(link.target_id, link.link_type, link.context) for link in links] == [
        (str(target_id), "SUPPORTS", "The source records the target as supporting evidence.")
    ]
    assert receipt.operation == "create_link"
    assert set(receipt.affected_ids) == {source_id, target_id}
    assert receipt.mutation_event_id is not None
    connection = db_manager.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM memory_link_revisions").fetchone()[0] == 1
    revision = connection.execute(
        "SELECT source_id, target_id, link_type, context, before_exists, after_exists "
        "FROM memory_link_revisions"
    ).fetchone()
    assert tuple(revision) == (
        str(source_id),
        str(target_id),
        "SUPPORTS",
        "The source records the target as supporting evidence.",
        0,
        1,
    )


def test_create_link_replay_is_idempotent(db_manager: DatabaseManager) -> None:
    repository, run, source_id, target_id = _seed_link_endpoints(db_manager)
    source = repository.get_memory(str(source_id))
    target = repository.get_memory(str(target_id))
    assert source is not None and target is not None
    action = _link_action(source_id, target_id, record_token(source), record_token(target))
    executor = CurationExecutor(SQLiteCurationActionStore(db_manager))

    first = executor.execute_create_link(action, run_id=run.run_id, source_type=source.type, target_type=target.type)
    replay = executor.execute_create_link(action, run_id=run.run_id, source_type=source.type, target_type=target.type)

    assert replay == first
    connection = db_manager.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM links").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM memory_mutation_events").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM memory_link_revisions").fetchone()[0] == 1
    assert len(repository.get_links(str(source_id))) == 1


def test_create_link_rejects_protection_before_mutation(db_manager: DatabaseManager) -> None:
    repository, run, source_id, target_id = _seed_link_endpoints(db_manager)
    source = repository.get_memory(str(source_id))
    target = repository.get_memory(str(target_id))
    assert source is not None and target is not None

    with pytest.raises(CurationPolicyRejection):
        CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_create_link(
            _link_action(source_id, target_id, record_token(source), record_token(target)),
            run_id=run.run_id,
            source_type=source.type,
            target_type=target.type,
            protections=[ProtectionMode.MANUAL_REVIEW_REQUIRED],
        )

    assert repository.get_links(str(source_id)) == []
    assert db_manager.get_connection().execute("SELECT COUNT(*) FROM memory_mutation_events").fetchone()[0] == 0


def test_create_link_rejects_generic_vocabulary_without_exact_evidence_or_preconditions(
    db_manager: DatabaseManager,
) -> None:
    repository, run, source_id, target_id = _seed_link_endpoints(db_manager)
    source = repository.get_memory(str(source_id))
    target = repository.get_memory(str(target_id))
    assert source is not None and target is not None
    action = CreateLinkAction(
        action_id=uuid4(),
        source_id=source_id,
        target_id=target_id,
        confidence=1,
        rationale="architecture and testing are related",
        evidence=[
            EvidenceRef(
                link=LinkAssertion(
                    source_id=source_id,
                    target_id=target_id,
                    link_type="SUPPORTS",
                    context="The source supports the target.",
                )
            )
        ],
        link_type="SUPPORTS",
        context="The source supports the target.",
        preconditions=ActionPreconditions(
            record_tokens={source_id: record_token(source), target_id: record_token(target)}
        ),
    )

    with pytest.raises(CurationActionFatalError, match="absent-link"):
        CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_create_link(
            action,
            run_id=run.run_id,
            source_type=source.type,
            target_type=target.type,
        )

    assert repository.get_links(str(source_id)) == []
    assert db_manager.get_connection().execute("SELECT COUNT(*) FROM memory_mutation_events").fetchone()[0] == 0


def test_create_link_rejects_context_bearing_absent_precondition_before_mutation(
    db_manager: DatabaseManager,
) -> None:
    repository, run, source_id, target_id = _seed_link_endpoints(db_manager)
    source = repository.get_memory(str(source_id))
    target = repository.get_memory(str(target_id))
    assert source is not None and target is not None
    action = _link_action(
        source_id,
        target_id,
        record_token(source),
        record_token(target),
    ).model_copy(
        update={
            "preconditions": ActionPreconditions.model_construct(
                record_tokens={source_id: record_token(source), target_id: record_token(target)},
                absent_links=[
                    LinkAssertion(
                        source_id=source_id,
                        target_id=target_id,
                        link_type="SUPPORTS",
                        context="Existing relationship context.",
                    )
                ],
            )
        }
    )
    repository.add_link(str(source_id), str(target_id), "SUPPORTS", "Different existing context.")

    with pytest.raises(CurationActionFatalError, match="must omit relationship context"):
        CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_create_link(
            action,
            run_id=run.run_id,
            source_type=source.type,
            target_type=target.type,
        )

    assert repository.get_links(str(source_id))[0].context == "Different existing context."


def test_create_link_missing_endpoint_fails_before_history(db_manager: DatabaseManager) -> None:
    repository, run, source_id, _target_id = _seed_link_endpoints(db_manager)
    source = repository.get_memory(str(source_id))
    assert source is not None
    missing_id = uuid4()
    action = _link_action(source_id, missing_id, record_token(source), "missing")

    with pytest.raises(CurationActionStaleError):
        CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_create_link(
            action,
            run_id=run.run_id,
            source_type=source.type,
            target_type="fact",
        )

    assert repository.get_links(str(source_id)) == []
    assert db_manager.get_connection().execute("SELECT COUNT(*) FROM memory_mutation_events").fetchone()[0] == 0


def test_rewrite_updates_authoritative_state_and_records_history(db_manager: DatabaseManager) -> None:
    repository, run, memory_id = _seed(db_manager)
    before = repository.get_memory(str(memory_id))
    assert before is not None

    receipt = CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_rewrite(
        _rewrite_action(memory_id, record_token(before), content="Rewritten content."),
        run_id=run.run_id,
        memory_type=before.type,
    )

    after = repository.get_memory(str(memory_id))
    assert after is not None
    assert after.content == "Rewritten content."
    assert receipt.operation == "rewrite_memory"
    assert receipt.affected_ids == [memory_id]
    connection = db_manager.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM memory_mutation_events").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM memory_record_revisions").fetchone()[0] == 1


def test_remove_link_removes_exact_edge_through_the_transaction_path(db_manager: DatabaseManager) -> None:
    repository, run, source_id, target_id = _seed_link_endpoints(db_manager)
    source = repository.get_memory(str(source_id))
    target = repository.get_memory(str(target_id))
    assert source is not None and target is not None
    executor = CurationExecutor(SQLiteCurationActionStore(db_manager))
    create_receipt = executor.execute_create_link(
        _link_action(
            source_id,
            target_id,
            record_token(source),
            record_token(target),
            link_type="DEPENDS_ON",
            context="link to remove",
        ),
        run_id=run.run_id,
        source_type=source.type,
        target_type=target.type,
    )

    remove_receipt = executor.execute_remove_link(
        _remove_link_action(source_id, target_id, record_token(source), record_token(target)),
        run_id=run.run_id,
        source_type=source.type,
        target_type=target.type,
    )

    assert create_receipt.operation == "create_link"
    assert remove_receipt.operation == "remove_link"
    assert repository.get_links(str(source_id)) == []
    connection = db_manager.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM memory_link_revisions").fetchone()[0] == 2


def test_merge_updates_canonical_archives_source_and_records_lineage(db_manager: DatabaseManager) -> None:
    repository, run, canonical_id = _seed(db_manager)
    source_id = uuid4()
    repository.create_memory(
        "Source title",
        "Source content.",
        ["workspace"],
        memory_id=str(source_id),
        summary="Source summary",
        tags=["source"],
        memory_type="fact",
    )
    canonical = repository.get_memory(str(canonical_id))
    source = repository.get_memory(str(source_id))
    assert canonical is not None and source is not None

    receipt = CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_merge(
        _merge_action(canonical_id, source_id, record_token(canonical), record_token(source)),
        run_id=run.run_id,
        memory_types={canonical_id: canonical.type, source_id: source.type},
    )

    updated_canonical = repository.get_memory(str(canonical_id))
    archived_source = repository.get_memory(str(source_id))
    assert updated_canonical is not None and archived_source is not None
    assert updated_canonical.content == "Merged content."
    assert archived_source.status == "archived"
    assert receipt.operation == "merge_memories"
    assert repository.get_links(str(canonical_id))[0].link_type == "SUPERSEDES"
    connection = db_manager.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM memory_mutation_events").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM memory_record_revisions").fetchone()[0] == 2
    assert connection.execute("SELECT COUNT(*) FROM memory_link_revisions").fetchone()[0] == 1


def test_merge_preserves_canonical_summary_when_action_omits_one(db_manager: DatabaseManager) -> None:
    repository, run, canonical_id = _seed(db_manager)
    source_id = uuid4()
    repository.create_memory(
        "Source title",
        "Source content.",
        ["workspace"],
        memory_id=str(source_id),
        summary="Source summary",
        tags=["source"],
        memory_type="fact",
    )
    canonical = repository.get_memory(str(canonical_id))
    source = repository.get_memory(str(source_id))
    assert canonical is not None and source is not None
    action = _merge_action(
        canonical_id,
        source_id,
        record_token(canonical),
        record_token(source),
    ).model_copy(update={"summary": None})

    CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_merge(
        action,
        run_id=run.run_id,
        memory_types={canonical_id: canonical.type, source_id: source.type},
    )

    updated_canonical = repository.get_memory(str(canonical_id))
    assert updated_canonical is not None
    assert updated_canonical.summary == "Original summary"


def test_split_creates_children_and_preserves_original_lineage(db_manager: DatabaseManager) -> None:
    repository, run, memory_id = _seed(db_manager)
    before = repository.get_memory(str(memory_id))
    assert before is not None

    receipt = CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_split(
        _split_action(memory_id, record_token(before)),
        run_id=run.run_id,
        memory_type=before.type,
    )

    after = repository.get_memory(str(memory_id))
    assert after is not None
    child_ids = sorted((str(value) for value in receipt.affected_ids if value != memory_id), key=lambda value: value.encode("utf-8"))

    def _part_index(child_id: str) -> int:
        child = repository.get_memory(child_id)
        assert child is not None
        part_index = child.metadata.get("split_part_index")
        assert isinstance(part_index, int)
        return part_index

    def _metadata_string_list(record: Any, key: str) -> list[str]:
        metadata = record.metadata
        value = metadata.get(key)
        assert isinstance(value, list)
        assert all(isinstance(item, str) for item in value)
        return value

    ordered_child_ids = sorted(child_ids, key=_part_index)
    split_child_count = after.metadata.get("split_child_count")
    assert split_child_count == 2
    assert sorted(_metadata_string_list(after, "split_child_memory_ids"), key=lambda value: value.encode("utf-8")) == child_ids
    for child_id in child_ids:
        child = repository.get_memory(child_id)
        assert child is not None
        assert child.metadata.get("split_from_memory_id") == str(memory_id)
        assert child.metadata.get("split_group_id") == str(receipt.action_id)
        assert _metadata_string_list(child, "split_child_memory_ids") == ordered_child_ids
        assert sorted(_metadata_string_list(child, "split_sibling_memory_ids"), key=lambda value: value.encode("utf-8")) == [
            other_id for other_id in ordered_child_ids if other_id != child_id
        ]
        assert repository.get_links(child_id)[0].target_id == str(memory_id)
    assert receipt.operation == "split_memory"
    connection = db_manager.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM memory_mutation_events").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM memory_record_revisions").fetchone()[0] == 2
    assert connection.execute("SELECT COUNT(*) FROM memory_link_revisions").fetchone()[0] == 2


def test_archive_marks_record_archived_and_replays_cleanly(db_manager: DatabaseManager) -> None:
    repository, run, memory_id = _seed(db_manager)
    before = repository.get_memory(str(memory_id))
    assert before is not None
    action = _archive_action(memory_id, record_token(before))
    executor = CurationExecutor(SQLiteCurationActionStore(db_manager))

    first = executor.execute_archive(action, run_id=run.run_id, memory_type=before.type)
    replay = executor.execute_archive(action, run_id=run.run_id, memory_type=before.type)

    after = repository.get_memory(str(memory_id))
    assert after is not None
    assert after.status == "archived"
    assert first == replay
    assert first.operation == "archive_memory"
    assert first.affected_ids == [memory_id]
