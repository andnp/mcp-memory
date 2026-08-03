from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from mcp_memory.core.curation_executor import CurationExecutor
from mcp_memory.core.curation_identity import record_token
from mcp_memory.core.curation_models import (
    ActionPreconditions,
    ArchiveMemoryAction,
    ClaimMapping,
    ClaimManifest,
    CreateLinkAction,
    CurationVerificationDescriptor,
    EvidenceRef,
    LinkAssertion,
    MergeMemoriesAction,
    NormalizeMemoryAction,
    RemoveLinkAction,
    RewriteMemoryAction,
    SplitMemoryAction,
)
from mcp_memory.core.curation_verifier import CurationVerifier, _state_token
from mcp_memory.curation_action_store import SQLiteCurationActionStore
from mcp_memory.curation_store import CurationReceiptState, CurationRun, CurationRunState, SQLiteCurationStore
from mcp_memory.relational.repository import RelationalMemoryReadContext, RelationalMemoryRepository
from mcp_memory.utils.db import DatabaseManager


pytestmark = pytest.mark.small


class _CountingMaintenanceReader:
    def __init__(self, repository: RelationalMemoryRepository) -> None:
        self.repository = repository
        self.peek_calls: list[str] = []

    def peek_memory(self, memory_id: str) -> RelationalMemoryReadContext | None:
        self.peek_calls.append(memory_id)
        return self.repository.peek_memory(memory_id)

    def search_memories_for_maintenance(
        self,
        query: str,
        *,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        limit: int = 50,
    ) -> list[RelationalMemoryReadContext]:
        return self.repository.search_memories_for_maintenance(
            query,
            workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            include_superseded=include_superseded,
            limit=limit,
        )


def _seed(db_manager: DatabaseManager) -> tuple[RelationalMemoryRepository, CurationRun, UUID]:
    repository = RelationalMemoryRepository(db_manager)
    memory_id = uuid4()
    repository.create_memory(
        "Original title",
        "Immutable content.",
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


def _normalize_action(memory_id: UUID, token: str) -> NormalizeMemoryAction:
    return NormalizeMemoryAction(
        action_id=uuid4(),
        target_id=memory_id,
        confidence=1,
        rationale="this prose must not be used as verification evidence",
        preconditions=ActionPreconditions(record_tokens={memory_id: token}),
        title="Specific title",
    )


def _link_action(
    source_id: UUID,
    target_id: UUID,
    source_token: str,
    target_token: str,
    *,
    link_type: str = "SUPPORTS",
    context: str = "exact typed edge",
) -> CreateLinkAction:
    absent = LinkAssertion(source_id=source_id, target_id=target_id, link_type=link_type)
    edge = LinkAssertion(
        source_id=source_id,
        target_id=target_id,
        link_type=link_type,
        context=context,
    )
    return CreateLinkAction(
        action_id=uuid4(),
        source_id=source_id,
        target_id=target_id,
        confidence=1,
        rationale="untrusted planner prose",
        evidence=[EvidenceRef(link=edge)],
        preconditions=ActionPreconditions(
            record_tokens={source_id: source_token, target_id: target_token},
            absent_links=[absent],
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


def _rewrite_action(memory_id: UUID, token: str) -> RewriteMemoryAction:
    return RewriteMemoryAction(
        action_id=uuid4(),
        target_id=memory_id,
        confidence=1,
        rationale="rewrite the record",
        content="Rewritten content.",
        claim_manifest=_claim_manifest(("rewritten claim", [memory_id])),
        evidence=[EvidenceRef(memory_id=memory_id)],
        preconditions=ActionPreconditions(record_tokens={memory_id: token}),
    )


def _remove_link_action(source_id: UUID, target_id: UUID, source_token: str, target_token: str) -> RemoveLinkAction:
    edge = LinkAssertion(source_id=source_id, target_id=target_id, link_type="DEPENDS_ON")
    return RemoveLinkAction(
        action_id=uuid4(),
        source_id=source_id,
        target_id=target_id,
        confidence=1,
        rationale="remove the link",
        evidence=[EvidenceRef(link=edge)],
        preconditions=ActionPreconditions(
            record_tokens={source_id: source_token, target_id: target_token},
            required_links=[edge],
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


def _verifier(
    db_manager: DatabaseManager,
    repository: RelationalMemoryRepository,
) -> tuple[CurationVerifier, _CountingMaintenanceReader]:
    reader = _CountingMaintenanceReader(repository)
    return CurationVerifier(SQLiteCurationStore(db_manager), reader), reader


def test_normalize_verifies_exact_after_token_from_fresh_maintenance_read(db_manager: DatabaseManager) -> None:
    repository, run, memory_id = _seed(db_manager)
    before = repository.get_memory(str(memory_id))
    assert before is not None
    action = _normalize_action(memory_id, record_token(before))
    receipt = CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_normalize(
        action,
        run_id=run.run_id,
        memory_type=before.type,
    )

    verifier, reader = _verifier(db_manager, repository)
    verified = verifier.verify(receipt, action)

    assert verified.status is CurationReceiptState.VERIFIED
    assert verified.error_code is None
    assert reader.peek_calls == [str(memory_id)]


def test_descriptor_remove_link_rejects_remaining_endpoint_edge_with_other_context(
    db_manager: DatabaseManager,
) -> None:
    repository, run, source_id = _seed(db_manager)
    target_id = uuid4()
    repository.create_memory(
        "Target",
        "Target content.",
        ["workspace"],
        memory_id=str(target_id),
        memory_type="fact",
    )
    repository.add_link(str(source_id), str(target_id), "DEPENDS_ON", "remaining context")
    source = repository.get_memory(str(source_id))
    target = repository.get_memory(str(target_id))
    assert source is not None and target is not None
    action = RemoveLinkAction(
        action_id=uuid4(),
        source_id=source_id,
        target_id=target_id,
        link_type="DEPENDS_ON",
        context="removed context",
        confidence=1,
        rationale="remove the link",
        evidence=[
            EvidenceRef(
                link=LinkAssertion(
                    source_id=source_id,
                    target_id=target_id,
                    link_type="DEPENDS_ON",
                    context="removed context",
                )
            )
        ],
        preconditions=ActionPreconditions(
            record_tokens={source_id: record_token(source), target_id: record_token(target)},
            required_links=[
                LinkAssertion(source_id=source_id, target_id=target_id, link_type="DEPENDS_ON")
            ],
        ),
    )
    receipt = CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_remove_link(
        action,
        run_id=run.run_id,
        source_type=source.type,
        target_type=target.type,
    )
    repository.add_link(str(source_id), str(target_id), "DEPENDS_ON", "remaining context")
    current_source = repository.peek_memory(str(source_id))
    current_target = repository.peek_memory(str(target_id))
    assert current_source is not None and current_target is not None
    descriptor = CurationVerificationDescriptor(
        operation="remove_link",
        target_ids=[source_id, target_id],
        source_id=source_id,
        target_id=target_id,
        link_type="DEPENDS_ON",
        context="removed context",
        exists=False,
    )
    updated = receipt.model_copy(
        update={
            "after_token": _state_token(
                {str(source_id): current_source.record, str(target_id): current_target.record},
                sorted([str(source_id), str(target_id)]),
            ),
            "verification_descriptor": descriptor,
        }
    )
    store = SQLiteCurationStore(db_manager)
    assert store.transition_receipt(
        run.run_id,
        receipt.action_id,
        CurationReceiptState.APPLIED_UNVERIFIED,
        updated,
    ) is not None
    verifier, _reader = _verifier(db_manager, repository)
    verified = verifier.verify_descriptor(updated, descriptor)
    assert verified.status is CurationReceiptState.VERIFICATION_FAILED
    assert verified.error_code == "edge_still_present"


def test_normalize_mismatch_is_verification_failed_and_terminal(db_manager: DatabaseManager) -> None:
    repository, run, memory_id = _seed(db_manager)
    before = repository.get_memory(str(memory_id))
    assert before is not None
    action = _normalize_action(memory_id, record_token(before))
    receipt = CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_normalize(
        action,
        run_id=run.run_id,
        memory_type=before.type,
    )
    repository.update_memory(str(memory_id), title="changed after commit")

    verifier, reader = _verifier(db_manager, repository)
    failed = verifier.verify(receipt, action)
    replay = verifier.verify(receipt, action)

    assert failed.status is CurationReceiptState.VERIFICATION_FAILED
    assert failed.error_code == "after_token_mismatch"
    assert replay == failed
    assert len(reader.peek_calls) == 1


def test_create_link_requires_exact_edge_in_fresh_maintenance_read(db_manager: DatabaseManager) -> None:
    repository, run, source_id = _seed(db_manager)
    target_id = uuid4()
    repository.create_memory("Target", "Target content.", ["workspace"], memory_id=str(target_id), memory_type="fact")
    source = repository.get_memory(str(source_id))
    target = repository.get_memory(str(target_id))
    assert source is not None and target is not None
    action = _link_action(source_id, target_id, record_token(source), record_token(target))
    receipt = CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_create_link(
        action,
        run_id=run.run_id,
        source_type=source.type,
        target_type=target.type,
    )

    verifier, reader = _verifier(db_manager, repository)
    verified = verifier.verify(receipt, action)

    assert verified.status is CurationReceiptState.VERIFIED
    assert set(reader.peek_calls) == {str(source_id), str(target_id)}


def test_create_link_missing_edge_fails_without_using_rationale(db_manager: DatabaseManager) -> None:
    repository, run, source_id = _seed(db_manager)
    target_id = uuid4()
    repository.create_memory("Target", "Target content.", ["workspace"], memory_id=str(target_id), memory_type="fact")
    source = repository.get_memory(str(source_id))
    target = repository.get_memory(str(target_id))
    assert source is not None and target is not None
    action = _link_action(source_id, target_id, record_token(source), record_token(target))
    receipt = CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_create_link(
        action,
        run_id=run.run_id,
        source_type=source.type,
        target_type=target.type,
    )
    assert repository.remove_link(str(source_id), str(target_id), "SUPPORTS")

    verifier, _reader = _verifier(db_manager, repository)
    failed = verifier.verify(receipt, action)

    assert failed.status is CurationReceiptState.VERIFICATION_FAILED
    assert failed.error_code == "edge_missing"


def test_rewrite_verifies_exact_after_token_from_fresh_read(db_manager: DatabaseManager) -> None:
    repository, run, memory_id = _seed(db_manager)
    before = repository.get_memory(str(memory_id))
    assert before is not None
    action = _rewrite_action(memory_id, record_token(before))
    receipt = CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_rewrite(
        action,
        run_id=run.run_id,
        memory_type=before.type,
    )

    verifier, reader = _verifier(db_manager, repository)
    verified = verifier.verify(receipt, action)

    assert verified.status is CurationReceiptState.VERIFIED
    assert reader.peek_calls == [str(memory_id)]


def test_remove_link_verifies_edge_absence_from_fresh_reads(db_manager: DatabaseManager) -> None:
    repository, run, source_id = _seed(db_manager)
    target_id = uuid4()
    repository.create_memory("Target", "Target content.", ["workspace"], memory_id=str(target_id), memory_type="fact")
    source = repository.get_memory(str(source_id))
    target = repository.get_memory(str(target_id))
    assert source is not None and target is not None
    executor = CurationExecutor(SQLiteCurationActionStore(db_manager))
    create_action = _link_action(
        source_id,
        target_id,
        record_token(source),
        record_token(target),
        link_type="DEPENDS_ON",
        context="link to remove",
    )
    executor.execute_create_link(
        create_action,
        run_id=run.run_id,
        source_type=source.type,
        target_type=target.type,
    )
    action = _remove_link_action(source_id, target_id, record_token(source), record_token(target))
    receipt = executor.execute_remove_link(
        action,
        run_id=run.run_id,
        source_type=source.type,
        target_type=target.type,
    )

    verifier, reader = _verifier(db_manager, repository)
    verified = verifier.verify(receipt, action)

    assert verified.status is CurationReceiptState.VERIFIED
    assert set(reader.peek_calls) == {str(source_id), str(target_id)}


def test_merge_verifies_canonical_archival_and_supercedes_lineage(db_manager: DatabaseManager) -> None:
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
    action = _merge_action(canonical_id, source_id, record_token(canonical), record_token(source))
    receipt = CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_merge(
        action,
        run_id=run.run_id,
        memory_types={canonical_id: canonical.type, source_id: source.type},
    )

    verifier, reader = _verifier(db_manager, repository)
    verified = verifier.verify(receipt, action)

    assert verified.status is CurationReceiptState.VERIFIED
    assert set(reader.peek_calls) == {str(canonical_id), str(source_id)}


def test_split_verifies_children_and_split_lineage(db_manager: DatabaseManager) -> None:
    repository, run, memory_id = _seed(db_manager)
    before = repository.get_memory(str(memory_id))
    assert before is not None
    action = _split_action(memory_id, record_token(before))
    receipt = CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_split(
        action,
        run_id=run.run_id,
        memory_type=before.type,
    )

    verifier, reader = _verifier(db_manager, repository)
    verified = verifier.verify(receipt, action)

    assert verified.status is CurationReceiptState.VERIFIED
    assert set(reader.peek_calls) == set(str(value) for value in receipt.affected_ids)


def test_archive_verifies_archived_status_from_fresh_read(db_manager: DatabaseManager) -> None:
    repository, run, memory_id = _seed(db_manager)
    before = repository.get_memory(str(memory_id))
    assert before is not None
    action = _archive_action(memory_id, record_token(before))
    receipt = CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_archive(
        action,
        run_id=run.run_id,
        memory_type=before.type,
    )

    verifier, reader = _verifier(db_manager, repository)
    verified = verifier.verify(receipt, action)

    assert verified.status is CurationReceiptState.VERIFIED
    assert reader.peek_calls == [str(memory_id)]
