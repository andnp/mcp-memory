from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from mcp_memory.core.curation_executor import CurationExecutor
from mcp_memory.core.curation_identity import record_token
from mcp_memory.core.curation_models import ActionPreconditions, CreateLinkAction, EvidenceRef, LinkAssertion, NormalizeMemoryAction
from mcp_memory.core.curation_verifier import CurationVerifier
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


def _link_action(source_id: UUID, target_id: UUID, source_token: str, target_token: str) -> CreateLinkAction:
    edge = LinkAssertion(
        source_id=source_id,
        target_id=target_id,
        link_type="SUPPORTS",
        context="exact typed edge",
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
            absent_links=[edge],
        ),
        link_type="SUPPORTS",
        context="exact typed edge",
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
