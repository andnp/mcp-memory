from __future__ import annotations

from mcp_memory.core.direct_mutation_evidence import (
    DirectMutationEntityDelta,
    DirectMutationEvidence,
    DirectMutationOutcome,
    reconcile_direct_mutation_evidence,
)
from mcp_memory.storage.direct_mutation_evidence_store import SQLiteDirectMutationEvidenceStore
from mcp_memory.utils.db import DatabaseManager


def _evidence() -> DirectMutationEvidence:
    started = DirectMutationEvidence.start(
        task_id="task-1", execution_epoch=2, session_id="session-1", call_id="call-1",
        sequence=1, tool_name="internal_update_memory_record", arguments={"memory_id": "m-1"},
    )
    return started.finish(
        payload={"status": "ok", "record": {"id": "m-1", "updated_at": "after"}},
        ledger_entry={"tool_name": "internal_update_memory_record", "call_id": "call-1", "task_id": "task-1", "execution_epoch": 2, "status": "success"},
        deltas=(DirectMutationEntityDelta("record", "m-1", "before", "after", True, True),),
    )


def test_reconciliation_requires_ledger_delta_and_postcondition() -> None:
    evidence = _evidence()
    verified = reconcile_direct_mutation_evidence(
        evidence, semantic_postcondition=True,
    )
    assert verified.outcome == DirectMutationOutcome.APPLIED_VERIFIED
    assert reconcile_direct_mutation_evidence(evidence, ledger_entry={}, semantic_postcondition=True).outcome == DirectMutationOutcome.LEDGER_INVALID
    assert reconcile_direct_mutation_evidence(evidence, semantic_postcondition=None).outcome == DirectMutationOutcome.APPLIED_UNVERIFIED


def test_sqlite_evidence_save_is_idempotent(tmp_path) -> None:
    db = DatabaseManager(tmp_path / "memory.db")
    store = SQLiteDirectMutationEvidenceStore(db)
    evidence = _evidence()
    assert store.append(evidence) == evidence
    assert store.append(evidence) == evidence
    assert store.get_by_idempotency_key(evidence.idempotency_key) == evidence
    assert store.list_for_execution("task-1", 2) == [evidence]
