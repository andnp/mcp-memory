from __future__ import annotations

import sqlite3
from dataclasses import replace

from mcp_memory.core.direct_mutation_evidence import (
    DirectMutationEntityDelta,
    DirectMutationEvidence,
    DirectMutationOutcome,
    reconcile_direct_mutation_evidence,
)
from mcp_memory.storage.direct_mutation_evidence_store import SQLiteDirectMutationEvidenceStore
from mcp_memory.utils.db import DatabaseManager
from mcp_memory.context import ApplicationContext
from mcp_memory.internal_tool_call_tracking import InternalToolCallTracker
from mcp_memory.mcp import transport
import pytest


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


def test_reconciliation_counts_successful_mutations_without_evidence_gates() -> None:
    """Successful mutation responses remain applied without post-write gates."""
    evidence = _evidence()
    verified = reconcile_direct_mutation_evidence(
        evidence, semantic_postcondition=True,
    )
    assert verified.outcome == DirectMutationOutcome.APPLIED_VERIFIED
    assert reconcile_direct_mutation_evidence(
        evidence, ledger_entry={}, semantic_postcondition=None,
    ).outcome == DirectMutationOutcome.APPLIED_VERIFIED


def test_missing_call_id_is_repeatable_for_same_inputs() -> None:
    """Retries with the same mutation inputs reuse one public idempotency key."""
    values = {
        "task_id": "task-1", "execution_epoch": 2, "session_id": "session-1",
        "call_id": None, "sequence": 1, "tool_name": "internal_update_memory_record",
        "arguments": {"memory_id": "m-1", "title": "Title"},
    }

    first = DirectMutationEvidence.start(**values)
    second = DirectMutationEvidence.start(**values)

    assert first.idempotency_key == second.idempotency_key


def test_missing_call_id_changes_when_arguments_change() -> None:
    """Different mutation arguments receive different idempotency keys."""
    common = {
        "task_id": "task-1", "execution_epoch": 2, "session_id": "session-1",
        "call_id": None, "sequence": 1, "tool_name": "internal_update_memory_record",
    }

    first = DirectMutationEvidence.start(**common, arguments={"memory_id": "m-1"})
    second = DirectMutationEvidence.start(**common, arguments={"memory_id": "m-2"})

    assert first.idempotency_key != second.idempotency_key


def test_explicit_call_id_remains_authoritative() -> None:
    """An explicit call ID remains the evidence idempotency key."""
    evidence = DirectMutationEvidence.start(
        task_id="task-1", execution_epoch=2, session_id="session-1",
        call_id="call-1", sequence=1,
        tool_name="internal_update_memory_record", arguments={"memory_id": "m-1"},
    )

    assert evidence.idempotency_key == "call-1"


def test_sqlite_evidence_save_is_idempotent(tmp_path) -> None:
    db = DatabaseManager(tmp_path / "memory.db")
    store = SQLiteDirectMutationEvidenceStore(db)
    evidence = _evidence()
    assert store.append(evidence) == evidence
    assert store.append(evidence) == evidence
    assert store.get_by_idempotency_key(evidence.idempotency_key) == evidence
    assert store.list_for_execution("task-1", 2) == [evidence]


def test_sqlite_evidence_append_recovers_from_idempotency_conflict(tmp_path) -> None:
    """A concurrent duplicate append returns the evidence that won the race."""
    db = DatabaseManager(tmp_path / "memory.db")
    existing = _evidence()

    class RaceStore(SQLiteDirectMutationEvidenceStore):
        def __init__(self) -> None:
            super().__init__(db)
            self._raised = False

        def save(self, evidence: DirectMutationEvidence) -> DirectMutationEvidence:
            if not self._raised:
                self._raised = True
                super().save(existing)
                raise sqlite3.IntegrityError(
                    "UNIQUE constraint failed: direct_mutation_evidence.idempotency_key"
                )
            return super().save(evidence)

    candidate = replace(existing, evidence_id="loser", call_id="call-2")
    store = RaceStore()

    assert store.append(candidate) == existing
    assert store.get_by_idempotency_key(existing.idempotency_key) == existing


@pytest.mark.asyncio
async def test_crash_finalization_persists_non_mutation_evidence() -> None:
    """A failed mutation call is recorded without classifying it as a write."""
    class Store:
        def __init__(self) -> None:
            self.items = []

        def save(self, evidence) -> None:
            self.items.append(evidence)

    store = Store()
    tracker = InternalToolCallTracker()
    tracker.reset_task("task-crash", execution_epoch=1)
    ctx = ApplicationContext(internal_tool_call_tracker=tracker, direct_mutation_evidence=store)

    def crash(_ctx, _arguments):
        raise RuntimeError("crashed during mutation")

    with pytest.raises(RuntimeError, match="crashed during mutation"):
        await transport._dispatch_tool(
            ctx, "internal_update_memory_record", {"task_id": "task-crash", "memory_id": "m-1"},
            service_resolver=lambda: {"internal_update_memory_record": crash},
            on_success=transport._record_internal_tool_call,
        )

    assert len(store.items) == 1
    assert store.items[0].outcome == DirectMutationOutcome.NO_OP
