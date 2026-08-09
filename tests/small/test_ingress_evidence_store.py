from __future__ import annotations

import json
from dataclasses import replace
from typing import cast

import pytest

from mcp_memory.core.ingress_evidence import (
    IngressActionReceipt,
    IngressBatchEvidence,
    IngressReceiptStatus,
    IngressSourceSnapshot,
    SourceCoverage,
    SourceCoverageOutcome,
)
from mcp_memory.core.ports.ingress import (
    IngressActionReceiptIdentityConflictError,
    SourceCoverageAssignmentConflictError,
)
from mcp_memory.storage.ingress_evidence_store import (
    SQLiteIngressActionReceiptStore,
    SQLiteIngressBatchEvidenceStore,
    SQLiteSourceCoverageStore,
)
from mcp_memory.utils.db import DatabaseManager

pytestmark = pytest.mark.small


def _batch(batch_id: str, sequence: int = 1) -> IngressBatchEvidence:
    return IngressBatchEvidence(
        batch_id=batch_id,
        task_id="task-1",
        execution_epoch=4,
        batch_sequence=sequence,
        claimed_entry_ids=("entry-2", "entry-1"),
        source_fingerprint="source-fingerprint",
        source_entries=(
            IngressSourceSnapshot(
                entry_id="entry-1",
                workspace_ids=("workspace-b", "workspace-a"),
                timestamp="2026-08-08T12:00:00+00:00",
                content_digest="digest-1",
                snapshot={"z": 1, "a": "value"},
            ),
        ),
        provider_route="provider/model",
        execution_mode="agentic_mcp",
        policy_version="policy-1",
        schema_version="schema-1",
        claimed_at="2026-08-08T12:00:01+00:00",
        finalized_at=None,
        grouping_strategy=None,
        grouping_fallback_reason="fallback was not needed",
    )


def _receipt(action_id: str, batch_id: str = "batch-1") -> IngressActionReceipt:
    return IngressActionReceipt(
        action_id=action_id,
        batch_id=batch_id,
        operation="append",
        entry_ids=("entry-1",),
        target_ids=("memory-1",),
        canonical_payload_digest="payload-digest",
        status=IngressReceiptStatus.APPLIED_UNVERIFIED,
        mutation_evidence_id="mutation-1",
        created_at="2026-08-08T12:00:02+00:00",
        terminalized_at=None,
        error_code=None,
        before_revision_tokens={"memory-1": "before"},
        after_revision_tokens={"memory-1": "after"},
    )


def test_batch_store_round_trips_replay_fields_and_deterministic_json(db_manager: DatabaseManager) -> None:
    """Batch persistence restores nested snapshots, tuples, and nullable replay fields."""
    store = SQLiteIngressBatchEvidenceStore(db_manager)
    evidence = _batch("batch-1")

    assert store.save(evidence) == evidence
    assert store.get(evidence.batch_id) == evidence
    raw = db_manager.get_connection().execute(
        "SELECT claimed_entry_ids_json, source_entries_json FROM ingress_batch_evidence"
    ).fetchone()
    assert raw["claimed_entry_ids_json"] == '["entry-2","entry-1"]'
    assert json.loads(raw["source_entries_json"])[0]["snapshot"] == {"a": "value", "z": 1}


def test_batch_store_lists_by_execution_sequence(db_manager: DatabaseManager) -> None:
    """Execution listing is ordered by the committed batch sequence."""
    store = SQLiteIngressBatchEvidenceStore(db_manager)
    store.save(_batch("batch-2", sequence=2))
    store.save(_batch("batch-1", sequence=1))

    assert [item.batch_id for item in store.list_for_execution("task-1", 4)] == ["batch-1", "batch-2"]
    assert store.list_for_execution("missing", 4) == []


def test_receipt_store_round_trips_enum_mappings_and_nulls(db_manager: DatabaseManager) -> None:
    """Receipt persistence preserves status values, revision maps, and terminal nulls."""
    store = SQLiteIngressActionReceiptStore(db_manager)
    receipt = _receipt("action-1")

    store.save(receipt)

    assert store.get("action-1") == receipt
    assert store.list_for_batch("batch-1") == [receipt]
    row = db_manager.get_connection().execute(
        "SELECT status, before_revision_tokens_json, terminalized_at, error_code "
        "FROM ingress_action_receipts WHERE action_id = ?",
        ("action-1",),
    ).fetchone()
    assert row["status"] == "applied_unverified"
    assert row["before_revision_tokens_json"] == '{"memory-1":"before"}'
    assert row["terminalized_at"] is None
    assert row["error_code"] is None


def test_receipt_store_reserves_replays_and_rejects_payload_collisions(
    db_manager: DatabaseManager,
) -> None:
    """Reservation preserves the first receipt and fails closed on digest drift.

    A matching digest reuses the stored row, while a divergent digest does not
    replace it or create a second action receipt.
    """
    store = SQLiteIngressActionReceiptStore(db_manager)
    receipt = _receipt("action-1")

    assert store.reserve(receipt) == receipt
    replay = replace(receipt, status=IngressReceiptStatus.FAILED, error_code="late")
    assert store.reserve(replay) == receipt

    collision = replace(receipt, canonical_payload_digest="different-payload")
    with pytest.raises(IngressActionReceiptIdentityConflictError):
        store.reserve(collision)

    assert store.get("action-1") == receipt
    assert db_manager.get_connection().execute(
        "SELECT COUNT(*) FROM ingress_action_receipts WHERE action_id = ?", ("action-1",)
    ).fetchone()[0] == 1


def test_receipt_store_concurrent_reservations_converge_on_one_row(
    db_manager: DatabaseManager,
) -> None:
    """Concurrent duplicate reservations converge on one committed receipt.

    The real SQLite connections exercise the write-lock boundary used by
    independent worker threads.
    """
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    store = SQLiteIngressActionReceiptStore(db_manager)
    receipt = _receipt("action-1")
    barrier = Barrier(2)

    def reserve_after_barrier() -> IngressActionReceipt:
        barrier.wait()
        return store.reserve(receipt)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: reserve_after_barrier(), range(2)))

    assert results == [receipt, receipt]
    assert db_manager.get_connection().execute(
        "SELECT COUNT(*) FROM ingress_action_receipts WHERE action_id = ?", ("action-1",)
    ).fetchone()[0] == 1


def test_coverage_store_preserves_requested_entry_order_and_updates_rows(
    db_manager: DatabaseManager,
) -> None:
    """Coverage listing follows requested IDs and save replaces the same entry."""
    store = SQLiteSourceCoverageStore(db_manager)
    first = SourceCoverage("entry-1", SourceCoverageOutcome.CREATED, "action-1")
    second = SourceCoverage("entry-2", SourceCoverageOutcome.IGNORED, reason="duplicate")
    store.save(first)
    store.save(second)
    updated = SourceCoverage("entry-1", SourceCoverageOutcome.APPENDED, "action-2", "replayed")

    assert store.save(updated) == updated
    assert store.get("entry-1") == updated
    assert store.list_for_entries(("entry-2", "missing", "entry-1")) == [second, updated]
    assert store.list_for_entries(()) == []


def test_coverage_store_assigns_terminal_once_and_replays_exactly(db_manager: DatabaseManager) -> None:
    """Terminal assignment preserves the winner and rejects a different action."""
    store = SQLiteSourceCoverageStore(db_manager)
    first = SourceCoverage("entry-1", SourceCoverageOutcome.CREATED, "action-1", "created")

    assert store.assign_terminal(first) == first
    assert store.assign_terminal(replace(first, reason="replay detail")) == first

    with pytest.raises(SourceCoverageAssignmentConflictError):
        store.assign_terminal(SourceCoverage("entry-1", SourceCoverageOutcome.APPENDED, "action-2"))

    assert store.get("entry-1") == first


def test_coverage_store_rejects_released_unhandled_assignment(db_manager: DatabaseManager) -> None:
    """Journal release projection cannot be persisted as terminal coverage."""
    store = SQLiteSourceCoverageStore(db_manager)
    released = SourceCoverage("entry-1", cast(SourceCoverageOutcome, "released_unhandled"), None)

    with pytest.raises(ValueError, match="not terminal"):
        store.assign_terminal(released)
