from types import SimpleNamespace
from typing import cast

import pytest

from mcp_memory.core.ingest_claim_lifecycle import _finalize_claimed_ingest_entries
from mcp_memory.core.ingress_evidence import (
    IngressActionReceipt,
    IngressBatchEvidence,
    IngressReceiptStatus,
    IngressSourceSnapshot,
    SourceCoverage,
    SourceCoverageOutcome,
)
from mcp_memory.context import ApplicationContext


pytestmark = pytest.mark.small


class _Journal:
    def __init__(self, claimed_ids: list[int]) -> None:
        self.claimed_ids = claimed_ids
        self.moved_ids: list[int] = []
        self.released_ids: list[int] = []

    def get_claimed_entry_ids(self, task_id: str) -> list[int]:
        return self.claimed_ids

    def move_claimed_entry_ids_to_recoverable(self, task_id: str, entry_ids: list[int]) -> list[int]:
        self.moved_ids = list(entry_ids)
        return list(entry_ids)

    def release_claimed_entry_ids(self, task_id: str, entry_ids: list[int]) -> list[int]:
        self.released_ids = list(entry_ids)
        return list(entry_ids)


class _SourceCoverage:
    def __init__(self, records: list[SourceCoverage]) -> None:
        self.records = records

    def list_for_entries(self, entry_ids: tuple[str, ...]) -> list[SourceCoverage]:
        return [record for record in self.records if record.entry_id in entry_ids]


class _ActionReceipts:
    def __init__(self, records: list[IngressActionReceipt]) -> None:
        self.records = {record.action_id: record for record in records}

    def get(self, action_id: str) -> IngressActionReceipt | None:
        return self.records.get(action_id)


class _Batches:
    def __init__(self, records: list[IngressBatchEvidence]) -> None:
        self.records = {record.batch_id: record for record in records}

    def get(self, batch_id: str) -> IngressBatchEvidence | None:
        return self.records.get(batch_id)


def _context(
    journal: _Journal,
    records: list[SourceCoverage] | None = None,
    receipts: list[IngressActionReceipt] | None = None,
    batches: list[IngressBatchEvidence] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        journal=journal,
        source_coverage=None if records is None else _SourceCoverage(records),
        ingress_action_receipts=None if receipts is None else _ActionReceipts(receipts),
        ingress_batch_evidence=None if batches is None else _Batches(batches),
    )


def _receipt(action_id: str, entry_id: int, batch_id: str = "batch-1") -> IngressActionReceipt:
    return IngressActionReceipt(
        action_id=action_id,
        batch_id=batch_id,
        operation="create",
        entry_ids=(str(entry_id),),
        target_ids=(),
        canonical_payload_digest="digest",
        status=IngressReceiptStatus.APPLIED_UNVERIFIED,
        mutation_evidence_id="event-1",
        created_at="2026-08-08T12:00:00+00:00",
    )


def _batch(batch_id: str = "batch-1", task_id: str = "task-1") -> IngressBatchEvidence:
    return IngressBatchEvidence(
        batch_id=batch_id,
        task_id=task_id,
        execution_epoch=1,
        batch_sequence=1,
        claimed_entry_ids=("1",),
        source_fingerprint="fingerprint",
        source_entries=(
            IngressSourceSnapshot(
                entry_id="1",
                workspace_ids=("workspace",),
                timestamp="2026-08-08T12:00:00+00:00",
                content_digest="content",
            ),
        ),
        provider_route="test",
        execution_mode="test",
        policy_version="v1",
        schema_version="1",
        claimed_at="2026-08-08T12:00:00+00:00",
    )


def test_finalize_reports_atomic_pre_finalized_entries() -> None:
    """Atomic recoverable entries remain visible in the later task result."""
    journal = _Journal([3, 4])
    ctx = _context(
        journal,
        [SourceCoverage("1", SourceCoverageOutcome.CREATED, "action-1")],
        [_receipt("action-1", 1)],
        [_batch()],
    )

    result = _finalize_claimed_ingest_entries(cast(ApplicationContext, ctx), task_id="task-1", handled_entry_ids=[1, 3])

    assert result == ([3, 4, 1], [1, 3], [4])
    assert journal.moved_ids == [3]


def test_finalize_preserves_legacy_claim_finalization() -> None:
    """Claims without atomic evidence use the legacy move and release flow."""
    journal = _Journal([1, 2])

    result = _finalize_claimed_ingest_entries(cast(ApplicationContext, _context(journal)), task_id="task-1", handled_entry_ids=[1])

    assert result == ([1, 2], [1], [2])


def test_finalize_releases_unhandled_claims() -> None:
    """Unhandled claims return to pending while handled claims become recoverable."""
    journal = _Journal([1, 2])

    result = _finalize_claimed_ingest_entries(cast(ApplicationContext, _context(journal)), task_id="task-1", handled_entry_ids=[])

    assert result == ([1, 2], [], [1, 2])


def test_finalize_does_not_release_another_tasks_claim() -> None:
    """Finalization only passes claims still owned by this task to release."""
    journal = _Journal([1])
    ctx = _context(
        journal,
        [SourceCoverage("2", SourceCoverageOutcome.APPENDED, "action-2")],
        [_receipt("action-2", 2, "batch-2")],
        [_batch("batch-2", "other-task")],
    )

    result = _finalize_claimed_ingest_entries(cast(ApplicationContext, ctx), task_id="task-1", handled_entry_ids=[2])

    assert result == ([1], [], [1])
    assert 2 not in journal.released_ids
