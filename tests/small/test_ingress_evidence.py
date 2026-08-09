from __future__ import annotations

from typing import get_type_hints

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
    IngressActionReceiptRepository,
    IngressBatchEvidenceRepository,
    SourceCoverageRepository,
)

pytestmark = pytest.mark.small


def test_receipt_status_and_coverage_outcome_are_separate_contracts() -> None:
    """Receipt state and source quality disposition expose distinct vocabularies."""
    assert {status.value for status in IngressReceiptStatus} == {
        "applied_unverified",
        "no_op",
        "stale",
        "failed",
        "unobserved",
    }
    assert {outcome.value for outcome in SourceCoverageOutcome} == {
        "created",
        "appended",
        "matched_existing",
        "ignored",
        "no_mutation",
        "unobserved",
    }
    assert IngressReceiptStatus.NO_OP.value != SourceCoverageOutcome.NO_MUTATION.value


def test_evidence_records_preserve_replay_and_terminal_metadata() -> None:
    """Batch, receipt, and coverage records retain the reviewed evidence fields."""
    source = IngressSourceSnapshot(
        entry_id="entry-1",
        workspace_ids=("workspace-1",),
        timestamp="2026-08-08T12:00:00+00:00",
        content_digest="digest-1",
        snapshot={"content": "bounded"},
    )
    batch = IngressBatchEvidence(
        batch_id="batch-1",
        task_id="task-1",
        execution_epoch=3,
        batch_sequence=2,
        claimed_entry_ids=("entry-1",),
        source_fingerprint="fingerprint-1",
        source_entries=(source,),
        provider_route="provider/model",
        execution_mode="agentic_mcp",
        policy_version="policy-1",
        schema_version="schema-1",
        claimed_at="2026-08-08T12:00:01+00:00",
    )
    receipt = IngressActionReceipt(
        action_id="action-1",
        batch_id=batch.batch_id,
        operation="append",
        entry_ids=batch.claimed_entry_ids,
        target_ids=("memory-1",),
        canonical_payload_digest="payload-1",
        status=IngressReceiptStatus.APPLIED_UNVERIFIED,
        mutation_evidence_id="mutation-1",
        created_at="2026-08-08T12:00:02+00:00",
        terminalized_at="2026-08-08T12:00:03+00:00",
    )
    coverage = SourceCoverage("entry-1", SourceCoverageOutcome.APPENDED, receipt.action_id)

    assert batch.source_entries == (source,)
    assert receipt.status is IngressReceiptStatus.APPLIED_UNVERIFIED
    assert receipt.mutation_evidence_id == "mutation-1"
    assert coverage.outcome is SourceCoverageOutcome.APPENDED
    assert coverage.action_id == receipt.action_id


def test_ports_expose_typed_record_contracts() -> None:
    """Repository protocols name their record inputs and outputs explicitly."""
    assert get_type_hints(IngressBatchEvidenceRepository.save)["evidence"] is IngressBatchEvidence
    assert get_type_hints(IngressActionReceiptRepository.save)["receipt"] is IngressActionReceipt
    assert get_type_hints(SourceCoverageRepository.save)["coverage"] is SourceCoverage
