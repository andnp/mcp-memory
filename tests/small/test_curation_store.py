from datetime import UTC, datetime
from uuid import uuid4

import pytest

from mcp_memory.core.curation_models import CurationRunOutcome
from mcp_memory.curation_store import (
    CurationActionReceipt,
    CurationCandidateState,
    CurationReceiptState,
    CurationRun,
    CurationRunState,
    terminalize_run,
    transition_receipt,
    transition_run,
)


def make_run() -> CurationRun:
    return CurationRun(run_id=uuid4(), frontier_key="frontier", context_fingerprint="context")


def test_run_transitions_are_explicit_and_terminalization_is_first_write_wins() -> None:
    run = make_run()
    run = transition_run(run, CurationRunState.PLANNING)
    run = transition_run(run, CurationRunState.EXECUTING)
    finished_at = datetime.now(UTC)
    run = terminalize_run(run, CurationRunOutcome.APPLIED, terminalized_at=finished_at)

    late = terminalize_run(run, CurationRunOutcome.CANCELLED)
    assert late is run
    assert late.outcome is CurationRunOutcome.APPLIED
    assert late.terminalized_at == finished_at


def test_invalid_run_transition_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid curation run transition"):
        transition_run(make_run(), CurationRunState.TERMINAL, outcome=CurationRunOutcome.NO_OP)


def test_receipt_terminal_state_cannot_be_replaced() -> None:
    receipt = CurationActionReceipt(
        run_id=uuid4(), action_id=uuid4(), operation="normalize_memory",
        status=CurationReceiptState.APPLIED_UNVERIFIED,
    )
    verified = transition_receipt(receipt, CurationReceiptState.VERIFIED)
    late = transition_receipt(verified, CurationReceiptState.FAILED, error_code="late_failure")
    assert late is verified
    assert late.status is CurationReceiptState.VERIFIED
    failed = transition_receipt(receipt, CurationReceiptState.FAILED, error_code="failed")
    assert failed.state is CurationReceiptState.FAILED


def test_receipt_is_compact_and_candidate_defaults_are_isolated() -> None:
    receipt = CurationActionReceipt(
        run_id=uuid4(), action_id=uuid4(), operation="rewrite_memory",
        affected_ids=[uuid4()], status=CurationReceiptState.VERIFIED,
        before_token="before", after_token="after",
    )
    assert "content" not in receipt.model_dump()
    first = CurationCandidateState(memory_id=uuid4())
    second = CurationCandidateState(memory_id=uuid4())
    first.consecutive_no_op_count = 1
    assert second.consecutive_no_op_count == 0
