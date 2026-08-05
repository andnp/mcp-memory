from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID, uuid4

import pytest

from mcp_memory.core.curation_models import CurationBudgetUsage, CurationRunOutcome
from mcp_memory.curation_store import (
    MAX_CURATION_READ_LIMIT,
    CandidateDisposition,
    CurationActionReceipt,
    CurationCandidateState,
    CurationReceiptIdentityConflictError,
    CurationReceiptState,
    CurationRun,
    CurationRunState,
)


class CurationRepositoryLike(Protocol):
    def create_run(self, run: CurationRun) -> CurationRun: ...

    def get_run(self, run_id: UUID) -> CurationRun | None: ...

    def transition_run(
        self,
        run_id: UUID,
        expected_state: CurationRunState,
        run: CurationRun,
    ) -> CurationRun | None: ...

    def terminalize_run(
        self,
        run_id: UUID,
        expected_state: CurationRunState,
        outcome: CurationRunOutcome,
    ) -> CurationRun | None: ...

    def put_receipt(self, receipt: CurationActionReceipt) -> CurationActionReceipt: ...

    def get_receipt(self, run_id: UUID, action_id: UUID) -> CurationActionReceipt | None: ...

    def transition_receipt(
        self,
        run_id: UUID,
        action_id: UUID,
        expected_state: CurationReceiptState,
        receipt: CurationActionReceipt,
    ) -> CurationActionReceipt | None: ...

    def get_candidate_state(self, memory_id: UUID) -> CurationCandidateState | None: ...

    def list_candidate_states(
        self,
        *,
        disposition: CandidateDisposition | None = None,
        limit: int = 100,
    ) -> Sequence[CurationCandidateState]: ...

    def put_candidate_state(self, state: CurationCandidateState) -> CurationCandidateState: ...

    def is_candidate_in_cooldown(self, memory_id: UUID, *, now: datetime | None = None) -> bool: ...

    def list_runs(self, *, limit: int = 100) -> Sequence[CurationRun]: ...

    def list_receipts(self, run_id: UUID, *, limit: int = 100) -> Sequence[CurationActionReceipt]: ...


RepositoryFactory = Callable[[], CurationRepositoryLike]


def _run() -> CurationRun:
    return CurationRun(
        run_id=uuid4(),
        frontier_key="frontier",
        context_fingerprint="context",
        budget_usage=CurationBudgetUsage(premium_requests=1),
        disclosure_audit={
            "version": 1,
            "provider_trust_class": "external",
            "records": [
                {
                    "memory_id": str(uuid4()),
                    "decision": "redact",
                    "fields": [{"field": "content", "decision": "redact"}],
                }
            ],
        },
    )


def _receipt(run_id: UUID) -> CurationActionReceipt:
    return CurationActionReceipt(
        run_id=run_id,
        action_id=uuid4(),
        operation="normalize_memory",
        affected_ids=[uuid4()],
        status=CurationReceiptState.APPLIED_UNVERIFIED,
        before_token="before",
        intent_hash="intent-hash",
    )


def assert_curation_repository_contract(make_repository: RepositoryFactory) -> None:
    """Assert the backend-neutral curation ledger contract."""
    repository = make_repository()

    # Runs follow the complete lifecycle, and conditional writes reject stale
    # transitions while preserving the first terminal result.
    run = repository.create_run(_run())
    planning = run.model_copy(update={"state": CurationRunState.PLANNING})
    assert repository.transition_run(run.run_id, CurationRunState.CREATED, planning) == planning

    executing = planning.model_copy(update={"state": CurationRunState.EXECUTING})
    assert repository.transition_run(run.run_id, CurationRunState.PLANNING, executing) == executing
    assert repository.transition_run(run.run_id, CurationRunState.CREATED, run) is None

    verifying = executing.model_copy(update={"state": CurationRunState.VERIFYING})
    assert repository.transition_run(run.run_id, CurationRunState.EXECUTING, verifying) == verifying
    terminal = repository.terminalize_run(
        run.run_id,
        CurationRunState.VERIFYING,
        CurationRunOutcome.APPLIED,
    )
    assert terminal is not None
    assert terminal.state is CurationRunState.TERMINAL
    assert terminal.outcome is CurationRunOutcome.APPLIED
    assert terminal.terminalized_at is not None
    assert repository.get_run(run.run_id) == terminal
    assert terminal.budget_usage.premium_requests == 1
    assert repository.terminalize_run(
        run.run_id,
        CurationRunState.VERIFYING,
        CurationRunOutcome.CANCELLED,
    ) == terminal
    assert repository.transition_run(
        run.run_id,
        CurationRunState.TERMINAL,
        run.model_copy(update={"state": CurationRunState.TERMINAL, "outcome": CurationRunOutcome.CANCELLED}),
    ) == terminal

    # Replaying an action returns its original receipt; changing its identity
    # is rejected, and a terminal receipt cannot be replaced by a late write.
    receipt = repository.put_receipt(_receipt(run.run_id))
    assert repository.get_receipt(run.run_id, receipt.action_id) == receipt
    replay = repository.put_receipt(
        receipt.model_copy(update={"status": CurationReceiptState.VERIFIED})
    )
    assert replay == receipt
    with pytest.raises(CurationReceiptIdentityConflictError):
        repository.put_receipt(receipt.model_copy(update={"operation": "create_link"}))

    verified = receipt.model_copy(
        update={
            "status": CurationReceiptState.VERIFIED,
            "verified_at": datetime(2026, 1, 1, tzinfo=UTC),
        }
    )
    assert repository.transition_receipt(
        run.run_id,
        receipt.action_id,
        CurationReceiptState.APPLIED_UNVERIFIED,
        verified,
    ) == verified
    assert repository.transition_receipt(
        run.run_id,
        receipt.action_id,
        CurationReceiptState.APPLIED_UNVERIFIED,
        verified.model_copy(update={"status": CurationReceiptState.FAILED, "error_code": "late"}),
    ) == verified

    # Candidate writes retain the no-op stabilization and cooldown state, with
    # the boundary itself treated as no longer being in cooldown.
    memory_id = uuid4()
    cooldown_until = datetime(2026, 1, 1, 0, 5, tzinfo=UTC)
    candidate = CurationCandidateState(
        memory_id=memory_id,
        last_observed_revision_token="revision-1",
        disposition=CandidateDisposition.COOLDOWN,
        consecutive_no_op_count=2,
        cooldown_until=cooldown_until,
        last_disposition_reason="unchanged",
        last_frontier_key="frontier",
        last_run_id=run.run_id,
        escalation_count=1,
        last_escalated_strategy="cold",
        last_considered_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        last_considered_strategy="cold-storage",
        last_mutation_family="curator",
        last_mutated_at=datetime(2026, 1, 1, 0, 2, tzinfo=UTC),
        coverage_evidence_json={"reason": "unchanged", "source": "cold-storage"},
    )
    assert repository.put_candidate_state(candidate) == candidate
    assert repository.get_candidate_state(memory_id) == candidate
    escalated = candidate.model_copy(
        update={
            "memory_id": uuid4(),
            "disposition": CandidateDisposition.ESCALATED,
            "last_disposition_reason": "retrieval_regression",
            "escalation_count": 2,
        }
    )
    assert repository.put_candidate_state(escalated) == escalated
    assert repository.list_candidate_states(
        disposition=CandidateDisposition.ESCALATED,
        limit=1,
    ) == [escalated]
    assert repository.is_candidate_in_cooldown(
        memory_id,
        now=cooldown_until - timedelta(seconds=1),
    )
    assert not repository.is_candidate_in_cooldown(memory_id, now=cooldown_until)

    stabilized = candidate.model_copy(
        update={
            "last_observed_revision_token": "revision-2",
            "disposition": CandidateDisposition.ACTIONED,
            "consecutive_no_op_count": 0,
            "cooldown_until": None,
        }
    )
    assert repository.put_candidate_state(stabilized) == stabilized
    assert repository.get_candidate_state(memory_id) == stabilized
    assert not repository.is_candidate_in_cooldown(memory_id, now=cooldown_until)

    # Both collection reads reject non-positive limits and cap oversized
    # requests at the shared maximum.
    for _ in range(MAX_CURATION_READ_LIMIT):
        repository.create_run(_run())
    assert len(repository.list_runs(limit=MAX_CURATION_READ_LIMIT + 1)) == MAX_CURATION_READ_LIMIT
    with pytest.raises(ValueError):
        repository.list_runs(limit=0)

    receipt_run = repository.create_run(_run())
    for _ in range(MAX_CURATION_READ_LIMIT):
        repository.put_receipt(_receipt(receipt_run.run_id))
    assert len(repository.list_receipts(receipt_run.run_id, limit=MAX_CURATION_READ_LIMIT + 1)) == MAX_CURATION_READ_LIMIT
    with pytest.raises(ValueError):
        repository.list_receipts(receipt_run.run_id, limit=0)
