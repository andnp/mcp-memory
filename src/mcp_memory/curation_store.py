"""Pure curation-ledger rows, transitions, and repository protocol."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Protocol, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from mcp_memory.core.curation_models import CurationBudgetUsage, CurationRunOutcome


class CurationStoreModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CurationRunState(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    TERMINAL = "terminal"


class CurationReceiptState(StrEnum):
    APPLIED_UNVERIFIED = "applied_unverified"
    VERIFIED = "verified"
    REJECTED = "rejected"
    STALE = "stale"
    FAILED = "failed"


class CandidateDisposition(StrEnum):
    PENDING = "pending"
    RETAINED = "retained"
    ACTIONED = "actioned"
    COOLDOWN = "cooldown"
    ESCALATED = "escalated"


class CurationRun(CurationStoreModel):
    run_id: UUID
    task_id: UUID | None = None
    work_item_id: UUID | None = None
    frontier_key: str
    selector_strategy: str | None = None
    context_fingerprint: str
    planner_id: str | None = None
    provider_id: str | None = None
    model_id: str | None = None
    policy_version: str = "1"
    schema_version: int = 1
    state: CurationRunState = CurationRunState.CREATED
    outcome: CurationRunOutcome | None = None
    plan_id: UUID | None = None
    rejection_codes: list[str] = Field(default_factory=list)
    retry_reason: str | None = None
    budget_usage: CurationBudgetUsage = Field(default_factory=CurationBudgetUsage)
    created_at: datetime | None = None
    terminalized_at: datetime | None = None


class CurationActionReceipt(CurationStoreModel):
    """Compact execution evidence; content belongs in mutation-history revisions."""

    run_id: UUID
    action_id: UUID
    operation: str
    affected_ids: list[UUID] = Field(default_factory=list)
    status: CurationReceiptState
    before_token: str | None = None
    after_token: str | None = None
    mutation_event_id: UUID | None = None
    error_code: str | None = None
    applied_at: datetime | None = None
    verified_at: datetime | None = None

    @property
    def state(self) -> CurationReceiptState:
        return self.status


class CurationCandidateState(CurationStoreModel):
    memory_id: UUID
    last_observed_revision_token: str | None = None
    disposition: CandidateDisposition = CandidateDisposition.PENDING
    consecutive_no_op_count: int = Field(default=0, ge=0)
    cooldown_until: datetime | None = None
    last_disposition_reason: str | None = None
    last_frontier_key: str | None = None
    last_run_id: UUID | None = None
    escalation_count: int = Field(default=0, ge=0)
    last_escalated_strategy: str | None = None


_RUN_TRANSITIONS: dict[CurationRunState, frozenset[CurationRunState]] = {
    CurationRunState.CREATED: frozenset({CurationRunState.PLANNING}),
    CurationRunState.PLANNING: frozenset({CurationRunState.EXECUTING, CurationRunState.TERMINAL}),
    CurationRunState.EXECUTING: frozenset({CurationRunState.VERIFYING, CurationRunState.TERMINAL}),
    CurationRunState.VERIFYING: frozenset({CurationRunState.TERMINAL}),
    CurationRunState.TERMINAL: frozenset(),
}


def is_terminal_run(run: CurationRun) -> bool:
    return run.state is CurationRunState.TERMINAL


def transition_run(
    run: CurationRun,
    state: CurationRunState,
    *,
    outcome: CurationRunOutcome | None = None,
    terminalized_at: datetime | None = None,
) -> CurationRun:
    """Apply a legal run transition, preserving an existing terminal result."""
    if is_terminal_run(run):
        return run
    if state not in _RUN_TRANSITIONS[run.state]:
        raise ValueError(f"invalid curation run transition: {run.state} -> {state}")
    if state is CurationRunState.TERMINAL and outcome is None:
        raise ValueError("terminal curation runs require an outcome")
    return run.model_copy(
        update={
            "state": state,
            "outcome": outcome,
            "terminalized_at": terminalized_at if state is CurationRunState.TERMINAL else None,
        }
    )


def terminalize_run(
    run: CurationRun,
    outcome: CurationRunOutcome,
    *,
    terminalized_at: datetime | None = None,
) -> CurationRun:
    """Conditionally terminalize a run: the first terminal outcome wins."""
    return transition_run(
        run,
        CurationRunState.TERMINAL,
        outcome=outcome,
        terminalized_at=terminalized_at,
    )


_RECEIPT_TRANSITIONS: dict[CurationReceiptState, frozenset[CurationReceiptState]] = {
    CurationReceiptState.APPLIED_UNVERIFIED: frozenset(
        {
            CurationReceiptState.VERIFIED,
            CurationReceiptState.REJECTED,
            CurationReceiptState.STALE,
            CurationReceiptState.FAILED,
        }
    ),
    CurationReceiptState.VERIFIED: frozenset(),
    CurationReceiptState.REJECTED: frozenset(),
    CurationReceiptState.STALE: frozenset(),
    CurationReceiptState.FAILED: frozenset(),
}


def is_terminal_receipt(receipt: CurationActionReceipt) -> bool:
    return receipt.status is not CurationReceiptState.APPLIED_UNVERIFIED


def transition_receipt(
    receipt: CurationActionReceipt,
    state: CurationReceiptState,
    *,
    verified_at: datetime | None = None,
    error_code: str | None = None,
) -> CurationActionReceipt:
    """Conditionally advance a receipt; terminal receipt state is immutable."""
    if is_terminal_receipt(receipt):
        return receipt
    if state not in _RECEIPT_TRANSITIONS[receipt.status]:
        raise ValueError(f"invalid curation receipt transition: {receipt.status} -> {state}")
    return receipt.model_copy(update={"status": state, "verified_at": verified_at, "error_code": error_code})


class CurationRepository(Protocol):
    """Backend-neutral protocol for the curation ledger."""

    def create_run(self, run: CurationRun) -> CurationRun: ...

    def get_run(self, run_id: UUID) -> CurationRun | None: ...

    def transition_run(self, run_id: UUID, expected_state: CurationRunState, run: CurationRun) -> CurationRun | None: ...

    def terminalize_run(
        self, run_id: UUID, expected_state: CurationRunState, outcome: CurationRunOutcome
    ) -> CurationRun | None: ...

    def put_receipt(self, receipt: CurationActionReceipt) -> CurationActionReceipt: ...

    def get_receipt(self, run_id: UUID, action_id: UUID) -> CurationActionReceipt | None: ...

    def transition_receipt(
        self, run_id: UUID, action_id: UUID, expected_state: CurationReceiptState, receipt: CurationActionReceipt
    ) -> CurationActionReceipt | None: ...

    def get_candidate_state(self, memory_id: UUID) -> CurationCandidateState | None: ...

    def put_candidate_state(self, state: CurationCandidateState) -> CurationCandidateState: ...

    def list_receipts(self, run_id: UUID) -> Sequence[CurationActionReceipt]: ...
