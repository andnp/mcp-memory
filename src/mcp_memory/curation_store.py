"""Pure curation-ledger rows, transitions, and repository protocol."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from enum import StrEnum
from collections.abc import Sequence
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mcp_memory.core.curation_models import (
    CurationBudgetUsage,
    CurationRunOutcome,
    CurationVerificationDescriptor,
)
from mcp_memory.utils.db import DatabaseManager


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
    VERIFICATION_FAILED = "verification_failed"
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
    disclosure_audit: dict[str, Any] = Field(default_factory=dict)
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
    intent_hash: str | None = None
    verification_descriptor: CurationVerificationDescriptor | None = None
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
            CurationReceiptState.VERIFICATION_FAILED,
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

    def list_candidate_states(
        self,
        *,
        disposition: CandidateDisposition | None = None,
        limit: int = 100,
    ) -> Sequence[CurationCandidateState]: ...

    def put_candidate_state(self, state: CurationCandidateState) -> CurationCandidateState: ...

    def list_runs(self, *, limit: int = 100) -> Sequence[CurationRun]: ...

    def list_receipts(self, run_id: UUID, *, limit: int = 100) -> Sequence[CurationActionReceipt]: ...


MAX_CURATION_READ_LIMIT = 1000


class CurationReceiptIdentityConflictError(ValueError):
    """Raised when an action ID is reused for a different receipt identity."""


class CurationReceiptHydrationError(ValueError):
    """Raised when persisted receipt data cannot be hydrated safely."""


class SQLiteCurationStore:
    """SQLite persistence for the curation ledger.

    The store only persists run, receipt, and candidate state.  It deliberately
    does not invoke action callbacks or mutate memory records.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db = db_manager

    def create_run(self, run: CurationRun) -> CurationRun:
        normalized = run.model_copy(update={"created_at": run.created_at or _now()})
        conn = self._db.get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO curation_runs (
                    run_id, task_id, work_item_id, frontier_key, selector_strategy,
                    context_fingerprint, planner_id, provider_id, model_id,
                    policy_version, schema_version, state, outcome, plan_id,
                    rejection_codes_json, retry_reason, budget_usage_json,
                    disclosure_audit_json, created_at, terminalized_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _run_values(normalized),
            )
        return normalized

    def get_run(self, run_id: UUID) -> CurationRun | None:
        row = self._db.get_connection().execute(
            "SELECT * FROM curation_runs WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
        return None if row is None else _run_from_row(row)

    def list_runs(self, *, limit: int = 100) -> Sequence[CurationRun]:
        bounded_limit = _bounded_limit(limit)
        rows = self._db.get_connection().execute(
            "SELECT * FROM curation_runs ORDER BY created_at DESC, run_id DESC LIMIT ?",
            (bounded_limit,),
        ).fetchall()
        return [_run_from_row(row) for row in rows]

    def transition_run(
        self,
        run_id: UUID,
        expected_state: CurationRunState,
        run: CurationRun,
    ) -> CurationRun | None:
        existing = self.get_run(run_id)
        if existing is None:
            return None
        if existing.state is CurationRunState.TERMINAL:
            return existing
        normalized = run.model_copy(
            update={
                "run_id": run_id,
                "created_at": run.created_at or existing.created_at,
            }
        )
        conn = self._db.get_connection()
        with conn:
            cursor = conn.execute(
                """
                UPDATE curation_runs
                SET task_id = ?, work_item_id = ?, frontier_key = ?, selector_strategy = ?,
                    context_fingerprint = ?, planner_id = ?, provider_id = ?, model_id = ?,
                    policy_version = ?, schema_version = ?, state = ?, outcome = ?, plan_id = ?,
                    rejection_codes_json = ?, retry_reason = ?, budget_usage_json = ?,
                    disclosure_audit_json = ?, created_at = ?, terminalized_at = ?
                WHERE run_id = ? AND state = ?
                """,
                [*_run_values(normalized)[1:], str(run_id), str(expected_state)],
            )
        if cursor.rowcount != 1:
            current = self.get_run(run_id)
            return current if current is not None and current.state is CurationRunState.TERMINAL else None
        return self.get_run(run_id)

    def terminalize_run(
        self,
        run_id: UUID,
        expected_state: CurationRunState,
        outcome: CurationRunOutcome,
    ) -> CurationRun | None:
        terminalized_at = _now()
        conn = self._db.get_connection()
        with conn:
            cursor = conn.execute(
                """
                UPDATE curation_runs
                SET state = ?, outcome = ?, terminalized_at = ?
                WHERE run_id = ? AND state = ? AND terminalized_at IS NULL
                """,
                (
                    str(CurationRunState.TERMINAL),
                    str(outcome),
                    _datetime_text(terminalized_at),
                    str(run_id),
                    str(expected_state),
                ),
            )
        stored = self.get_run(run_id)
        if stored is None:
            return None
        if cursor.rowcount == 1 or stored.state is CurationRunState.TERMINAL:
            return stored
        return None

    def put_receipt(self, receipt: CurationActionReceipt) -> CurationActionReceipt:
        if receipt.intent_hash is None:
            raise ValueError("curation action receipt intent_hash is required")
        normalized = receipt.model_copy(
            update={
                "applied_at": receipt.applied_at or (_now() if receipt.status is CurationReceiptState.APPLIED_UNVERIFIED else None),
            }
        )
        conn = self._db.get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO curation_action_receipts (
                        run_id, action_id, operation, affected_ids_json, status,
                        before_token, after_token, mutation_event_id, intent_hash,
                        verification_descriptor_json, error_code, applied_at, verified_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    _receipt_values(normalized),
                )
        except sqlite3.IntegrityError:
            existing = self.get_receipt(receipt.run_id, receipt.action_id)
            if existing is None:
                raise
            if not _receipt_identity_matches(existing, receipt):
                raise CurationReceiptIdentityConflictError(
                    f"curation action receipt identity is already in use for {receipt.run_id}/{receipt.action_id}"
                )
            return existing
        return normalized

    def get_receipt(self, run_id: UUID, action_id: UUID) -> CurationActionReceipt | None:
        row = self._db.get_connection().execute(
            """
            SELECT * FROM curation_action_receipts
            WHERE run_id = ? AND action_id = ?
            """,
            (str(run_id), str(action_id)),
        ).fetchone()
        return None if row is None else _receipt_from_row(row)

    def list_receipts(self, run_id: UUID, *, limit: int = 100) -> Sequence[CurationActionReceipt]:
        bounded_limit = _bounded_limit(limit)
        rows = self._db.get_connection().execute(
            """
            SELECT * FROM curation_action_receipts
            WHERE run_id = ?
            ORDER BY rowid ASC
            LIMIT ?
            """,
            (str(run_id), bounded_limit),
        ).fetchall()
        return [_receipt_from_row(row) for row in rows]

    def transition_receipt(
        self,
        run_id: UUID,
        action_id: UUID,
        expected_state: CurationReceiptState,
        receipt: CurationActionReceipt,
    ) -> CurationActionReceipt | None:
        if receipt.intent_hash is None:
            raise ValueError("curation action receipt intent_hash is required")
        existing = self.get_receipt(run_id, action_id)
        if existing is None:
            return None
        if is_terminal_receipt(existing):
            return existing
        conn = self._db.get_connection()
        with conn:
            cursor = conn.execute(
                """
                UPDATE curation_action_receipts
                SET operation = ?, affected_ids_json = ?, status = ?, before_token = ?,
                    after_token = ?, mutation_event_id = ?, intent_hash = ?,
                    verification_descriptor_json = ?, error_code = ?,
                    applied_at = ?, verified_at = ?
                WHERE run_id = ? AND action_id = ? AND status = ?
                """,
                [
                    *_receipt_values(receipt)[2:],
                    str(run_id),
                    str(action_id),
                    str(expected_state),
                ],
            )
        stored = self.get_receipt(run_id, action_id)
        if stored is None:
            return None
        if cursor.rowcount == 1 or stored.status is not expected_state:
            return stored
        return None

    def get_candidate_state(self, memory_id: UUID) -> CurationCandidateState | None:
        row = self._db.get_connection().execute(
            "SELECT * FROM curation_candidate_state WHERE memory_id = ?",
            (str(memory_id),),
        ).fetchone()
        return None if row is None else _candidate_from_row(row)

    def list_candidate_states(
        self,
        *,
        disposition: CandidateDisposition | None = None,
        limit: int = 100,
    ) -> Sequence[CurationCandidateState]:
        bounded_limit = _bounded_limit(limit)
        query = "SELECT * FROM curation_candidate_state"
        params: list[object] = []
        if disposition is not None:
            query += " WHERE disposition = ?"
            params.append(str(disposition))
        query += " ORDER BY escalation_count DESC, last_run_id DESC LIMIT ?"
        params.append(bounded_limit)
        rows = self._db.get_connection().execute(query, params).fetchall()
        return [_candidate_from_row(row) for row in rows]

    def put_candidate_state(self, state: CurationCandidateState) -> CurationCandidateState:
        conn = self._db.get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO curation_candidate_state (
                    memory_id, last_observed_revision_token, disposition,
                    consecutive_no_op_count, cooldown_until, last_disposition_reason,
                    last_frontier_key, last_run_id, escalation_count, last_escalated_strategy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    last_observed_revision_token = excluded.last_observed_revision_token,
                    disposition = excluded.disposition,
                    consecutive_no_op_count = excluded.consecutive_no_op_count,
                    cooldown_until = excluded.cooldown_until,
                    last_disposition_reason = excluded.last_disposition_reason,
                    last_frontier_key = excluded.last_frontier_key,
                    last_run_id = excluded.last_run_id,
                    escalation_count = excluded.escalation_count,
                    last_escalated_strategy = excluded.last_escalated_strategy
                """,
                _candidate_values(state),
            )
        return state

    def is_candidate_in_cooldown(self, memory_id: UUID, *, now: datetime | None = None) -> bool:
        state = self.get_candidate_state(memory_id)
        return state is not None and state.cooldown_until is not None and state.cooldown_until > (now or _now())


SQLiteCurationRepository = SQLiteCurationStore


def _bounded_limit(limit: int) -> int:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    return min(limit, MAX_CURATION_READ_LIMIT)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _datetime_text(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _datetime_value(value: object) -> datetime | None:
    return None if value is None else datetime.fromisoformat(str(value))


def _uuid_text(value: UUID | None) -> str | None:
    return None if value is None else str(value)


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _json_value(value: object, *, default: Any) -> Any:
    return default if value is None else json.loads(str(value))


def _run_values(run: CurationRun) -> tuple[object, ...]:
    return (
        str(run.run_id),
        _uuid_text(run.task_id),
        _uuid_text(run.work_item_id),
        run.frontier_key,
        run.selector_strategy,
        run.context_fingerprint,
        run.planner_id,
        run.provider_id,
        run.model_id,
        run.policy_version,
        run.schema_version,
        str(run.state),
        None if run.outcome is None else str(run.outcome),
        _uuid_text(run.plan_id),
        _json_text(run.rejection_codes),
        run.retry_reason,
        _json_text(run.budget_usage.model_dump(mode="json")),
        _json_text(run.disclosure_audit),
        _datetime_text(run.created_at),
        _datetime_text(run.terminalized_at),
    )


def _run_from_row(row: sqlite3.Row) -> CurationRun:
    return CurationRun(
        run_id=UUID(str(row["run_id"])),
        task_id=None if row["task_id"] is None else UUID(str(row["task_id"])),
        work_item_id=None if row["work_item_id"] is None else UUID(str(row["work_item_id"])),
        frontier_key=str(row["frontier_key"]),
        selector_strategy=row["selector_strategy"],
        context_fingerprint=str(row["context_fingerprint"]),
        planner_id=row["planner_id"],
        provider_id=row["provider_id"],
        model_id=row["model_id"],
        policy_version=str(row["policy_version"]),
        schema_version=int(row["schema_version"]),
        state=CurationRunState(str(row["state"])),
        outcome=None if row["outcome"] is None else CurationRunOutcome(str(row["outcome"])),
        plan_id=None if row["plan_id"] is None else UUID(str(row["plan_id"])),
        rejection_codes=list(_json_value(row["rejection_codes_json"], default=[])),
        retry_reason=row["retry_reason"],
        budget_usage=CurationBudgetUsage.model_validate(_json_value(row["budget_usage_json"], default={})),
        disclosure_audit=dict(_json_value(row["disclosure_audit_json"], default={})),
        created_at=_datetime_value(row["created_at"]),
        terminalized_at=_datetime_value(row["terminalized_at"]),
    )


def _receipt_values(receipt: CurationActionReceipt) -> tuple[object, ...]:
    return (
        str(receipt.run_id),
        str(receipt.action_id),
        receipt.operation,
        _json_text([str(memory_id) for memory_id in receipt.affected_ids]),
        str(receipt.status),
        receipt.before_token,
        receipt.after_token,
        _uuid_text(receipt.mutation_event_id),
        receipt.intent_hash,
        None
        if receipt.verification_descriptor is None
        else _json_text(receipt.verification_descriptor.model_dump(mode="json")),
        receipt.error_code,
        _datetime_text(receipt.applied_at),
        _datetime_text(receipt.verified_at),
    )


def _receipt_from_row(row: sqlite3.Row) -> CurationActionReceipt:
    descriptor_json = (
        row["verification_descriptor_json"]
        if "verification_descriptor_json" in row.keys()
        else None
    )
    try:
        descriptor = (
            None
            if descriptor_json is None
            else CurationVerificationDescriptor.model_validate(_json_value(descriptor_json, default={}))
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise CurationReceiptHydrationError("descriptor_hydration_failed") from exc
    return CurationActionReceipt(
        run_id=UUID(str(row["run_id"])),
        action_id=UUID(str(row["action_id"])),
        operation=str(row["operation"]),
        affected_ids=[UUID(value) for value in _json_value(row["affected_ids_json"], default=[])],
        status=CurationReceiptState(str(row["status"])),
        before_token=row["before_token"],
        after_token=row["after_token"],
        mutation_event_id=None if row["mutation_event_id"] is None else UUID(str(row["mutation_event_id"])),
        intent_hash=row["intent_hash"],
        verification_descriptor=descriptor,
        error_code=row["error_code"],
        applied_at=_datetime_value(row["applied_at"]),
        verified_at=_datetime_value(row["verified_at"]),
    )


def _receipt_identity_matches(left: CurationActionReceipt, right: CurationActionReceipt) -> bool:
    return (
        left.operation == right.operation
        and left.affected_ids == right.affected_ids
        and left.before_token == right.before_token
        and (left.intent_hash is None or right.intent_hash is None or left.intent_hash == right.intent_hash)
        and (
            left.verification_descriptor is None
            or right.verification_descriptor is None
            or left.verification_descriptor == right.verification_descriptor
        )
    )


def _candidate_values(state: CurationCandidateState) -> tuple[object, ...]:
    return (
        str(state.memory_id),
        state.last_observed_revision_token,
        str(state.disposition),
        state.consecutive_no_op_count,
        _datetime_text(state.cooldown_until),
        state.last_disposition_reason,
        state.last_frontier_key,
        _uuid_text(state.last_run_id),
        state.escalation_count,
        state.last_escalated_strategy,
    )


def _candidate_from_row(row: sqlite3.Row) -> CurationCandidateState:
    return CurationCandidateState(
        memory_id=UUID(str(row["memory_id"])),
        last_observed_revision_token=row["last_observed_revision_token"],
        disposition=CandidateDisposition(str(row["disposition"])),
        consecutive_no_op_count=int(row["consecutive_no_op_count"]),
        cooldown_until=_datetime_value(row["cooldown_until"]),
        last_disposition_reason=row["last_disposition_reason"],
        last_frontier_key=row["last_frontier_key"],
        last_run_id=None if row["last_run_id"] is None else UUID(str(row["last_run_id"])),
        escalation_count=int(row["escalation_count"]),
        last_escalated_strategy=row["last_escalated_strategy"],
    )
