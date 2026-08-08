from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from collections.abc import Iterator, Sequence
from typing import Any, cast
from uuid import UUID

from pydantic import ValidationError

from mcp_memory.core.curation_models import (
    CurationBudgetUsage,
    CurationRunOutcome,
    CurationVerificationDescriptor,
)
from mcp_memory.curation_store import (
    MAX_CURATION_READ_LIMIT,
    CandidateDisposition,
    CurationActionReceipt,
    CurationCandidateState,
    CurationReceiptIdentityConflictError,
    CurationReceiptHydrationError,
    CurationReceiptState,
    CurationRun,
    CurationRunState,
    is_terminal_receipt,
)
from mcp_memory.storage.session import DbConnectionLike, SessionManager


class PostgresCurationStore:
    """Postgres persistence for the curation ledger."""

    def __init__(self, session_manager: SessionManager[DbConnectionLike]) -> None:
        self._sessions = session_manager

    def create_run(self, run: CurationRun) -> CurationRun:
        normalized = run.model_copy(update={"created_at": run.created_at or _now()})
        with self._transaction() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO curation_runs (
                        run_id, task_id, work_item_id, frontier_key, selector_strategy,
                        context_fingerprint, planner_id, provider_id, model_id,
                        policy_version, schema_version, state, outcome, plan_id,
                        rejection_codes_json, retry_reason, budget_usage_json,
                        disclosure_audit_json, created_at, terminalized_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s::jsonb, %s, %s)
                    """,
                    _run_values(normalized),
                )
        return normalized

    def get_run(self, run_id: UUID) -> CurationRun | None:
        row = self._fetchone(_RUN_SELECT + " WHERE run_id = %s", (str(run_id),))
        return None if row is None else _run_from_row(row)

    def list_runs(self, *, limit: int = 100) -> list[CurationRun]:
        bounded_limit = _bounded_limit(limit)
        rows = self._fetchall(
            _RUN_SELECT + " ORDER BY created_at DESC, run_id DESC LIMIT %s",
            (bounded_limit,),
        )
        return [_run_from_row(row) for row in rows]

    def transition_run(
        self,
        run_id: UUID,
        expected_state: CurationRunState,
        run: CurationRun,
    ) -> CurationRun | None:
        with self._transaction() as connection:
            with connection.cursor() as cursor:
                current_row = _fetchone_on_cursor(
                    cursor, _RUN_SELECT + " WHERE run_id = %s", (str(run_id),)
                )
                if current_row is None:
                    return None
                current = _run_from_row(current_row)
                if current.state is CurationRunState.TERMINAL:
                    return current
                normalized = run.model_copy(
                    update={"run_id": run_id, "created_at": run.created_at or current.created_at}
                )
                cursor.execute(
                    """
                    UPDATE curation_runs
                    SET task_id = %s, work_item_id = %s, frontier_key = %s, selector_strategy = %s,
                        context_fingerprint = %s, planner_id = %s, provider_id = %s, model_id = %s,
                        policy_version = %s, schema_version = %s, state = %s, outcome = %s, plan_id = %s,
                        rejection_codes_json = %s::jsonb, retry_reason = %s, budget_usage_json = %s::jsonb,
                        disclosure_audit_json = %s::jsonb, created_at = %s, terminalized_at = %s
                    WHERE run_id = %s AND state = %s
                    """,
                    (*_run_values(normalized)[1:], str(run_id), str(expected_state)),
                )
                updated = int(getattr(cursor, "rowcount", 0) or 0)
                stored_row = _fetchone_on_cursor(
                    cursor, _RUN_SELECT + " WHERE run_id = %s", (str(run_id),)
                )
                if stored_row is None:
                    return None
                stored = _run_from_row(stored_row)
                if updated == 1 or stored.state is CurationRunState.TERMINAL:
                    return stored
                return None

    def terminalize_run(
        self,
        run_id: UUID,
        expected_state: CurationRunState,
        outcome: CurationRunOutcome,
    ) -> CurationRun | None:
        with self._transaction() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE curation_runs
                    SET state = %s, outcome = %s, terminalized_at = %s
                    WHERE run_id = %s AND state = %s AND terminalized_at IS NULL
                    """,
                    (
                        str(CurationRunState.TERMINAL),
                        str(outcome),
                        _datetime_text(_now()),
                        str(run_id),
                        str(expected_state),
                    ),
                )
                updated = int(getattr(cursor, "rowcount", 0) or 0)
                row = _fetchone_on_cursor(cursor, _RUN_SELECT + " WHERE run_id = %s", (str(run_id),))
                if row is None:
                    return None
                stored = _run_from_row(row)
                return stored if updated == 1 or stored.state is CurationRunState.TERMINAL else None

    def put_receipt(self, receipt: CurationActionReceipt) -> CurationActionReceipt:
        if receipt.intent_hash is None:
            raise ValueError("curation action receipt intent_hash is required")
        normalized = receipt.model_copy(
            update={
                "applied_at": receipt.applied_at
                or (_now() if receipt.status is CurationReceiptState.APPLIED_UNVERIFIED else None),
            }
        )
        with self._transaction() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO curation_action_receipts (
                        run_id, action_id, operation, affected_ids_json, status,
                        before_token, after_token, mutation_event_id, intent_hash,
                        verification_descriptor_json, error_code,
                        applied_at, verified_at
                    ) VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                    ON CONFLICT (run_id, action_id) DO NOTHING
                    """,
                    _receipt_values(normalized),
                )
                row = _fetchone_on_cursor(
                    cursor,
                    _RECEIPT_SELECT + " WHERE run_id = %s AND action_id = %s",
                    (str(receipt.run_id), str(receipt.action_id)),
                )
                if row is None:
                    raise RuntimeError("curation action receipt insert was not persisted")
                stored = _receipt_from_row(row)
                if not _receipt_identity_matches(stored, receipt):
                    raise CurationReceiptIdentityConflictError(
                        f"curation action receipt identity is already in use for {receipt.run_id}/{receipt.action_id}"
                    )
                return stored

    def get_receipt(self, run_id: UUID, action_id: UUID) -> CurationActionReceipt | None:
        row = self._fetchone(
            _RECEIPT_SELECT + " WHERE run_id = %s AND action_id = %s",
            (str(run_id), str(action_id)),
        )
        return None if row is None else _receipt_from_row(row)

    def list_receipts(self, run_id: UUID, *, limit: int = 100) -> list[CurationActionReceipt]:
        bounded_limit = _bounded_limit(limit)
        rows = self._fetchall(
            _RECEIPT_SELECT + " WHERE run_id = %s ORDER BY action_id ASC LIMIT %s",
            (str(run_id), bounded_limit),
        )
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
        with self._transaction() as connection:
            with connection.cursor() as cursor:
                row = _fetchone_on_cursor(
                    cursor,
                    _RECEIPT_SELECT + " WHERE run_id = %s AND action_id = %s",
                    (str(run_id), str(action_id)),
                )
                if row is None:
                    return None
                existing = _receipt_from_row(row)
                if is_terminal_receipt(existing):
                    return existing
                cursor.execute(
                    """
                    UPDATE curation_action_receipts
                    SET operation = %s, affected_ids_json = %s::jsonb, status = %s,
                        before_token = %s, after_token = %s, mutation_event_id = %s,
                        intent_hash = %s, verification_descriptor_json = %s::jsonb,
                        error_code = %s, applied_at = %s, verified_at = %s
                    WHERE run_id = %s AND action_id = %s AND status = %s
                    """,
                    (
                        *_receipt_values(receipt)[2:],
                        str(run_id),
                        str(action_id),
                        str(expected_state),
                    ),
                )
                updated = int(getattr(cursor, "rowcount", 0) or 0)
                stored_row = _fetchone_on_cursor(
                    cursor,
                    _RECEIPT_SELECT + " WHERE run_id = %s AND action_id = %s",
                    (str(run_id), str(action_id)),
                )
                if stored_row is None:
                    return None
                stored = _receipt_from_row(stored_row)
                return stored if updated == 1 or stored.status is not expected_state else None

    def get_candidate_state(self, memory_id: UUID) -> CurationCandidateState | None:
        row = self._fetchone(_CANDIDATE_SELECT + " WHERE memory_id = %s", (str(memory_id),))
        return None if row is None else _candidate_from_row(row)

    def list_candidate_states(
        self,
        *,
        disposition: CandidateDisposition | None = None,
        limit: int = 100,
    ) -> Sequence[CurationCandidateState]:
        bounded_limit = _bounded_limit(limit)
        query = _CANDIDATE_SELECT
        params: list[object] = []
        if disposition is not None:
            query += " WHERE disposition = %s"
            params.append(str(disposition))
        query += " ORDER BY escalation_count DESC, last_run_id DESC LIMIT %s"
        params.append(bounded_limit)
        return [_candidate_from_row(row) for row in self._fetchall(query, tuple(params))]

    def put_candidate_state(self, state: CurationCandidateState) -> CurationCandidateState:
        with self._transaction() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO curation_candidate_state (
                        memory_id, last_observed_revision_token, disposition,
                        consecutive_no_op_count, cooldown_until, last_disposition_reason,
                        last_frontier_key, last_run_id, escalation_count, last_escalated_strategy,
                        last_considered_at, last_considered_strategy, last_mutation_family,
                        last_mutated_at, coverage_evidence_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (memory_id) DO UPDATE SET
                        last_observed_revision_token = EXCLUDED.last_observed_revision_token,
                        disposition = EXCLUDED.disposition,
                        consecutive_no_op_count = EXCLUDED.consecutive_no_op_count,
                        cooldown_until = EXCLUDED.cooldown_until,
                        last_disposition_reason = EXCLUDED.last_disposition_reason,
                        last_frontier_key = EXCLUDED.last_frontier_key,
                        last_run_id = EXCLUDED.last_run_id,
                        escalation_count = EXCLUDED.escalation_count,
                    last_escalated_strategy = EXCLUDED.last_escalated_strategy
                    ,last_considered_at = EXCLUDED.last_considered_at
                    ,last_considered_strategy = EXCLUDED.last_considered_strategy
                    ,last_mutation_family = EXCLUDED.last_mutation_family
                    ,last_mutated_at = EXCLUDED.last_mutated_at
                    ,coverage_evidence_json = EXCLUDED.coverage_evidence_json
                    """,
                    _candidate_values(state),
                )
        return state

    def compare_and_set_candidate_state(
        self,
        memory_id: UUID,
        expected_token: str | None,
        state: CurationCandidateState,
    ) -> CurationCandidateState | None:
        if state.memory_id != memory_id:
            raise ValueError("candidate state memory_id does not match CAS key")
        with self._transaction() as connection:
            with connection.cursor() as cursor:
                if expected_token is None:
                    cursor.execute(
                        """
                        INSERT INTO curation_candidate_state (
                            memory_id, last_observed_revision_token, disposition,
                            consecutive_no_op_count, cooldown_until, last_disposition_reason,
                            last_frontier_key, last_run_id, escalation_count, last_escalated_strategy,
                            last_considered_at, last_considered_strategy, last_mutation_family,
                            last_mutated_at, coverage_evidence_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        ON CONFLICT (memory_id) DO UPDATE SET
                            last_observed_revision_token = EXCLUDED.last_observed_revision_token,
                            disposition = EXCLUDED.disposition,
                            consecutive_no_op_count = EXCLUDED.consecutive_no_op_count,
                            cooldown_until = EXCLUDED.cooldown_until,
                            last_disposition_reason = EXCLUDED.last_disposition_reason,
                            last_frontier_key = EXCLUDED.last_frontier_key,
                            last_run_id = EXCLUDED.last_run_id,
                            escalation_count = EXCLUDED.escalation_count,
                            last_escalated_strategy = EXCLUDED.last_escalated_strategy,
                            last_considered_at = EXCLUDED.last_considered_at,
                            last_considered_strategy = EXCLUDED.last_considered_strategy,
                            last_mutation_family = EXCLUDED.last_mutation_family,
                            last_mutated_at = EXCLUDED.last_mutated_at,
                            coverage_evidence_json = EXCLUDED.coverage_evidence_json
                        WHERE curation_candidate_state.last_observed_revision_token IS NULL
                        """,
                        _candidate_values(state),
                    )
                    return state if int(getattr(cursor, "rowcount", 0) or 0) == 1 else None
                cursor.execute(
                    """
                    UPDATE curation_candidate_state
                    SET last_observed_revision_token = %s, disposition = %s,
                        consecutive_no_op_count = %s, cooldown_until = %s,
                        last_disposition_reason = %s, last_frontier_key = %s,
                        last_run_id = %s, escalation_count = %s,
                        last_escalated_strategy = %s, last_considered_at = %s,
                        last_considered_strategy = %s, last_mutation_family = %s,
                        last_mutated_at = %s, coverage_evidence_json = %s::jsonb
                    WHERE memory_id = %s AND last_observed_revision_token = %s
                    """,
                    (*_candidate_values(state)[1:], str(memory_id), expected_token),
                )
                return state if int(getattr(cursor, "rowcount", 0) or 0) == 1 else None

    def is_candidate_in_cooldown(self, memory_id: UUID, *, now: datetime | None = None) -> bool:
        state = self.get_candidate_state(memory_id)
        return state is not None and state.cooldown_until is not None and state.cooldown_until > (now or _now())

    @contextmanager
    def _transaction(self) -> Iterator[DbConnectionLike]:
        with self._sessions.open_connection() as connection:
            try:
                yield connection
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _fetchone(self, query: str, params: tuple[object, ...]) -> tuple[object, ...] | None:
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                return _fetchone_on_cursor(cursor, query, params)

    def _fetchall(self, query: str, params: tuple[object, ...]) -> list[tuple[object, ...]]:
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()


_RUN_SELECT = """
SELECT run_id, task_id, work_item_id, frontier_key, selector_strategy,
       context_fingerprint, planner_id, provider_id, model_id, policy_version,
       schema_version, state, outcome, plan_id, rejection_codes_json,
       retry_reason, budget_usage_json, disclosure_audit_json, created_at, terminalized_at
FROM curation_runs
"""
_RECEIPT_SELECT = """
SELECT run_id, action_id, operation, affected_ids_json, status, before_token,
       after_token, mutation_event_id, intent_hash, verification_descriptor_json,
       error_code, applied_at, verified_at
FROM curation_action_receipts
"""
_CANDIDATE_SELECT = """
SELECT memory_id, last_observed_revision_token, disposition, consecutive_no_op_count,
       cooldown_until, last_disposition_reason, last_frontier_key, last_run_id,
       escalation_count, last_escalated_strategy, last_considered_at,
       last_considered_strategy, last_mutation_family, last_mutated_at,
       coverage_evidence_json
FROM curation_candidate_state
"""


def _fetchone_on_cursor(cursor, query: str, params: tuple[object, ...]) -> tuple[object, ...] | None:
    cursor.execute(query, params)
    return cursor.fetchone()


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
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _json_value(value: object, *, default: object) -> object:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, (Mapping, list)):
        return value
    return value


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
        _datetime_text(state.last_considered_at),
        state.last_considered_strategy,
        state.last_mutation_family,
        _datetime_text(state.last_mutated_at),
        _json_text(state.coverage_evidence_json),
    )


def _run_from_row(row: tuple[object, ...]) -> CurationRun:
    return CurationRun(
        run_id=UUID(str(row[0])),
        task_id=None if row[1] is None else UUID(str(row[1])),
        work_item_id=None if row[2] is None else UUID(str(row[2])),
        frontier_key=str(row[3]),
        selector_strategy=None if row[4] is None else str(row[4]),
        context_fingerprint=str(row[5]),
        planner_id=None if row[6] is None else str(row[6]),
        provider_id=None if row[7] is None else str(row[7]),
        model_id=None if row[8] is None else str(row[8]),
        policy_version=str(row[9]),
        schema_version=int(str(row[10])),
        state=CurationRunState(str(row[11])),
        outcome=None if row[12] is None else CurationRunOutcome(str(row[12])),
        plan_id=None if row[13] is None else UUID(str(row[13])),
        rejection_codes=[str(value) for value in _json_list(row[14])],
        retry_reason=None if row[15] is None else str(row[15]),
        budget_usage=CurationBudgetUsage.model_validate(_json_value(row[16], default={})),
        disclosure_audit=cast(dict[str, Any], _json_value(row[17], default={})),
        created_at=_datetime_value(row[18]),
        terminalized_at=_datetime_value(row[19]),
    )


def _receipt_from_row(row: tuple[object, ...]) -> CurationActionReceipt:
    try:
        descriptor = (
            None
            if row[9] is None
            else CurationVerificationDescriptor.model_validate(_json_value(row[9], default={}))
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise CurationReceiptHydrationError("descriptor_hydration_failed") from exc
    return CurationActionReceipt(
        run_id=UUID(str(row[0])),
        action_id=UUID(str(row[1])),
        operation=str(row[2]),
        affected_ids=[UUID(str(value)) for value in _json_list(row[3])],
        status=CurationReceiptState(str(row[4])),
        before_token=None if row[5] is None else str(row[5]),
        after_token=None if row[6] is None else str(row[6]),
        mutation_event_id=None if row[7] is None else UUID(str(row[7])),
        intent_hash=None if row[8] is None else str(row[8]),
        verification_descriptor=descriptor,
        error_code=None if row[10] is None else str(row[10]),
        applied_at=_datetime_value(row[11]),
        verified_at=_datetime_value(row[12]),
    )


def _candidate_from_row(row: tuple[object, ...]) -> CurationCandidateState:
    return CurationCandidateState(
        memory_id=UUID(str(row[0])),
        last_observed_revision_token=None if row[1] is None else str(row[1]),
        disposition=CandidateDisposition(str(row[2])),
        consecutive_no_op_count=int(str(row[3])),
        cooldown_until=_datetime_value(row[4]),
        last_disposition_reason=None if row[5] is None else str(row[5]),
        last_frontier_key=None if row[6] is None else str(row[6]),
        last_run_id=None if row[7] is None else UUID(str(row[7])),
        escalation_count=int(str(row[8])),
        last_escalated_strategy=None if row[9] is None else str(row[9]),
        last_considered_at=_datetime_value(row[10]),
        last_considered_strategy=None if row[11] is None else str(row[11]),
        last_mutation_family=None if row[12] is None else str(row[12]),
        last_mutated_at=_datetime_value(row[13]),
        coverage_evidence_json=cast(dict[str, Any], _json_value(row[14], default={})),
    )


def _receipt_identity_matches(left: CurationActionReceipt, right: CurationActionReceipt) -> bool:
    return (
        left.operation == right.operation
        and left.affected_ids == right.affected_ids
        and left.before_token == right.before_token
        and (
            left.intent_hash is None
            or right.intent_hash is None
            or left.intent_hash == right.intent_hash
        )
        and (
            left.verification_descriptor is None
            or right.verification_descriptor is None
            or left.verification_descriptor == right.verification_descriptor
        )
    )


def _json_list(value: object) -> list[object]:
    decoded = _json_value(value, default=[])
    return decoded if isinstance(decoded, list) else []


PostgresCurationRepository = PostgresCurationStore
