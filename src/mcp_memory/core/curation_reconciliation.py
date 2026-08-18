"""Deterministic recovery for committed but unverified curation actions."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TypeVar
from uuid import UUID

from mcp_memory.context import TaskQueueProtocol
from mcp_memory.core.curation_models import NormalizeMemoryAction
from mcp_memory.core.curation_verifier import (
    CurationVerificationError,
    CurationVerifier,
    LowRiskCurationAction,
)
from mcp_memory.core.ports.curation import (
    CurationActionReceipt,
    CurationReceiptHydrationError,
    CurationReceiptState,
    CurationRepository,
    CurationRun,
    CurationRunOutcome,
    CurationRunState,
)
from mcp_memory.core.ports.maintenance import MaintenanceReadRepositoryLike

CURATION_RECONCILIATION_LOCK_RETRY_ATTEMPTS = 3
CURATION_RECONCILIATION_LOCK_RETRY_DELAY_SECONDS = 0.05
CURATION_RECONCILIATION_RUN_LIMIT = 200
CURATION_RECONCILIATION_STALE_AFTER_SECONDS = 300.0


class CurationReconciliationDisposition(StrEnum):
    CONTINUE = "continue"
    DEFER = "defer"
    BLOCK = "block"


class CurationProviderAttribution(StrEnum):
    EXACT = "exact"
    MISSING = "missing"
    MISMATCHED = "mismatched"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class ProviderAttemptIdentity:
    """Run-level identity carried by one persisted provider attempt."""

    task_id: str | None
    execution_epoch: int | None
    attempt_identity: str | None


@dataclass(frozen=True, slots=True)
class CurationProviderAttributionResult:
    disposition: CurationProviderAttribution
    reason_code: str
    attempt_identities: tuple[str, ...] = ()


def reconcile_provider_attempts(
    run: CurationRun,
    attempts: Iterable[ProviderAttemptIdentity | object],
) -> CurationProviderAttributionResult:
    """Attribute provider attempts to a run without using action-level IDs."""
    if run.task_id is None or run.execution_epoch is None:
        return CurationProviderAttributionResult(
            CurationProviderAttribution.MISSING,
            "run_execution_identity_missing",
        )

    expected = (str(run.task_id), run.execution_epoch)
    observations: dict[str, tuple[str, int]] = {}
    missing_count = 0
    mismatched_count = 0
    exact_count = 0
    ambiguous = False
    for attempt in attempts:
        task_id = _attempt_text(attempt, "task_id")
        execution_epoch = _attempt_epoch(attempt)
        attempt_identity = _attempt_text(attempt, "attempt_identity")
        if task_id is None or execution_epoch is None or attempt_identity is None:
            missing_count += 1
            continue
        observation = (task_id, execution_epoch)
        previous = observations.get(attempt_identity)
        if previous is not None and previous != observation:
            ambiguous = True
        observations[attempt_identity] = observation
        if observation == expected:
            exact_count += 1
        else:
            mismatched_count += 1

    identities = tuple(sorted(observations))
    if ambiguous or (exact_count and (mismatched_count or missing_count)):
        return CurationProviderAttributionResult(
            CurationProviderAttribution.AMBIGUOUS,
            "provider_attempt_attribution_ambiguous",
            identities,
        )
    if mismatched_count:
        return CurationProviderAttributionResult(
            CurationProviderAttribution.MISMATCHED,
            "provider_attempt_attribution_mismatched",
            identities,
        )
    if missing_count or not observations:
        return CurationProviderAttributionResult(
            CurationProviderAttribution.MISSING,
            "provider_attempt_attribution_missing",
            identities,
        )
    return CurationProviderAttributionResult(
        CurationProviderAttribution.EXACT,
        "provider_attempt_attribution_exact",
        identities,
    )


def _attempt_text(attempt: ProviderAttemptIdentity | object, name: str) -> str | None:
    value = getattr(attempt, name, None)
    if value is None or not str(value):
        return None
    return str(value)


def _attempt_epoch(attempt: ProviderAttemptIdentity | object) -> int | None:
    value = getattr(attempt, "execution_epoch", None)
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class CurationReconciliationOutcome:
    """The deterministic decision a caller must make for one reconciliation pass."""

    disposition: CurationReconciliationDisposition
    reason_code: str
    run_id: UUID | None = None
    work_item_id: UUID | None = None
    verified_receipt_count: int = 0
    retry_delay_seconds: float | None = None
    run: CurationRun | None = None


ActionResolver = Callable[[CurationRun, CurationActionReceipt], LowRiskCurationAction | None]
_T = TypeVar("_T")


class CurationReconciler:
    """Verify each applied receipt at most once and terminalize its run safely."""

    def __init__(
        self,
        curation_store: CurationRepository,
        maintenance_reads: MaintenanceReadRepositoryLike,
        *,
        action_resolver: ActionResolver | None = None,
        sleep: Callable[[float], None] = time.sleep,
        task_queue: TaskQueueProtocol | None = None,
        clock: Callable[[], float] = time.time,
        stale_after_seconds: float = CURATION_RECONCILIATION_STALE_AFTER_SECONDS,
    ) -> None:
        self._curation_store = curation_store
        self._verifier = CurationVerifier(curation_store, maintenance_reads)
        self._action_resolver = action_resolver or _default_action_resolver
        self._sleep = sleep
        self._task_queue = task_queue
        self._clock = clock
        self._stale_after_seconds = stale_after_seconds

    def reconcile(self) -> list[CurationReconciliationOutcome]:
        """Reconcile all non-terminal runs with committed unverified receipts."""
        try:
            runs = self._with_lock_retries(
                lambda: self._curation_store.list_runs(limit=CURATION_RECONCILIATION_RUN_LIMIT)
            )
        except sqlite3.OperationalError as exc:
            if _is_locked(exc):
                return [_defer(None, "sqlite_locked")]
            raise

        outcomes: list[CurationReconciliationOutcome] = []
        for run in runs:
            if run.state is CurationRunState.TERMINAL:
                continue
            outcome = self.reconcile_run(run)
            if outcome is not None:
                outcomes.append(outcome)
        return outcomes

    def reconcile_run(self, run: CurationRun) -> CurationReconciliationOutcome | None:
        try:
            receipts = self._with_lock_retries(lambda: self._curation_store.list_receipts(run.run_id))
        except CurationReceiptHydrationError:
            return self._finish_failed_run(run, "descriptor_hydration_failed", 0)
        except sqlite3.OperationalError as exc:
            if _is_locked(exc):
                return _defer(run, "sqlite_locked")
            return _block(run, "receipt_read_failed")

        pending = [
            receipt
            for receipt in receipts
            if receipt.status is CurationReceiptState.APPLIED_UNVERIFIED
        ]
        if not pending:
            return self._finalize_if_possible(run, receipts, verified_receipt_count=0)

        verified_count = 0
        for receipt in pending:
            try:
                if receipt.verification_descriptor is not None:
                    descriptor = receipt.verification_descriptor
                    result = self._with_lock_retries(
                        lambda: self._verifier.verify_descriptor(receipt, descriptor)
                    )
                else:
                    try:
                        action = self._action_resolver(run, receipt)
                    except Exception:
                        return self._finish_failed_run(run, "action_resolution_failed", verified_count)
                    if action is None:
                        return self._finish_failed_run(run, "action_unavailable", verified_count)
                    resolved_action = action
                    result = self._with_lock_retries(lambda: self._verifier.verify(receipt, resolved_action))
            except sqlite3.OperationalError as exc:
                if _is_locked(exc):
                    return _defer(run, "sqlite_locked")
                return self._finish_failed_run(run, "verification_storage_failed", verified_count)
            except CurationVerificationError:
                return self._finish_failed_run(run, "verification_failed", verified_count)
            except Exception:
                return self._finish_failed_run(run, "verification_failed", verified_count)
            if result.status is CurationReceiptState.VERIFIED:
                verified_count += 1

        try:
            final_receipts = self._with_lock_retries(lambda: self._curation_store.list_receipts(run.run_id))
        except CurationReceiptHydrationError:
            return self._finish_failed_run(run, "descriptor_hydration_failed", verified_count)
        except sqlite3.OperationalError as exc:
            if _is_locked(exc):
                return _defer(run, "sqlite_locked")
            return _block(run, "receipt_read_failed")
        return self._finalize_if_possible(run, final_receipts, verified_receipt_count=verified_count)

    def before_curator_work(self) -> CurationReconciliationOutcome:
        """Return the strongest typed gate result before a curator handler runs."""
        outcomes = self.reconcile()
        blocked = next(
            (outcome for outcome in outcomes if outcome.disposition is CurationReconciliationDisposition.BLOCK),
            None,
        )
        if blocked is not None:
            return blocked
        deferred = next(
            (outcome for outcome in outcomes if outcome.disposition is CurationReconciliationDisposition.DEFER),
            None,
        )
        if deferred is not None:
            return deferred
        return CurationReconciliationOutcome(CurationReconciliationDisposition.CONTINUE, "reconciled")

    def _finalize_if_possible(
        self,
        run: CurationRun,
        receipts: Sequence[CurationActionReceipt],
        *,
        verified_receipt_count: int,
    ) -> CurationReconciliationOutcome | None:
        if not receipts:
            return self._reconcile_empty_run(run)
        if any(receipt.status is CurationReceiptState.APPLIED_UNVERIFIED for receipt in receipts):
            return _defer(run, "receipt_unresolved")
        failed = any(receipt.status is not CurationReceiptState.VERIFIED for receipt in receipts)
        outcome = CurationRunOutcome.VERIFICATION_FAILED if failed else CurationRunOutcome.APPLIED
        try:
            terminal = self._with_lock_retries(
                lambda: self._curation_store.terminalize_run(run.run_id, run.state, outcome)
            )
        except sqlite3.OperationalError as exc:
            if _is_locked(exc):
                return _defer(run, "sqlite_locked")
            return _block(run, "run_terminalization_failed")
        if terminal is None:
            terminal = self._curation_store.get_run(run.run_id)
        if terminal is None:
            return _block(run, "run_missing")
        if terminal.outcome is not None and terminal.outcome is not outcome:
            return _continue_after_terminal(run, terminal, verified_receipt_count)
        if failed or terminal.outcome is CurationRunOutcome.VERIFICATION_FAILED:
            return _block(run, "verification_failed", verified_receipt_count, terminal)
        return CurationReconciliationOutcome(
            CurationReconciliationDisposition.CONTINUE,
            "run_finalized",
            run_id=run.run_id,
            work_item_id=run.work_item_id,
            verified_receipt_count=verified_receipt_count,
            run=terminal,
        )

    def _reconcile_empty_run(self, run: CurationRun) -> CurationReconciliationOutcome | None:
        """Recover provider/task lifecycle failures that produced no receipts."""
        task_queue = self._task_queue
        if task_queue is not None and run.task_id is not None:
            try:
                task = self._with_lock_retries(lambda: task_queue.get_task(str(run.task_id)))
            except (KeyError, ValueError):
                task = None
            except sqlite3.OperationalError as exc:
                if _is_locked(exc):
                    return _defer(run, "sqlite_locked")
                raise
            if task is not None:
                if task.status in {"pending", "running"}:
                    return None
                if task.status == "cancelled":
                    return self._terminalize_empty_run(run, CurationRunOutcome.CANCELLED, "task_cancelled")
                if task.status in {"failed", "completed"}:
                    return self._terminalize_empty_run(run, CurationRunOutcome.PROVIDER_FAILED, "task_finished_without_run")
                return None

        if not self._is_stale(run.created_at):
            return None
        return self._terminalize_empty_run(run, CurationRunOutcome.PROVIDER_FAILED, "stale_task_missing")

    def _is_stale(self, created_at: datetime | None) -> bool:
        if created_at is None:
            return False
        age_seconds = max(self._clock() - created_at.timestamp(), 0.0)
        return age_seconds >= self._stale_after_seconds

    def _terminalize_empty_run(
        self,
        run: CurationRun,
        outcome: CurationRunOutcome,
        reason_code: str,
    ) -> CurationReconciliationOutcome:
        try:
            terminal = self._with_lock_retries(
                lambda: self._curation_store.terminalize_run(run.run_id, run.state, outcome)
            )
        except sqlite3.OperationalError as exc:
            if _is_locked(exc):
                return _defer(run, "sqlite_locked")
            return _block(run, "run_terminalization_failed")
        if terminal is None:
            terminal = self._curation_store.get_run(run.run_id)
        if terminal is None:
            return _block(run, "run_missing")
        if terminal.outcome is not None and terminal.outcome is not outcome:
            return _continue_after_terminal(run, terminal, 0)
        return CurationReconciliationOutcome(
            CurationReconciliationDisposition.CONTINUE,
            reason_code,
            run_id=run.run_id,
            work_item_id=run.work_item_id,
            run=terminal,
        )

    def _finish_failed_run(
        self,
        run: CurationRun,
        reason_code: str,
        verified_receipt_count: int,
    ) -> CurationReconciliationOutcome:
        try:
            terminal = self._with_lock_retries(
                lambda: self._curation_store.terminalize_run(
                    run.run_id, run.state, CurationRunOutcome.VERIFICATION_FAILED
                )
            )
        except sqlite3.OperationalError as exc:
            if _is_locked(exc):
                return _defer(run, "sqlite_locked")
            return _block(run, "run_terminalization_failed", verified_receipt_count)
        if terminal is not None and terminal.outcome is not CurationRunOutcome.VERIFICATION_FAILED:
            return _continue_after_terminal(run, terminal, verified_receipt_count)
        return _block(run, reason_code, verified_receipt_count, terminal)

    def _with_lock_retries(self, callback: Callable[[], _T]) -> _T:
        for attempt in range(CURATION_RECONCILIATION_LOCK_RETRY_ATTEMPTS):
            try:
                return callback()
            except sqlite3.OperationalError as exc:
                if not _is_locked(exc) or attempt + 1 >= CURATION_RECONCILIATION_LOCK_RETRY_ATTEMPTS:
                    raise
                self._sleep(CURATION_RECONCILIATION_LOCK_RETRY_DELAY_SECONDS * (attempt + 1))
        raise RuntimeError("curation reconciliation retries exhausted")


def _default_action_resolver(_run: CurationRun, receipt: CurationActionReceipt) -> LowRiskCurationAction | None:
    if receipt.operation != "normalize_memory" or len(receipt.affected_ids) != 1:
        return None
    return NormalizeMemoryAction(
        action_id=receipt.action_id,
        target_id=receipt.affected_ids[0],
        confidence=1.0,
        rationale="daemon recovery postcondition verification",
        # Recovery only needs the typed operation and target; the verifier does
        # not consult the proposed metadata. Keep the action valid without
        # implying a real metadata change.
        tags=[],
    )


def _is_locked(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).lower()
    return "database is locked" in message or "database is busy" in message


def _defer(run: CurationRun | None, reason_code: str) -> CurationReconciliationOutcome:
    return CurationReconciliationOutcome(
        CurationReconciliationDisposition.DEFER,
        reason_code,
        run_id=None if run is None else run.run_id,
        work_item_id=None if run is None else run.work_item_id,
        retry_delay_seconds=CURATION_RECONCILIATION_LOCK_RETRY_DELAY_SECONDS,
    )


def _block(
    run: CurationRun,
    reason_code: str,
    verified_receipt_count: int = 0,
    terminal: CurationRun | None = None,
) -> CurationReconciliationOutcome:
    return CurationReconciliationOutcome(
        CurationReconciliationDisposition.BLOCK,
        reason_code,
        run_id=run.run_id,
        work_item_id=run.work_item_id,
        verified_receipt_count=verified_receipt_count,
        run=terminal,
    )


def _continue_after_terminal(
    run: CurationRun,
    terminal: CurationRun,
    verified_receipt_count: int,
) -> CurationReconciliationOutcome:
    return CurationReconciliationOutcome(
        CurationReconciliationDisposition.CONTINUE,
        "run_already_terminal",
        run_id=run.run_id,
        work_item_id=run.work_item_id,
        verified_receipt_count=verified_receipt_count,
        run=terminal,
    )


__all__ = [
    "ActionResolver",
    "CurationProviderAttribution",
    "CurationProviderAttributionResult",
    "CurationReconciliationDisposition",
    "CurationReconciliationOutcome",
    "CurationReconciler",
    "CURATION_RECONCILIATION_LOCK_RETRY_ATTEMPTS",
    "CURATION_RECONCILIATION_LOCK_RETRY_DELAY_SECONDS",
    "CURATION_RECONCILIATION_STALE_AFTER_SECONDS",
    "ProviderAttemptIdentity",
    "reconcile_provider_attempts",
]
