from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from inspect import isawaitable, iscoroutinefunction
import logging
import threading
import time
from typing import Any, cast

import mcp_memory.core.tasks as task_queue_module
from mcp_memory.context import TaskRuntimeContext
from mcp_memory.core.maintenance_idle import (
    build_idle_pause_result,
    should_pause_autonomous_recurring_maintenance,
)
from mcp_memory.core._recovery_actions import RecoveryAction
from mcp_memory.core.curation_reconciliation import (
    CurationReconciliationDisposition,
    CurationReconciler,
)
from mcp_memory.core.recurring_jitter import compute_recurring_jitter_seconds
from mcp_memory.core.system1_scheduling import schedule_system1_ingest, schedule_system1_ingest_continuation
from mcp_memory.core.task_results import TaskRunResultSource
from mcp_memory.core.task_handlers import (
    AUTONOMOUS_RECURRING_TASK_INTERVAL_SECONDS,
    CURATOR_TASK_NAME,
    SYSTEM1_INGEST_TASK_NAME,
    task_priority,
)
from mcp_memory.core.ports.tasks import TaskRecord
from mcp_memory.core.ports.providers import (
    NullProviderUsagePort,
    ProviderUsagePort,
    TaskExecutionAttemptPort,
    TaskExecutionAttemptRecordLike,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RecoveredTaskReconciliationPolicy:
    is_retryable_interruption: bool
    termination_reason: str | None = None

    @classmethod
    def terminal(cls) -> _RecoveredTaskReconciliationPolicy:
        return cls(is_retryable_interruption=False)

    @classmethod
    def retryable_interruption(cls, *, termination_reason: str) -> _RecoveredTaskReconciliationPolicy:
        return cls(
            is_retryable_interruption=True,
            termination_reason=termination_reason,
        )

    def reconcile(self, worker: RuntimeTaskWorker, task: TaskRecord) -> None:
        if self.is_retryable_interruption:
            termination_reason = self.termination_reason
            if termination_reason is None:
                raise ValueError("Retryable interruption reconciliation requires a termination reason")
            worker._reconcile_retryable_interruption(task, termination_reason=termination_reason)
            return
        worker._reconcile_terminal_task_state(task)


@dataclass(frozen=True)
class _RecoveredTaskOutcome:
    task: TaskRecord
    reconciliation_policy: _RecoveredTaskReconciliationPolicy


@dataclass(frozen=True)
class _RunningTaskRecoveryPlan:
    action: RecoveryAction
    recovered_at: float
    error_text: str | None = None
    retry_delay_seconds: float | None = None
    retry_termination_reason: str | None = None

    @classmethod
    def finalize_cancellation(cls, *, recovered_at: float) -> _RunningTaskRecoveryPlan:
        return cls(action=RecoveryAction.FINALIZE_CANCELLATION, recovered_at=recovered_at)

    @classmethod
    def retry_dead_subprocess(
        cls,
        *,
        subprocess_pid: int,
        recovered_at: float,
        retry_delay_seconds: float,
    ) -> _RunningTaskRecoveryPlan:
        return cls(
            action=RecoveryAction.RETRY_DEAD_SUBPROCESS,
            recovered_at=recovered_at,
            error_text=f"Provider subprocess {subprocess_pid} exited unexpectedly",
            retry_delay_seconds=retry_delay_seconds,
            retry_termination_reason="provider_subprocess_exited_retry",
        )

    @classmethod
    def retry_abandoned(
        cls,
        *,
        recovered_at: float,
        retry_delay_seconds: float,
    ) -> _RunningTaskRecoveryPlan:
        return cls(
            action=RecoveryAction.RETRY_DEAD_SUBPROCESS,
            recovered_at=recovered_at,
            error_text="Task was abandoned without an active provider subprocess",
            retry_delay_seconds=retry_delay_seconds,
            retry_termination_reason="abandoned_no_subprocess_retry",
        )

    @classmethod
    def fail_abandoned(cls, *, recovered_at: float) -> _RunningTaskRecoveryPlan:
        return cls(
            action=RecoveryAction.FAIL_ABANDONED,
            recovered_at=recovered_at,
            error_text="Task was abandoned without an active provider subprocess",
        )


class RuntimeTaskWorker:
    def __init__(
        self,
        ctx: TaskRuntimeContext,
        handlers: dict[str, Callable[[Any, TaskRecord], Any]] | None = None,
        handler_factory: Callable[[], dict[str, Callable[[Any, TaskRecord], Any]]] | None = None,
        poll_interval_seconds: float = 0.1,
        retry_delay_seconds: float = 0.0,
        abandoned_recovery_interval_seconds: float = 30.0,
        abandoned_task_stale_after_seconds: float = 60.0,
        curation_reconciler: CurationReconciler | None = None,
    ) -> None:
        self._ctx = ctx
        self._handlers = handlers or {}
        self._handler_factory = handler_factory
        self._poll_interval_seconds = poll_interval_seconds
        self._retry_delay_seconds = retry_delay_seconds
        self._abandoned_recovery_interval_seconds = abandoned_recovery_interval_seconds
        self._abandoned_task_stale_after_seconds = abandoned_task_stale_after_seconds
        self._curation_reconciler = curation_reconciler
        self._stop_event = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None
        self._reconciliation_runner: asyncio.Task[None] | None = None
        injected_provider_usage = getattr(ctx, "provider_usage", None)
        self._provider_usage: ProviderUsagePort = (
            injected_provider_usage
            if injected_provider_usage is not None
            else NullProviderUsagePort()
        )
        self._next_abandoned_recovery_at = 0.0
        self._reconciliation_lock = asyncio.Lock()
        # Deliberately process-local so restart recovery still sees stale attempts.
        self._owned_task_attempts: set[tuple[str, int]] = set()
        self._owned_task_attempts_lock = threading.Lock()
        self._last_reconciliation_snapshot: tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...], tuple[str, ...]] | None = None

    async def start(self) -> None:
        if self._runner is not None and not self._runner.done():
            return

        self._stop_event = asyncio.Event()
        self._next_abandoned_recovery_at = 0.0
        logger.info(
            "Starting runtime task worker",
            extra={
                "workspace_id": getattr(self._ctx, "workspace_id", None),
                "handler_names": sorted(self._handlers),
                "handler_count": len(self._handlers),
            },
        )
        self._runner = asyncio.create_task(self._run_loop())
        self._reconciliation_runner = asyncio.create_task(self._run_reconciliation_loop())

    async def stop(self, grace_period_seconds: float) -> None:
        self._stop_event.set()
        runner = self._runner
        reconciliation_runner = self._reconciliation_runner
        active_runners = [task for task in (runner, reconciliation_runner) if task is not None]
        if not active_runners:
            return

        try:
            await asyncio.wait_for(
                asyncio.shield(asyncio.gather(*active_runners, return_exceptions=True)),
                timeout=grace_period_seconds,
            )
        except TimeoutError:
            for active_runner in active_runners:
                active_runner.cancel()
            await asyncio.gather(*active_runners, return_exceptions=True)
        finally:
            self._runner = None
            self._reconciliation_runner = None

    async def _run_loop(self) -> None:
        task_queue = getattr(self._ctx, "task_queue", None)
        if task_queue is None:
            return

        await self._run_reconciliation_pass(reason="startup")

        while not self._stop_event.is_set():
            await self._recover_abandoned_tasks(task_queue)
            try:
                task = await asyncio.to_thread(
                    task_queue.claim_next,
                )
            except Exception:
                logger.exception("Runtime task worker failed while claiming the next task")
                await asyncio.sleep(self._poll_interval_seconds)
                continue
            if task is None:
                await asyncio.sleep(self._poll_interval_seconds)
                continue

            try:
                await self._process_task_with_reconciliation(task)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(
                    "Runtime task worker hit an unexpected error while processing a task",
                    extra={
                        "task_id": task.id,
                        "task_name": task.task_name,
                        "workspace_id": task.workspace_id,
                    },
                )
                await self._recover_unexpected_task_error(task, exc)

    async def _process_task_with_reconciliation(self, task: TaskRecord) -> None:
        owned_attempt = (task.id, task.execution_epoch)
        with self._owned_task_attempts_lock:
            self._owned_task_attempts.add(owned_attempt)
        processing_task: asyncio.Task[None] | None = None
        try:
            processing_task = asyncio.create_task(self._process_task(task))
            while True:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(processing_task),
                        timeout=self._abandoned_recovery_interval_seconds,
                    )
                    return
                except asyncio.TimeoutError:
                    task_queue = getattr(self._ctx, "task_queue", None)
                    if task_queue is None:
                        continue
                    current_task = await asyncio.to_thread(task_queue.get_task, task.id)
                    if current_task.status != "running" or current_task.execution_epoch != task.execution_epoch:
                        processing_task.cancel()
                        await asyncio.gather(processing_task, return_exceptions=True)
                        return
                    await asyncio.to_thread(
                        task_queue.touch_running_task,
                        task.id,
                        execution_epoch=task.execution_epoch,
                    )
                    await self._recover_abandoned_tasks(task_queue)
                    current_task = await asyncio.to_thread(task_queue.get_task, task.id)
                    if current_task.status == "running" and current_task.execution_epoch == task.execution_epoch:
                        continue
                    processing_task.cancel()
                    await asyncio.gather(processing_task, return_exceptions=True)
                    return
        finally:
            if processing_task is not None and not processing_task.done():
                processing_task.cancel()
                await asyncio.gather(processing_task, return_exceptions=True)
            with self._owned_task_attempts_lock:
                self._owned_task_attempts.discard(owned_attempt)

    async def _recover_unexpected_task_error(self, task: TaskRecord, exc: Exception) -> None:
        task_queue = getattr(self._ctx, "task_queue", None)
        if task_queue is None:
            return

        try:
            current = await asyncio.to_thread(task_queue.get_task, task.id)
        except Exception:
            logger.exception(
                "Runtime task worker could not load task state after an unexpected error",
                extra={"task_id": task.id, "task_name": task.task_name},
            )
            return

        if current.status != "running":
            return

        try:
            failed_task = await asyncio.to_thread(
                task_queue.fail,
                task.id,
                f"Unhandled runtime task worker error: {exc}",
                self._retry_delay_seconds,
                None,
                current.execution_epoch,
            )
        except Exception:
            logger.exception(
                "Runtime task worker could not mark task failed after an unexpected error",
                extra={"task_id": task.id, "task_name": task.task_name},
            )
            return

        await asyncio.to_thread(self._reconcile_terminal_task_state, failed_task)
        await self._schedule_follow_up(task, failed_task)

    async def _recover_abandoned_tasks(self, task_queue) -> None:
        current_time = time.time()
        if current_time < self._next_abandoned_recovery_at:
            return
        self._next_abandoned_recovery_at = current_time + self._abandoned_recovery_interval_seconds
        del task_queue
        await self._run_reconciliation_pass(now=current_time, reason="worker_loop")

    async def _run_reconciliation_loop(self) -> None:
        task_queue = getattr(self._ctx, "task_queue", None)
        if task_queue is None:
            return

        while not self._stop_event.is_set():
            try:
                await self._run_reconciliation_pass(reason="periodic")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Runtime task worker reconciliation pass failed")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._abandoned_recovery_interval_seconds,
                )
            except TimeoutError:
                continue

    async def _run_reconciliation_pass(
        self,
        *,
        now: float | None = None,
        reason: str,
    ) -> None:
        task_queue = getattr(self._ctx, "task_queue", None)
        if task_queue is None:
            return

        async with self._reconciliation_lock:
            current_time = time.time() if now is None else now
            await asyncio.to_thread(self._recover_leaked_work_items)
            recovered_tasks = await asyncio.to_thread(self._recover_running_tasks, current_time)
            recovered_task_ids: list[str] = []
            for recovered_task in recovered_tasks:
                recovered_task_ids.append(recovered_task.task.id)
                await asyncio.to_thread(self._reconcile_recovered_task_state, recovered_task)

            reconciled_conversation_ids = await asyncio.to_thread(
                self._reconcile_orphaned_running_conversations,
            )
            curation_outcomes = []
            if self._curation_reconciler is not None:
                curation_outcomes = await asyncio.to_thread(self._curation_reconciler.reconcile)

            journal = getattr(self._ctx, "journal", None)
            released_claim_ids: list[int] = []
            if journal is not None:
                released_claim_ids = await asyncio.to_thread(journal.release_orphaned_claims)

            pending_tasks = await asyncio.to_thread(task_queue.list_tasks, "pending", None, 200)
            overdue_pending = sorted(
                [
                    queued_task
                    for queued_task in pending_tasks
                    if queued_task.available_at <= current_time
                    and (current_time - queued_task.available_at) >= self._abandoned_task_stale_after_seconds
                ],
                key=lambda queued_task: current_time - queued_task.available_at,
                reverse=True,
            )
            overdue_summaries = [queued_task.id for queued_task in overdue_pending[:5]]
            snapshot = (
                tuple(recovered_task_ids),
                tuple(reconciled_conversation_ids),
                tuple(released_claim_ids),
                tuple(overdue_summaries),
            )
            should_log = bool(
                recovered_task_ids
                or reconciled_conversation_ids
                or released_claim_ids
                or overdue_pending
                or any(
                    outcome.disposition is not CurationReconciliationDisposition.CONTINUE
                    for outcome in curation_outcomes
                )
            )
            if should_log and snapshot != self._last_reconciliation_snapshot:
                logger.info(
                    "Runtime reconciliation pass completed",
                    extra={
                        "workspace_id": getattr(self._ctx, "workspace_id", None),
                        "reason": reason,
                        "recovered_task_count": len(recovered_task_ids),
                        "recovered_task_ids": recovered_task_ids,
                        "reconciled_conversation_count": len(reconciled_conversation_ids),
                        "reconciled_conversation_ids": reconciled_conversation_ids,
                        "released_orphaned_claim_count": len(released_claim_ids),
                        "released_orphaned_claim_ids": released_claim_ids,
                        "overdue_pending_count": len(overdue_pending),
                        "overdue_pending_task_ids": overdue_summaries,
                        "curation_reconciliation_outcomes": [
                            {
                                "disposition": outcome.disposition.value,
                                "reason_code": outcome.reason_code,
                                "run_id": None if outcome.run_id is None else str(outcome.run_id),
                            }
                            for outcome in curation_outcomes
                        ],
                    },
                )
            self._last_reconciliation_snapshot = snapshot if should_log else None

    def _reconcile_orphaned_running_conversations(self) -> list[str]:
        task_queue = getattr(self._ctx, "task_queue", None)
        if task_queue is None:
            return []

        reconciled_request_ids: list[str] = []
        running_conversations = self._provider_usage.list_conversations(status="running", limit=200)
        for conversation in running_conversations:
            if conversation.task_id is None:
                continue
            try:
                task = task_queue.get_task(conversation.task_id)
            except Exception:
                self._provider_usage.finalize_running_conversation(
                    request_id=conversation.request_id,
                    status="error",
                    error_text="Conversation task was missing during reconciliation",
                    reason_category="recovery",
                    reason_code="task_missing",
                    completed_at=max(conversation.completed_at, time.time()),
                )
                reconciled_request_ids.append(conversation.request_id)
                continue
            if task.status == "running":
                continue
            status = "success" if task.status == "completed" else "cancelled" if task.status == "cancelled" else "error"
            error_text = None if status == "success" else (task.last_error or f"Task finished with status {task.status}")
            reason_category = None if status == "success" else "cancellation" if status == "cancelled" else "recovery"
            reason_code = None if status == "success" else "task_cancelled" if status == "cancelled" else f"task_{task.status}"
            self._provider_usage.finalize_running_conversation(
                request_id=conversation.request_id,
                status=status,
                error_text=error_text,
                reason_category=reason_category,
                reason_code=reason_code,
                completed_at=task.completed_at or task.updated_at,
            )
            reconciled_request_ids.append(conversation.request_id)
        return reconciled_request_ids

    async def _process_task(self, task: TaskRecord) -> None:
        task_queue = getattr(self._ctx, "task_queue", None)
        if task_queue is None:
            return

        handler = self._handlers.get(task.task_name)
        if handler is None:
            logger.error(
                "Task handler missing; attempting refresh",
                extra={
                    "task_id": task.id,
                    "task_name": task.task_name,
                    "workspace_id": task.workspace_id,
                    "handler_names": sorted(self._handlers),
                    "handler_count": len(self._handlers),
                },
            )
            handler = self._refresh_handler(task.task_name)
        if handler is None:
            await asyncio.to_thread(
                task_queue.fail_permanently,
                task.id,
                f"No task handler registered for {task.task_name}",
                None,
                task.execution_epoch,
            )
            return

        journal = getattr(self._ctx, "journal", None)
        if journal is not None:
            idle_state = await asyncio.to_thread(
                should_pause_autonomous_recurring_maintenance,
                task,
                journal,
            )
            if idle_state is not None:
                completed_task = await asyncio.to_thread(
                    task_queue.complete,
                    task.id,
                    None,
                    build_idle_pause_result(task, idle_state),
                    task.execution_epoch,
                )
                await asyncio.to_thread(self._reconcile_terminal_task_state, completed_task)
                return

        if task.task_name == CURATOR_TASK_NAME and self._curation_reconciler is not None:
            curation_outcome = await asyncio.to_thread(self._curation_reconciler.before_curator_work)
            if curation_outcome.disposition is CurationReconciliationDisposition.BLOCK:
                blocked_task = await asyncio.to_thread(
                    task_queue.fail_permanently,
                    task.id,
                    f"Curation reconciliation blocked work: {curation_outcome.reason_code}",
                    None,
                    task.execution_epoch,
                )
                await asyncio.to_thread(self._reconcile_terminal_task_state, blocked_task)
                return
            if curation_outcome.disposition is CurationReconciliationDisposition.DEFER:
                delay = curation_outcome.retry_delay_seconds or self._retry_delay_seconds
                deferred_task = await asyncio.to_thread(
                    task_queue.retry_running_task,
                    task.id,
                    f"Curation reconciliation deferred work: {curation_outcome.reason_code}",
                    time.time() + max(delay, 0.0),
                    task.execution_epoch,
                )
                await asyncio.to_thread(
                    self._reconcile_retryable_interruption,
                    deferred_task,
                    termination_reason=f"curation_reconciliation_{curation_outcome.reason_code}",
                )
                return

        try:
            if iscoroutinefunction(handler):
                result = handler(self._ctx, task)
            else:
                result = await asyncio.to_thread(handler, self._ctx, task)
            if isawaitable(result):
                result = await result
        except asyncio.CancelledError:
            current_task = await asyncio.to_thread(task_queue.get_task, task.id)
            if current_task.status != "running":
                return
            if await asyncio.to_thread(task_queue.is_cancellation_requested, task.id):
                cancelled_task = await asyncio.to_thread(
                    task_queue.finalize_cancellation,
                    task.id,
                    cancelled_at=None,
                    execution_epoch=task.execution_epoch,
                )
                await asyncio.to_thread(self._reconcile_terminal_task_state, cancelled_task)
            elif self._stop_event.is_set():
                retried_task = await asyncio.to_thread(
                    task_queue.retry_running_task,
                    task.id,
                    "Task interrupted during daemon shutdown; retrying",
                    None,
                    task.execution_epoch,
                )
                await asyncio.to_thread(
                    self._reconcile_retryable_interruption,
                    retried_task,
                    termination_reason="daemon_shutdown_retry",
                )
            else:
                failed_task = await asyncio.to_thread(
                    task_queue.fail_permanently,
                    task.id,
                    "Task interrupted during worker shutdown",
                    None,
                    task.execution_epoch,
                )
                await asyncio.to_thread(self._reconcile_terminal_task_state, failed_task)
            raise
        except Exception as exc:
            if await asyncio.to_thread(task_queue.is_cancellation_requested, task.id):
                cancelled_task = await asyncio.to_thread(
                    task_queue.finalize_cancellation,
                    task.id,
                    cancelled_at=None,
                    execution_epoch=task.execution_epoch,
                )
                await asyncio.to_thread(self._reconcile_terminal_task_state, cancelled_task)
                return
            retry_delay_seconds = self._retry_delay_seconds
            requested_retry_delay = getattr(exc, "retry_delay_seconds", None)
            if isinstance(requested_retry_delay, (int, float)):
                retry_delay_seconds = max(float(requested_retry_delay), 0.0)
            failed_task = await asyncio.to_thread(
                task_queue.fail,
                task.id,
                str(exc),
                retry_delay_seconds,
                None,
                task.execution_epoch,
            )
            await asyncio.to_thread(self._reconcile_terminal_task_state, failed_task)
            await self._schedule_follow_up(task, failed_task)
            return

        normalized_result = result if isinstance(result, Mapping) else {}
        if await asyncio.to_thread(task_queue.is_cancellation_requested, task.id):
            cancelled_task = await asyncio.to_thread(
                task_queue.finalize_cancellation,
                task.id,
                cancelled_at=None,
                execution_epoch=task.execution_epoch,
            )
            await asyncio.to_thread(self._reconcile_terminal_task_state, cancelled_task)
            await self._schedule_follow_up(task, cancelled_task, normalized_result)
            return
        completed_task = await asyncio.to_thread(
            task_queue.complete,
            task.id,
            None,
            normalized_result,
            task.execution_epoch,
        )
        await asyncio.to_thread(self._reconcile_terminal_task_state, completed_task)
        await self._schedule_follow_up(task, completed_task, normalized_result)

    def _recover_running_tasks(self, current_time: float) -> list[_RecoveredTaskOutcome]:
        task_queue = getattr(self._ctx, "task_queue", None)
        if task_queue is None:
            return []
        attempt_repository = getattr(self._ctx, "task_execution_attempts", None)
        with self._owned_task_attempts_lock:
            owned_task_attempts = set(self._owned_task_attempts)
        recovered: list[_RecoveredTaskOutcome] = []
        running_tasks = task_queue.list_tasks(status="running", workspace_id=None, limit=200)
        for task in running_tasks:
            if (task.id, task.execution_epoch) in owned_task_attempts:
                continue
            recovered_task = self._recover_running_task(
                task_queue,
                task,
                attempt_repository=attempt_repository,
                current_time=current_time,
            )
            if recovered_task is not None:
                recovered.append(recovered_task)
        return recovered

    def _recover_running_task(
        self,
        task_queue,
        task: TaskRecord,
        *,
        attempt_repository: TaskExecutionAttemptPort | None,
        current_time: float,
    ) -> _RecoveredTaskOutcome | None:
        attempt = self._get_running_task_attempt(task, attempt_repository=attempt_repository)
        plan = self._classify_running_task_recovery(task, attempt=attempt, current_time=current_time)
        if plan is None:
            return None
        return self._apply_running_task_recovery_plan(task_queue, task, plan)

    def _get_running_task_attempt(
        self,
        task: TaskRecord,
        *,
        attempt_repository,
    ) -> TaskExecutionAttemptRecordLike | None:
        if attempt_repository is None:
            return None
        try:
            return attempt_repository.get_attempt(task_id=task.id, execution_epoch=task.execution_epoch)
        except ValueError:
            return None

    def _classify_running_task_recovery(
        self,
        task: TaskRecord,
        *,
        attempt: TaskExecutionAttemptRecordLike | None,
        current_time: float,
    ) -> _RunningTaskRecoveryPlan | None:
        recent_activity_at = max(
            task.updated_at,
            task.started_at or task.updated_at,
            task.claimed_at or task.updated_at,
        )
        subprocess_pid = task.subprocess_pid
        if attempt is not None:
            attempt_activity_at = max(
                attempt.started_at,
                attempt.last_heartbeat_at or attempt.started_at,
            )
            recent_activity_at = max(recent_activity_at, attempt_activity_at)
            if attempt.subprocess_pid is not None:
                subprocess_pid = attempt.subprocess_pid

        is_stale = max(current_time - recent_activity_at, 0.0) >= self._abandoned_task_stale_after_seconds
        if subprocess_pid is not None:
            if task_queue_module._is_process_alive(subprocess_pid):
                return None
            if not is_stale:
                return None
            if task.cancellation_requested_at is not None:
                return _RunningTaskRecoveryPlan.finalize_cancellation(recovered_at=current_time)
            return _RunningTaskRecoveryPlan.retry_dead_subprocess(
                subprocess_pid=subprocess_pid,
                recovered_at=current_time,
                retry_delay_seconds=self._retry_delay_seconds,
            )
        if task.cancellation_requested_at is not None:
            return _RunningTaskRecoveryPlan.finalize_cancellation(recovered_at=current_time)
        if not is_stale:
            return None
        return _RunningTaskRecoveryPlan.retry_abandoned(
            recovered_at=current_time,
            retry_delay_seconds=self._retry_delay_seconds,
        )

    def _apply_running_task_recovery_plan(
        self,
        task_queue,
        task: TaskRecord,
        plan: _RunningTaskRecoveryPlan,
    ) -> _RecoveredTaskOutcome:
        if plan.action is RecoveryAction.FINALIZE_CANCELLATION:
            return self._recover_task_outcome(
                task_queue,
                lambda: task_queue.finalize_cancellation(
                    task.id,
                    cancelled_at=plan.recovered_at,
                    execution_epoch=task.execution_epoch,
                ),
                preferred_reconciliation_policy=_RecoveredTaskReconciliationPolicy.terminal(),
            )
        if plan.action is RecoveryAction.RETRY_DEAD_SUBPROCESS:
            return self._recover_task_outcome(
                task_queue,
                lambda: task_queue.fail(
                    task.id,
                    plan.error_text,
                    plan.retry_delay_seconds,
                    failed_at=plan.recovered_at,
                    execution_epoch=task.execution_epoch,
                ),
                preferred_reconciliation_policy=_RecoveredTaskReconciliationPolicy.retryable_interruption(
                    termination_reason=plan.retry_termination_reason or "provider_subprocess_exited_retry",
                ),
            )
        return self._recover_task_outcome(
            task_queue,
            lambda: task_queue.fail_permanently(
                task.id,
                plan.error_text,
                failed_at=plan.recovered_at,
                execution_epoch=task.execution_epoch,
            ),
            preferred_reconciliation_policy=_RecoveredTaskReconciliationPolicy.terminal(),
        )

    def _recover_task_outcome(
        self,
        task_queue,
        callback: Callable[[], TaskRecord],
        *,
        preferred_reconciliation_policy: _RecoveredTaskReconciliationPolicy,
    ) -> _RecoveredTaskOutcome:
        task = self._recover_terminal_task(task_queue, callback)
        reconciliation_policy = preferred_reconciliation_policy
        if preferred_reconciliation_policy.is_retryable_interruption and task.status != "pending":
            reconciliation_policy = _RecoveredTaskReconciliationPolicy.terminal()
        return _RecoveredTaskOutcome(
            task=task,
            reconciliation_policy=reconciliation_policy,
        )

    def _recover_terminal_task(self, task_queue, callback: Callable[[], TaskRecord]) -> TaskRecord:
        retry_transition = getattr(task_queue, "_retry_recovery_transition", None)
        if callable(retry_transition):
            return cast(Callable[[Callable[[], TaskRecord]], TaskRecord], retry_transition)(callback)
        return callback()

    def _reconcile_terminal_task_state(self, task: TaskRecord) -> None:
        self._reconcile_task_execution_attempt(task)
        self._reconcile_task_conversations(task)
        self._release_task_work_items(task)
        self._release_task_embedding_repairs(task)

    def _reconcile_recovered_task_state(self, outcome: _RecoveredTaskOutcome) -> None:
        outcome.reconciliation_policy.reconcile(self, outcome.task)

    def _reconcile_retryable_interruption(self, task: TaskRecord, *, termination_reason: str) -> None:
        self._finish_task_execution_attempt(
            task,
            status="error",
            error_text=task.last_error,
            termination_reason=termination_reason,
        )
        self._provider_usage.reconcile_running_task_conversations(
            task_id=task.id,
            status="error",
            error_text=task.last_error,
            reason_category="recovery",
            reason_code=termination_reason,
            completed_at=task.updated_at,
        )
        self._release_task_work_items(task)
        self._release_task_embedding_repairs(task)

    def _reconcile_task_execution_attempt(self, task: TaskRecord) -> None:
        if task.status not in {"completed", "failed", "cancelled"}:
            return
        attempt_status = "success" if task.status == "completed" else "cancelled" if task.status == "cancelled" else "error"
        error_text = None if task.status == "completed" else task.last_error
        termination_reason = None if task.status == "completed" else f"task_{task.status}"
        self._finish_task_execution_attempt(
            task,
            status=attempt_status,
            error_text=error_text,
            termination_reason=termination_reason,
        )

    def _finish_task_execution_attempt(
        self,
        task: TaskRecord,
        *,
        status: str,
        error_text: str | None,
        termination_reason: str | None,
    ) -> None:
        attempt_repository = getattr(self._ctx, "task_execution_attempts", None)
        if attempt_repository is None:
            return
        try:
            attempt_repository.finish_attempt(
                task_id=task.id,
                execution_epoch=task.execution_epoch,
                status=status,
                completed_at=task.completed_at or task.updated_at,
                error_text=error_text,
                termination_reason=termination_reason,
            )
        except ValueError:
            return

    def _reconcile_task_conversations(self, task: TaskRecord) -> None:
        if task.status not in {"failed", "cancelled"}:
            return
        conversation_status = "cancelled" if task.status == "cancelled" else "error"
        self._provider_usage.reconcile_running_task_conversations(
            task_id=task.id,
            status=conversation_status,
            error_text=task.last_error,
            completed_at=task.completed_at or task.updated_at,
        )

    def _release_task_work_items(self, task: TaskRecord) -> None:
        work_items = getattr(self._ctx, "work_items", None)
        if work_items is None:
            return
        released_at = task.completed_at or task.updated_at
        for work_item in work_items.list_items(status="running", lease_owner=task.id, limit=100):
            try:
                work_items.release_item(work_item.id, released_at=released_at)
            except Exception:
                logger.exception(
                    "Runtime task worker failed to release a leaked work item for a terminal task",
                    extra={
                        "task_id": task.id,
                        "task_name": task.task_name,
                        "task_status": task.status,
                        "work_item_id": work_item.id,
                        "work_item_family": work_item.family_key,
                    },
                )

    def _release_task_embedding_repairs(self, task: TaskRecord) -> None:
        embedding_repair_queue = getattr(self._ctx, "embedding_repair_queue", None)
        if embedding_repair_queue is None:
            return
        released_at = task.completed_at or task.updated_at
        for repair_item in embedding_repair_queue.list_items(status="running", lease_owner=task.id, limit=100):
            try:
                embedding_repair_queue.release_item(repair_item.id, released_at=released_at)
            except Exception:
                logger.exception(
                    "Runtime task worker failed to release a leaked embedding repair item for a terminal task",
                    extra={
                        "task_id": task.id,
                        "task_name": task.task_name,
                        "task_status": task.status,
                        "repair_item_id": repair_item.id,
                        "memory_id": repair_item.memory_id,
                    },
                )

    def _recover_leaked_work_items(self) -> None:
        task_queue = getattr(self._ctx, "task_queue", None)
        work_items = getattr(self._ctx, "work_items", None)
        if task_queue is None or work_items is None:
            self._recover_leaked_embedding_repairs()
            return
        for work_item in work_items.list_items(status="running", limit=200):
            lease_owner = work_item.lease_owner
            if not isinstance(lease_owner, str) or not lease_owner:
                self._release_recovered_work_item(work_items, work_item.id, work_item.family_key)
                continue
            try:
                owner_task = task_queue.get_task(lease_owner)
            except Exception:
                self._release_recovered_work_item(work_items, work_item.id, work_item.family_key)
                continue
            if owner_task.status != "running":
                self._release_recovered_work_item(work_items, work_item.id, work_item.family_key)
        self._recover_leaked_embedding_repairs()

    def _release_recovered_work_item(self, work_items, work_item_id: str, family_key: str) -> None:
        try:
            work_items.release_item(work_item_id)
        except Exception:
            logger.exception(
                "Runtime task worker failed to recover a leaked running work item",
                extra={
                    "work_item_id": work_item_id,
                    "work_item_family": family_key,
                },
            )

    def _recover_leaked_embedding_repairs(self) -> None:
        task_queue = getattr(self._ctx, "task_queue", None)
        embedding_repair_queue = getattr(self._ctx, "embedding_repair_queue", None)
        if task_queue is None or embedding_repair_queue is None:
            return
        for repair_item in embedding_repair_queue.list_items(status="running", limit=200):
            lease_owner = repair_item.lease_owner
            if not isinstance(lease_owner, str) or not lease_owner:
                self._release_recovered_embedding_repair(embedding_repair_queue, repair_item.id, repair_item.memory_id)
                continue
            try:
                owner_task = task_queue.get_task(lease_owner)
            except Exception:
                self._release_recovered_embedding_repair(embedding_repair_queue, repair_item.id, repair_item.memory_id)
                continue
            if owner_task.status != "running":
                self._release_recovered_embedding_repair(embedding_repair_queue, repair_item.id, repair_item.memory_id)

    def _release_recovered_embedding_repair(self, embedding_repair_queue, item_id: str, memory_id: str) -> None:
        try:
            embedding_repair_queue.release_item(item_id)
        except Exception:
            logger.exception(
                "Runtime task worker failed to recover a leaked running embedding repair item",
                extra={
                    "repair_item_id": item_id,
                    "memory_id": memory_id,
                },
            )

    async def _schedule_follow_up(
        self,
        task: TaskRecord,
        terminal_task: TaskRecord,
        run_result: TaskRunResultSource = None,
    ) -> None:
        task_queue = getattr(self._ctx, "task_queue", None)
        if task_queue is None:
            return

        if task.task_name == SYSTEM1_INGEST_TASK_NAME and terminal_task.status == "completed":
            journal = getattr(self._ctx, "journal", None)
            if journal is not None:
                schedule_ingest = schedule_system1_ingest
                if _should_schedule_ingest_continuation(run_result):
                    schedule_ingest = schedule_system1_ingest_continuation
                await asyncio.to_thread(
                    schedule_ingest,
                    task_queue,
                    journal,
                    task.workspace_id,
                    now=terminal_task.completed_at or terminal_task.updated_at,
                )
            return

        interval_seconds = AUTONOMOUS_RECURRING_TASK_INTERVAL_SECONDS.get(task.task_name)
        if interval_seconds is None or terminal_task.status not in {"completed", "failed"}:
            return

        jitter_seconds = compute_recurring_jitter_seconds(interval_seconds)
        next_available_at = (terminal_task.completed_at or terminal_task.updated_at) + interval_seconds + jitter_seconds
        await asyncio.to_thread(
            task_queue.enqueue_unique,
            task.task_name,
            {
                "workspace_id": None,
                "trigger": "recurring_follow_up",
                "interval_seconds": interval_seconds,
                "jitter_seconds": jitter_seconds,
            },
            None,
            task_priority(task.task_name),
            3,
            next_available_at,
        )

    def _refresh_handler(self, task_name: str):
        if self._handler_factory is None:
            return None
        refreshed = self._handler_factory()
        if not refreshed:
            logger.error("Task handler refresh returned no handlers", extra={"task_name": task_name})
            return None
        self._handlers = refreshed
        logger.warning(
            "Refreshed runtime task handlers",
            extra={
                "task_name": task_name,
                "handler_names": sorted(self._handlers),
                "handler_count": len(self._handlers),
            },
        )
        return self._handlers.get(task_name)


def _should_schedule_ingest_continuation(run_result: TaskRunResultSource) -> bool:
    if not isinstance(run_result, Mapping):
        return False
    pending_remaining = run_result.get("pending_remaining")
    meaningful_actions = run_result.get("meaningful_actions")
    return (
        isinstance(pending_remaining, int)
        and not isinstance(pending_remaining, bool)
        and pending_remaining > 0
        and isinstance(meaningful_actions, int)
        and not isinstance(meaningful_actions, bool)
        and meaningful_actions > 0
    )
