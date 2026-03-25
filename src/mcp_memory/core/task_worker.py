from __future__ import annotations

import asyncio
from collections.abc import Callable
from inspect import isawaitable, iscoroutinefunction
import logging
import time
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.maintenance_idle import (
    build_idle_pause_result,
    should_pause_autonomous_recurring_maintenance,
)
from mcp_memory.core.recurring_jitter import compute_recurring_jitter_seconds
from mcp_memory.core.system1_scheduling import schedule_system1_ingest
from mcp_memory.core.task_handlers import RECURRING_TASK_INTERVAL_SECONDS, SYSTEM1_INGEST_TASK_NAME, task_priority
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.provider_usage_store import ProviderUsageRepository


logger = logging.getLogger(__name__)


class RuntimeTaskWorker:
    def __init__(
        self,
        ctx: ApplicationContext,
        handlers: dict[str, Callable[[ApplicationContext, TaskRecord], Any]] | None = None,
        handler_factory: Callable[[], dict[str, Callable[[ApplicationContext, TaskRecord], Any]]] | None = None,
        poll_interval_seconds: float = 0.1,
        retry_delay_seconds: float = 0.0,
        abandoned_recovery_interval_seconds: float = 30.0,
        abandoned_task_stale_after_seconds: float = 60.0,
    ) -> None:
        self._ctx = ctx
        self._handlers = handlers or {}
        self._handler_factory = handler_factory
        self._poll_interval_seconds = poll_interval_seconds
        self._retry_delay_seconds = retry_delay_seconds
        self._abandoned_recovery_interval_seconds = abandoned_recovery_interval_seconds
        self._abandoned_task_stale_after_seconds = abandoned_task_stale_after_seconds
        self._stop_event = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None
        self._reconciliation_runner: asyncio.Task[None] | None = None
        self._provider_usage = ProviderUsageRepository(ctx.db_manager, workspace_id=None)
        self._next_abandoned_recovery_at = 0.0
        self._reconciliation_lock = asyncio.Lock()
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
        processing_task = asyncio.create_task(self._process_task(task))
        try:
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
                    await self._recover_abandoned_tasks(task_queue)
                    current_task = await asyncio.to_thread(task_queue.get_task, task.id)
                    if current_task.status == "running":
                        continue
                    processing_task.cancel()
                    await asyncio.gather(processing_task, return_exceptions=True)
                    return
        finally:
            if not processing_task.done():
                processing_task.cancel()
                await asyncio.gather(processing_task, return_exceptions=True)

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
            recovered_tasks = await asyncio.to_thread(
                task_queue.recover_abandoned_running_tasks,
                stale_after_seconds=self._abandoned_task_stale_after_seconds,
                now=current_time,
            )
            recovered_task_ids: list[str] = []
            for recovered_task in recovered_tasks:
                recovered_task_ids.append(recovered_task.id)
                await asyncio.to_thread(self._reconcile_terminal_task_state, recovered_task)

            reconciled_conversation_ids = await asyncio.to_thread(
                self._reconcile_orphaned_running_conversations,
            )

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
            should_log = bool(recovered_task_ids or reconciled_conversation_ids or released_claim_ids or overdue_pending)
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
                )
                await asyncio.to_thread(self._reconcile_terminal_task_state, completed_task)
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
                cancelled_task = await asyncio.to_thread(task_queue.finalize_cancellation, task.id)
                await asyncio.to_thread(self._reconcile_terminal_task_state, cancelled_task)
            else:
                failed_task = await asyncio.to_thread(
                    task_queue.fail_permanently,
                    task.id,
                    "Task interrupted during worker shutdown",
                )
                await asyncio.to_thread(self._reconcile_terminal_task_state, failed_task)
            raise
        except Exception as exc:
            if await asyncio.to_thread(task_queue.is_cancellation_requested, task.id):
                cancelled_task = await asyncio.to_thread(task_queue.finalize_cancellation, task.id)
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
            )
            await asyncio.to_thread(self._reconcile_terminal_task_state, failed_task)
            await self._schedule_follow_up(task, failed_task)
            return

        normalized_result = result if isinstance(result, dict) else {}
        if await asyncio.to_thread(task_queue.is_cancellation_requested, task.id):
            cancelled_task = await asyncio.to_thread(task_queue.finalize_cancellation, task.id)
            await asyncio.to_thread(self._reconcile_terminal_task_state, cancelled_task)
            await self._schedule_follow_up(task, cancelled_task)
            return
        completed_task = await asyncio.to_thread(task_queue.complete, task.id, None, normalized_result)
        await asyncio.to_thread(self._reconcile_terminal_task_state, completed_task)
        await self._schedule_follow_up(task, completed_task)

    def _reconcile_terminal_task_state(self, task: TaskRecord) -> None:
        self._reconcile_task_conversations(task)
        self._release_task_work_items(task)
        self._release_task_embedding_repairs(task)

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

    async def _schedule_follow_up(self, task: TaskRecord, terminal_task: TaskRecord) -> None:
        task_queue = getattr(self._ctx, "task_queue", None)
        if task_queue is None:
            return

        if task.task_name == SYSTEM1_INGEST_TASK_NAME and terminal_task.status == "completed":
            journal = getattr(self._ctx, "journal", None)
            if journal is not None:
                await asyncio.to_thread(
                    schedule_system1_ingest,
                    task_queue,
                    journal,
                    task.workspace_id,
                    now=terminal_task.completed_at or terminal_task.updated_at,
                )
            return

        interval_seconds = RECURRING_TASK_INTERVAL_SECONDS.get(task.task_name)
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
