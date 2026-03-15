from __future__ import annotations

import asyncio
from collections.abc import Callable
from inspect import isawaitable
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.system1_scheduling import schedule_system1_ingest
from mcp_memory.core.task_handlers import RECURRING_TASK_INTERVAL_SECONDS, SYSTEM1_INGEST_TASK_NAME
from mcp_memory.core.tasks import TaskRecord


class RuntimeTaskWorker:
    def __init__(
        self,
        ctx: ApplicationContext,
        handlers: dict[str, Callable[[ApplicationContext, TaskRecord], Any]] | None = None,
        poll_interval_seconds: float = 0.1,
        retry_delay_seconds: float = 0.0,
    ) -> None:
        self._ctx = ctx
        self._handlers = handlers or {}
        self._poll_interval_seconds = poll_interval_seconds
        self._retry_delay_seconds = retry_delay_seconds
        self._stop_event = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._runner is not None and not self._runner.done():
            return

        self._stop_event = asyncio.Event()
        self._runner = asyncio.create_task(self._run_loop())

    async def stop(self, grace_period_seconds: float) -> None:
        self._stop_event.set()
        runner = self._runner
        if runner is None:
            return

        try:
            await asyncio.wait_for(asyncio.shield(runner), timeout=grace_period_seconds)
        except TimeoutError:
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)
        finally:
            self._runner = None

    async def _run_loop(self) -> None:
        task_queue = getattr(self._ctx, "task_queue", None)
        if task_queue is None:
            return

        while not self._stop_event.is_set():
            task = await asyncio.to_thread(
                task_queue.claim_next,
                workspace_id=getattr(self._ctx, "workspace_id", None),
            )
            if task is None:
                await asyncio.sleep(self._poll_interval_seconds)
                continue

            await self._process_task(task)

    async def _process_task(self, task: TaskRecord) -> None:
        task_queue = getattr(self._ctx, "task_queue", None)
        if task_queue is None:
            return

        handler = self._handlers.get(task.task_name)
        if handler is None:
            await asyncio.to_thread(
                task_queue.fail_permanently,
                task.id,
                f"No task handler registered for {task.task_name}",
            )
            return

        try:
            result = handler(self._ctx, task)
            if isawaitable(result):
                result = await result
        except Exception as exc:
            failed_task = await asyncio.to_thread(
                task_queue.fail,
                task.id,
                str(exc),
                self._retry_delay_seconds,
            )
            await self._schedule_follow_up(task, failed_task)
            return

        normalized_result = result if isinstance(result, dict) else {}
        completed_task = await asyncio.to_thread(task_queue.complete, task.id, None, normalized_result)
        await self._schedule_follow_up(task, completed_task)

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

        next_available_at = (terminal_task.completed_at or terminal_task.updated_at) + interval_seconds
        await asyncio.to_thread(
            task_queue.enqueue_unique,
            task.task_name,
            {
                "workspace_id": task.workspace_id,
                "trigger": "recurring_follow_up",
                "interval_seconds": interval_seconds,
            },
            task.workspace_id,
            100,
            3,
            next_available_at,
        )