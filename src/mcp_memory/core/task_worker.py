from __future__ import annotations

import asyncio
from collections.abc import Callable
from inspect import isawaitable
from typing import Any

from mcp_memory.context import ApplicationContext
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
            task = await asyncio.to_thread(task_queue.claim_next)
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
                await result
        except Exception as exc:
            await asyncio.to_thread(
                task_queue.fail,
                task.id,
                str(exc),
                self._retry_delay_seconds,
            )
            return

        await asyncio.to_thread(task_queue.complete, task.id)