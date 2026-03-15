from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_memory.config import Config
from mcp_memory.context import ApplicationContext
from mcp_memory.core.journal import System1Journal
from mcp_memory.core.tasks import SQLiteTaskQueue, TaskRecord, TaskRunSummary
from mcp_memory.utils.db import DatabaseManager


@dataclass(frozen=True)
class RuntimeInfoFacade:
    workspace_id: str | None
    workspace_root: Path | None
    memory_path: Path | None
    db_path: Path | None
    ai_provider_name: str | None
    ai_model_name: str | None
    runtime_active: bool
    client_count: int
    task_queue_enabled: bool

    @classmethod
    def from_context(
        cls,
        ctx: ApplicationContext,
        controller: Any | None = None,
    ) -> RuntimeInfoFacade:
        config: Config | None = ctx.config
        db_manager: DatabaseManager | None = ctx.db_manager
        return cls(
            workspace_id=ctx.workspace_id,
            workspace_root=ctx.workspace_root,
            memory_path=ctx.memory_path,
            db_path=db_manager.db_path if db_manager is not None else None,
            ai_provider_name=config.ai.provider if config is not None else None,
            ai_model_name=config.ai.model if config is not None else None,
            runtime_active=bool(getattr(controller, "has_runtime", False)),
            client_count=int(getattr(controller, "client_count", 0)),
            task_queue_enabled=ctx.task_queue is not None,
        )


@dataclass(frozen=True)
class JournalFacade:
    journal: System1Journal | None

    @classmethod
    def from_context(cls, ctx: ApplicationContext) -> JournalFacade:
        return cls(journal=ctx.journal)

    def count_by_status(self) -> dict[str, int]:
        if self.journal is None:
            return {}
        return self.journal.count_by_status()


@dataclass(frozen=True)
class TaskQueueFacade:
    task_queue: SQLiteTaskQueue | None

    @classmethod
    def from_context(cls, ctx: ApplicationContext) -> TaskQueueFacade:
        return cls(task_queue=ctx.task_queue)

    def count_by_status(self) -> dict[str, int]:
        if self.task_queue is None:
            return {}
        return self.task_queue.count_by_status()

    def list_tasks(
        self,
        *,
        status: str | None = None,
        workspace_id: str | None = None,
        limit: int = 20,
    ) -> list[TaskRecord]:
        if self.task_queue is None:
            return []
        return self.task_queue.list_tasks(
            status=status,
            workspace_id=workspace_id,
            limit=limit,
        )

    def enqueue(
        self,
        *,
        task_name: str,
        data: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        priority: int = 100,
        max_retries: int = 3,
        available_at: float | None = None,
        task_id: str | None = None,
    ) -> TaskRecord:
        if self.task_queue is None:
            raise ValueError("task_queue_not_initialized")
        return self.task_queue.enqueue(
            task_name=task_name,
            data=data,
            workspace_id=workspace_id,
            priority=priority,
            max_retries=max_retries,
            available_at=available_at,
            task_id=task_id,
        )

    def enqueue_unique(
        self,
        *,
        task_name: str,
        data: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        priority: int = 100,
        max_retries: int = 3,
        available_at: float | None = None,
    ) -> tuple[TaskRecord, bool]:
        if self.task_queue is None:
            raise ValueError("task_queue_not_initialized")
        return self.task_queue.enqueue_unique(
            task_name=task_name,
            data=data,
            workspace_id=workspace_id,
            priority=priority,
            max_retries=max_retries,
            available_at=available_at,
        )

    def summarize_task_runs(
        self,
        task_names: list[str],
        *,
        workspace_id: str | None = None,
    ) -> list[TaskRunSummary]:
        if self.task_queue is None:
            return []
        return self.task_queue.summarize_task_runs(
            task_names,
            workspace_id=workspace_id,
        )

    def request_cancel(
        self,
        task_id: str,
        *,
        cancelled_by: str,
        reason: str,
    ) -> TaskRecord:
        if self.task_queue is None:
            raise ValueError("task_queue_not_initialized")
        return self.task_queue.request_cancel(
            task_id,
            cancelled_by=cancelled_by,
            reason=reason,
        )

    def get_task(self, task_id: str) -> TaskRecord:
        if self.task_queue is None:
            raise ValueError("task_queue_not_initialized")
        return self.task_queue.get_task(task_id)