from __future__ import annotations

from pathlib import Path

from mcp_memory.context import ApplicationContext
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.management.models import (
    CompactMemoryRecord,
    HealthPayload,
    JournalSummary,
    MemoryDetailPayload,
    OverviewCounts,
    OverviewPayload,
    StorageSummary,
    TaskListPayload,
    TaskStatusSummary,
)
from mcp_memory.mcp.services import record_to_payload


class ManagementService:
    def __init__(self, ctx: ApplicationContext, controller) -> None:
        self._ctx = ctx
        self._controller = controller

    def get_health(self):
        db_path = None
        if self._ctx.db_manager is not None:
            db_path = str(self._ctx.db_manager.db_path)

        return HealthPayload(
            status="ok",
            project_name=self._ctx.project_name,
            memory_path=str(self._ctx.memory_path) if self._ctx.memory_path is not None else None,
            db_path=db_path,
            runtime_active=self._controller.has_runtime,
            client_count=self._controller.client_count,
            task_queue_enabled=self._ctx.task_queue is not None,
        )

    def get_overview(self, recent_limit: int = 10, failed_limit: int = 10):
        records = []
        if self._ctx.repository is not None:
            records = self._ctx.repository.list_memories(limit=500)

        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}
        for record in records:
            by_type[record.type] = by_type.get(record.type, 0) + 1
            by_status[record.status] = by_status.get(record.status, 0) + 1

        recent_records = [self._compact_record(record) for record in records[:recent_limit]]
        task_counts = self._ctx.task_queue.count_by_status() if self._ctx.task_queue is not None else {}
        failed_tasks = self._serialize_tasks(
            self._ctx.task_queue.list_tasks(status="failed", limit=failed_limit)
            if self._ctx.task_queue is not None
            else []
        )
        sqlite_bytes = 0
        sqlite_path = None
        if self._ctx.db_manager is not None:
            sqlite_path = self._ctx.db_manager.db_path
            if sqlite_path.exists():
                sqlite_bytes = sqlite_path.stat().st_size

        journal_counts = self._ctx.journal.count_by_status() if self._ctx.journal is not None else {}

        return OverviewPayload(
            memories=OverviewCounts(total=len(records), by_type=by_type, by_status=by_status),
            recent_memories=recent_records,
            tasks=TaskStatusSummary(
                by_status=task_counts,
                failed_count=task_counts.get("failed", 0),
            ),
            failed_tasks=failed_tasks,
            journal=JournalSummary(pending_count=journal_counts.get("pending", 0)),
            storage=StorageSummary(
                sqlite_bytes=sqlite_bytes,
                sqlite_path=str(sqlite_path) if sqlite_path is not None else None,
            ),
        )

    def list_tasks(
        self,
        status: str | None = None,
        workspace_id: str | None = None,
        limit: int = 20,
    ):
        if self._ctx.task_queue is None:
            return TaskListPayload(tasks=[])
        tasks = self._ctx.task_queue.list_tasks(status=status, workspace_id=workspace_id, limit=limit)
        return TaskListPayload(tasks=self._serialize_tasks(tasks))

    def list_memories(
        self,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ):
        if self._ctx.repository is None:
            return []
        records = self._ctx.repository.list_memories(
            workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            limit=limit,
        )
        return [self._compact_record(record) for record in records]

    def get_memory_detail(self, memory_id: str):
        if self._ctx.repository is None:
            raise ValueError("repository_not_initialized")

        record = self._ctx.repository.get_memory(memory_id)
        if record is None:
            raise ValueError("memory_not_found")

        outgoing = self._ctx.repository.get_links(memory_id, direction="outgoing")
        incoming = self._ctx.repository.get_links(memory_id, direction="incoming")

        superseded = []
        for link in outgoing:
            if link.link_type != "SUPERSEDES":
                continue
            target = self._ctx.repository.get_memory(link.target_id)
            if target is not None:
                superseded.append(record_to_payload(target))

        return MemoryDetailPayload(
            record=record_to_payload(record),
            relationships={
                "incoming": [self._serialize_link(link) for link in incoming],
                "outgoing": [self._serialize_link(link) for link in outgoing],
            },
            superseded=superseded,
        )

    def load_dashboard_html(self):
        static_path = Path(__file__).with_name("static") / "index.html"
        return static_path.read_text(encoding="utf-8")

    def _compact_record(self, record):
        return CompactMemoryRecord(
            id=record.id,
            title=record.title,
            summary=record.summary,
            type=record.type,
            status=record.status,
            updated_at=record.updated_at,
            workspace_ids=list(record.workspace_ids),
            tags=list(record.tags),
        )

    def _serialize_tasks(self, tasks: list[TaskRecord]):
        return [
            {
                "id": task.id,
                "task_name": task.task_name,
                "workspace_id": task.workspace_id,
                "status": task.status,
                "priority": task.priority,
                "retries_count": task.retries_count,
                "max_retries": task.max_retries,
                "created_at": task.created_at,
                "updated_at": task.updated_at,
                "available_at": task.available_at,
                "claimed_at": task.claimed_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "last_error": task.last_error,
            }
            for task in tasks
        ]

    def _serialize_link(self, link):
        return {
            "source_id": link.source_id,
            "target_id": link.target_id,
            "link_type": link.link_type,
            "context": link.context,
        }