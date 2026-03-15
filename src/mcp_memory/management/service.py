from __future__ import annotations

from pathlib import Path

from mcp_memory.context import ApplicationContext
from mcp_memory.core import MemoryPipeline
from mcp_memory.management.models import (
    HealthPayload,
    JournalSummary,
    MemoryListPayload,
    MemoryDetailPayload,
    OverviewCounts,
    OverviewPayload,
    StorageSummary,
    TaskListPayload,
    TaskStatusSummary,
)
from mcp_memory.serialization import (
    compact_memory_record_payload,
    link_payload,
    memory_record_payload,
    task_payload,
)


class ManagementService:
    def __init__(self, ctx: ApplicationContext, controller) -> None:
        pipeline = MemoryPipeline.from_context(ctx, controller)
        self._runtime_info = pipeline.runtime_info
        self._journal = pipeline.journal
        self._task_queue = pipeline.task_queue
        self._memory_queries = pipeline.memory_queries
        self._dashboard_static_path = Path(__file__).with_name("static") / "index.html"

    def get_health(self):
        return HealthPayload(
            status="ok",
            workspace_id=self._runtime_info.workspace_id,
            workspace_root=str(self._runtime_info.workspace_root) if self._runtime_info.workspace_root is not None else None,
            memory_path=str(self._runtime_info.memory_path) if self._runtime_info.memory_path is not None else None,
            db_path=str(self._runtime_info.db_path) if self._runtime_info.db_path is not None else None,
            runtime_active=self._runtime_info.runtime_active,
            client_count=self._runtime_info.client_count,
            task_queue_enabled=self._runtime_info.task_queue_enabled,
        )

    def get_overview(self, recent_limit: int = 10, failed_limit: int = 10):
        records = [] if self._memory_queries is None else self._memory_queries.list_memories(limit=500)

        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}
        for record in records:
            by_type[record.type] = by_type.get(record.type, 0) + 1
            by_status[record.status] = by_status.get(record.status, 0) + 1

        recent_records = [compact_memory_record_payload(record) for record in records[:recent_limit]]
        task_counts = self._task_queue.count_by_status()
        failed_tasks = [
            task_payload(task)
            for task in self._task_queue.list_tasks(status="failed", limit=failed_limit)
        ]
        sqlite_bytes = 0
        sqlite_path = self._runtime_info.db_path
        if sqlite_path is not None and sqlite_path.exists():
            sqlite_bytes = sqlite_path.stat().st_size

        journal_counts = self._journal.count_by_status()

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

    def get_memory_detail(self, memory_id: str):
        if self._memory_queries is None:
            raise ValueError("repository_not_initialized")

        record = self._memory_queries.get_memory(memory_id)
        if record is None:
            raise ValueError("memory_not_found")

        outgoing = self._memory_queries.get_links(memory_id, direction="outgoing")
        incoming = self._memory_queries.get_links(memory_id, direction="incoming")
        superseded = [
            memory_record_payload(target)
            for target in self._memory_queries.get_superseded_records(memory_id)
        ]

        return MemoryDetailPayload(
            record=memory_record_payload(record),
            relationships={
                "incoming": [link_payload(link) for link in incoming],
                "outgoing": [link_payload(link) for link in outgoing],
            },
            superseded=superseded,
        )

    def list_tasks(
        self,
        status: str | None = None,
        workspace_id: str | None = None,
        limit: int = 20,
    ) -> TaskListPayload:
        tasks = self._task_queue.list_tasks(
            status=status,
            workspace_id=workspace_id,
            limit=limit,
        )
        return TaskListPayload(tasks=[task_payload(task) for task in tasks])

    def list_memories(
        self,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> MemoryListPayload:
        if self._memory_queries is None:
            return MemoryListPayload()
        records = self._memory_queries.list_memories(
            workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            limit=limit,
        )
        return MemoryListPayload(
            records=[compact_memory_record_payload(record) for record in records]
        )

    def load_dashboard_html(self):
        return self._dashboard_static_path.read_text(encoding="utf-8")
