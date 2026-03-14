from __future__ import annotations

from mcp_memory.core.agent_runtime import (
    SYSTEM1_INGEST_TASK_NAME,
    SYSTEM1_INGEST_THRESHOLD,
)
from mcp_memory.core.journal import System1Journal
from mcp_memory.core.tasks import SQLiteTaskQueue


class RecordThoughtOperation:
    def __init__(
        self,
        journal: System1Journal,
        task_queue: SQLiteTaskQueue | None,
        workspace_id: str | None,
    ) -> None:
        self._journal = journal
        self._task_queue = task_queue
        self._workspace_id = workspace_id

    def execute(self, content: str) -> dict:
        entry = self._journal.record(content, workspace_id=self._workspace_id)
        payload = {
            "status": "recorded",
            "entry": entry.to_dict(),
        }

        if self._task_queue is not None:
            pending_count = self._journal.count_by_status().get("pending", 0)
            if pending_count >= SYSTEM1_INGEST_THRESHOLD:
                ingest_task, created = self._task_queue.enqueue_unique(
                    task_name=SYSTEM1_INGEST_TASK_NAME,
                    workspace_id=self._workspace_id,
                    data={
                        "workspace_id": self._workspace_id,
                        "trigger": "system1_threshold",
                        "pending_count": pending_count,
                    },
                )
                payload["ingest_task"] = {
                    "id": ingest_task.id,
                    "status": ingest_task.status,
                    "task_name": ingest_task.task_name,
                    "workspace_id": ingest_task.workspace_id,
                    "created": created,
                }

        return payload


class GetPendingThoughtsOperation:
    def __init__(self, journal: System1Journal) -> None:
        self._journal = journal

    def execute(self, limit: int) -> dict:
        return {
            "status": "ok",
            "entries": [entry.to_dict() for entry in self._journal.get_pending(limit=limit)],
        }