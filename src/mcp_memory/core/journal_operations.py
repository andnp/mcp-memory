from __future__ import annotations

from mcp_memory.core.system1_scheduling import schedule_system1_ingest
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
            scheduled = schedule_system1_ingest(self._task_queue, self._journal, self._workspace_id)
            if scheduled is not None and scheduled.trigger == "system1_threshold":
                ingest_task = scheduled.task
                created = scheduled.created
                payload["ingest_task"] = {
                    "id": ingest_task.id,
                    "status": ingest_task.status,
                    "task_name": ingest_task.task_name,
                    "workspace_id": ingest_task.workspace_id,
                    "created": created,
                }

        return payload