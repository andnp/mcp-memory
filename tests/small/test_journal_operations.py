from __future__ import annotations

from pathlib import Path

import pytest

from mcp_memory.core import journal_operations as journal_operations_module
from mcp_memory.core.journal import System1Journal
from mcp_memory.core.tasks import SQLiteTaskQueue
from mcp_memory.storage.shared_read_cache import SharedReadCache


pytestmark = pytest.mark.small


def test_flush_record_thought_writeback_outbox_resumes_maintenance_and_schedules_ingest_for_flushed_workspaces(
    db_manager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = System1Journal(db_manager)
    queue = SQLiteTaskQueue(db_manager)
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")

    paused_task = queue.enqueue(
        "memory-curator",
        priority=95,
        available_at=0.0,
        task_id="paused-memory-curator",
    )
    assert queue.claim_next(now=1.0) is not None
    queue.complete(
        paused_task.id,
        completed_at=2.0,
        run_result={
            "paused_for_idle": True,
            "interval_seconds": 3600.0,
            "task_priority": 95,
        },
    )

    assert cache.enqueue_record_thought_outbox_entry(
        content="queued workspace a thought",
        workspace_id="workspace-a",
        timestamp=10.0,
        max_entries=10,
    ) is not None
    assert cache.enqueue_record_thought_outbox_entry(
        content="queued workspace b thought",
        workspace_id="workspace-b",
        timestamp=11.0,
        max_entries=10,
    ) is not None

    scheduled_workspace_ids: list[str | None] = []
    original_schedule_system1_ingest = journal_operations_module.schedule_system1_ingest

    def _tracking_schedule_system1_ingest(task_queue, scheduled_journal, workspace_id, *, suppression_config=None):
        scheduled_workspace_ids.append(workspace_id)
        return original_schedule_system1_ingest(
            task_queue,
            scheduled_journal,
            workspace_id,
            suppression_config=suppression_config,
        )

    monkeypatch.setattr(
        journal_operations_module,
        "schedule_system1_ingest",
        _tracking_schedule_system1_ingest,
    )

    result = journal_operations_module.flush_record_thought_writeback_outbox(
        journal,
        task_queue=queue,
        suppression_config=None,
        writeback_cache=cache,
    )

    resumed_task = queue.find_open_task("memory-curator", None)
    ingest_task = queue.find_open_task_any_workspace("ingest-system1")
    pending_entries = journal.get_pending(limit=10)

    assert result.flushed_count == 2
    assert result.flushed_workspace_ids == ("workspace-a", "workspace-b")
    assert scheduled_workspace_ids == ["workspace-a", "workspace-b"]
    assert cache.count_record_thought_outbox_entries() == 0
    assert {entry.content for entry in pending_entries} == {
        "queued workspace a thought",
        "queued workspace b thought",
    }
    assert resumed_task is not None
    assert resumed_task.status == "pending"
    assert resumed_task.data["trigger"] == "recurring_resume"
    assert ingest_task is not None
    assert ingest_task.status == "pending"
    assert ingest_task.data["trigger"].startswith("system1_debounce")
    assert ingest_task.data["journal_workspace_id"] == "*"