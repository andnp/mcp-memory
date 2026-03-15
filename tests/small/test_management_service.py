from types import SimpleNamespace
import logging

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.tasks import SQLiteTaskQueue
from mcp_memory.management.service import ManagementService
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.runtime_logging import SQLiteStructuredLogHandler


pytestmark = pytest.mark.small


def test_management_service_overview_and_memory_detail(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    handler = SQLiteStructuredLogHandler(
        db_manager=db_manager,
        workspace_id="workspace-a",
        source="daemon",
    )

    primary = repository.create_memory(
        title="Dashboard plan",
        content="Expose overview metrics and a memory detail view.",
        workspace_ids=["workspace-a"],
        memory_type="plan",
        tags=["dashboard", "api"],
        status="active",
    )
    secondary = repository.create_memory(
        title="Dashboard legacy plan",
        content="An older plan that has been replaced.",
        workspace_ids=["workspace-a"],
        memory_type="plan",
        tags=["dashboard"],
        status="stale",
    )
    assert primary is not None and secondary is not None

    repository.add_link(primary.id, secondary.id, "SUPERSEDES", "Replaced during epic 6")

    task = task_queue.enqueue(
        "summarize-memory",
        task_id="task-1",
        workspace_id="workspace-a",
        available_at=0.0,
    )
    assert task_queue.claim_next(now=10.0) is not None
    task_queue.fail_permanently(task.id, "summary provider offline", failed_at=11.0)
    handler.emit(
        logging.LogRecord(
            name="mcp_memory.tests",
            level=logging.WARNING,
            pathname=__file__,
            lineno=42,
            msg="daemon log row",
            args=(),
            exc_info=None,
        )
    )

    ctx = ApplicationContext(
        workspace_id="workspace-a",
        memory_path=db_manager.db_path.parent,
        db_manager=db_manager,
        repository=repository,
        task_queue=task_queue,
    )
    controller = SimpleNamespace(has_runtime=True, client_count=1)
    service = ManagementService(ctx, controller)

    overview = service.get_overview()
    detail = service.get_memory_detail(primary.id)
    health = service.get_health()
    filtered_logs = service.list_logs(query="daemon", source="daemon")
    summary = service.summarize_logs(source="daemon")

    assert overview.memories.total == 2
    assert overview.memories.by_status == {"active": 1, "stale": 1}
    assert overview.memory_metrics.total_memories == 2
    assert overview.memory_metrics.thought_buffer_entries == 0
    assert overview.agent_runs[0].task_name == "ingest-system1"
    assert overview.tasks.failed_count == 1
    assert overview.failed_tasks[0]["last_error"] == "summary provider offline"
    assert overview.recent_logs[0].message == "daemon log row"
    assert overview.recent_logs[0].source == "daemon"
    assert filtered_logs.logs[0].logger_name == "mcp_memory.tests"
    assert summary.total == 1
    assert summary.by_level == {"WARNING": 1}
    assert summary.by_source == {"daemon": 1}
    assert detail.record["id"] == primary.id
    assert detail.relationships["outgoing"][0]["link_type"] == "SUPERSEDES"
    assert detail.superseded[0]["id"] == secondary.id
    assert overview.failed_tasks[0]["status"] == "failed"
    assert health.runtime_active is True
    assert health.workspace_id == "workspace-a"
