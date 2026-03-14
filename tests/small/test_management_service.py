from types import SimpleNamespace

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.tasks import SQLiteTaskQueue
from mcp_memory.management.service import ManagementService
from mcp_memory.relational.repository import RelationalMemoryRepository


pytestmark = pytest.mark.small


def test_management_service_overview_and_memory_detail(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)

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
    tasks = service.list_tasks(status="failed", limit=5)
    health = service.get_health()

    assert overview.memories.total == 2
    assert overview.memories.by_status == {"active": 1, "stale": 1}
    assert overview.tasks.failed_count == 1
    assert overview.failed_tasks[0]["last_error"] == "summary provider offline"
    assert detail.record["id"] == primary.id
    assert detail.relationships["outgoing"][0]["link_type"] == "SUPERSEDES"
    assert detail.superseded[0]["id"] == secondary.id
    assert tasks.tasks[0]["status"] == "failed"
    assert health.runtime_active is True
    assert health.workspace_id == "workspace-a"
