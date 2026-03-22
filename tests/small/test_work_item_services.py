from __future__ import annotations

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.mcp.internal_tools import get_internal_maintenance_tools
from mcp_memory.mcp.transport import internal_tool_services
from mcp_memory.work_item_store import (
    EXECUTION_LANE_AGENTIC,
    SQLiteWorkItemRepository,
    WORK_FAMILY_MEMORY_TAGGING,
)


pytestmark = pytest.mark.small


def _build_ctx(db_manager) -> ApplicationContext:
    return ApplicationContext(
        workspace_id="workspace-a",
        db_manager=db_manager,
        work_items=SQLiteWorkItemRepository(db_manager),
    )


def test_internal_get_work_batch_claims_family_specific_work_items(db_manager) -> None:
    ctx = _build_ctx(db_manager)
    assert ctx.work_items is not None
    first, created_first = ctx.work_items.enqueue_unique(
        family_key=WORK_FAMILY_MEMORY_TAGGING,
        execution_lane=EXECUTION_LANE_AGENTIC,
        workspace_id="workspace-a",
        idempotency_key="memory_tagging:one",
        payload={"memory_id": "memory-one"},
    )
    second, created_second = ctx.work_items.enqueue_unique(
        family_key=WORK_FAMILY_MEMORY_TAGGING,
        execution_lane=EXECUTION_LANE_AGENTIC,
        workspace_id="workspace-a",
        idempotency_key="memory_tagging:two",
        payload={"memory_id": "memory-two"},
    )
    assert created_first is True and created_second is True

    payload = internal_tool_services()["internal_get_work_batch"](
        ctx,
        {
            "task_id": "taxonomist-work-batch",
            "family_key": WORK_FAMILY_MEMORY_TAGGING,
            "execution_lane": EXECUTION_LANE_AGENTIC,
            "limit": 2,
        },
    )

    assert payload["status"] == "ok"
    assert [record["id"] for record in payload["records"]] == [first.id, second.id]
    assert [record["payload"] for record in payload["records"]] == [
        {"memory_id": "memory-one"},
        {"memory_id": "memory-two"},
    ]
    running_items = ctx.work_items.list_items(status="running", family_key=WORK_FAMILY_MEMORY_TAGGING)
    assert [record.lease_owner for record in running_items] == ["taxonomist-work-batch", "taxonomist-work-batch"]


def test_internal_get_work_batch_tool_is_exposed_for_maintenance_agents() -> None:
    tool_names = {tool.name for tool in get_internal_maintenance_tools()}
    assert "internal_get_work_batch" in tool_names
