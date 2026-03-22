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
    assert "internal_heartbeat_work_item" in tool_names
    assert "internal_complete_work_item" in tool_names
    assert "internal_defer_work_item" in tool_names
    assert "internal_release_work_item" in tool_names


def test_internal_work_item_lifecycle_tools_heartbeat_complete_and_reclaim_expired_leases(db_manager) -> None:
    ctx = _build_ctx(db_manager)
    assert ctx.work_items is not None
    created, is_new = ctx.work_items.enqueue_unique(
        family_key=WORK_FAMILY_MEMORY_TAGGING,
        execution_lane=EXECUTION_LANE_AGENTIC,
        workspace_id="workspace-a",
        idempotency_key="memory_tagging:lifecycle",
        payload={"memory_id": "memory-lifecycle"},
    )
    assert is_new is True

    first_claim = internal_tool_services()["internal_get_work_batch"](
        ctx,
        {
            "task_id": "taxonomist-first",
            "family_key": WORK_FAMILY_MEMORY_TAGGING,
            "execution_lane": EXECUTION_LANE_AGENTIC,
            "limit": 1,
            "lease_ttl_seconds": 10,
        },
    )
    claimed_record = first_claim["records"][0]
    first_expiry = claimed_record["lease_expires_at"]
    assert claimed_record["id"] == created.id

    heartbeat_payload = internal_tool_services()["internal_heartbeat_work_item"](
        ctx,
        {
            "task_id": "taxonomist-first",
            "work_item_id": created.id,
            "lease_ttl_seconds": 20,
        },
    )
    assert heartbeat_payload["status"] == "ok"
    assert heartbeat_payload["record"]["lease_expires_at"] > first_expiry

    reclaimed = ctx.work_items.claim_batch(
        family_key=WORK_FAMILY_MEMORY_TAGGING,
        execution_lane=EXECUTION_LANE_AGENTIC,
        lease_owner="taxonomist-second",
        limit=1,
        workspace_id="workspace-a",
        now=heartbeat_payload["record"]["lease_expires_at"] + 1,
    )
    assert [record.id for record in reclaimed] == [created.id]
    assert reclaimed[0].lease_owner == "taxonomist-second"
    assert reclaimed[0].attempt_count == 2

    complete_payload = internal_tool_services()["internal_complete_work_item"](
        ctx,
        {"work_item_id": created.id},
    )
    assert complete_payload["status"] == "ok"
    assert complete_payload["record"]["id"] == created.id
    assert complete_payload["record"]["status"] == "completed"
    assert complete_payload["record"]["completed_at"] is not None
    assert complete_payload["record"]["last_error"] is None


def test_internal_work_item_lifecycle_tools_defer_and_release(db_manager) -> None:
    ctx = _build_ctx(db_manager)
    assert ctx.work_items is not None
    first, first_created = ctx.work_items.enqueue_unique(
        family_key=WORK_FAMILY_MEMORY_TAGGING,
        execution_lane=EXECUTION_LANE_AGENTIC,
        workspace_id="workspace-a",
        idempotency_key="memory_tagging:defer",
        payload={"memory_id": "memory-defer"},
    )
    second, second_created = ctx.work_items.enqueue_unique(
        family_key=WORK_FAMILY_MEMORY_TAGGING,
        execution_lane=EXECUTION_LANE_AGENTIC,
        workspace_id="workspace-a",
        idempotency_key="memory_tagging:release",
        payload={"memory_id": "memory-release"},
    )
    assert first_created is True and second_created is True

    claimed = internal_tool_services()["internal_get_work_batch"](
        ctx,
        {
            "task_id": "taxonomist-batch",
            "family_key": WORK_FAMILY_MEMORY_TAGGING,
            "execution_lane": EXECUTION_LANE_AGENTIC,
            "limit": 2,
        },
    )
    assert [record["id"] for record in claimed["records"]] == [first.id, second.id]

    deferred = internal_tool_services()["internal_defer_work_item"](
        ctx,
        {
            "work_item_id": first.id,
            "error": "provider_rate_limited",
            "retry_delay_seconds": 30,
        },
    )
    released = internal_tool_services()["internal_release_work_item"](
        ctx,
        {"work_item_id": second.id},
    )

    assert deferred["status"] == "ok"
    assert deferred["record"]["id"] == first.id
    assert deferred["record"]["status"] == "deferred"
    assert deferred["record"]["last_error"] == "provider_rate_limited"

    assert released == {
        "status": "ok",
        "record": {
            "id": second.id,
            "status": "pending",
            "lease_owner": None,
            "lease_expires_at": None,
        },
    }
