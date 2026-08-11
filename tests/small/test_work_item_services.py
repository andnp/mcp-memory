from __future__ import annotations

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.ports.work_items import EXECUTION_LANE_AGENTIC, WORK_FAMILY_MEMORY_TAGGING
from mcp_memory.core.task_handlers import CURATOR_TASK_NAME, TAXONOMIST_TASK_NAME
from mcp_memory.core.tasks import SQLiteTaskQueue
from mcp_memory.mcp.internal_tools import get_internal_maintenance_tools
from mcp_memory.mcp.transport import internal_tool_services
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.storage.sqlite_work_item_store import SQLiteWorkItemRepository
from tests.small.work_item_repository_contract import (
    assert_claim_batch_orders_ready_items,
    assert_enqueue_unique_deduplicates_idempotency_keys,
    assert_heartbeat_extends_leases_and_allows_expired_reclaim,
    assert_release_defer_and_complete_items,
)


pytestmark = pytest.mark.small


def test_sqlite_work_item_repository_deduplicates_idempotency_keys(db_manager) -> None:
    assert_enqueue_unique_deduplicates_idempotency_keys(
        lambda: SQLiteWorkItemRepository(db_manager)
    )


def test_sqlite_work_item_repository_claim_batch_orders_ready_items(db_manager) -> None:
    assert_claim_batch_orders_ready_items(
        lambda: SQLiteWorkItemRepository(db_manager)
    )


def test_sqlite_work_item_repository_heartbeat_and_reclaim_share_backend_contract(db_manager) -> None:
    assert_heartbeat_extends_leases_and_allows_expired_reclaim(
        lambda: SQLiteWorkItemRepository(db_manager)
    )


def test_sqlite_work_item_repository_release_defer_and_complete_share_backend_contract(db_manager) -> None:
    assert_release_defer_and_complete_items(
        lambda: SQLiteWorkItemRepository(db_manager)
    )


def _build_ctx(db_manager) -> ApplicationContext:
    return ApplicationContext(
        workspace_id="workspace-a",
        db_manager=db_manager,
        repository=RelationalMemoryRepository(db_manager),
        task_queue=SQLiteTaskQueue(db_manager),
        work_items=SQLiteWorkItemRepository(db_manager),
    )


def _start_running_task(ctx: ApplicationContext, *, task_name: str, task_id: str, workspace_id: str | None) -> None:
    assert ctx.task_queue is not None
    ctx.task_queue.enqueue(
        task_name,
        workspace_id=workspace_id,
        task_id=task_id,
        available_at=0.0,
        data={} if workspace_id is None else {"workspace_id": workspace_id},
    )
    claimed = ctx.task_queue.claim_next(now=0.0)
    assert claimed is not None
    assert claimed.id == task_id


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
            "workspace_id": "workspace-a",
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


def test_internal_get_work_batch_uses_running_task_scope_for_global_tasks(db_manager) -> None:
    ctx = _build_ctx(db_manager)
    assert ctx.work_items is not None
    _start_running_task(
        ctx,
        task_name=TAXONOMIST_TASK_NAME,
        task_id="taxonomist-global-task",
        workspace_id=None,
    )
    global_item, created_global = ctx.work_items.enqueue_unique(
        family_key=WORK_FAMILY_MEMORY_TAGGING,
        execution_lane=EXECUTION_LANE_AGENTIC,
        workspace_id=None,
        idempotency_key="memory_tagging:global",
        payload={"memory_id": "memory-global"},
    )
    _, created_local = ctx.work_items.enqueue_unique(
        family_key=WORK_FAMILY_MEMORY_TAGGING,
        execution_lane=EXECUTION_LANE_AGENTIC,
        workspace_id="workspace-a",
        idempotency_key="memory_tagging:local",
        payload={"memory_id": "memory-local"},
    )
    assert created_global is True and created_local is True

    payload = internal_tool_services()["internal_get_work_batch"](
        ctx,
        {
            "task_id": "taxonomist-global-task",
            "family_key": WORK_FAMILY_MEMORY_TAGGING,
            "execution_lane": EXECUTION_LANE_AGENTIC,
            "limit": 2,
        },
    )

    assert payload["status"] == "ok"
    assert [record["id"] for record in payload["records"]] == [global_item.id]
    assert payload["records"][0]["workspace_id"] is None


def test_internal_get_work_batch_treats_global_workspace_argument_as_global_scope(db_manager) -> None:
    ctx = _build_ctx(db_manager)
    assert ctx.work_items is not None
    global_item, created = ctx.work_items.enqueue_unique(
        family_key=WORK_FAMILY_MEMORY_TAGGING,
        execution_lane=EXECUTION_LANE_AGENTIC,
        workspace_id=None,
        idempotency_key="memory_tagging:legacy-global",
        payload={"memory_id": "memory-global"},
    )
    assert created is True

    payload = internal_tool_services()["internal_get_work_batch"](
        ctx,
        {
            "task_id": "taxonomist-global-argument",
            "family_key": WORK_FAMILY_MEMORY_TAGGING,
            "execution_lane": EXECUTION_LANE_AGENTIC,
            "workspace_id": "global",
            "limit": 1,
        },
    )

    assert payload["status"] == "ok"
    assert [record["id"] for record in payload["records"]] == [global_item.id]


def test_internal_list_memory_records_defaults_to_global_listing(db_manager) -> None:
    ctx = _build_ctx(db_manager)
    assert ctx.repository is not None
    remote = ctx.repository.create_memory(
        title="Remote workspace memory",
        content="Important shared detail.",
        memory_type="fact",
        workspace_ids=["workspace-b"],
        tags=["shared"],
    )
    assert remote is not None

    payload = internal_tool_services()["internal_list_memory_records"](
        ctx,
        {"limit": 10, "status": "active"},
    )

    assert payload["status"] == "ok"
    assert remote.id in [record["id"] for record in payload["records"]]


def test_internal_get_next_curator_batch_uses_running_task_scope_for_global_tasks(db_manager) -> None:
    ctx = _build_ctx(db_manager)
    assert ctx.repository is not None
    _start_running_task(
        ctx,
        task_name=CURATOR_TASK_NAME,
        task_id="curator-global-task",
        workspace_id=None,
    )
    remote = ctx.repository.create_memory(
        title="Remote workspace memory",
        content="Shared structural cleanup candidate.",
        memory_type="fact",
        workspace_ids=["workspace-b"],
        tags=["shared"],
    )
    assert remote is not None

    payload = internal_tool_services()["internal_get_next_curator_batch"](
        ctx,
        {
            "task_id": "curator-global-task",
            "exclude_memory_ids": [],
            "limit": 4,
        },
    )

    assert payload["status"] == "ok"
    assert remote.id in [record["id"] for record in payload["records"]]


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
            "workspace_id": "workspace-a",
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
            "workspace_id": "workspace-a",
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
