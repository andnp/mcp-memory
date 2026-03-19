from __future__ import annotations

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.journal import System1Journal
from mcp_memory.core.task_handlers import SUMMARIZE_MEMORY_PRIORITY, SUMMARIZE_MEMORY_TASK_NAME, SYSTEM1_INGEST_TASK_NAME
from mcp_memory.core.tasks import SQLiteTaskQueue
from mcp_memory.mcp.internal_ingest_services import (
    INGEST_ENTRY_DISPOSITIONS_TASK_DATA_KEY,
    INGEST_HANDLED_ENTRY_IDS_TASK_DATA_KEY,
    INGEST_TOOL_INVOCATIONS_TASK_DATA_KEY,
)
from mcp_memory.mcp.internal_tools import get_internal_maintenance_tools
from mcp_memory.mcp.transport import internal_tool_services
from mcp_memory.relational.repository import RelationalMemoryRepository


pytestmark = pytest.mark.small


def _build_ctx(db_manager) -> ApplicationContext:
    return ApplicationContext(
        workspace_id="workspace-a",
        db_manager=db_manager,
        journal=System1Journal(db_manager),
        repository=RelationalMemoryRepository(db_manager),
        task_queue=SQLiteTaskQueue(db_manager),
    )


def _start_running_ingest_task(ctx: ApplicationContext, task_id: str):
    assert ctx.task_queue is not None
    ctx.task_queue.enqueue(
        SYSTEM1_INGEST_TASK_NAME,
        workspace_id=ctx.workspace_id,
        available_at=0.0,
        task_id=task_id,
    )
    task = ctx.task_queue.claim_next(now=0.0)
    assert task is not None
    assert task.id == task_id
    return task


@pytest.mark.parametrize(
    ("entry_ids", "detail"),
    [
        ([], "entry_ids must contain at least one entry id"),
        ([0], "entry_ids must contain only positive integer ids"),
        (["abc"], "entry_ids must contain only positive integer ids"),
        ([True], "entry_ids must contain only positive integer ids"),
    ],
)
def test_internal_ingest_create_rejects_malformed_entry_ids(db_manager, entry_ids, detail: str) -> None:
    ctx = _build_ctx(db_manager)
    _start_running_ingest_task(ctx, "ingest-invalid-payload")

    payload = internal_tool_services()["internal_ingest_create_memory"](
        ctx,
        {
            "task_id": "ingest-invalid-payload",
            "entry_ids": entry_ids,
            "title": "Invalid payload",
            "content": "Some content",
        },
    )

    assert payload == {
        "status": "error",
        "error": "invalid_ingest_payload",
        "detail": detail,
    }


def test_internal_ingest_create_normalizes_mixed_duplicate_entry_ids_and_records_side_effects(db_manager) -> None:
    ctx = _build_ctx(db_manager)
    _start_running_ingest_task(ctx, "ingest-create-task")
    assert ctx.task_queue is not None
    assert ctx.repository is not None

    payload = internal_tool_services()["internal_ingest_create_memory"](
        ctx,
        {
            "task_id": "ingest-create-task",
            "entry_ids": [11, "11", " 12 ", 12, "013"],
            "title": "Scenario-rich ingest coverage",
            "content": "Keep the ingest create contract behavior-preserving.",
            "workspace_ids": ["workspace-a"],
            "tags": ["Testing", "testing", "ingest_flow"],
            "metadata": {"source": "unit-test"},
        },
    )

    assert payload["status"] == "ok"
    assert payload["handled_entry_ids"] == [11, 12, 13]
    record = ctx.repository.get_memory(payload["record"]["id"])
    assert record is not None
    assert record.metadata == {
        "source": "unit-test",
        "source_entry_ids": [11, 12, 13],
        "ingest_task_id": "ingest-create-task",
    }
    assert record.tags == ["auto-ingested", "ingest-flow", "system1", "testing"]

    task = ctx.task_queue.get_task("ingest-create-task")
    assert task.data[INGEST_HANDLED_ENTRY_IDS_TASK_DATA_KEY] == [11, 12, 13]
    assert task.data[INGEST_ENTRY_DISPOSITIONS_TASK_DATA_KEY] == [
        {
            "entry_id": 11,
            "disposition": "created",
            "memory_id": record.id,
            "memory_title": record.title,
        },
        {
            "entry_id": 12,
            "disposition": "created",
            "memory_id": record.id,
            "memory_title": record.title,
        },
        {
            "entry_id": 13,
            "disposition": "created",
            "memory_id": record.id,
            "memory_title": record.title,
        },
    ]
    assert task.data[INGEST_TOOL_INVOCATIONS_TASK_DATA_KEY] == [
        {
            "tool_name": "internal_ingest_create_memory",
            "mutation": True,
        }
    ]

    summary_task = ctx.task_queue.find_open_task(SUMMARIZE_MEMORY_TASK_NAME, ctx.workspace_id)
    assert summary_task is not None
    assert summary_task.data == {"memory_id": record.id}
    assert summary_task.priority == SUMMARIZE_MEMORY_PRIORITY


def test_internal_ingest_append_alias_preserves_side_effects_and_target_errors(db_manager) -> None:
    ctx = _build_ctx(db_manager)
    _start_running_ingest_task(ctx, "ingest-append-task")
    assert ctx.repository is not None
    assert ctx.task_queue is not None

    missing = internal_tool_services()["internal_ingest_append_memory"](
        ctx,
        {
            "memory_id": "missing-memory",
            "content": "Append content",
            "task_id": "ingest-append-task",
            "entry_ids": [21],
        },
    )
    assert missing == {"status": "error", "error": "memory_not_found"}

    archived = ctx.repository.create_memory(
        title="Archived target",
        content="Old content",
        workspace_ids=[ctx.workspace_id or "workspace-a"],
        memory_type="fact",
        status="archived",
    )
    assert archived is not None
    inactive = internal_tool_services()["internal_ingest_append_memory"](
        ctx,
        {
            "memory_id": archived.id,
            "content": "Append content",
            "task_id": "ingest-append-task",
            "entry_ids": [21],
        },
    )
    assert inactive == {"status": "error", "error": "memory_not_active"}

    target = ctx.repository.create_memory(
        title="Active target",
        content="Canonical memory body.",
        workspace_ids=[ctx.workspace_id or "workspace-a"],
        memory_type="fact",
        tags=["existing"],
        metadata={"appended_entry_ids": [20]},
    )
    assert target is not None

    payload = internal_tool_services()["internal_append_to_existing_memory_for_ingest"](
        ctx,
        {
            "memory_id": target.id,
            "content": "  Newly appended ingest detail.  ",
            "task_id": "ingest-append-task",
            "entry_ids": [21, "21", "022"],
            "workspace_ids": ["workspace-b"],
            "tags": ["Existing", "new_tag"],
            "metadata": {"note": "kept"},
        },
    )

    assert payload["status"] == "ok"
    assert payload["handled_entry_ids"] == [21, 22]

    updated = ctx.repository.get_memory(target.id)
    assert updated is not None
    assert updated.content.endswith("Newly appended ingest detail.")
    assert updated.metadata == {
        "appended_entry_ids": [20, 21, 22],
        "ingest_task_id": "ingest-append-task",
        "note": "kept",
    }
    assert updated.tags == ["existing", "new-tag", "system1-appended"]
    assert updated.workspace_ids == ["workspace-a", "workspace-b"]

    task = ctx.task_queue.get_task("ingest-append-task")
    assert task.data[INGEST_HANDLED_ENTRY_IDS_TASK_DATA_KEY] == [21, 22]
    assert task.data[INGEST_ENTRY_DISPOSITIONS_TASK_DATA_KEY] == [
        {
            "entry_id": 21,
            "disposition": "appended",
            "memory_id": target.id,
            "memory_title": target.title,
        },
        {
            "entry_id": 22,
            "disposition": "appended",
            "memory_id": target.id,
            "memory_title": target.title,
        },
    ]
    assert task.data[INGEST_TOOL_INVOCATIONS_TASK_DATA_KEY] == [
        {
            "tool_name": "internal_ingest_append_memory",
            "mutation": True,
        }
    ]


def test_internal_get_next_ingest_batch_keeps_payload_shape_and_non_mutating_tool_tracking(db_manager) -> None:
    ctx = _build_ctx(db_manager)
    _start_running_ingest_task(ctx, "ingest-batch-task")
    assert ctx.journal is not None
    assert ctx.task_queue is not None

    first = ctx.journal.record("first pending thought", workspace_id=ctx.workspace_id)
    second = ctx.journal.record("second pending thought", workspace_id=ctx.workspace_id)

    payload = internal_tool_services()["internal_get_next_ingest_batch"](
        ctx,
        {
            "task_id": "ingest-batch-task",
            "batch_size": 10,
            "grouping_strategy": "fifo",
        },
    )

    assert payload["status"] == "ok"
    assert payload["task_id"] == "ingest-batch-task"
    assert payload["requested_grouping_strategy"] == "fifo"
    assert payload["grouping_strategy_used"] == "fifo"
    assert payload["grouping_fallback_reason"] is None
    assert payload["claimed_entry_ids"] == [first.id, second.id]
    assert payload["pending_remaining"] == 0
    assert payload["has_more"] is False
    assert payload["group_count"] == len(payload["groups"])
    assert payload["groups"] == [
        {
            "group_index": 0,
            "entries": [
                {
                    "id": first.id,
                    "content": "first pending thought",
                    "workspace_id": ctx.workspace_id,
                    "timestamp": first.timestamp,
                    "status": "claimed",
                },
                {
                    "id": second.id,
                    "content": "second pending thought",
                    "workspace_id": ctx.workspace_id,
                    "timestamp": second.timestamp,
                    "status": "claimed",
                },
            ],
        }
    ]

    task = ctx.task_queue.get_task("ingest-batch-task")
    assert task.data[INGEST_TOOL_INVOCATIONS_TASK_DATA_KEY] == [
        {
            "tool_name": "internal_get_next_ingest_batch",
            "mutation": False,
        }
    ]


def test_internal_tool_registry_keeps_ingest_names_and_aliases_stable() -> None:
    services = internal_tool_services()
    tool_names = {tool.name for tool in get_internal_maintenance_tools()}

    assert "internal_get_next_ingest_batch" in tool_names
    assert "internal_ingest_append_memory" in tool_names
    assert "internal_ingest_create_memory" in tool_names
    assert "internal_ingest_append_memory" in services
    assert "internal_ingest_create_memory" in services
    assert "internal_append_to_existing_memory_for_ingest" in services
    assert "internal_create_memory_record_for_ingest" in services
    assert services["internal_ingest_append_memory"] is services["internal_append_to_existing_memory_for_ingest"]
    assert services["internal_ingest_create_memory"] is services["internal_create_memory_record_for_ingest"]