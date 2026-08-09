from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.ingest_claim_lifecycle import (
    _record_ingest_tool_invocation,
    _record_successful_ingest_entry_dispositions,
    _record_successful_ingest_entry_ids,
    _record_touched_memory_ids,
    _recorded_ingest_run_metadata,
    _reset_recorded_ingest_handled_entry_ids,
)
from mcp_memory.core.journal import System1Journal
from mcp_memory.core.task_submission import enqueue_summary_refresh_task
from mcp_memory.core.task_handlers import SUMMARIZE_MEMORY_PRIORITY, SUMMARIZE_MEMORY_TASK_NAME, SYSTEM1_INGEST_TASK_NAME
from mcp_memory.core.tasks import SQLiteTaskQueue
from mcp_memory.mcp.internal_service_support import _enqueue_summary_task
from mcp_memory.mcp.internal_ingest_services import (
    INGEST_ENTRY_DISPOSITIONS_TASK_DATA_KEY,
    INGEST_HANDLED_ENTRY_IDS_TASK_DATA_KEY,
    INGEST_TOOL_INVOCATIONS_TASK_DATA_KEY,
)
from mcp_memory.mcp.internal_tools import get_internal_maintenance_tools
from mcp_memory.mcp.transport import internal_tool_services
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.storage.ingress_evidence_store import SQLiteIngressActionReceiptStore
from mcp_memory.storage.ingress_mutation_transaction import SQLiteIngressMutationStore


pytestmark = pytest.mark.small


@dataclass
class _TaskQueueOnlyContext:
    task_queue: SQLiteTaskQueue | None


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


def _start_running_ingest_task_on_queue(queue: SQLiteTaskQueue, task_id: str, *, workspace_id: str = "workspace-a"):
    queue.enqueue(
        SYSTEM1_INGEST_TASK_NAME,
        workspace_id=workspace_id,
        available_at=0.0,
        task_id=task_id,
    )
    task = queue.claim_next(now=0.0)
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
        "created_via_ingest": True,
        "source": "unit-test",
        "source_entry_ids": [11, 12, 13],
        "ingest_task_id": "ingest-create-task",
    }
    assert record.tags == ["ingest-flow", "testing"]

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
    assert summary_task is None
    assert record.summary


def test_internal_ingest_create_persists_shadow_action_receipt(db_manager) -> None:
    """Shadow mode returns stable action metadata after a successful create."""
    ctx = _build_ctx(db_manager)
    cast(Any, ctx).config = SimpleNamespace(ingress_evidence_mode="shadow")
    ctx.ingress_action_receipts = SQLiteIngressActionReceiptStore(db_manager)
    _start_running_ingest_task(ctx, "ingest-receipt-task")

    payload = internal_tool_services()["internal_ingest_create_memory"](
        ctx,
        {
            "task_id": "ingest-receipt-task",
            "batch_id": "batch-stable",
            "entry_ids": [31, 32],
            "title": "Receipt-backed ingest",
            "content": "The successful mutation emits stable ingress receipt metadata.",
        },
    )

    assert payload["ingress_action_id"]
    assert payload["ingress_payload_digest"]
    receipt = ctx.ingress_action_receipts.get(payload["ingress_action_id"])
    assert receipt is not None
    assert receipt.batch_id == "batch-stable"
    assert receipt.status.value == "applied_unverified"
    assert receipt.canonical_payload_digest == payload["ingress_payload_digest"]


def test_internal_ingest_create_enforce_rejects_before_domain_write(db_manager) -> None:
    """Enforce mode rejects before creating a memory without atomic coordination."""
    ctx = _build_ctx(db_manager)
    cast(Any, ctx).config = SimpleNamespace(ingress_evidence_mode="enforce")
    ctx.ingress_action_receipts = SQLiteIngressActionReceiptStore(db_manager)
    _start_running_ingest_task(ctx, "ingest-enforce-task")
    assert ctx.repository is not None
    before = ctx.repository.list_memory_ids()

    payload = internal_tool_services()["internal_ingest_create_memory"](
        ctx,
        {
            "task_id": "ingest-enforce-task",
            "batch_id": "batch-enforce",
            "entry_ids": [41],
            "title": "Rejected enforce mutation",
            "content": "This must not be written before atomic coordination exists.",
        },
    )

    assert payload == {"status": "error", "error": "atomic_boundary_required"}
    assert ctx.repository.list_memory_ids() == before


def test_internal_ingest_create_enforce_uses_atomic_boundary_and_preserves_side_effects(db_manager) -> None:
    """Enforce create commits the memory and returns the atomic receipt metadata."""
    ctx = _build_ctx(db_manager)
    cast(Any, ctx).config = SimpleNamespace(ingress_evidence_mode="enforce")
    cast(Any, ctx).ingress_mutation_transaction = SQLiteIngressMutationStore(db_manager)
    _start_running_ingest_task(ctx, "ingest-atomic-create")

    payload = internal_tool_services()["internal_ingest_create_memory"](
        ctx,
        {
            "task_id": "ingest-atomic-create",
            "batch_id": "batch-atomic-create",
            "entry_ids": [61, 62],
            "title": "Atomic create",
            "content": "The service commits domain and ingress evidence together.",
            "tags": ["atomic"],
        },
    )

    assert payload["status"] == "ok"
    assert payload["handled_entry_ids"] == [61, 62]
    assert payload["ingress_action_id"]
    assert payload["ingress_payload_digest"]
    assert ctx.repository is not None
    assert ctx.repository.get_memory(payload["record"]["id"]) is not None
    connection = db_manager.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM ingress_source_coverage").fetchone()[0] == 2
    assert connection.execute("SELECT COUNT(*) FROM memory_mutation_events").fetchone()[0] == 1


def test_internal_ingest_append_enforce_uses_atomic_boundary(db_manager) -> None:
    """Enforce append uses the transaction domain surface for its mutation."""
    ctx = _build_ctx(db_manager)
    cast(Any, ctx).config = SimpleNamespace(ingress_evidence_mode="enforce")
    cast(Any, ctx).ingress_mutation_transaction = SQLiteIngressMutationStore(db_manager)
    _start_running_ingest_task(ctx, "ingest-atomic-append")
    assert ctx.repository is not None
    target = ctx.repository.create_memory(
        title="Atomic append target",
        content="Before",
        workspace_ids=[ctx.workspace_id or "workspace-a"],
        tags=["existing"],
    )
    assert target is not None

    payload = internal_tool_services()["internal_ingest_append_memory"](
        ctx,
        {
            "memory_id": target.id,
            "content": "After",
            "task_id": "ingest-atomic-append",
            "batch_id": "batch-atomic-append",
            "entry_ids": [63],
            "tags": ["new_tag"],
        },
    )

    assert payload["status"] == "ok"
    assert payload["handled_entry_ids"] == [63]
    updated = ctx.repository.get_memory(target.id)
    assert updated is not None
    assert updated.content == "Before\n\nAfter"
    assert updated.metadata["appended_via_ingest"] is True
    assert updated.tags == ["existing", "new-tag"]


class _CountingIngressMutationTransaction:
    def __init__(self, store: SQLiteIngressMutationStore) -> None:
        self.store = store
        self.callback_calls = 0

    def execute(self, **kwargs: Any) -> Any:
        apply = kwargs["apply"]

        def counted_apply(domain: Any) -> Any:
            self.callback_calls += 1
            return apply(domain)

        return self.store.execute(**{**kwargs, "apply": counted_apply})


def test_internal_ingest_enforce_exact_replay_skips_domain_callback(db_manager) -> None:
    """An exact enforce replay returns the same response without reapplying."""
    ctx = _build_ctx(db_manager)
    cast(Any, ctx).config = SimpleNamespace(ingress_evidence_mode="enforce")
    boundary = _CountingIngressMutationTransaction(SQLiteIngressMutationStore(db_manager))
    cast(Any, ctx).ingress_mutation_transaction = boundary
    _start_running_ingest_task(ctx, "ingest-atomic-replay")
    arguments = {
        "task_id": "ingest-atomic-replay",
        "batch_id": "batch-atomic-replay",
        "entry_ids": [64],
        "title": "Replayable atomic create",
        "content": "Only one domain callback should run.",
    }

    first = internal_tool_services()["internal_ingest_create_memory"](ctx, arguments)
    replay = internal_tool_services()["internal_ingest_create_memory"](ctx, arguments)

    assert first == replay
    assert boundary.callback_calls == 1


def test_internal_ingest_enforce_divergent_replay_fails_closed(db_manager) -> None:
    """A divergent enforce replay returns a collision without another memory."""
    ctx = _build_ctx(db_manager)
    cast(Any, ctx).config = SimpleNamespace(ingress_evidence_mode="enforce")
    boundary = _CountingIngressMutationTransaction(SQLiteIngressMutationStore(db_manager))
    cast(Any, ctx).ingress_mutation_transaction = boundary
    _start_running_ingest_task(ctx, "ingest-atomic-collision")
    base_arguments = {
        "task_id": "ingest-atomic-collision",
        "batch_id": "batch-atomic-collision",
        "entry_ids": [65],
        "title": "Collision original",
        "content": "Original payload.",
    }
    first = internal_tool_services()["internal_ingest_create_memory"](ctx, base_arguments)
    assert first["status"] == "ok"
    assert ctx.repository is not None
    before_ids = ctx.repository.list_memory_ids()

    collision = internal_tool_services()["internal_ingest_create_memory"](
        ctx,
        {**base_arguments, "content": "Divergent payload."},
    )

    assert collision["status"] == "error"
    assert collision["error"] == "ingress_action_identity_collision"
    assert ctx.repository.list_memory_ids() == before_ids
    assert boundary.callback_calls == 1


def test_internal_ingest_enforce_requires_batch_id_even_with_transaction_boundary(db_manager) -> None:
    """Enforce mode rejects a mutation without a batch identity."""
    ctx = _build_ctx(db_manager)
    cast(Any, ctx).config = SimpleNamespace(ingress_evidence_mode="enforce")
    boundary = _CountingIngressMutationTransaction(SQLiteIngressMutationStore(db_manager))
    cast(Any, ctx).ingress_mutation_transaction = boundary
    _start_running_ingest_task(ctx, "ingest-missing-batch")
    assert ctx.repository is not None
    before = ctx.repository.list_memory_ids()

    payload = internal_tool_services()["internal_ingest_create_memory"](
        ctx,
        {
            "task_id": "ingest-missing-batch",
            "entry_ids": [66],
            "title": "Missing batch",
            "content": "This must not mutate.",
        },
    )

    assert payload == {"status": "error", "error": "atomic_boundary_required"}
    assert ctx.repository.list_memory_ids() == before
    assert boundary.callback_calls == 0


def test_internal_ingest_enforce_requires_transaction_boundary(db_manager) -> None:
    """Enforce mode rejects a mutation without a transaction resource."""
    ctx = _build_ctx(db_manager)
    cast(Any, ctx).config = SimpleNamespace(ingress_evidence_mode="enforce")
    _start_running_ingest_task(ctx, "ingest-missing-transaction")
    assert ctx.repository is not None
    before = ctx.repository.list_memory_ids()

    payload = internal_tool_services()["internal_ingest_create_memory"](
        ctx,
        {
            "task_id": "ingest-missing-transaction",
            "batch_id": "batch-missing-transaction",
            "entry_ids": [67],
            "title": "Missing transaction",
            "content": "This must not mutate.",
        },
    )

    assert payload == {"status": "error", "error": "atomic_boundary_required"}
    assert ctx.repository.list_memory_ids() == before


def test_enqueue_summary_task_coalesces_open_work_and_can_requeue_after_completion(db_manager) -> None:
    ctx = _build_ctx(db_manager)
    assert ctx.repository is not None
    assert ctx.task_queue is not None

    record = ctx.repository.create_memory(
        title="Summary refresh target",
        content="Deterministic summaries should avoid duplicate queue churn.",
        workspace_ids=[ctx.workspace_id or "workspace-a"],
        memory_type="observation",
    )
    assert record is not None

    _enqueue_summary_task(ctx, record.id, list(record.workspace_ids))
    _enqueue_summary_task(ctx, record.id, list(record.workspace_ids))

    summary_tasks = ctx.task_queue.list_tasks(
        status="pending",
        workspace_id=None,
        limit=10,
    )
    summary_tasks = [task for task in summary_tasks if task.task_name == SUMMARIZE_MEMORY_TASK_NAME]
    assert len(summary_tasks) == 1
    assert summary_tasks[0].workspace_id is None
    assert summary_tasks[0].data == {"memory_id": record.id}
    assert summary_tasks[0].priority == SUMMARIZE_MEMORY_PRIORITY

    claimed = ctx.task_queue.claim_next(now=summary_tasks[0].available_at)
    assert claimed is not None
    assert claimed.task_name == SUMMARIZE_MEMORY_TASK_NAME
    ctx.task_queue.complete(claimed.id, completed_at=claimed.claimed_at or summary_tasks[0].available_at)

    _enqueue_summary_task(ctx, record.id, list(record.workspace_ids))

    requeued = ctx.task_queue.find_open_task_with_data_any_workspace(
        SUMMARIZE_MEMORY_TASK_NAME,
        data_fields={"memory_id": record.id},
    )
    assert requeued is not None
    assert requeued.id != claimed.id


def test_enqueue_summary_refresh_task_dedupes_globally_across_workspace_contexts(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    ctx_a = ApplicationContext(workspace_id="workspace-a", db_manager=db_manager, task_queue=queue)
    ctx_b = ApplicationContext(workspace_id="workspace-b", db_manager=db_manager, task_queue=queue)

    first = enqueue_summary_refresh_task(ctx_a, memory_id="memory-shared")
    second = enqueue_summary_refresh_task(ctx_b, memory_id="memory-shared")

    assert first is not None
    assert second is not None
    assert first.id == second.id
    assert first.workspace_id is None
    assert second.workspace_id is None
    assert first.data == {"memory_id": "memory-shared"}


def test_enqueue_summary_refresh_task_accepts_queue_only_context(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    ctx = _TaskQueueOnlyContext(task_queue=queue)

    first = enqueue_summary_refresh_task(ctx, memory_id="memory-shared")
    second = enqueue_summary_refresh_task(ctx, memory_id="memory-shared")

    assert first is not None
    assert second is not None
    assert first.id == second.id
    assert first.workspace_id is None
    assert first.data == {"memory_id": "memory-shared"}


def test_ingest_task_data_helpers_accept_queue_only_context(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    _start_running_ingest_task_on_queue(queue, "ingest-task-data-only")
    ctx = _TaskQueueOnlyContext(task_queue=queue)

    _record_successful_ingest_entry_ids(ctx, task_id="ingest-task-data-only", entry_ids=[11, 12])
    _record_successful_ingest_entry_dispositions(
        ctx,
        task_id="ingest-task-data-only",
        entry_dispositions=[
            {
                "entry_id": 11,
                "disposition": "created",
                "memory_id": "memory-1",
                "memory_title": "Created memory",
            },
            {
                "entry_id": 12,
                "disposition": "appended",
                "memory_id": "memory-2",
                "memory_title": "Appended memory",
                "reason": "duplicate observation",
            },
        ],
    )
    _record_ingest_tool_invocation(
        ctx,
        task_id="ingest-task-data-only",
        tool_name="internal_ingest_create_memory",
        mutation=True,
    )
    _record_touched_memory_ids(
        ctx,
        task_id="ingest-task-data-only",
        memory_ids=["memory-2", "memory-1", " ", "memory-1"],
    )

    assert _recorded_ingest_run_metadata(ctx, "ingest-task-data-only") == {
        "handled_entry_ids": [11, 12],
        "entry_dispositions": [
            {
                "entry_id": 11,
                "disposition": "created",
                "memory_id": "memory-1",
                "memory_title": "Created memory",
            },
            {
                "entry_id": 12,
                "disposition": "appended",
                "memory_id": "memory-2",
                "memory_title": "Appended memory",
                "reason": "duplicate observation",
            },
        ],
        "touched_memory_ids": ["memory-1", "memory-2"],
        "tool_usage": {
            "tool_calls_executed": 1,
            "mutations": 1,
            "tool_names_used": ["internal_ingest_create_memory"],
        },
    }

    _reset_recorded_ingest_handled_entry_ids(ctx, "ingest-task-data-only")

    assert _recorded_ingest_run_metadata(ctx, "ingest-task-data-only") == {
        "handled_entry_ids": [],
        "entry_dispositions": [],
        "touched_memory_ids": [],
        "tool_usage": {
            "tool_calls_executed": 0,
            "mutations": 0,
            "tool_names_used": [],
        },
    }


def test_internal_ingest_create_rejects_routine_completion_trace_payload(db_manager) -> None:
    ctx = _build_ctx(db_manager)
    _start_running_ingest_task(ctx, "ingest-completion-trace")

    payload = internal_tool_services()["internal_ingest_create_memory"](
        ctx,
        {
            "task_id": "ingest-completion-trace",
            "entry_ids": [31],
            "title": "task_complete: deduplicator 123",
            "content": "Task: deduplicator\nTask ID: 123\nTask name: deduplicator\nMerged: 0\nArchived: 0\nAbsorbed_observations: 0",
        },
    )

    assert payload == {
        "status": "error",
        "error": "low_value_memory_rejected",
        "detail": "routine completion/status traces must use task_complete instead of creating memories",
    }


def test_internal_ingest_create_emits_warnings_for_untagged_generic_observation(db_manager) -> None:
    ctx = _build_ctx(db_manager)
    _start_running_ingest_task(ctx, "ingest-warning-task")

    payload = internal_tool_services()["internal_ingest_create_memory"](
        ctx,
        {
            "task_id": "ingest-warning-task",
            "entry_ids": [41],
            "title": "Broad observation",
            "content": "A durable but loosely structured observation.",
            "summary": "Covers several related findings.",
        },
    )

    assert payload["status"] == "ok"
    assert payload["warnings"] == ["generic_summary", "observation_missing_tags"]


def test_internal_ingest_create_warns_for_added_summary_boilerplate(db_manager) -> None:
    ctx = _build_ctx(db_manager)
    _start_running_ingest_task(ctx, "ingest-added-summary-warning")

    payload = internal_tool_services()["internal_ingest_create_memory"](
        ctx,
        {
            "task_id": "ingest-added-summary-warning",
            "entry_ids": [42],
            "title": "Specific observation",
            "content": "A durable observation with a concrete detail.",
            "summary": "Added several related findings.",
        },
    )

    assert payload["status"] == "ok"
    assert payload["warnings"] == ["generic_summary", "observation_missing_tags"]


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
        "appended_via_ingest": True,
        "appended_entry_ids": [20, 21, 22],
        "ingest_task_id": "ingest-append-task",
        "note": "kept",
    }
    assert updated.tags == ["existing", "new-tag"]
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


def test_internal_ingest_append_rejects_routine_completion_trace_and_warns_on_oversized_memory(db_manager) -> None:
    ctx = _build_ctx(db_manager)
    _start_running_ingest_task(ctx, "ingest-append-guardrails")
    assert ctx.repository is not None

    target = ctx.repository.create_memory(
        title="Durable target",
        content="Base body.",
        workspace_ids=[ctx.workspace_id or "workspace-a"],
        memory_type="observation",
        tags=["durable"],
    )
    assert target is not None

    rejected = internal_tool_services()["internal_ingest_append_memory"](
        ctx,
        {
            "memory_id": target.id,
            "content": "Task ID: dedup-123\nTask name: deduplicator\nCompletion marker\nMerged: 0\nArchived: 0",
            "task_id": "ingest-append-guardrails",
            "entry_ids": [51],
            "summary": "Task ID: dedup-123",
        },
    )

    assert rejected == {
        "status": "error",
        "error": "low_value_memory_rejected",
        "detail": "routine completion/status traces must use task_complete instead of creating memories",
    }

    warned = internal_tool_services()["internal_ingest_append_memory"](
        ctx,
        {
            "memory_id": target.id,
            "content": "x" * 4_100,
            "task_id": "ingest-append-guardrails",
            "entry_ids": [52],
        },
    )

    assert warned["status"] == "ok"
    assert warned["warnings"] == ["memory_needs_split"]


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

    assert "task_complete" in tool_names
    assert "internal_task_complete" in tool_names
    assert "internal_get_next_ingest_batch" in tool_names
    assert "internal_get_next_curator_batch" in tool_names
    assert "internal_ingest_append_memory" in tool_names
    assert "internal_ingest_create_memory" in tool_names
    assert "internal_ingest_append_memory" in services
    assert "internal_ingest_create_memory" in services
    assert "task_complete" in services
    assert "internal_append_to_existing_memory_for_ingest" in services
    assert "internal_create_memory_record_for_ingest" in services
    assert services["internal_ingest_append_memory"] is services["internal_append_to_existing_memory_for_ingest"]
    assert services["internal_ingest_create_memory"] is services["internal_create_memory_record_for_ingest"]


def test_internal_task_complete_returns_completion_ack(db_manager) -> None:
    ctx = _build_ctx(db_manager)

    payload = internal_tool_services()["internal_task_complete"](
        ctx,
        {
            "task_id": "curator-task-1",
            "task_name": "memory-curator",
            "summary": "Completed one merge and one retag.",
        },
    )

    assert payload == {
        "status": "ok",
        "completion_recorded": True,
    }


def test_internal_create_memory_record_rejects_routine_completion_trace_payload(db_manager) -> None:
    ctx = _build_ctx(db_manager)

    payload = internal_tool_services()["internal_create_memory_record"](
        ctx,
        {
            "title": "task_complete_record: curator abc",
            "content": "Task: memory-curator\nTask ID: abc\nTask name: memory-curator\nSummary: no-op pass\nMerged: 0\nArchived: 0",
        },
    )

    assert payload == {
        "status": "error",
        "error": "low_value_memory_rejected",
        "detail": "routine completion/status traces must use task_complete instead of creating memories",
    }


def test_internal_append_and_update_services_apply_guardrails(db_manager) -> None:
    ctx = _build_ctx(db_manager)
    assert ctx.repository is not None

    record = ctx.repository.create_memory(
        title="Working memory",
        content="Initial durable content.",
        workspace_ids=[ctx.workspace_id or "workspace-a"],
        memory_type="observation",
        tags=["quality"],
    )
    assert record is not None

    append_rejected = internal_tool_services()["internal_append_memory_content"](
        ctx,
        {
            "memory_id": record.id,
            "content": "Task ID: curator-2\nTask name: memory-curator\nCompletion marker\nArchived: 0",
            "summary": "Task ID: curator-2",
        },
    )
    assert append_rejected == {
        "status": "error",
        "error": "low_value_memory_rejected",
        "detail": "routine completion/status traces must use task_complete instead of creating memories",
    }

    append_warned = internal_tool_services()["internal_append_memory_content"](
        ctx,
        {
            "memory_id": record.id,
            "content": "y" * 4_100,
        },
    )
    assert append_warned["status"] == "ok"
    assert append_warned["warnings"] == ["memory_needs_split"]

    update_rejected = internal_tool_services()["internal_update_memory_record"](
        ctx,
        {
            "memory_id": record.id,
            "title": "task_complete: curator-2",
            "content": "Task ID: curator-2\nTask name: memory-curator\nMerged: 0\nArchived: 0",
            "summary": "Task ID: curator-2",
        },
    )
    assert update_rejected == {
        "status": "error",
        "error": "low_value_memory_rejected",
        "detail": "routine completion/status traces must use task_complete instead of creating memories",
    }

    update_warned = internal_tool_services()["internal_update_memory_record"](
        ctx,
        {
            "memory_id": record.id,
            "content": "z" * 4_100,
            "summary": "Covers several related findings.",
            "tags": [],
        },
    )
    assert update_warned["status"] == "ok"
    assert update_warned["warnings"] == ["generic_summary", "observation_missing_tags", "memory_needs_split"]


def test_task_complete_alias_returns_completion_ack(db_manager) -> None:
    ctx = _build_ctx(db_manager)

    payload = internal_tool_services()["task_complete"](
        ctx,
        {
            "task_id": "curator-task-2",
            "task_name": "memory-curator",
            "summary": "Completed one archive.",
        },
    )

    assert payload == {
        "status": "ok",
        "completion_recorded": True,
    }
