from __future__ import annotations

import asyncio

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.journal import System1Journal
from mcp_memory.core.system1_scheduling import schedule_system1_ingest
from mcp_memory.core.task_worker import RuntimeTaskWorker
from mcp_memory.core.tasks import SQLiteTaskQueue


pytestmark = pytest.mark.small


def test_sqlite_task_queue_enqueue_and_claim_order(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)

    late = queue.enqueue(
        "late-task",
        priority=50,
        available_at=20.0,
        task_id="late-task",
    )
    early = queue.enqueue(
        "early-task",
        data={"kind": "ingest"},
        workspace_id="workspace-a",
        priority=10,
        available_at=10.0,
        task_id="early-task",
    )

    claimed = queue.claim_next(now=15.0)

    assert claimed is not None
    assert claimed.id == early.id
    assert claimed.task_name == "early-task"
    assert claimed.data == {"kind": "ingest"}
    assert claimed.workspace_id == "workspace-a"
    assert claimed.status == "running"
    assert claimed.claimed_at == 15.0
    assert queue.claim_next(now=15.0) is None

    untouched = queue.get_task(late.id)
    assert untouched.status == "pending"
    assert untouched.available_at == 20.0


def test_sqlite_task_queue_claim_next_can_filter_by_workspace(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)

    task_a = queue.enqueue(
        "workspace-a-task",
        workspace_id="workspace-a",
        available_at=0.0,
        task_id="workspace-a-task",
    )
    task_b = queue.enqueue(
        "workspace-b-task",
        workspace_id="workspace-b",
        available_at=0.0,
        task_id="workspace-b-task",
    )

    claimed = queue.claim_next(now=1.0, workspace_id="workspace-a")

    assert claimed is not None
    assert claimed.id == task_a.id
    assert queue.get_task(task_b.id).status == "pending"


def test_sqlite_task_queue_complete_marks_terminal_state(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    task = queue.enqueue(
        "complete-me",
        available_at=20.0,
        task_id="complete-me",
    )

    claimed = queue.claim_next(now=25.0)
    completed = queue.complete(task.id, completed_at=30.0)

    assert claimed is not None
    assert completed.status == "completed"
    assert completed.completed_at == 30.0
    assert completed.last_error is None
    assert queue.count_by_status() == {"completed": 1}
    task_runs = queue.list_task_runs(task_name="complete-me")
    assert len(task_runs) == 1
    assert task_runs[0].status == "completed"
    assert task_runs[0].duration_seconds == 5.0


def test_sqlite_task_queue_fail_requeues_before_dead_letter(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    task = queue.enqueue(
        "retry-me",
        max_retries=3,
        available_at=10.0,
        task_id="retry-me",
    )

    first_claim = queue.claim_next(now=10.0)
    failed = queue.fail(task.id, "temporary failure", retry_delay_seconds=5.0, failed_at=12.0)

    assert first_claim is not None
    assert failed.status == "pending"
    assert failed.retries_count == 1
    assert failed.available_at == 17.0
    assert failed.last_error == "temporary failure"
    assert queue.claim_next(now=16.0) is None

    second_claim = queue.claim_next(now=17.0)
    assert second_claim is not None
    assert second_claim.id == task.id
    assert second_claim.status == "running"


def test_sqlite_task_queue_fail_transitions_to_dead_letter_at_retry_limit(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    task = queue.enqueue(
        "poison-pill",
        max_retries=2,
        available_at=5.0,
        task_id="poison-pill",
    )

    assert queue.claim_next(now=5.0) is not None
    retried = queue.fail(task.id, "first failure", failed_at=6.0)
    assert retried.status == "pending"

    assert queue.claim_next(now=6.0) is not None
    dead_letter = queue.fail(task.id, "second failure", failed_at=7.0)

    assert dead_letter.status == "failed"
    assert dead_letter.retries_count == 2
    assert dead_letter.completed_at == 7.0
    assert dead_letter.last_error == "second failure"
    assert queue.count_by_status() == {"failed": 1}
    task_runs = queue.list_task_runs(task_name="poison-pill")
    assert [task_run.status for task_run in task_runs] == ["failed", "retry"]


def test_sqlite_task_queue_summarize_task_runs_aggregates_status_and_compression(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    task = queue.enqueue(
        "defragmenter",
        workspace_id="workspace-a",
        available_at=10.0,
        task_id="defragmenter-1",
    )

    assert queue.claim_next(now=10.0) is not None
    queue.complete(task.id, completed_at=14.0, run_result={"lines_compressed": 7})

    summary = queue.summarize_task_runs(["defragmenter"], workspace_id="workspace-a")[0]

    assert summary.task_name == "defragmenter"
    assert summary.total_runs == 1
    assert summary.completed_runs == 1
    assert summary.failed_runs == 0
    assert summary.retry_runs == 0
    assert summary.avg_duration_seconds == 4.0
    assert summary.total_lines_compressed == 7


def test_sqlite_task_queue_rejects_completion_for_non_running_task(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    task = queue.enqueue("idle-task", task_id="idle-task")

    with pytest.raises(ValueError, match="not running"):
        queue.complete(task.id)


def test_sqlite_task_queue_enqueue_unique_reuses_existing_open_task(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)

    first, first_created = queue.enqueue_unique(
        "ingest-system1",
        workspace_id="workspace-a",
        data={"pending_count": 3},
    )
    second, second_created = queue.enqueue_unique(
        "ingest-system1",
        workspace_id="workspace-a",
        data={"pending_count": 4},
    )

    assert first_created is True
    assert second_created is False
    assert second.id == first.id
    assert queue.count_by_status() == {"pending": 1}


def test_schedule_system1_ingest_debounces_then_pulls_forward_at_threshold(
    db_manager,
    monkeypatch,
) -> None:
    queue = SQLiteTaskQueue(db_manager)
    journal = System1Journal(db_manager)

    now = 100.0
    monkeypatch.setattr("mcp_memory.core.journal.time.time", lambda: now)
    journal.record("note 0", workspace_id="workspace-a")

    scheduled = schedule_system1_ingest(queue, journal, "workspace-a", now=now)

    assert scheduled is not None
    assert scheduled.created is True
    assert scheduled.trigger == "system1_debounce"
    assert scheduled.task.available_at == 400.0

    for index in range(1, 10):
        now += 1.0
        monkeypatch.setattr("mcp_memory.core.journal.time.time", lambda current=now: current)
        journal.record(f"note {index}", workspace_id="workspace-a")

    accelerated = schedule_system1_ingest(queue, journal, "workspace-a", now=now)

    assert accelerated is not None
    assert accelerated.created is False
    assert accelerated.trigger == "system1_threshold"
    assert accelerated.task.id == scheduled.task.id
    assert accelerated.task.available_at == now


@pytest.mark.asyncio
async def test_runtime_task_worker_completes_claimed_tasks(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    seen_payloads: list[dict[str, str]] = []

    def handle_ingest(ctx: ApplicationContext, task) -> None:
        seen_payloads.append(task.data)

    ctx = ApplicationContext(db_manager=db_manager, task_queue=queue)
    task = queue.enqueue(
        "ingest-system1",
        data={"memory_id": "123"},
        available_at=0.0,
        task_id="ingest-system1",
    )
    worker = RuntimeTaskWorker(
        ctx,
        handlers={"ingest-system1": handle_ingest},
        poll_interval_seconds=0.01,
    )

    await worker.start()
    for _ in range(20):
        if queue.get_task(task.id).status == "completed":
            break
        await asyncio.sleep(0.01)
    await worker.stop(0.05)

    completed = queue.get_task(task.id)
    assert completed.status == "completed"
    assert seen_payloads == [{"memory_id": "123"}]
    task_runs = queue.list_task_runs(task_name="ingest-system1")
    assert len(task_runs) == 1
    assert task_runs[0].status == "completed"


@pytest.mark.asyncio
async def test_runtime_task_worker_retries_and_dead_letters_failures(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    attempts: list[str] = []

    def fail_task(ctx: ApplicationContext, task) -> None:
        attempts.append(task.id)
        raise RuntimeError("boom")

    ctx = ApplicationContext(db_manager=db_manager, task_queue=queue)
    task = queue.enqueue(
        "failing-task",
        available_at=0.0,
        max_retries=2,
        task_id="failing-task",
    )
    worker = RuntimeTaskWorker(
        ctx,
        handlers={"failing-task": fail_task},
        poll_interval_seconds=0.01,
        retry_delay_seconds=0.0,
    )

    await worker.start()
    for _ in range(50):
        if queue.get_task(task.id).status == "failed":
            break
        await asyncio.sleep(0.01)
    await worker.stop(0.05)

    failed = queue.get_task(task.id)
    assert failed.status == "failed"
    assert failed.retries_count == 2
    assert failed.last_error == "boom"
    assert attempts == ["failing-task", "failing-task"]
    task_runs = queue.list_task_runs(task_name="failing-task")
    assert [task_run.status for task_run in task_runs] == ["failed", "retry"]


@pytest.mark.asyncio
async def test_runtime_task_worker_dead_letters_unknown_tasks(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    ctx = ApplicationContext(db_manager=db_manager, task_queue=queue)
    task = queue.enqueue(
        "unknown-task",
        available_at=0.0,
        max_retries=5,
        task_id="unknown-task",
    )
    worker = RuntimeTaskWorker(ctx, handlers={}, poll_interval_seconds=0.01)

    await worker.start()
    for _ in range(20):
        if queue.get_task(task.id).status == "failed":
            break
        await asyncio.sleep(0.01)
    await worker.stop(0.05)

    failed = queue.get_task(task.id)
    assert failed.status == "failed"
    assert failed.retries_count == 5
    assert failed.last_error == "No task handler registered for unknown-task"


@pytest.mark.asyncio
async def test_runtime_task_workers_only_claim_matching_workspace_tasks(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    seen_by_a: list[str] = []
    seen_by_b: list[str] = []

    def handle_for_a(ctx: ApplicationContext, task) -> None:
        seen_by_a.append(task.id)

    def handle_for_b(ctx: ApplicationContext, task) -> None:
        seen_by_b.append(task.id)

    task_a = queue.enqueue(
        "shared-task",
        workspace_id="workspace-a",
        available_at=0.0,
        task_id="task-a",
    )
    task_b = queue.enqueue(
        "shared-task",
        workspace_id="workspace-b",
        available_at=0.0,
        task_id="task-b",
    )

    worker_a = RuntimeTaskWorker(
        ApplicationContext(db_manager=db_manager, task_queue=queue, workspace_id="workspace-a"),
        handlers={"shared-task": handle_for_a},
        poll_interval_seconds=0.01,
    )
    worker_b = RuntimeTaskWorker(
        ApplicationContext(db_manager=db_manager, task_queue=queue, workspace_id="workspace-b"),
        handlers={"shared-task": handle_for_b},
        poll_interval_seconds=0.01,
    )

    await worker_a.start()
    await worker_b.start()
    for _ in range(50):
        if queue.get_task(task_a.id).status == "completed" and queue.get_task(task_b.id).status == "completed":
            break
        await asyncio.sleep(0.01)
    await worker_a.stop(0.05)
    await worker_b.stop(0.05)

    assert queue.get_task(task_a.id).status == "completed"
    assert queue.get_task(task_b.id).status == "completed"
    assert seen_by_a == ["task-a"]
    assert seen_by_b == ["task-b"]