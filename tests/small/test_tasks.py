from __future__ import annotations

import asyncio
from datetime import datetime
import threading

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.journal import System1Journal
from mcp_memory.core.system1_scheduling import schedule_system1_ingest
from mcp_memory.core.task_handlers import (
    SYSTEM1_AUTO_INGEST_RATE_LIMIT_SECONDS,
    SYSTEM1_INGEST_PRIORITY,
    SYSTEM1_INGEST_TASK_NAME,
)
from mcp_memory.core.task_worker import RuntimeTaskWorker
from mcp_memory.core.tasks import SQLiteTaskQueue
from mcp_memory.provider_usage_store import ProviderUsageRepository


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


def test_sqlite_task_queue_can_cancel_pending_task(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    task = queue.enqueue("cancel-me", task_id="cancel-me")

    cancelled = queue.request_cancel(
        task.id,
        cancelled_by="cli",
        reason="operator_cancelled",
        requested_at=12.0,
    )

    assert cancelled.status == "cancelled"
    assert cancelled.cancelled_by == "cli"
    assert cancelled.cancellation_reason == "operator_cancelled"
    task_runs = queue.list_task_runs(task_name="cancel-me")
    assert [task_run.status for task_run in task_runs] == ["cancelled"]


def test_sqlite_task_queue_tracks_running_subprocess_and_finalizes_cancellation(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    task = queue.enqueue("cancel-running", task_id="cancel-running", available_at=0.0)

    assert queue.claim_next(now=5.0) is not None
    tracked = queue.set_running_process(task.id, subprocess_pid=9999, request_id="req-1", updated_at=6.0)
    requested = queue.request_cancel(task.id, cancelled_by="cli", reason="timeout triage", requested_at=7.0)
    cancelled = queue.finalize_cancellation(task.id, cancelled_at=8.0)

    assert tracked.subprocess_pid == 9999
    assert tracked.active_request_id == "req-1"
    assert requested.cancellation_requested_at == 7.0
    assert cancelled.status == "cancelled"
    assert cancelled.subprocess_pid is None
    assert cancelled.active_request_id is None
    assert cancelled.last_error == "timeout triage"


def test_sqlite_task_queue_touch_running_task_refreshes_updated_at(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    task = queue.enqueue("touch-me", task_id="touch-me", available_at=0.0)

    assert queue.claim_next(now=5.0) is not None
    touched = queue.touch_running_task(task.id, updated_at=7.5)

    assert touched.status == "running"
    assert touched.updated_at == pytest.approx(7.5)


def test_sqlite_task_queue_does_not_recover_dead_subprocess_before_stale_threshold(
    db_manager,
    monkeypatch,
) -> None:
    queue = SQLiteTaskQueue(db_manager)
    task = queue.enqueue("pid-dead-task", task_id="pid-dead-task", available_at=0.0)

    assert queue.claim_next(now=950.0) is not None
    queue.set_running_process(task.id, subprocess_pid=9999, request_id="req-dead", updated_at=951.0)
    monkeypatch.setattr("mcp_memory.core.tasks._is_process_alive", lambda pid: False)

    recovered = queue.recover_abandoned_running_tasks(now=1000.0, stale_after_seconds=100.0)

    assert recovered == []
    running = queue.get_task(task.id)
    assert running.status == "running"
    assert running.subprocess_pid == 9999
    assert running.last_error is None


def test_sqlite_task_queue_fails_dead_stale_subprocess_task(db_manager, monkeypatch) -> None:
    queue = SQLiteTaskQueue(db_manager)
    task = queue.enqueue("pid-dead-task", task_id="pid-dead-task", available_at=0.0)

    assert queue.claim_next(now=10.0) is not None
    queue.set_running_process(task.id, subprocess_pid=9999, request_id="req-dead", updated_at=12.0)
    monkeypatch.setattr("mcp_memory.core.tasks._is_process_alive", lambda pid: False)

    recovered = queue.recover_abandoned_running_tasks(now=1000.0, stale_after_seconds=300.0)

    assert [item.id for item in recovered] == [task.id]
    failed = queue.get_task(task.id)
    assert failed.status == "failed"
    assert failed.last_error == "Provider subprocess 9999 exited unexpectedly"


def test_sqlite_task_queue_recent_progress_prevents_dead_subprocess_recovery(
    db_manager,
    monkeypatch,
) -> None:
    queue = SQLiteTaskQueue(db_manager)
    task = queue.enqueue("pid-dead-task", task_id="pid-heartbeat-task", available_at=0.0)

    assert queue.claim_next(now=10.0) is not None
    queue.set_running_process(task.id, subprocess_pid=9999, request_id="req-live", updated_at=12.0)
    queue.touch_running_task(task.id, updated_at=950.0)
    monkeypatch.setattr("mcp_memory.core.tasks._is_process_alive", lambda pid: False)

    recovered = queue.recover_abandoned_running_tasks(now=1000.0, stale_after_seconds=100.0)

    assert recovered == []
    running = queue.get_task(task.id)
    assert running.status == "running"
    assert running.updated_at == pytest.approx(950.0)


def test_sqlite_task_queue_cancels_dead_stale_subprocess_when_cancellation_requested(
    db_manager,
    monkeypatch,
) -> None:
    queue = SQLiteTaskQueue(db_manager)
    task = queue.enqueue("pid-dead-task", task_id="pid-dead-task", available_at=0.0)

    assert queue.claim_next(now=10.0) is not None
    queue.set_running_process(task.id, subprocess_pid=9999, request_id="req-dead", updated_at=12.0)
    queue.request_cancel(task.id, cancelled_by="cli", reason="operator_cancelled", requested_at=20.0)
    monkeypatch.setattr("mcp_memory.core.tasks._is_process_alive", lambda pid: False)

    recovered = queue.recover_abandoned_running_tasks(now=1000.0, stale_after_seconds=300.0)

    assert [item.id for item in recovered] == [task.id]
    cancelled = queue.get_task(task.id)
    assert cancelled.status == "cancelled"
    assert cancelled.last_error == "operator_cancelled"
    assert cancelled.subprocess_pid is None


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
    assert scheduled.task.available_at == 3700.0
    assert scheduled.task.priority == SYSTEM1_INGEST_PRIORITY

    for index in range(1, 40):
        now += 1.0
        monkeypatch.setattr("mcp_memory.core.journal.time.time", lambda current=now: current)
        journal.record(f"note {index}", workspace_id="workspace-a")

    accelerated = schedule_system1_ingest(queue, journal, "workspace-a", now=now)

    assert accelerated is not None
    assert accelerated.created is False
    assert accelerated.trigger == "system1_threshold"
    assert accelerated.task.id == scheduled.task.id
    assert accelerated.task.available_at == now
    assert accelerated.task.priority == SYSTEM1_INGEST_PRIORITY


def test_schedule_system1_ingest_resets_debounce_on_new_activity_below_threshold(
    db_manager,
    monkeypatch,
) -> None:
    queue = SQLiteTaskQueue(db_manager)
    journal = System1Journal(db_manager)

    now = 100.0
    monkeypatch.setattr("mcp_memory.core.journal.time.time", lambda: now)
    journal.record("note 0", workspace_id="workspace-a")

    initial = schedule_system1_ingest(queue, journal, "workspace-a", now=now)

    assert initial is not None
    assert initial.trigger == "system1_debounce"
    assert initial.task.available_at == 3700.0

    now = 160.0
    monkeypatch.setattr("mcp_memory.core.journal.time.time", lambda: now)
    journal.record("note 1", workspace_id="workspace-a")

    rescheduled = schedule_system1_ingest(queue, journal, "workspace-a", now=now)

    assert rescheduled is not None
    assert rescheduled.created is False
    assert rescheduled.task.id == initial.task.id
    assert rescheduled.trigger == "system1_debounce"
    assert rescheduled.task.available_at == 3760.0


def test_schedule_system1_ingest_threshold_respects_rate_limit_boundary(
    db_manager,
    monkeypatch,
) -> None:
    queue = SQLiteTaskQueue(db_manager)
    journal = System1Journal(db_manager)

    recent_task = queue.enqueue(
        SYSTEM1_INGEST_TASK_NAME,
        workspace_id=None,
        data={"workspace_id": None},
        available_at=0.0,
        task_id="recent-completed-ingest",
    )
    assert queue.claim_next(now=90.0) is not None
    queue.complete(recent_task.id, completed_at=100.0)

    now = 1000.0
    monkeypatch.setattr("mcp_memory.core.journal.time.time", lambda: now)
    journal.record("note 0", workspace_id="workspace-a")

    initial = schedule_system1_ingest(queue, journal, "workspace-a", now=now)

    assert initial is not None
    assert initial.trigger == "system1_debounce"
    assert initial.task.available_at == 4600.0

    for index in range(1, 40):
        now += 1.0
        monkeypatch.setattr("mcp_memory.core.journal.time.time", lambda current=now: current)
        journal.record(f"note {index}", workspace_id="workspace-a")

    accelerated = schedule_system1_ingest(queue, journal, "workspace-a", now=now)

    assert accelerated is not None
    assert accelerated.created is False
    assert accelerated.task.id == initial.task.id
    assert accelerated.trigger == "system1_threshold_rate_limited"
    assert accelerated.task.available_at == 100.0 + SYSTEM1_AUTO_INGEST_RATE_LIMIT_SECONDS


def test_schedule_system1_ingest_does_not_delay_pending_manual_task(
    db_manager,
    monkeypatch,
) -> None:
    queue = SQLiteTaskQueue(db_manager)
    journal = System1Journal(db_manager)

    recent_task = queue.enqueue(
        SYSTEM1_INGEST_TASK_NAME,
        workspace_id=None,
        data={"workspace_id": None},
        available_at=0.0,
        task_id="recent-manual-success",
    )
    assert queue.claim_next(now=90.0) is not None
    queue.complete(recent_task.id, completed_at=100.0)

    manual_task = queue.enqueue(
        SYSTEM1_INGEST_TASK_NAME,
        workspace_id="workspace-a",
        data={"workspace_id": "workspace-a"},
        available_at=150.0,
        task_id="manual-pending-ingest",
    )

    now = 200.0
    monkeypatch.setattr("mcp_memory.core.journal.time.time", lambda: now)
    journal.record("note 0", workspace_id="workspace-a")

    scheduled = schedule_system1_ingest(queue, journal, "workspace-a", now=now)

    assert scheduled is not None
    assert scheduled.created is False
    assert scheduled.task.id == manual_task.id
    assert scheduled.task.available_at == 150.0
    assert scheduled.task.data == {"workspace_id": "workspace-a"}


def test_schedule_system1_ingest_uses_manual_success_as_cooldown_anchor(
    db_manager,
    monkeypatch,
) -> None:
    queue = SQLiteTaskQueue(db_manager)
    journal = System1Journal(db_manager)

    manual_task = queue.enqueue(
        SYSTEM1_INGEST_TASK_NAME,
        workspace_id=None,
        data={"workspace_id": None},
        available_at=0.0,
        task_id="manual-success-anchor",
    )
    assert queue.claim_next(now=490.0) is not None
    queue.complete(manual_task.id, completed_at=500.0)

    now = 550.0
    for index in range(40):
        entry_time = now + index
        monkeypatch.setattr("mcp_memory.core.journal.time.time", lambda current=entry_time: current)
        journal.record(f"note {index}", workspace_id="workspace-a")

    scheduled = schedule_system1_ingest(queue, journal, "workspace-a", now=now + 39.0)

    assert scheduled is not None
    assert scheduled.created is True
    assert scheduled.trigger == "system1_threshold_rate_limited"
    assert scheduled.task.available_at == 500.0 + SYSTEM1_AUTO_INGEST_RATE_LIMIT_SECONDS


def test_schedule_system1_ingest_respects_active_suppression_window(
    db_manager,
    monkeypatch,
) -> None:
    from mcp_memory.config import IngestSuppressionConfig, IngestSuppressionWindow

    queue = SQLiteTaskQueue(db_manager)
    journal = System1Journal(db_manager)

    now = datetime(2026, 3, 17, 22, 15).astimezone().timestamp()
    monkeypatch.setattr("mcp_memory.core.journal.time.time", lambda: now)
    journal.record("quiet-hours note", workspace_id="workspace-a")

    scheduled = schedule_system1_ingest(
        queue,
        journal,
        "workspace-a",
        now=now,
        suppression_config=IngestSuppressionConfig(
            enabled=True,
            windows=[IngestSuppressionWindow(start_hour=22, end_hour=6)],
        ),
    )

    assert scheduled is not None
    assert scheduled.trigger == "system1_debounce_suppressed"
    assert scheduled.task.available_at == pytest.approx(datetime(2026, 3, 18, 6, 0).astimezone().timestamp())


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
    for _ in range(100):
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
async def test_runtime_task_worker_runs_sync_handlers_off_loop(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    thread_ids: list[int] = []
    main_thread_id = threading.get_ident()

    def sync_handler(ctx: ApplicationContext, task) -> None:
        thread_ids.append(threading.get_ident())

    ctx = ApplicationContext(db_manager=db_manager, task_queue=queue)
    task = queue.enqueue(
        'sync-task',
        available_at=0.0,
        task_id='sync-task',
    )
    claimed = queue.claim_next(now=1.0)
    assert claimed is not None

    worker = RuntimeTaskWorker(
        ctx,
        handlers={'sync-task': sync_handler},
        poll_interval_seconds=0.01,
    )

    await worker._process_task(claimed)  # noqa: SLF001

    completed = queue.get_task(task.id)
    assert completed.status == 'completed'
    assert thread_ids
    assert thread_ids[0] != main_thread_id


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
async def test_runtime_task_worker_honors_exception_specific_retry_delay(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)

    class _RetryLaterError(RuntimeError):
        def __init__(self) -> None:
            super().__init__("retry later")
            self.retry_delay_seconds = 12.0

    def fail_task(ctx: ApplicationContext, task) -> None:
        raise _RetryLaterError()

    ctx = ApplicationContext(db_manager=db_manager, task_queue=queue)
    task = queue.enqueue(
        "failing-task",
        available_at=0.0,
        max_retries=2,
        task_id="failing-task-delayed",
    )
    worker = RuntimeTaskWorker(
        ctx,
        handlers={"failing-task": fail_task},
        poll_interval_seconds=0.01,
        retry_delay_seconds=0.0,
    )

    await worker.start()
    for _ in range(50):
        if queue.get_task(task.id).status == "pending" and queue.get_task(task.id).retries_count == 1:
            break
        await asyncio.sleep(0.01)
    await worker.stop(0.05)

    failed_once = queue.get_task(task.id)
    assert failed_once.status == "pending"
    assert failed_once.retries_count == 1
    assert failed_once.available_at == pytest.approx(failed_once.updated_at + 12.0)


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
async def test_runtime_task_worker_claims_tasks_across_workspace_ids(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    seen: list[tuple[str, str | None]] = []

    def handle_shared(ctx: ApplicationContext, task) -> None:
        seen.append((task.id, task.workspace_id))

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

    worker = RuntimeTaskWorker(
        ApplicationContext(db_manager=db_manager, task_queue=queue, workspace_id="workspace-a"),
        handlers={"shared-task": handle_shared},
        poll_interval_seconds=0.01,
    )

    await worker.start()
    for _ in range(50):
        if queue.get_task(task_a.id).status == "completed" and queue.get_task(task_b.id).status == "completed":
            break
        await asyncio.sleep(0.01)
    await worker.stop(0.05)

    assert queue.get_task(task_a.id).status == "completed"
    assert queue.get_task(task_b.id).status == "completed"
    assert seen == [("task-a", "workspace-a"), ("task-b", "workspace-b")]


@pytest.mark.asyncio
async def test_runtime_task_worker_recovers_abandoned_running_tasks_on_start(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    ctx = ApplicationContext(db_manager=db_manager, task_queue=queue, workspace_id="workspace-a")
    task = queue.enqueue(
        "orphaned-task",
        workspace_id="workspace-a",
        available_at=0.0,
        task_id="orphaned-task",
    )
    assert queue.claim_next(now=1.0, workspace_id="workspace-a") is not None

    worker = RuntimeTaskWorker(
        ctx,
        handlers={"orphaned-task": lambda ctx, task: None},
        poll_interval_seconds=0.01,
    )

    original = SQLiteTaskQueue.recover_abandoned_running_tasks

    def recover(self, **kwargs):
        return original(self, stale_after_seconds=0.0, now=2.0, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(SQLiteTaskQueue, "recover_abandoned_running_tasks", recover)
    try:
        await worker.start()
        for _ in range(20):
            if queue.get_task(task.id).status == "failed":
                break
            await asyncio.sleep(0.01)
        await worker.stop(0.05)
    finally:
        monkeypatch.undo()

    assert queue.get_task(task.id).status == "failed"


@pytest.mark.asyncio
async def test_runtime_task_worker_releases_orphaned_journal_claims_on_start(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    journal = System1Journal(db_manager)
    ctx = ApplicationContext(db_manager=db_manager, task_queue=queue, journal=journal, workspace_id="workspace-a")
    task = queue.enqueue(
        "orphaned-task",
        workspace_id="workspace-a",
        available_at=0.0,
        task_id="orphaned-task",
    )
    assert queue.claim_next(now=1.0, workspace_id="workspace-a") is not None

    entry = journal.record("orphaned claimed thought", workspace_id="workspace-a")
    claimed = journal.claim_pending(task_id=task.id, limit=1, workspace_id="workspace-a", claimed_at=2.0)
    assert [item.id for item in claimed] == [entry.id]

    worker = RuntimeTaskWorker(
        ctx,
        handlers={"orphaned-task": lambda ctx, task: None},
        poll_interval_seconds=0.01,
    )

    original = SQLiteTaskQueue.recover_abandoned_running_tasks

    def recover(self, **kwargs):
        return original(self, stale_after_seconds=0.0, now=3.0, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(SQLiteTaskQueue, "recover_abandoned_running_tasks", recover)
    try:
        await worker.start()
        for _ in range(20):
            if queue.get_task(task.id).status == "failed":
                break
            await asyncio.sleep(0.01)
        await worker.stop(0.05)
    finally:
        monkeypatch.undo()

    assert queue.get_task(task.id).status == "failed"
    assert [pending.id for pending in journal.get_pending(workspace_id="workspace-a")] == [entry.id]


@pytest.mark.asyncio
async def test_runtime_task_worker_reconciles_running_conversation_for_recovered_dead_subprocess_task(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    ctx = ApplicationContext(db_manager=db_manager, task_queue=queue, workspace_id="workspace-a")
    task = queue.enqueue(
        "ingest-system1",
        workspace_id="workspace-a",
        available_at=0.0,
        task_id="orphaned-provider-task",
    )
    assert queue.claim_next(now=1.0, workspace_id="workspace-a") is not None
    queue.set_running_process(task.id, subprocess_pid=9999, request_id="req-orphaned", updated_at=2.0)
    repository.record_conversation(
        request_id="req-orphaned",
        attempt=1,
        task_name="ingest-system1",
        task_id=task.id,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
        subprocess_pid=9999,
        prompt_text="ingest pending thoughts",
        response_text="",
        parsed=None,
        status="running",
        error_text=None,
        started_at=1.0,
        completed_at=1.0,
    )

    worker = RuntimeTaskWorker(
        ctx,
        handlers={"ingest-system1": lambda ctx, task: None},
        poll_interval_seconds=0.01,
    )

    original = SQLiteTaskQueue.recover_abandoned_running_tasks

    def recover(self, **kwargs):
        return original(self, stale_after_seconds=0.0, now=5.0, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("mcp_memory.core.tasks._is_process_alive", lambda pid: False)
    monkeypatch.setattr(SQLiteTaskQueue, "recover_abandoned_running_tasks", recover)
    try:
        await worker.start()
        for _ in range(20):
            if queue.get_task(task.id).status == "failed":
                break
            await asyncio.sleep(0.01)
        await worker.stop(0.05)
    finally:
        monkeypatch.undo()

    failed = queue.get_task(task.id)
    conversation = repository.get_conversation("req-orphaned")[0]

    assert failed.status == "failed"
    assert failed.last_error == "Provider subprocess 9999 exited unexpectedly"
    assert conversation.status == "error"
    assert conversation.task_id == task.id
    assert conversation.error_text == "Provider subprocess 9999 exited unexpectedly"


@pytest.mark.asyncio
async def test_runtime_task_worker_finalizes_requested_cancellation_on_cancelled_error(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    ctx = ApplicationContext(db_manager=db_manager, task_queue=queue, workspace_id="workspace-a")
    task = queue.enqueue(
        "cancel-me",
        workspace_id="workspace-a",
        available_at=0.0,
        task_id="cancel-me",
    )
    claimed = queue.claim_next(now=1.0, workspace_id="workspace-a")
    assert claimed is not None
    queue.request_cancel(task.id, cancelled_by="cli", reason="operator_cancelled", requested_at=2.0)

    async def cancelled_handler(ctx: ApplicationContext, task) -> None:
        raise asyncio.CancelledError

    worker = RuntimeTaskWorker(
        ctx,
        handlers={"cancel-me": cancelled_handler},
        poll_interval_seconds=0.01,
    )

    with pytest.raises(asyncio.CancelledError):
        await worker._process_task(claimed)  # noqa: SLF001

    cancelled = queue.get_task(task.id)
    assert cancelled.status == "cancelled"
    assert cancelled.cancellation_reason == "operator_cancelled"


@pytest.mark.asyncio
async def test_runtime_task_worker_refreshes_missing_handler_on_demand(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    seen: list[str] = []

    def recovered_handler(ctx: ApplicationContext, task) -> None:
        seen.append(task.id)

    task = queue.enqueue(
        "memory-curator",
        workspace_id="workspace-a",
        available_at=0.0,
        task_id="refresh-handler-task",
    )
    claimed = queue.claim_next(now=1.0, workspace_id="workspace-a")
    assert claimed is not None

    worker = RuntimeTaskWorker(
        ApplicationContext(db_manager=db_manager, task_queue=queue, workspace_id="workspace-a"),
        handlers={},
        handler_factory=lambda: {"memory-curator": recovered_handler},
        poll_interval_seconds=0.01,
    )

    await worker._process_task(claimed)  # noqa: SLF001

    completed = queue.get_task(task.id)
    assert completed.status == "completed"
    assert seen == ["refresh-handler-task"]
