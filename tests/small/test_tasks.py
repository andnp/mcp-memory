from __future__ import annotations

import asyncio
from datetime import datetime
import sqlite3
import threading
import time

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.journal import System1Journal
from mcp_memory.core.journal_operations import RecordThoughtOperation
from mcp_memory.core.maintenance_idle import (
    AUTONOMOUS_MAINTENANCE_IDLE_THRESHOLD_SECONDS,
    resume_paused_recurring_maintenance,
)
from mcp_memory.core.system1_scheduling import schedule_system1_ingest
from mcp_memory.core.task_handlers import (
    CURATOR_TASK_NAME,
    SYSTEM1_AUTO_INGEST_RATE_LIMIT_SECONDS,
    SYSTEM1_INGEST_PRIORITY,
    SYSTEM1_INGEST_TASK_NAME,
    RECURRING_TASK_INTERVAL_SECONDS,
)
from mcp_memory.core import tasks as tasks_module
from mcp_memory.core.task_worker import RuntimeTaskWorker
from mcp_memory.core.tasks import SQLiteTaskQueue, TaskRecord
from mcp_memory.provider_usage_store import ProviderUsageRepository
from mcp_memory.task_execution_store import TaskExecutionAttemptRepository
from mcp_memory.utils.db import SQLITE_BUSY_TIMEOUT_MILLISECONDS


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
    assert claimed.execution_epoch == 1
    assert claimed.claimed_at == 15.0
    assert queue.claim_next(now=15.0) is None

    untouched = queue.get_task(late.id)
    assert untouched.status == "pending"
    assert untouched.execution_epoch == 0
    assert untouched.available_at == 20.0


def test_sqlite_task_queue_claim_next_increments_execution_epoch_per_attempt(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    task = queue.enqueue("epoch-task", task_id="epoch-task", available_at=0.0, max_retries=3)

    first_claim = queue.claim_next(now=1.0)
    assert first_claim is not None
    assert first_claim.id == task.id
    assert first_claim.execution_epoch == 1

    retried = queue.fail(task.id, "temporary failure", failed_at=2.0, retry_delay_seconds=0.0)
    assert retried.status == "pending"
    assert retried.execution_epoch == 1

    second_claim = queue.claim_next(now=3.0)

    assert second_claim is not None
    assert second_claim.id == task.id
    assert second_claim.execution_epoch == 2


def test_sqlite_task_queue_rejects_stale_execution_epoch_updates(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    task = queue.enqueue("stale-epoch-task", task_id="stale-epoch-task", available_at=0.0, max_retries=3)

    first_claim = queue.claim_next(now=1.0)
    assert first_claim is not None
    queue.fail(task.id, "retry me", failed_at=2.0, retry_delay_seconds=0.0, execution_epoch=first_claim.execution_epoch)

    second_claim = queue.claim_next(now=3.0)
    assert second_claim is not None
    assert second_claim.execution_epoch == 2

    with pytest.raises(ValueError, match="execution epoch 1"):
        queue.touch_running_task(task.id, updated_at=4.0, execution_epoch=1)

    with pytest.raises(ValueError, match="execution epoch 1"):
        queue.complete(task.id, completed_at=5.0, execution_epoch=1)

    running = queue.get_task(task.id)
    assert running.status == "running"
    assert running.execution_epoch == 2


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
    assert second_claim.last_error is None


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


def test_database_manager_waits_for_short_write_lock_release(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    locker = sqlite3.connect(str(db_manager.db_path), check_same_thread=False, timeout=0.01)
    locker.execute("PRAGMA journal_mode=WAL;")
    locker.execute("BEGIN IMMEDIATE")

    def release_lock() -> None:
        time.sleep(0.1)
        locker.commit()
        locker.close()

    releaser = threading.Thread(target=release_lock)
    releaser.start()
    started_at = time.perf_counter()
    try:
        task = queue.enqueue("delayed-write", task_id="delayed-write")
    finally:
        releaser.join()
    elapsed_seconds = time.perf_counter() - started_at

    assert task.id == "delayed-write"
    assert elapsed_seconds >= 0.08
    assert elapsed_seconds < (SQLITE_BUSY_TIMEOUT_MILLISECONDS / 1000.0)


def test_sqlite_task_queue_extend_running_task_data_int_list_preserves_concurrent_updates(
    db_manager,
    monkeypatch,
) -> None:
    queue = SQLiteTaskQueue(db_manager)
    task = queue.enqueue("concurrent-data-merge", task_id="concurrent-data-merge")
    other_queue = SQLiteTaskQueue(db_manager)

    first_decode_started = threading.Event()
    release_first_decode = threading.Event()
    original_decode = tasks_module._decode_json_object
    decode_calls = 0
    decode_lock = threading.Lock()

    def controlled_decode(value):
        nonlocal decode_calls
        with decode_lock:
            decode_calls += 1
            call_number = decode_calls
        if call_number == 1:
            first_decode_started.set()
            assert release_first_decode.wait(timeout=5.0)
        return original_decode(value)

    monkeypatch.setattr(tasks_module, "_decode_json_object", controlled_decode)

    errors: list[BaseException] = []

    def write_values(target_queue: SQLiteTaskQueue, values: list[int]) -> None:
        try:
            target_queue.extend_running_task_data_int_list(
                task.id,
                field_name="ingest_handled_entry_ids",
                values=values,
            )
        except BaseException as exc:  # pragma: no cover - failure path captured in assertion
            errors.append(exc)

    first_writer = threading.Thread(target=write_values, args=(queue, [101]), name="first-writer")
    second_writer = threading.Thread(target=write_values, args=(other_queue, [202]), name="second-writer")

    first_writer.start()
    assert first_decode_started.wait(timeout=5.0)

    second_writer.start()
    time.sleep(0.05)
    release_first_decode.set()

    first_writer.join()
    second_writer.join()

    assert errors == []
    updated = queue.get_task(task.id)
    assert updated.data["ingest_handled_entry_ids"] == [101, 202]


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


def test_sqlite_task_queue_retries_recovery_transition_after_transient_database_lock(
    db_manager,
    monkeypatch,
) -> None:
    queue = SQLiteTaskQueue(db_manager)
    task = queue.enqueue("pid-dead-task", task_id="pid-dead-task-retry", available_at=0.0)

    assert queue.claim_next(now=10.0) is not None
    queue.set_running_process(task.id, subprocess_pid=9999, request_id="req-dead", updated_at=12.0)
    monkeypatch.setattr("mcp_memory.core.tasks._is_process_alive", lambda pid: False)

    attempts = 0
    original_fail_permanently = queue.fail_permanently

    def flaky_fail_permanently(task_id: str, error: str, failed_at: float | None = None):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise sqlite3.OperationalError("database is locked")
        return original_fail_permanently(task_id, error, failed_at=failed_at)

    monkeypatch.setattr(queue, "fail_permanently", flaky_fail_permanently)

    recovered = queue.recover_abandoned_running_tasks(now=1000.0, stale_after_seconds=300.0)

    assert attempts == 2
    assert [item.id for item in recovered] == [task.id]
    failed = queue.get_task(task.id)
    assert failed.status == "failed"
    assert failed.last_error == "Provider subprocess 9999 exited unexpectedly"


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


def test_schedule_system1_ingest_uses_one_global_auto_task_across_workspaces(
    db_manager,
    monkeypatch,
) -> None:
    queue = SQLiteTaskQueue(db_manager)
    journal = System1Journal(db_manager)

    first_now = 100.0
    monkeypatch.setattr("mcp_memory.core.journal.time.time", lambda: first_now)
    journal.record("note a", workspace_id="workspace-a")

    first = schedule_system1_ingest(queue, journal, "workspace-a", now=first_now)

    assert first is not None
    assert first.created is True
    assert first.task.workspace_id is None
    assert first.task.data["workspace_id"] == "workspace-a"
    assert first.task.data["journal_workspace_id"] == "*"

    second_now = 160.0
    monkeypatch.setattr("mcp_memory.core.journal.time.time", lambda: second_now)
    journal.record("note b", workspace_id="workspace-b")

    second = schedule_system1_ingest(queue, journal, "workspace-b", now=second_now)

    assert second is not None
    assert second.created is False
    assert second.task.id == first.task.id
    assert second.task.workspace_id is None
    assert second.task.data["workspace_id"] == "workspace-b"
    assert second.task.data["journal_workspace_id"] == "*"
    assert second.task.available_at == 3760.0


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
async def test_runtime_task_worker_pauses_idle_autonomous_recurring_maintenance(db_manager, monkeypatch) -> None:
    queue = SQLiteTaskQueue(db_manager)
    journal = System1Journal(db_manager)
    seen: list[str] = []

    monkeypatch.setattr("mcp_memory.core.journal.time.time", lambda: 100.0)
    journal.record("old maintenance anchor", workspace_id="workspace-a")

    task = queue.enqueue(
        CURATOR_TASK_NAME,
        workspace_id=None,
        data={
            "workspace_id": None,
            "trigger": "recurring_follow_up",
            "interval_seconds": RECURRING_TASK_INTERVAL_SECONDS[CURATOR_TASK_NAME],
        },
        available_at=0.0,
        task_id="idle-curator",
    )
    claimed = queue.claim_next(now=100.0 + AUTONOMOUS_MAINTENANCE_IDLE_THRESHOLD_SECONDS + 1.0)
    assert claimed is not None

    monkeypatch.setattr("mcp_memory.core.maintenance_idle.time.time", lambda: 100.0 + AUTONOMOUS_MAINTENANCE_IDLE_THRESHOLD_SECONDS + 1.0)
    worker = RuntimeTaskWorker(
        ApplicationContext(db_manager=db_manager, task_queue=queue, journal=journal),
        handlers={CURATOR_TASK_NAME: lambda ctx, task: seen.append(task.id)},
        poll_interval_seconds=0.01,
    )

    await worker._process_task(claimed)  # noqa: SLF001

    completed = queue.get_task(task.id)
    run = queue.list_task_runs(task_id=task.id)[0]

    assert completed.status == "completed"
    assert seen == []
    assert queue.find_open_task(CURATOR_TASK_NAME, None) is None
    assert run.result["paused_for_idle"] is True
    assert run.result["last_thought_at"] == pytest.approx(100.0)
    assert run.result["idle_seconds"] == pytest.approx(AUTONOMOUS_MAINTENANCE_IDLE_THRESHOLD_SECONDS + 1.0)


@pytest.mark.asyncio
async def test_runtime_task_worker_does_not_pause_manual_maintenance_runs_when_idle(db_manager, monkeypatch) -> None:
    queue = SQLiteTaskQueue(db_manager)
    journal = System1Journal(db_manager)
    seen: list[str] = []

    monkeypatch.setattr("mcp_memory.core.journal.time.time", lambda: 100.0)
    journal.record("old maintenance anchor", workspace_id="workspace-a")

    task = queue.enqueue(
        CURATOR_TASK_NAME,
        workspace_id=None,
        data={"workspace_id": None},
        available_at=0.0,
        task_id="manual-curator",
    )
    claimed = queue.claim_next(now=100.0 + AUTONOMOUS_MAINTENANCE_IDLE_THRESHOLD_SECONDS + 1.0)
    assert claimed is not None

    monkeypatch.setattr("mcp_memory.core.maintenance_idle.time.time", lambda: 100.0 + AUTONOMOUS_MAINTENANCE_IDLE_THRESHOLD_SECONDS + 1.0)
    worker = RuntimeTaskWorker(
        ApplicationContext(db_manager=db_manager, task_queue=queue, journal=journal),
        handlers={CURATOR_TASK_NAME: lambda ctx, task: seen.append(task.id)},
        poll_interval_seconds=0.01,
    )

    await worker._process_task(claimed)  # noqa: SLF001

    completed = queue.get_task(task.id)
    run = queue.list_task_runs(task_id=task.id)[0]

    assert completed.status == "completed"
    assert seen == ["manual-curator"]
    assert run.result == {}


def test_record_thought_operation_resumes_paused_recurring_maintenance(db_manager, monkeypatch) -> None:
    queue = SQLiteTaskQueue(db_manager)
    journal = System1Journal(db_manager)
    monkeypatch.setattr("mcp_memory.core.maintenance_idle.compute_recurring_jitter_seconds", lambda interval_seconds: 0.0)

    paused = queue.enqueue(
        CURATOR_TASK_NAME,
        workspace_id=None,
        data={
            "workspace_id": None,
            "trigger": "recurring_follow_up",
            "interval_seconds": RECURRING_TASK_INTERVAL_SECONDS[CURATOR_TASK_NAME],
        },
        available_at=0.0,
        task_id="paused-curator",
    )
    assert queue.claim_next(now=10.0) is not None
    queue.complete(
        paused.id,
        completed_at=20.0,
        run_result={
            "paused_for_idle": True,
            "idle_seconds": AUTONOMOUS_MAINTENANCE_IDLE_THRESHOLD_SECONDS + 5.0,
            "last_thought_at": 5.0,
        },
    )

    monkeypatch.setattr("mcp_memory.core.journal.time.time", lambda: 200.0)
    payload = RecordThoughtOperation(journal, queue, "workspace-a").execute("fresh thought")
    resumed = queue.find_open_task(CURATOR_TASK_NAME, None)

    assert resumed is not None
    assert resumed.status == "pending"
    assert resumed.available_at == pytest.approx(200.0)
    assert resumed.data["trigger"] == "recurring_resume"
    assert payload["resumed_maintenance_tasks"] == [
        {
            "id": resumed.id,
            "status": "pending",
            "task_name": CURATOR_TASK_NAME,
            "workspace_id": None,
        }
    ]


def test_resume_paused_recurring_maintenance_applies_jitter(db_manager, monkeypatch) -> None:
    queue = SQLiteTaskQueue(db_manager)

    paused = queue.enqueue(
        CURATOR_TASK_NAME,
        workspace_id=None,
        data={
            "workspace_id": None,
            "trigger": "recurring_follow_up",
            "interval_seconds": RECURRING_TASK_INTERVAL_SECONDS[CURATOR_TASK_NAME],
        },
        available_at=0.0,
        task_id="paused-curator-jitter",
    )
    assert queue.claim_next(now=10.0) is not None
    queue.complete(
        paused.id,
        completed_at=20.0,
        run_result={
            "paused_for_idle": True,
            "idle_seconds": AUTONOMOUS_MAINTENANCE_IDLE_THRESHOLD_SECONDS + 5.0,
            "last_thought_at": 5.0,
            "interval_seconds": RECURRING_TASK_INTERVAL_SECONDS[CURATOR_TASK_NAME],
        },
    )

    monkeypatch.setattr("mcp_memory.core.maintenance_idle.compute_recurring_jitter_seconds", lambda interval_seconds: 12.0)

    resumed = resume_paused_recurring_maintenance(queue, now=200.0)[0]

    assert resumed.available_at == pytest.approx(212.0)
    assert resumed.data["trigger"] == "recurring_resume"
    assert resumed.data["jitter_seconds"] == pytest.approx(12.0)


@pytest.mark.asyncio
async def test_runtime_task_worker_recurring_follow_up_applies_jitter(db_manager, monkeypatch) -> None:
    queue = SQLiteTaskQueue(db_manager)
    ctx = ApplicationContext(db_manager=db_manager, task_queue=queue)
    monkeypatch.setattr("mcp_memory.core.task_worker.compute_recurring_jitter_seconds", lambda interval_seconds: 18.0)

    task = queue.enqueue(
        CURATOR_TASK_NAME,
        workspace_id=None,
        data={"workspace_id": None, "trigger": "recurring_schedule", "interval_seconds": RECURRING_TASK_INTERVAL_SECONDS[CURATOR_TASK_NAME]},
        available_at=0.0,
        task_id="curator-follow-up-jitter",
    )
    claimed = queue.claim_next(now=100.0)
    assert claimed is not None

    worker = RuntimeTaskWorker(ctx, handlers={CURATOR_TASK_NAME: lambda ctx, task: None}, poll_interval_seconds=0.01)
    completed_task = queue.complete(task.id, completed_at=140.0, run_result={"mutations": 1})

    await worker._schedule_follow_up(claimed, completed_task)  # noqa: SLF001

    follow_up = queue.find_open_task(CURATOR_TASK_NAME, None)
    assert follow_up is not None
    assert follow_up.id != task.id
    assert follow_up.available_at == pytest.approx(140.0 + RECURRING_TASK_INTERVAL_SECONDS[CURATOR_TASK_NAME] + 18.0)
    assert follow_up.data["trigger"] == "recurring_follow_up"
    assert follow_up.data["jitter_seconds"] == pytest.approx(18.0)


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
        kwargs.pop("stale_after_seconds", None)
        kwargs.pop("now", None)
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
        kwargs.pop("stale_after_seconds", None)
        kwargs.pop("now", None)
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
async def test_runtime_task_worker_logs_unchanged_overdue_pending_tasks_once(db_manager, caplog) -> None:
    queue = SQLiteTaskQueue(db_manager)
    ctx = ApplicationContext(db_manager=db_manager, task_queue=queue, workspace_id="workspace-a")
    overdue = queue.enqueue(
        "overdue-pending-task",
        workspace_id="workspace-a",
        available_at=0.0,
        task_id="overdue-pending-task",
    )
    worker = RuntimeTaskWorker(
        ctx,
        handlers={"overdue-pending-task": lambda context, queued_task: None},
        poll_interval_seconds=0.01,
        abandoned_task_stale_after_seconds=60.0,
    )

    with caplog.at_level("INFO"):
        await worker._run_reconciliation_pass(now=120.0, reason="periodic")  # noqa: SLF001
        await worker._run_reconciliation_pass(now=121.0, reason="periodic")  # noqa: SLF001

    matching_messages = [record for record in caplog.records if record.message == "Runtime reconciliation pass completed"]

    assert overdue.id == "overdue-pending-task"
    assert len(matching_messages) == 1
    assert matching_messages[0].__dict__["overdue_pending_count"] == 1
    assert matching_messages[0].__dict__["overdue_pending_task_ids"] == [overdue.id]


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
        kwargs.pop("stale_after_seconds", None)
        kwargs.pop("now", None)
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
async def test_runtime_task_worker_uses_attempt_heartbeat_to_keep_running_task_alive(db_manager, monkeypatch) -> None:
    queue = SQLiteTaskQueue(db_manager)
    attempt_repository = TaskExecutionAttemptRepository(db_manager, workspace_id="workspace-a")
    task = queue.enqueue(
        "heartbeat-task",
        workspace_id="workspace-a",
        available_at=0.0,
        task_id="attempt-heartbeat-task",
    )
    claimed = queue.claim_next(now=1.0, workspace_id="workspace-a")
    assert claimed is not None

    attempt_repository.start_attempt(
        task_id=task.id,
        execution_epoch=claimed.execution_epoch,
        task_name=task.task_name,
        request_id="req-attempt-heartbeat",
        subprocess_pid=9999,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
        started_at=1.0,
    )
    attempt_repository.heartbeat_attempt(
        task_id=task.id,
        execution_epoch=claimed.execution_epoch,
        heartbeat_at=950.0,
        subprocess_pid=9999,
    )

    ctx = ApplicationContext(
        db_manager=db_manager,
        task_queue=queue,
        task_execution_attempts=attempt_repository,
        workspace_id="workspace-a",
    )
    worker = RuntimeTaskWorker(ctx, handlers={"heartbeat-task": lambda context, queued_task: None})

    monkeypatch.setattr("mcp_memory.core.tasks._is_process_alive", lambda pid: False)

    await worker._run_reconciliation_pass(now=1000.0, reason="periodic")  # noqa: SLF001

    running = queue.get_task(task.id)
    attempt = attempt_repository.get_attempt(task_id=task.id, execution_epoch=claimed.execution_epoch)
    assert running.status == "running"
    assert attempt.status == "running"
    assert attempt.last_heartbeat_at == pytest.approx(950.0)


@pytest.mark.asyncio
async def test_runtime_task_worker_reconciles_attempt_for_recovered_dead_process(db_manager, monkeypatch) -> None:
    queue = SQLiteTaskQueue(db_manager)
    attempt_repository = TaskExecutionAttemptRepository(db_manager, workspace_id="workspace-a")
    task = queue.enqueue(
        "orphaned-attempt-task",
        workspace_id="workspace-a",
        available_at=0.0,
        task_id="orphaned-attempt-task",
    )
    claimed = queue.claim_next(now=1.0, workspace_id="workspace-a")
    assert claimed is not None

    attempt_repository.start_attempt(
        task_id=task.id,
        execution_epoch=claimed.execution_epoch,
        task_name=task.task_name,
        request_id="req-orphaned-attempt",
        subprocess_pid=9999,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
        started_at=1.0,
    )

    ctx = ApplicationContext(
        db_manager=db_manager,
        task_queue=queue,
        task_execution_attempts=attempt_repository,
        workspace_id="workspace-a",
    )
    worker = RuntimeTaskWorker(ctx, handlers={"orphaned-attempt-task": lambda context, queued_task: None})

    monkeypatch.setattr("mcp_memory.core.tasks._is_process_alive", lambda pid: False)

    await worker._run_reconciliation_pass(now=1000.0, reason="periodic")  # noqa: SLF001

    failed = queue.get_task(task.id)
    attempt = attempt_repository.get_attempt(task_id=task.id, execution_epoch=claimed.execution_epoch)
    assert failed.status == "failed"
    assert failed.last_error == "Provider subprocess 9999 exited unexpectedly"
    assert attempt.status == "error"
    assert attempt.completed_at == pytest.approx(1000.0)
    assert attempt.termination_reason == "task_failed"


@pytest.mark.asyncio
async def test_runtime_task_worker_cancels_stalled_task_after_reconciliation_and_drains_queue(
    db_manager,
    monkeypatch,
) -> None:
    queue = SQLiteTaskQueue(db_manager)
    ctx = ApplicationContext(db_manager=db_manager, task_queue=queue, workspace_id="workspace-a")
    stalled_task = queue.enqueue(
        SYSTEM1_INGEST_TASK_NAME,
        workspace_id="workspace-a",
        available_at=0.0,
        task_id="stalled-ingest-task",
    )
    follow_up_task = queue.enqueue(
        "after-task",
        workspace_id="workspace-a",
        available_at=0.0,
        task_id="after-stalled-ingest-task",
    )
    started = asyncio.Event()
    cancelled = asyncio.Event()
    parked = asyncio.Event()
    completed: list[str] = []

    async def stalled_handler(ctx: ApplicationContext, task) -> None:
        queue.set_running_process(task.id, subprocess_pid=9999, request_id="req-stalled-ingest", updated_at=2.0)
        started.set()
        try:
            await parked.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    def follow_up_handler(ctx: ApplicationContext, task) -> None:
        completed.append(task.id)

    monkeypatch.setattr("mcp_memory.core.tasks._is_process_alive", lambda pid: False)
    worker = RuntimeTaskWorker(
        ctx,
        handlers={
            SYSTEM1_INGEST_TASK_NAME: stalled_handler,
            "after-task": follow_up_handler,
        },
        poll_interval_seconds=0.01,
        abandoned_recovery_interval_seconds=0.01,
        abandoned_task_stale_after_seconds=0.0,
    )

    await worker.start()
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)
        for _ in range(100):
            stalled_status = queue.get_task(stalled_task.id).status
            follow_up_status = queue.get_task(follow_up_task.id).status
            if stalled_status == "failed" and follow_up_status == "completed":
                break
            await asyncio.sleep(0.01)
    finally:
        parked.set()
        await worker.stop(0.05)

    assert cancelled.is_set()
    assert queue.get_task(stalled_task.id).status == "failed"
    assert queue.get_task(stalled_task.id).last_error == "Provider subprocess 9999 exited unexpectedly"
    assert queue.get_task(follow_up_task.id).status == "completed"
    assert completed == [follow_up_task.id]


@pytest.mark.asyncio
async def test_runtime_task_worker_periodic_reconciliation_finalizes_stale_running_conversations(
    db_manager,
) -> None:
    queue = SQLiteTaskQueue(db_manager)
    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    ctx = ApplicationContext(db_manager=db_manager, task_queue=queue, workspace_id="workspace-a")
    task = queue.enqueue(
        SYSTEM1_INGEST_TASK_NAME,
        workspace_id="workspace-a",
        available_at=0.0,
        task_id="stale-running-conversation-task",
    )
    assert queue.claim_next(now=1.0, workspace_id="workspace-a") is not None
    failed_task = queue.fail_permanently(task.id, "Provider subprocess exited unexpectedly", failed_at=2.0)
    repository.record_conversation(
        request_id="req-stale-running-conversation",
        attempt=1,
        task_name=SYSTEM1_INGEST_TASK_NAME,
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
        handlers={SYSTEM1_INGEST_TASK_NAME: lambda context, queued_task: None},
        poll_interval_seconds=0.01,
        abandoned_recovery_interval_seconds=0.01,
        abandoned_task_stale_after_seconds=0.0,
    )

    await worker.start()
    try:
        for _ in range(100):
            conversation = repository.get_conversation("req-stale-running-conversation")[0]
            if conversation.status == "error":
                break
            await asyncio.sleep(0.01)
    finally:
        await worker.stop(0.05)

    conversation = repository.get_conversation("req-stale-running-conversation")[0]

    assert failed_task.status == "failed"
    assert conversation.status == "error"
    assert conversation.error_text == "Provider subprocess exited unexpectedly"
    assert conversation.reason_category == "recovery"
    assert conversation.reason_code == "task_failed"


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
async def test_runtime_task_worker_requests_shutdown_cancellation_for_running_tasks(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    ctx = ApplicationContext(db_manager=db_manager, task_queue=queue, workspace_id="workspace-a")
    task = queue.enqueue(
        "shutdown-me",
        workspace_id="workspace-a",
        available_at=0.0,
        task_id="shutdown-me",
    )

    started = asyncio.Event()
    released = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocking_handler(_ctx: ApplicationContext, queued_task: TaskRecord) -> None:
        started.set()
        try:
            await released.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    worker = RuntimeTaskWorker(
        ctx,
        handlers={"shutdown-me": blocking_handler},
        poll_interval_seconds=0.01,
        abandoned_recovery_interval_seconds=30.0,
    )

    await worker.start()
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)
        await worker.stop(0.05)
    finally:
        released.set()

    task_after_stop = queue.get_task(task.id)

    assert cancelled.is_set()
    assert task_after_stop.status == "cancelled"
    assert task_after_stop.cancellation_reason == "daemon_shutdown"
    assert task_after_stop.cancelled_by == "daemon"


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
