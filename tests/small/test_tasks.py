from __future__ import annotations

import pytest

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


def test_sqlite_task_queue_rejects_completion_for_non_running_task(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    task = queue.enqueue("idle-task", task_id="idle-task")

    with pytest.raises(ValueError, match="not running"):
        queue.complete(task.id)