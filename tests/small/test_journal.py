import pytest

from mcp_memory.core.journal import System1Journal


pytestmark = pytest.mark.small


def test_system1_journal_record_and_lifecycle(system1_journal) -> None:
    first = system1_journal.record("  remember to fix imports  ", workspace_id="workspace-a")
    system1_journal.record("write the first pytest suite", workspace_id="workspace-b")

    pending = system1_journal.get_pending()
    assert [entry.content for entry in pending] == [
        "remember to fix imports",
        "write the first pytest suite",
    ]
    assert [entry.workspace_id for entry in pending] == ["workspace-a", "workspace-b"]

    assert system1_journal.mark_processed([first.id]) == 1
    assert system1_journal.mark_archived([first.id]) == 1

    counts = system1_journal.count_by_status()
    assert counts["archived"] == 1
    assert counts["pending"] == 1


def test_system1_journal_rejects_empty_content(system1_journal) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        system1_journal.record("   ")


def test_system1_journal_workspace_queries_are_scoped(system1_journal) -> None:
    first = system1_journal.record("first workspace-a thought", workspace_id="workspace-a")
    system1_journal.record("workspace-b thought", workspace_id="workspace-b")
    second = system1_journal.record("second workspace-a thought", workspace_id="workspace-a")

    workspace_a_pending = system1_journal.get_pending(workspace_id="workspace-a")

    assert [entry.id for entry in workspace_a_pending] == [first.id, second.id]
    assert system1_journal.count_by_status(workspace_id="workspace-a") == {"pending": 2}
    assert system1_journal.get_oldest_pending_timestamp(workspace_id="workspace-a") == first.timestamp


def test_system1_journal_claims_release_and_delete_are_task_scoped(system1_journal) -> None:
    first = system1_journal.record("first claimed thought", workspace_id="workspace-a")
    second = system1_journal.record("second claimed thought", workspace_id="workspace-a")
    third = system1_journal.record("third pending thought", workspace_id="workspace-a")

    claimed = system1_journal.claim_pending(task_id="ingest-a", limit=2, workspace_id="workspace-a", claimed_at=10.0)

    assert [entry.id for entry in claimed] == [first.id, second.id]
    assert [entry.status for entry in claimed] == ["claimed", "claimed"]
    assert [entry.id for entry in system1_journal.get_pending(workspace_id="workspace-a")] == [third.id]
    assert system1_journal.count_by_status(workspace_id="workspace-a") == {"claimed": 2, "pending": 1}

    released_ids = system1_journal.release_claims("ingest-a")

    assert released_ids == [first.id, second.id]
    assert [entry.id for entry in system1_journal.get_pending(workspace_id="workspace-a")] == [first.id, second.id, third.id]

    claimed_again = system1_journal.claim_pending(task_id="ingest-b", limit=2, workspace_id="workspace-a", claimed_at=11.0)
    deleted_ids = system1_journal.delete_claims("ingest-b")

    assert [entry.id for entry in claimed_again] == [first.id, second.id]
    assert deleted_ids == [first.id, second.id]
    assert [entry.id for entry in system1_journal.get_pending(workspace_id="workspace-a")] == [third.id]


def test_system1_journal_release_orphaned_claims_only_releases_non_running_tasks(db_manager) -> None:
    from mcp_memory.core.tasks import SQLiteTaskQueue

    queue = SQLiteTaskQueue(db_manager)
    journal = System1Journal(db_manager)

    running_task = queue.enqueue("ingest-system1", available_at=0.0, task_id="running-ingest")
    orphaned_task = queue.enqueue("ingest-system1", available_at=0.0, task_id="orphaned-ingest")
    assert queue.claim_next(now=1.0) is not None
    assert queue.claim_next(now=2.0) is not None
    queue.fail_permanently(orphaned_task.id, "boom", failed_at=3.0)

    running_entry = journal.record("claimed by running task", workspace_id="workspace-a")
    orphaned_entry = journal.record("claimed by failed task", workspace_id="workspace-a")
    journal.claim_pending(task_id=running_task.id, limit=1, workspace_id="workspace-a", claimed_at=4.0)
    journal.claim_pending(task_id=orphaned_task.id, limit=1, workspace_id="workspace-a", claimed_at=5.0)

    released_ids = journal.release_orphaned_claims()

    assert released_ids == [orphaned_entry.id]
    assert [entry.id for entry in journal.get_pending(workspace_id="workspace-a")] == [orphaned_entry.id]
    assert journal.count_by_status(workspace_id="workspace-a") == {"claimed": 1, "pending": 1}
    assert running_entry.id not in released_ids
