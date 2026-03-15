import pytest


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