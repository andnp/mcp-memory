import pytest


pytestmark = pytest.mark.small


def test_system1_journal_record_and_lifecycle(system1_journal) -> None:
    first = system1_journal.record("  remember to fix imports  ", workspace_id="workspace-a")
    second = system1_journal.record("write the first pytest suite", workspace_id="workspace-b")

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