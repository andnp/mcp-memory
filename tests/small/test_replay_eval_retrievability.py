"""Contract coverage for excluding labels the snapshot cannot satisfy."""

from __future__ import annotations

from benchmarks.replay_eval.labels import RelevanceLabel
from benchmarks.replay_eval.retrievability import CorpusReach, partition_by_reach


def _label(memory_id: str, workspace_id: str | None = None) -> RelevanceLabel:
    return RelevanceLabel(
        query="q",
        workspace_id=workspace_id,
        memory_id=memory_id,
        logged_rank=1,
        observed_at=1.0,
    )


def _reach(active: set[str], workspaces: dict[str, set[str]]) -> CorpusReach:
    return CorpusReach(active_ids=active, workspaces=workspaces)


def test_an_active_in_workspace_label_is_kept() -> None:
    """Replay every label a search could still return."""
    reach = _reach({"m"}, {"m": {"w1"}})

    assert partition_by_reach([_label("m", "w1")], reach).retrievable == (
        _label("m", "w1"),
    )


def test_an_archived_memory_is_excluded_as_inactive() -> None:
    """Stop counting a memory the corpus no longer serves as a ranking miss."""
    result = partition_by_reach([_label("m", "w1")], _reach(set(), {"m": {"w1"}}))

    assert result.retrievable == ()
    assert result.inactive == 1
    assert result.out_of_workspace == 0


def test_a_memory_outside_the_labeled_workspace_is_excluded() -> None:
    """Respect the workspace filter the replayed search will apply."""
    result = partition_by_reach([_label("m", "w1")], _reach({"m"}, {"m": {"w2"}}))

    assert result.retrievable == ()
    assert result.inactive == 0
    assert result.out_of_workspace == 1


def test_a_global_label_ignores_workspace_membership() -> None:
    """Keep global searches, which apply no workspace filter."""
    result = partition_by_reach([_label("m", None)], _reach({"m"}, {}))

    assert len(result.retrievable) == 1


def test_an_inactive_memory_is_not_also_counted_out_of_workspace() -> None:
    """Attribute each exclusion to exactly one cause so the counts sum."""
    result = partition_by_reach([_label("m", "w1")], _reach(set(), {}))

    assert (result.inactive, result.out_of_workspace) == (1, 0)


def test_the_partition_reports_what_it_excluded() -> None:
    """State the exclusions in the artifact rather than silently shrinking it."""
    labels = [_label("keep", "w1"), _label("gone", "w1"), _label("moved", "w1")]
    reach = _reach({"keep", "moved"}, {"keep": {"w1"}, "moved": {"w2"}})

    assert partition_by_reach(labels, reach).to_mapping() == {
        "retrievable": 1,
        "excluded_inactive": 1,
        "excluded_out_of_workspace": 1,
    }


def test_an_empty_label_set_partitions_to_nothing() -> None:
    """Handle the degenerate input without special-casing at the call site."""
    result = partition_by_reach([], _reach(set(), {}))

    assert result.retrievable == ()
    assert result.inactive == 0
