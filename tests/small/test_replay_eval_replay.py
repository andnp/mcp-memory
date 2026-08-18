"""Contract coverage for replaying labeled queries through a variant."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from benchmarks.replay_eval.labels import RelevanceLabel
from benchmarks.replay_eval.replay import installed_variant, replay_labels
from benchmarks.replay_eval.variants import Variant, sigmoid
from mcp_memory.core import search_ranking


class RecordingSearch:
    """Return canned rankings and record which queries were issued."""

    def __init__(self, rankings: dict[str, list[str]]) -> None:
        self._rankings = rankings
        self.calls: list[tuple[str, str | None, int]] = []

    def __call__(self, query: str, workspace_id: str | None, limit: int) -> Sequence[str]:
        self.calls.append((query, workspace_id, limit))
        return self._rankings.get(query, [])[:limit]


def _label(query: str, memory_id: str, workspace_id: str | None = "w1") -> RelevanceLabel:
    return RelevanceLabel(
        query=query,
        workspace_id=workspace_id,
        memory_id=memory_id,
        logged_rank=1,
        observed_at=1_700_000_000.0,
    )


VARIANT = sigmoid()


def test_a_labeled_memory_is_located_by_its_position() -> None:
    """Record where the variant placed the memory that was read."""
    search = RecordingSearch({"q": ["a", "b", "target"]})

    observations = replay_labels([_label("q", "target")], VARIANT, search)

    assert observations[0].retrieved_rank == 3
    assert observations[0].variant == "sigmoid"


def test_a_memory_outside_the_results_is_recorded_as_unretrieved() -> None:
    """Distinguish a missing memory from a poorly ranked one."""
    search = RecordingSearch({"q": ["a", "b"]})

    observations = replay_labels([_label("q", "target")], VARIANT, search)

    assert observations[0].retrieved_rank is None


def test_labels_sharing_a_query_reuse_one_search() -> None:
    """Answer every label for a query from a single ranking."""
    search = RecordingSearch({"q": ["first", "second"]})
    labels = [_label("q", "first"), _label("q", "second")]

    observations = replay_labels(labels, VARIANT, search)

    assert len(search.calls) == 1
    assert sorted(o.retrieved_rank or 0 for o in observations) == [1, 2]


def test_the_same_query_in_different_workspaces_is_searched_separately() -> None:
    """Keep workspace scope part of the query identity."""
    search = RecordingSearch({"q": ["a"]})
    labels = [_label("q", "a", "w1"), _label("q", "a", "w2")]

    replay_labels(labels, VARIANT, search)

    assert len(search.calls) == 2
    assert {call[1] for call in search.calls} == {"w1", "w2"}


def test_every_label_produces_an_observation() -> None:
    """Never drop a label, so coverage cannot silently shrink."""
    search = RecordingSearch({"q1": ["a"], "q2": []})
    labels = [_label("q1", "a"), _label("q2", "b"), _label("q1", "missing")]

    assert len(replay_labels(labels, VARIANT, search)) == 3


def test_the_requested_limit_is_passed_to_the_search() -> None:
    """Bound the results the ranker is asked to return."""
    search = RecordingSearch({"q": ["a"]})

    replay_labels([_label("q", "a")], VARIANT, search, limit=25)

    assert search.calls[0][2] == 25


def test_progress_is_reported_per_distinct_query() -> None:
    """Let a long run report progress against the work actually done."""
    search = RecordingSearch({"q1": ["a"], "q2": ["b"]})
    seen: list[tuple[int, int]] = []

    replay_labels(
        [_label("q1", "a"), _label("q2", "b")],
        VARIANT,
        search,
        on_progress=lambda done, total: seen.append((done, total)),
    )

    assert seen == [(1, 2), (2, 2)]


def test_replay_rejects_a_non_positive_limit() -> None:
    """Reject a limit that cannot return a ranking."""
    with pytest.raises(ValueError, match="limit must be positive"):
        replay_labels([], VARIANT, RecordingSearch({}), limit=0)


def test_the_variant_calibration_is_active_during_replay() -> None:
    """Confirm the variant actually replaces the shipped curve."""
    marker = Variant(name="marker", description="constant", calibrate=lambda _: 0.5)
    observed: list[float] = []

    def search(query: str, workspace_id: str | None, limit: int) -> Sequence[str]:
        observed.append(
            search_ranking._calibrate_score(0.02, threshold=0.035, steepness=150.0)
        )
        return ["a"]

    replay_labels([_label("q", "a")], marker, search)

    assert observed == [0.5]


def test_the_shipped_curve_is_restored_after_replay() -> None:
    """Leave the process as it was found, even for later variants."""
    original = search_ranking._calibrate_score

    replay_labels([_label("q", "a")], VARIANT, RecordingSearch({"q": ["a"]}))

    assert search_ranking._calibrate_score is original


def test_the_shipped_curve_is_restored_after_a_failure() -> None:
    """Never leak a variant into the process when a search raises."""
    original = search_ranking._calibrate_score

    def failing(query: str, workspace_id: str | None, limit: int) -> Sequence[str]:
        raise RuntimeError("search exploded")

    with pytest.raises(RuntimeError, match="search exploded"):
        replay_labels([_label("q", "a")], VARIANT, failing)

    assert search_ranking._calibrate_score is original


def test_installed_variant_restores_on_exit() -> None:
    """Scope the swap to the block that needs it."""
    original = search_ranking._calibrate_score

    with installed_variant(VARIANT):
        assert search_ranking._calibrate_score is not original

    assert search_ranking._calibrate_score is original
