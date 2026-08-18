"""Contract coverage for implicit relevance labels."""

from __future__ import annotations

import pytest

from benchmarks.replay_eval.labels import RelevanceLabel


def _label(**overrides: object) -> RelevanceLabel:
    fields: dict[str, object] = {
        "query": "how does ranking work",
        "workspace_id": "workspace-1",
        "memory_id": "memory-1",
        "logged_rank": 3,
        "observed_at": 1_700_000_000.0,
    }
    fields.update(overrides)
    return RelevanceLabel(**fields)  # type: ignore[arg-type]


def test_label_retains_the_query_and_memory_it_pairs() -> None:
    """Keep the pairing that anchors a replay comparison."""
    label = _label()

    assert label.query == "how does ranking work"
    assert label.memory_id == "memory-1"
    assert label.logged_rank == 3
    assert label.dwell_ms is None


def test_label_accepts_a_global_workspace() -> None:
    """Allow labels from searches that were not workspace-scoped."""
    assert _label(workspace_id=None).workspace_id is None


@pytest.mark.parametrize("query", ["", "   "])
def test_label_rejects_a_blank_query(query: str) -> None:
    """Refuse a label that cannot be replayed as a search."""
    with pytest.raises(ValueError, match="query"):
        _label(query=query)


def test_label_rejects_a_blank_memory_id() -> None:
    """Refuse a label with no memory to locate in replay results."""
    with pytest.raises(ValueError, match="memory_id"):
        _label(memory_id="  ")


@pytest.mark.parametrize("rank", [0, -1])
def test_label_rejects_a_rank_below_one(rank: int) -> None:
    """Reject ranks that telemetry records as one-based."""
    with pytest.raises(ValueError, match="one-based"):
        _label(logged_rank=rank)


def test_label_rejects_negative_dwell() -> None:
    """Reject a dwell time that cannot weight relevance evidence."""
    with pytest.raises(ValueError, match="dwell_ms"):
        _label(dwell_ms=-1.0)


def test_label_accepts_zero_dwell() -> None:
    """Allow an immediate close, which is weak but valid evidence."""
    assert _label(dwell_ms=0.0).dwell_ms == 0.0
