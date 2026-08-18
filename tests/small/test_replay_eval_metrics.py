"""Contract coverage for aggregating replay observations."""

from __future__ import annotations

import pytest

from benchmarks.replay_eval.labels import RelevanceLabel
from benchmarks.replay_eval.metrics import summarize
from benchmarks.replay_eval.observations import ReplayObservation


def _observations(
    ranks: list[int | None], variant: str = "baseline"
) -> list[ReplayObservation]:
    return [
        ReplayObservation(
            label=RelevanceLabel(
                query=f"query {index}",
                workspace_id=None,
                memory_id=f"memory-{index}",
                logged_rank=1,
                observed_at=1_700_000_000.0,
            ),
            variant=variant,
            retrieved_rank=rank,
        )
        for index, rank in enumerate(ranks)
    ]


def test_mean_reciprocal_rank_averages_placements() -> None:
    """Average reciprocal ranks over every labeled query."""
    metrics = summarize("baseline", _observations([1, 2, 4]))

    assert metrics.mean_reciprocal_rank == pytest.approx((1.0 + 0.5 + 0.25) / 3)
    assert metrics.query_count == 3


def test_unretrieved_queries_lower_the_score_without_being_dropped() -> None:
    """Count an unretrieved memory as a scored zero, not a missing query."""
    metrics = summarize("baseline", _observations([1, None]))

    assert metrics.mean_reciprocal_rank == pytest.approx(0.5)
    assert metrics.query_count == 2
    assert metrics.unretrieved_rate == pytest.approx(0.5)


def test_hit_rates_are_reported_per_cutoff() -> None:
    """Report the share of labeled memories reaching each cutoff."""
    metrics = summarize("baseline", _observations([1, 3, 7, None]))

    assert metrics.hit_at[1] == pytest.approx(0.25)
    assert metrics.hit_at[3] == pytest.approx(0.5)
    assert metrics.hit_at[10] == pytest.approx(0.75)


def test_custom_cutoffs_replace_the_defaults() -> None:
    """Let a caller ask for the cutoffs its report needs."""
    metrics = summarize("baseline", _observations([2]), cutoffs=(2,))

    assert set(metrics.hit_at) == {2}


def test_summarize_rejects_an_empty_observation_set() -> None:
    """Refuse to report a mean over no queries."""
    with pytest.raises(ValueError, match="at least one observation"):
        summarize("baseline", [])


def test_summarize_rejects_observations_from_another_variant() -> None:
    """Refuse to attribute another variant's placements to this one."""
    mixed = _observations([1]) + _observations([2], variant="challenger")

    with pytest.raises(ValueError, match="other variants"):
        summarize("baseline", mixed)


def test_serialization_orders_cutoffs_for_stable_diffs() -> None:
    """Serialize deterministically so report diffs stay readable."""
    mapping = summarize("baseline", _observations([1]), cutoffs=(10, 1)).to_mapping()

    assert list(mapping["hit_at"]) == ["1", "10"]  # type: ignore[call-overload]
    assert mapping["variant"] == "baseline"
