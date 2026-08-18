"""Contract coverage for replayed placements of labeled memories."""

from __future__ import annotations

import pytest

from benchmarks.replay_eval.labels import RelevanceLabel
from benchmarks.replay_eval.observations import ReplayObservation

LABEL = RelevanceLabel(
    query="how does ranking work",
    workspace_id="workspace-1",
    memory_id="memory-1",
    logged_rank=4,
    observed_at=1_700_000_000.0,
)


def _observation(retrieved_rank: int | None, variant: str = "baseline") -> ReplayObservation:
    return ReplayObservation(label=LABEL, variant=variant, retrieved_rank=retrieved_rank)


@pytest.mark.parametrize(
    ("rank", "expected"),
    [(1, 1.0), (2, 0.5), (4, 0.25), (10, 0.1)],
)
def test_reciprocal_rank_rewards_higher_placements(rank: int, expected: float) -> None:
    """Score a labeled memory by where the variant placed it."""
    assert _observation(rank).reciprocal_rank == pytest.approx(expected)


def test_unretrieved_memory_scores_zero() -> None:
    """Score a memory the variant never surfaced as no credit at all."""
    assert _observation(None).reciprocal_rank == 0.0


def test_hit_at_includes_the_boundary_rank() -> None:
    """Count a placement exactly at k as a hit."""
    observation = _observation(5)

    assert observation.hit_at(5) is True
    assert observation.hit_at(4) is False


def test_unretrieved_memory_never_hits() -> None:
    """Never credit a hit for a memory that was not surfaced."""
    assert _observation(None).hit_at(100) is False


def test_hit_at_rejects_a_non_positive_cutoff() -> None:
    """Reject a cutoff that cannot describe a one-based placement."""
    with pytest.raises(ValueError, match="k must be positive"):
        _observation(1).hit_at(0)


@pytest.mark.parametrize("rank", [0, -3])
def test_observation_rejects_a_rank_below_one(rank: int) -> None:
    """Reject placements that replay reports as one-based."""
    with pytest.raises(ValueError, match="one-based"):
        _observation(rank)


def test_observation_requires_a_variant_name() -> None:
    """Refuse an observation that cannot be attributed to a variant."""
    with pytest.raises(ValueError, match="variant"):
        _observation(1, variant=" ")
