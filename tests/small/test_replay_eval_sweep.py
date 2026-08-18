"""Contract coverage for assembling a calibration-threshold sweep report."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from benchmarks.replay_eval.labels import RelevanceLabel
from benchmarks.replay_eval.sweep import sweep_thresholds


def _label(query: str, memory_id: str) -> RelevanceLabel:
    return RelevanceLabel(
        query=query,
        workspace_id="w1",
        memory_id=memory_id,
        logged_rank=1,
        observed_at=1_700_000_000.0,
    )


class ThresholdSearch:
    """A search whose results shift with the currently installed threshold.

    ``installed_variant`` patches ``_calibrate_score`` to dispatch to the
    active variant regardless of the threshold/steepness passed in, so probing
    it with a fixed score reveals which threshold is in effect.
    """

    def __call__(self, query: str, workspace_id: str | None, limit: int) -> Sequence[str]:
        import mcp_memory.core.search_ranking as search_ranking

        above_midpoint = search_ranking._calibrate_score(0.025, threshold=0, steepness=0) > 0.5
        return ["target"] if above_midpoint else ["other", "other2", "other3", "target"]


def test_the_report_covers_the_baseline_and_every_challenger() -> None:
    """Attribute a result to each threshold that was actually replayed."""
    labels = [_label(f"q{i}", "target") for i in range(30)]

    payload = sweep_thresholds(
        labels,
        baseline_threshold=0.035,
        challenger_thresholds=[0.02, 0.01],
        search=ThresholdSearch(),
        seed=1,
    )

    variants = cast("list[dict[str, Any]]", payload["variants"])
    names = {entry["variant"] for entry in variants}
    assert names == {"threshold_0.035", "threshold_0.02", "threshold_0.01"}


def test_the_baseline_is_the_first_threshold_given() -> None:
    """Compare every challenger against the threshold that was passed first."""
    labels = [_label(f"q{i}", "target") for i in range(10)]

    payload = sweep_thresholds(
        labels,
        baseline_threshold=0.035,
        challenger_thresholds=[0.02],
        search=ThresholdSearch(),
        seed=1,
    )

    assert payload["baseline"] == "threshold_0.035"


def test_a_challenger_that_ranks_the_memory_higher_shows_a_positive_delta() -> None:
    """Surface the delta the calibration retune actually cares about."""
    labels = [_label(f"q{i}", "target") for i in range(50)]

    payload = sweep_thresholds(
        labels,
        baseline_threshold=0.035,
        challenger_thresholds=[0.02],
        search=ThresholdSearch(),
        seed=2,
    )

    variants = cast("list[dict[str, Any]]", payload["variants"])
    challenger = next(v for v in variants if v["variant"] == "threshold_0.02")
    assert challenger["delta_vs_baseline"]["delta"] > 0.0
