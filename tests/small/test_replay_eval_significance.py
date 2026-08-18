"""Contract coverage for the paired bootstrap comparison."""

from __future__ import annotations

import pytest

from benchmarks.replay_eval.significance import paired_bootstrap


def _pairs(baseline: list[float], challenger: list[float]) -> list[tuple[float, float]]:
    return list(zip(baseline, challenger, strict=True))


def test_delta_is_the_mean_per_query_difference() -> None:
    """Report the challenger's average gain over the baseline."""
    result = paired_bootstrap(_pairs([0.2, 0.4], [0.4, 0.8]), seed=1)

    assert result.delta == pytest.approx(0.3)
    assert result.pair_count == 2


def test_a_consistent_gain_is_reported_as_significant() -> None:
    """Exclude no-change when every query improves by a similar amount."""
    baseline = [0.2 + 0.001 * index for index in range(200)]
    challenger = [value + 0.30 for value in baseline]

    result = paired_bootstrap(_pairs(baseline, challenger), seed=7)

    assert result.significant is True
    assert result.low > 0.0


def test_symmetric_noise_is_not_reported_as_significant() -> None:
    """Keep no-change inside the interval when gains and losses cancel.

    This is the guard that matters most: a reshuffling that helps as often as
    it hurts must not read as an improvement.
    """
    baseline = [0.5] * 200
    challenger = [0.5 + (0.2 if index % 2 else -0.2) for index in range(200)]

    result = paired_bootstrap(_pairs(baseline, challenger), seed=7)

    assert result.significant is False
    assert result.low < 0.0 < result.high


def test_a_consistent_regression_is_reported_as_significant() -> None:
    """Detect a challenger that is reliably worse, not only one that is better."""
    baseline = [0.6] * 200
    challenger = [0.3] * 200

    result = paired_bootstrap(_pairs(baseline, challenger), seed=7)

    assert result.significant is True
    assert result.high < 0.0


def test_the_same_seed_reproduces_the_interval() -> None:
    """Keep the comparison deterministic so reports are reproducible."""
    pairs = _pairs([0.1, 0.5, 0.9], [0.3, 0.4, 0.95])

    first = paired_bootstrap(pairs, seed=42)
    second = paired_bootstrap(pairs, seed=42)

    assert (first.low, first.high) == (second.low, second.high)


def test_a_different_seed_can_move_the_interval() -> None:
    """Confirm the interval is resampled rather than a fixed formula."""
    pairs = _pairs([0.1, 0.5, 0.9], [0.3, 0.4, 0.95])

    first = paired_bootstrap(pairs, seed=1, resamples=50)
    second = paired_bootstrap(pairs, seed=2, resamples=50)

    assert first.delta == pytest.approx(second.delta)
    assert (first.low, first.high) != (second.low, second.high)


def test_the_interval_brackets_the_observed_delta() -> None:
    """Keep the reported delta inside its own confidence interval."""
    baseline = [0.1 * index for index in range(20)]
    challenger = [value + 0.05 for value in baseline]

    result = paired_bootstrap(_pairs(baseline, challenger), seed=3)

    assert result.low <= result.delta <= result.high


def test_bootstrap_rejects_an_empty_comparison() -> None:
    """Refuse to report a delta over no queries."""
    with pytest.raises(ValueError, match="at least one pair"):
        paired_bootstrap([], seed=1)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 1.5])
def test_bootstrap_rejects_an_impossible_confidence(confidence: float) -> None:
    """Reject confidence levels that cannot describe an interval."""
    with pytest.raises(ValueError, match="confidence"):
        paired_bootstrap([(0.1, 0.2)], seed=1, confidence=confidence)


def test_bootstrap_rejects_a_non_positive_resample_count() -> None:
    """Reject a resample count that cannot produce a distribution."""
    with pytest.raises(ValueError, match="resamples"):
        paired_bootstrap([(0.1, 0.2)], seed=1, resamples=0)
