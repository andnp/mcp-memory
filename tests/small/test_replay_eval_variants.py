"""Contract coverage for the ranker variants under comparison."""

from __future__ import annotations

import pytest

from benchmarks.replay_eval.variants import (
    Variant,
    default_variants,
    noise_control,
    saturation,
    sigmoid,
    threshold_variant,
)

# The fused-score range the pipeline actually produces: a single lane at a deep
# rank through two lanes both at rank one.
OBSERVED_LOW = 0.0139
OBSERVED_HIGH = 0.0328


@pytest.mark.parametrize("variant", default_variants(), ids=lambda v: v.name)
def test_every_variant_maps_observed_scores_into_the_unit_range(
    variant: Variant,
) -> None:
    """Keep every calibration inside the range the composite score assumes."""
    for score in (0.0, OBSERVED_LOW, OBSERVED_HIGH, 1.0, 3.0):
        assert 0.0 <= variant.calibrate(score) <= 1.0


@pytest.mark.parametrize("variant", [sigmoid(), saturation()], ids=lambda v: v.name)
def test_uncontrolled_variants_rank_a_better_fused_score_higher(
    variant: Variant,
) -> None:
    """Preserve fused-score ordering, which is the point of a calibration."""
    assert variant.calibrate(OBSERVED_HIGH) > variant.calibrate(OBSERVED_LOW)


def test_saturation_reads_a_rank_one_single_lane_hit_as_even_odds() -> None:
    """Anchor the curve where its knob says it should sit."""
    assert saturation().calibrate(1.0 / 61.0) == pytest.approx(0.5)


def test_saturation_midpoint_tracks_the_fusion_constant() -> None:
    """Follow rrf_k so changing fusion cannot silently decalibrate the curve."""
    assert saturation(rrf_k=10.0).calibrate(1.0 / 11.0) == pytest.approx(0.5)


def test_sigmoid_never_reaches_even_odds_on_observed_scores() -> None:
    """Record that the incumbent midpoint sits above the achievable range."""
    assert sigmoid().calibrate(OBSERVED_HIGH) < 0.5


def test_the_noise_control_is_marked_as_a_control() -> None:
    """Let a report separate the control from the variants under test."""
    assert noise_control().is_control is True
    assert sigmoid().is_control is False


def test_the_noise_control_perturbs_the_baseline() -> None:
    """Confirm the control actually reshuffles rather than passing through."""
    baseline = sigmoid().calibrate
    control = noise_control(magnitude=0.05).calibrate
    scores = [OBSERVED_LOW + index * 0.0005 for index in range(40)]

    assert any(control(score) != baseline(score) for score in scores)


def test_the_noise_control_is_reproducible() -> None:
    """Keep the control deterministic so a run can be repeated exactly."""
    first = noise_control(seed=5).calibrate
    second = noise_control(seed=5).calibrate

    assert [first(s) for s in (0.01, 0.02, 0.03)] == [
        second(s) for s in (0.01, 0.02, 0.03)
    ]


def test_a_different_control_seed_perturbs_differently() -> None:
    """Allow independent control runs rather than one fixed shuffle."""
    scores = [OBSERVED_LOW + index * 0.0005 for index in range(40)]
    first = noise_control(seed=1).calibrate
    second = noise_control(seed=2).calibrate

    assert any(first(score) != second(score) for score in scores)


def test_a_zero_magnitude_control_matches_the_baseline() -> None:
    """Degrade to the baseline so the control's effect is attributable."""
    baseline = sigmoid().calibrate
    control = noise_control(magnitude=0.0).calibrate

    assert [control(s) for s in (0.01, 0.02, 0.03)] == [
        baseline(s) for s in (0.01, 0.02, 0.03)
    ]


def test_the_control_rejects_a_negative_magnitude() -> None:
    """Reject a perturbation size that has no meaning."""
    with pytest.raises(ValueError, match="magnitude"):
        noise_control(magnitude=-0.1)


def test_the_default_comparison_set_includes_the_control() -> None:
    """Never compare variants without the guard that validates the harness."""
    assert any(variant.is_control for variant in default_variants())


def test_variant_names_are_unique() -> None:
    """Keep variants attributable in a report."""
    names = [variant.name for variant in default_variants()]

    assert len(names) == len(set(names))


def test_variant_requires_a_name() -> None:
    """Refuse a variant a report could not attribute."""
    with pytest.raises(ValueError, match="name"):
        Variant(name=" ", description="d", calibrate=float)


def test_threshold_variant_names_reflect_their_threshold() -> None:
    """Let a sweep attribute each result back to its threshold."""
    assert threshold_variant(0.02).name != threshold_variant(0.03).name


def test_threshold_variant_calibrates_like_a_sigmoid_at_that_threshold() -> None:
    """Reuse the shipped curve rather than a second calibration formula."""
    assert threshold_variant(0.02).calibrate(0.025) == sigmoid(threshold=0.02).calibrate(0.025)


def test_threshold_variant_is_not_a_control() -> None:
    """Keep threshold sweeps out of the noise-control guard."""
    assert threshold_variant(0.02).is_control is False
