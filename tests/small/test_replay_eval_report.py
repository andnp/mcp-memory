"""Contract coverage for comparing replayed variants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from benchmarks.replay_eval.labels import RelevanceLabel
from benchmarks.replay_eval.observations import ReplayObservation
from benchmarks.replay_eval.report import (
    compare_variants,
    per_query_scores,
    sorted_query_keys,
)


def _observation(
    query: str, memory_id: str, variant: str, rank: int | None
) -> ReplayObservation:
    return ReplayObservation(
        label=RelevanceLabel(
            query=query,
            workspace_id="w1",
            memory_id=memory_id,
            logged_rank=1,
            observed_at=1_700_000_000.0,
        ),
        variant=variant,
        retrieved_rank=rank,
    )


def _run(variant: str, placements: Mapping[str, Sequence[tuple[str, int | None]]]):
    return [
        _observation(query, memory_id, variant, rank)
        for query, entries in placements.items()
        for memory_id, rank in entries
    ]


def test_labels_sharing_a_query_collapse_to_one_score() -> None:
    """Weight each query once regardless of how many labels it carries."""
    observations = _run("v", {"q": [("a", 1), ("b", 3)]})

    scores = per_query_scores(observations)

    assert len(scores) == 1
    assert scores[("q", "w1")] == pytest.approx((1.0 + 1 / 3) / 2)


def test_each_query_contributes_its_own_score() -> None:
    """Keep distinct queries as distinct evidence."""
    scores = per_query_scores(_run("v", {"q1": [("a", 1)], "q2": [("b", 2)]}))

    assert scores == {("q1", "w1"): 1.0, ("q2", "w1"): 0.5}


def test_the_baseline_has_no_delta_against_itself() -> None:
    """Leave the reference point without a self-comparison."""
    replays = {"base": _run("base", {"q": [("a", 1)]})}

    report = compare_variants(replays, baseline="base")

    assert report.comparisons[0].delta is None


def test_a_better_variant_reports_a_positive_delta() -> None:
    """Report a gain when the labeled memory ranks higher."""
    placements = {f"q{i}": [("a", 4)] for i in range(60)}
    improved = {f"q{i}": [("a", 1)] for i in range(60)}
    replays = {
        "base": _run("base", placements),
        "better": _run("better", improved),
    }

    report = compare_variants(replays, baseline="base", seed=3)
    better = next(c for c in report.comparisons if c.metrics.variant == "better")

    assert better.delta is not None
    assert better.delta.delta > 0
    assert better.delta.significant is True


def test_an_identical_variant_reports_no_change() -> None:
    """Report a zero delta when nothing about the ranking moved."""
    placements = {f"q{i}": [("a", 2)] for i in range(40)}
    replays = {
        "base": _run("base", placements),
        "same": _run("same", placements),
    }

    report = compare_variants(replays, baseline="base", seed=3)
    same = next(c for c in report.comparisons if c.metrics.variant == "same")

    assert same.delta is not None
    assert same.delta.delta == pytest.approx(0.0)
    assert same.delta.significant is False


def test_controls_are_marked_in_the_report() -> None:
    """Let a reader tell the harness's own guard from a real candidate."""
    replays = {
        "base": _run("base", {"q": [("a", 1)]}),
        "noise": _run("noise", {"q": [("a", 1)]}),
    }

    report = compare_variants(replays, baseline="base", controls=("noise",))
    noise = next(c for c in report.comparisons if c.metrics.variant == "noise")

    assert noise.is_control is True


def test_query_and_label_counts_are_reported() -> None:
    """State how much evidence the comparison rests on."""
    replays = {"base": _run("base", {"q1": [("a", 1), ("b", 2)], "q2": [("c", 1)]})}

    report = compare_variants(replays, baseline="base")

    assert report.query_count == 2
    assert report.label_count == 3


def test_a_missing_baseline_is_rejected() -> None:
    """Refuse a comparison with no reference point."""
    with pytest.raises(ValueError, match="was not replayed"):
        compare_variants({"other": _run("other", {"q": [("a", 1)]})}, baseline="base")


def test_variants_covering_different_queries_are_rejected() -> None:
    """Refuse to compare variants measured on different evidence."""
    replays = {
        "base": _run("base", {"q1": [("a", 1)]}),
        "partial": _run("partial", {"q2": [("a", 1)]}),
    }

    with pytest.raises(ValueError, match="different queries"):
        compare_variants(replays, baseline="base")


def test_serialization_reports_every_variant() -> None:
    """Serialize a complete comparison for a durable artifact."""
    replays = {
        "base": _run("base", {"q": [("a", 1)]}),
        "noise": _run("noise", {"q": [("a", 2)]}),
    }

    mapping = compare_variants(replays, baseline="base", controls=("noise",)).to_mapping()

    assert mapping["baseline"] == "base"
    variants = mapping["variants"]
    assert isinstance(variants, list)
    assert {v["variant"] for v in variants} == {"base", "noise"}  # type: ignore[index]


def test_query_keys_sort_with_and_without_a_workspace() -> None:
    """Order keys deterministically when a global search records no workspace."""
    keys = [("b", None), ("a", "w1"), ("a", None)]

    assert sorted_query_keys(keys) == [("a", None), ("a", "w1"), ("b", None)]


def test_a_comparison_spans_global_and_scoped_queries() -> None:
    """Compare variants whose labels mix workspace-scoped and global searches."""
    def run(variant: str, rank: int) -> list[ReplayObservation]:
        return [
            ReplayObservation(
                label=RelevanceLabel(
                    query=f"q{index}",
                    workspace_id=None if index % 2 else "w1",
                    memory_id="a",
                    logged_rank=1,
                    observed_at=1.0,
                ),
                variant=variant,
                retrieved_rank=rank,
            )
            for index in range(30)
        ]

    report = compare_variants(
        {"base": run("base", 4), "better": run("better", 1)},
        baseline="base",
        seed=1,
    )

    assert report.query_count == 30
