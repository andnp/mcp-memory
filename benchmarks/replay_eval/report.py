"""Compare replayed variants against a baseline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .metrics import VariantMetrics, summarize
from .observations import ReplayObservation
from .significance import DEFAULT_RESAMPLES, PairedDelta, paired_bootstrap

QueryKey = tuple[str, str | None]


def sorted_query_keys(keys: Iterable[QueryKey]) -> list[QueryKey]:
    """Order query keys deterministically despite an optional workspace.

    A global search records no workspace, so the raw tuples mix ``str`` with
    ``None`` and cannot be compared directly.
    """
    return sorted(keys, key=lambda key: (key[0], key[1] or ""))


@dataclass(frozen=True, slots=True)
class VariantComparison:
    """One variant's quality, and its measured gain over the baseline."""

    metrics: VariantMetrics
    is_control: bool
    delta: PairedDelta | None

    def to_mapping(self) -> dict[str, object]:
        """Serialize with stable ordering for report diffs."""
        return {
            **self.metrics.to_mapping(),
            "is_control": self.is_control,
            "delta_vs_baseline": None if self.delta is None else self.delta.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """Every variant measured over one labeled query set."""

    baseline: str
    query_count: int
    label_count: int
    comparisons: tuple[VariantComparison, ...]

    def to_mapping(self) -> dict[str, object]:
        """Serialize with stable ordering for report diffs."""
        return {
            "baseline": self.baseline,
            "query_count": self.query_count,
            "label_count": self.label_count,
            "variants": [item.to_mapping() for item in self.comparisons],
        }


def per_query_scores(
    observations: Sequence[ReplayObservation],
) -> dict[QueryKey, float]:
    """Average each query's labeled placements into one score.

    Labels that share a query are not independent observations of ranker
    quality, so they are collapsed before any interval is computed; otherwise a
    query that happens to carry many labels would count many times over.
    """
    grouped: dict[QueryKey, list[float]] = {}
    for observation in observations:
        key = (observation.label.query, observation.label.workspace_id)
        grouped.setdefault(key, []).append(observation.reciprocal_rank)
    return {key: sum(scores) / len(scores) for key, scores in grouped.items()}


def compare_variants(
    replays: Mapping[str, Sequence[ReplayObservation]],
    *,
    baseline: str,
    controls: Sequence[str] = (),
    seed: int = 0,
    resamples: int = DEFAULT_RESAMPLES,
) -> ComparisonReport:
    """Measure every variant against the baseline over identical queries."""
    if baseline not in replays:
        raise ValueError(f"baseline {baseline!r} was not replayed")

    scores = {name: per_query_scores(items) for name, items in replays.items()}
    baseline_scores = scores[baseline]
    for name, variant_scores in scores.items():
        if variant_scores.keys() != baseline_scores.keys():
            raise ValueError(f"variant {name!r} covered different queries than the baseline")

    ordered_keys = sorted_query_keys(baseline_scores)
    comparisons = [
        VariantComparison(
            metrics=summarize(name, replays[name]),
            is_control=name in controls,
            delta=(
                None
                if name == baseline
                else paired_bootstrap(
                    [(baseline_scores[key], scores[name][key]) for key in ordered_keys],
                    seed=seed,
                    resamples=resamples,
                )
            ),
        )
        for name in sorted(replays)
    ]
    return ComparisonReport(
        baseline=baseline,
        query_count=len(ordered_keys),
        label_count=len(replays[baseline]),
        comparisons=tuple(comparisons),
    )
