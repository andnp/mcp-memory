"""Replay labeled queries through a ranker variant."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager

from mcp_memory.core import search_ranking

from .labels import RelevanceLabel
from .observations import ReplayObservation
from .variants import Variant

DEFAULT_LIMIT = 10

# (query, workspace_id, limit) -> the ordered memory ids the ranker returned.
SearchFn = Callable[[str, str | None, int], Sequence[str]]


@contextmanager
def installed_variant(variant: Variant) -> Iterator[None]:
    """Run with one variant's calibration in place of the shipped curve.

    Every fused score reaches the composite score through this one function, so
    swapping it is enough to isolate the calibration and leaves the rest of the
    pipeline exactly as it runs in production.
    """
    original = search_ranking._calibrate_score

    def calibrate(rrf_score: float, *, threshold: float, steepness: float) -> float:
        del threshold, steepness
        return variant.calibrate(rrf_score)

    search_ranking._calibrate_score = calibrate
    try:
        yield
    finally:
        search_ranking._calibrate_score = original


def _group_by_query(
    labels: Iterable[RelevanceLabel],
) -> dict[tuple[str, str | None], list[RelevanceLabel]]:
    grouped: dict[tuple[str, str | None], list[RelevanceLabel]] = {}
    for label in labels:
        grouped.setdefault((label.query, label.workspace_id), []).append(label)
    return grouped


def replay_labels(
    labels: Iterable[RelevanceLabel],
    variant: Variant,
    search: SearchFn,
    *,
    limit: int = DEFAULT_LIMIT,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[ReplayObservation]:
    """Locate each labeled memory in the results its query now returns.

    Labels that share a query are answered from one search: the ranking does not
    depend on which label is being scored, and distinct queries are far fewer
    than labels.
    """
    if limit < 1:
        raise ValueError("limit must be positive")

    grouped = _group_by_query(labels)
    observations: list[ReplayObservation] = []
    with installed_variant(variant):
        for index, ((query, workspace_id), group) in enumerate(grouped.items(), start=1):
            ranked = list(search(query, workspace_id, limit))
            positions = {
                memory_id: rank for rank, memory_id in enumerate(ranked, start=1)
            }
            observations.extend(
                ReplayObservation(
                    label=label,
                    variant=variant.name,
                    retrieved_rank=positions.get(label.memory_id),
                )
                for label in group
            )
            if on_progress is not None:
                on_progress(index, len(grouped))
    return observations
