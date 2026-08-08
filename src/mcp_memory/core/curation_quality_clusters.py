"""Composite retrieval utility for affected memory clusters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClusterUtility:
    before: float
    after: float
    duplicate_density_before: float
    duplicate_density_after: float

    @property
    def delta(self) -> float:
        return round(
            (self.after - self.before)
            + (self.duplicate_density_before - self.duplicate_density_after),
            3,
        )

    @property
    def duplicate_density_delta(self) -> float:
        return round(self.duplicate_density_after - self.duplicate_density_before, 3)


def evaluate_cluster_utility(
    *,
    before_ids: Sequence[str],
    after_ids: Sequence[str],
    clusters: Sequence[Sequence[str]],
    top_k: int,
) -> ClusterUtility:
    if not clusters:
        return ClusterUtility(0.0, 0.0, 0.0, 0.0)
    return ClusterUtility(
        before=_cluster_score(before_ids, clusters, top_k),
        after=_cluster_score(after_ids, clusters, top_k),
        duplicate_density_before=_duplicate_density(before_ids, clusters),
        duplicate_density_after=_duplicate_density(after_ids, clusters),
    )


def _cluster_score(
    result_ids: Sequence[str], clusters: Sequence[Sequence[str]], top_k: int
) -> float:
    if not result_ids:
        return 0.0
    top_ids = set(result_ids[:top_k])
    scores: list[float] = []
    for cluster in clusters:
        cluster_ids = set(cluster)
        if not cluster_ids:
            continue
        ranks = [
            rank
            for rank, memory_id in enumerate(result_ids, start=1)
            if memory_id in cluster_ids
        ]
        coverage = len(top_ids & cluster_ids) / len(cluster_ids)
        best_rank = 0.0 if not ranks else 1.0 / ranks[0]
        scores.append((coverage + best_rank) / 2.0)
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def _duplicate_density(result_ids: Sequence[str], clusters: Sequence[Sequence[str]]) -> float:
    if not result_ids:
        return 0.0
    cluster_by_id = {
        memory_id: index
        for index, cluster in enumerate(clusters)
        for memory_id in cluster
    }
    labels = [
        cluster_by_id[memory_id]
        for memory_id in result_ids
        if memory_id in cluster_by_id
    ]
    if not labels:
        return 0.0
    return round(1.0 - (len(set(labels)) / len(labels)), 3)
