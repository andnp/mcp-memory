"""Aggregate replay observations into comparable variant metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .observations import ReplayObservation

HIT_CUTOFFS = (1, 3, 5, 10)


@dataclass(frozen=True, slots=True)
class VariantMetrics:
    """Retrieval quality for one variant over one labeled query set."""

    variant: str
    query_count: int
    mean_reciprocal_rank: float
    hit_at: dict[int, float]
    unretrieved_rate: float

    def to_mapping(self) -> dict[str, object]:
        """Serialize with stable ordering for report diffs."""
        return {
            "variant": self.variant,
            "query_count": self.query_count,
            "mean_reciprocal_rank": round(self.mean_reciprocal_rank, 6),
            "hit_at": {str(k): round(v, 6) for k, v in sorted(self.hit_at.items())},
            "unretrieved_rate": round(self.unretrieved_rate, 6),
        }


def summarize(
    variant: str,
    observations: Sequence[ReplayObservation],
    *,
    cutoffs: Sequence[int] = HIT_CUTOFFS,
) -> VariantMetrics:
    """Summarize one variant's placements across every labeled query."""
    if not observations:
        raise ValueError("summarize requires at least one observation")
    foreign = {item.variant for item in observations} - {variant}
    if foreign:
        raise ValueError(f"observations belong to other variants: {sorted(foreign)}")

    count = len(observations)
    return VariantMetrics(
        variant=variant,
        query_count=count,
        mean_reciprocal_rank=sum(item.reciprocal_rank for item in observations) / count,
        hit_at={
            cutoff: sum(item.hit_at(cutoff) for item in observations) / count
            for cutoff in cutoffs
        },
        unretrieved_rate=sum(item.retrieved_rank is None for item in observations) / count,
    )
