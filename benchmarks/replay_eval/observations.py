"""Where one ranker variant placed a labeled memory when replayed."""

from __future__ import annotations

from dataclasses import dataclass

from .labels import RelevanceLabel


@dataclass(frozen=True, slots=True)
class ReplayObservation:
    """One labeled query replayed through one variant.

    ``retrieved_rank`` is ``None`` when the variant did not surface the labeled
    memory at all, which is a distinct outcome from ranking it last: it means
    the memory fell outside the requested limit entirely.
    """

    label: RelevanceLabel
    variant: str
    retrieved_rank: int | None

    def __post_init__(self) -> None:
        """Reject placements telemetry and replay both express as one-based."""
        if not self.variant.strip():
            raise ValueError("variant must not be empty")
        if self.retrieved_rank is not None and self.retrieved_rank < 1:
            raise ValueError("retrieved_rank must be one-based")

    @property
    def reciprocal_rank(self) -> float:
        """Score this placement, treating an unretrieved memory as zero."""
        return 0.0 if self.retrieved_rank is None else 1.0 / self.retrieved_rank

    def hit_at(self, k: int) -> bool:
        """Report whether the labeled memory landed in the top ``k``."""
        if k < 1:
            raise ValueError("k must be positive")
        return self.retrieved_rank is not None and self.retrieved_rank <= k
