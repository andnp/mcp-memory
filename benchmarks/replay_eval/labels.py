"""Implicit relevance labels derived from retrieval telemetry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RelevanceLabel:
    """One query paired with a memory a caller read after searching for it.

    ``logged_rank`` records where the incumbent ranked the memory when it was
    read. It is a drift diagnostic rather than a baseline: the corpus has grown
    since, so a rank measured then is not comparable to a rank measured now.
    """

    query: str
    workspace_id: str | None
    memory_id: str
    logged_rank: int
    observed_at: float
    dwell_ms: float | None = None

    def __post_init__(self) -> None:
        """Reject labels that cannot anchor a replay comparison."""
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if not self.memory_id.strip():
            raise ValueError("memory_id must not be empty")
        if self.logged_rank < 1:
            raise ValueError("logged_rank must be one-based")
        if self.dwell_ms is not None and self.dwell_ms < 0:
            raise ValueError("dwell_ms must not be negative")
