"""Derive implicit relevance labels from retrieval telemetry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .labels import RelevanceLabel

DEFAULT_WINDOW_SECONDS = 300.0

# A read is attributed to the most recent search that surfaced the same memory
# within the window. Postgres syntax: the live store is Postgres, and the local
# SQLite file under the data directory is a stale cache rather than the corpus.
LABEL_QUERY = """
WITH reads AS (
    SELECT memory_id, created_at, duration_ms
    FROM memory_tool_events
    WHERE event_kind = 'read'
      AND memory_id IS NOT NULL
      AND created_at >= %(since)s
)
SELECT origin.query_text,
       origin.workspace_id,
       reads.memory_id,
       origin.result_rank,
       reads.created_at,
       reads.duration_ms
FROM reads
JOIN LATERAL (
    SELECT search.query_text, search.workspace_id, search.result_rank
    FROM memory_tool_events search
    WHERE search.event_kind = 'search'
      AND search.memory_id = reads.memory_id
      AND search.created_at <= reads.created_at
      AND reads.created_at - search.created_at <= %(window)s
    ORDER BY search.created_at DESC
    LIMIT 1
) origin ON true
"""


class LabelCursor(Protocol):
    """The read-only cursor surface label extraction depends on."""

    def execute(self, query: str, params: dict[str, Any], /) -> object: ...

    def fetchall(self) -> Sequence[Sequence[Any]]: ...


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Labels recovered from telemetry, with the rows that could not be used."""

    labels: tuple[RelevanceLabel, ...]
    skipped: int

    @property
    def considered(self) -> int:
        """Count every attributed row, including the unusable ones."""
        return len(self.labels) + self.skipped


def extract_labels(
    cursor: LabelCursor,
    *,
    since: float,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> ExtractionResult:
    """Recover labels from attributed read events, reporting unusable rows.

    Rows that cannot form a valid label are counted rather than dropped
    silently, so a report can never present partial coverage as full coverage.
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    cursor.execute(LABEL_QUERY, {"since": since, "window": window_seconds})
    labels: list[RelevanceLabel] = []
    skipped = 0
    for query, workspace_id, memory_id, rank, observed_at, dwell_ms in cursor.fetchall():
        try:
            labels.append(
                RelevanceLabel(
                    query=query or "",
                    workspace_id=workspace_id,
                    memory_id=memory_id or "",
                    logged_rank=int(rank),
                    observed_at=float(observed_at),
                    dwell_ms=None if dwell_ms is None else float(dwell_ms),
                )
            )
        except (ValueError, TypeError):
            skipped += 1
    return ExtractionResult(labels=tuple(labels), skipped=skipped)
