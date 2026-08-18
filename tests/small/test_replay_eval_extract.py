"""Contract coverage for deriving relevance labels from telemetry."""

from __future__ import annotations

from typing import Any

import pytest

from benchmarks.replay_eval.extract import LABEL_QUERY, extract_labels

ROW = ("how does ranking work", "workspace-1", "memory-1", 3, 1_700_000_000.0, 250.0)


class FakeCursor:
    """Record the executed statement and replay canned telemetry rows."""

    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows
        self.executed: list[tuple[str, dict[str, Any]]] = []

    def execute(self, query: str, params: dict[str, Any], /) -> None:
        self.executed.append((query, params))

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


def test_attributed_rows_become_labels() -> None:
    """Turn each attributed read into a replayable label."""
    result = extract_labels(FakeCursor([ROW]), since=1_699_000_000.0)

    assert result.skipped == 0
    label = result.labels[0]
    assert label.query == "how does ranking work"
    assert label.workspace_id == "workspace-1"
    assert label.memory_id == "memory-1"
    assert label.logged_rank == 3
    assert label.dwell_ms == 250.0


def test_a_global_search_yields_a_label_without_a_workspace() -> None:
    """Preserve searches that were not workspace-scoped."""
    row = (ROW[0], None, *ROW[2:])

    result = extract_labels(FakeCursor([row]), since=0.0)

    assert result.labels[0].workspace_id is None


def test_a_missing_dwell_is_preserved_as_unknown() -> None:
    """Distinguish an unrecorded dwell from a zero-length one."""
    row = (*ROW[:5], None)

    result = extract_labels(FakeCursor([row]), since=0.0)

    assert result.labels[0].dwell_ms is None


@pytest.mark.parametrize(
    "row",
    [
        ("", "workspace-1", "memory-1", 3, 1.0, None),
        ("query", "workspace-1", "", 3, 1.0, None),
        ("query", "workspace-1", "memory-1", 0, 1.0, None),
        ("query", "workspace-1", "memory-1", None, 1.0, None),
    ],
    ids=["blank-query", "blank-memory", "zero-rank", "null-rank"],
)
def test_unusable_rows_are_counted_rather_than_dropped(row: tuple[Any, ...]) -> None:
    """Count rows that cannot form a label so coverage stays honest."""
    result = extract_labels(FakeCursor([row]), since=0.0)

    assert result.labels == ()
    assert result.skipped == 1
    assert result.considered == 1


def test_usable_and_unusable_rows_are_reported_together() -> None:
    """Report both halves of a mixed batch."""
    result = extract_labels(FakeCursor([ROW, ("", None, "m", 1, 1.0, None)]), since=0.0)

    assert len(result.labels) == 1
    assert result.skipped == 1
    assert result.considered == 2


def test_the_window_and_cutoff_are_passed_to_the_query() -> None:
    """Bind the attribution window rather than interpolating it."""
    cursor = FakeCursor([])

    extract_labels(cursor, since=1_699_000_000.0, window_seconds=120.0)

    query, params = cursor.executed[0]
    assert query == LABEL_QUERY
    assert params == {"since": 1_699_000_000.0, "window": 120.0}


def test_extraction_rejects_a_non_positive_window() -> None:
    """Reject a window that cannot attribute any read to a search."""
    with pytest.raises(ValueError, match="window_seconds"):
        extract_labels(FakeCursor([]), since=0.0, window_seconds=0.0)
