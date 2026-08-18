"""Contract coverage for loading labels back from the artifact."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.replay_eval.artifact import load_labels


def _write(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _row(
    query: str = "q",
    workspace_id: str | None = "w1",
    memory_id: str = "m1",
    logged_rank: int = 1,
) -> dict[str, object]:
    return {
        "query": query,
        "workspace_id": workspace_id,
        "memory_id": memory_id,
        "logged_rank": logged_rank,
        "observed_at": 1_700_000_000.0,
        "dwell_ms": 250.0,
    }


def test_each_line_becomes_one_label(tmp_path: Path) -> None:
    """Round-trip the fields the extractor wrote."""
    path = _write(tmp_path, [_row(query="q1"), _row(query="q2")])

    loaded = load_labels(path)

    assert {label.query for label in loaded.labels} == {"q1", "q2"}
    assert loaded.duplicates_collapsed == 0


def test_a_repeated_triple_is_collapsed(tmp_path: Path) -> None:
    """Never let one triple double-count a query in replay."""
    path = _write(tmp_path, [_row(), _row()])

    loaded = load_labels(path)

    assert len(loaded.labels) == 1
    assert loaded.duplicates_collapsed == 1


def test_a_different_workspace_is_not_a_duplicate(tmp_path: Path) -> None:
    """Keep workspace scope part of the triple identity."""
    path = _write(tmp_path, [_row(workspace_id="w1"), _row(workspace_id="w2")])

    loaded = load_labels(path)

    assert len(loaded.labels) == 2
    assert loaded.duplicates_collapsed == 0


def test_a_repeated_triple_keeps_the_first_occurrence(tmp_path: Path) -> None:
    """Make the collapse deterministic rather than arbitrary."""
    path = _write(tmp_path, [_row(logged_rank=1), _row(logged_rank=9)])

    loaded = load_labels(path)

    assert loaded.labels[0].logged_rank == 1


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    """Tolerate trailing newlines without producing empty labels."""
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(_row()) + "\n\n", encoding="utf-8")

    loaded = load_labels(path)

    assert len(loaded.labels) == 1
