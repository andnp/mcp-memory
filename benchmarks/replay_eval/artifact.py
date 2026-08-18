"""Load relevance labels back from the extracted artifact."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .labels import RelevanceLabel


@dataclass(frozen=True, slots=True)
class LoadedLabels:
    """Labels read from the artifact, with duplicate triples collapsed."""

    labels: tuple[RelevanceLabel, ...]
    duplicates_collapsed: int


def load_labels(path: Path) -> LoadedLabels:
    """Read one label per line, keeping the first of any repeated triple.

    A (query, workspace_id, memory_id) triple can appear more than once if the
    same memory was read after the same query on separate occasions; only one
    observation per triple can anchor a replay comparison.
    """
    seen: dict[tuple[str, str | None, str], RelevanceLabel] = {}
    duplicates = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        label = RelevanceLabel(
            query=row["query"],
            workspace_id=row["workspace_id"],
            memory_id=row["memory_id"],
            logged_rank=row["logged_rank"],
            observed_at=row["observed_at"],
            dwell_ms=row["dwell_ms"],
        )
        key = (label.query, label.workspace_id, label.memory_id)
        if key in seen:
            duplicates += 1
            continue
        seen[key] = label
    return LoadedLabels(labels=tuple(seen.values()), duplicates_collapsed=duplicates)
