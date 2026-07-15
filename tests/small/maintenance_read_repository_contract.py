from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

from mcp_memory.relational.repository import RelationalMemoryRepository

T = TypeVar("T")


def assert_maintenance_read_preserves_telemetry(
    repository: RelationalMemoryRepository,
    memory_ids: Sequence[str],
    operation: Callable[[], T],
) -> T:
    """Run a maintenance read and assert its raw SQLite telemetry is unchanged."""
    before = _telemetry_snapshot(repository, memory_ids)
    result = operation()
    after = _telemetry_snapshot(repository, memory_ids)
    assert after == before
    return result


def _telemetry_snapshot(
    repository: RelationalMemoryRepository,
    memory_ids: Sequence[str],
) -> dict[str, tuple[str, str, str, str]]:
    normalized_ids = list(dict.fromkeys(memory_ids))
    if not normalized_ids:
        return {}
    placeholders = ",".join("?" for _ in normalized_ids)
    rows = repository._db.get_connection().execute(
        f"""
        SELECT id, quote(read_count), quote(access_score),
               quote(last_accessed_at), quote(last_surfaced_at)
        FROM memories
        WHERE id IN ({placeholders})
        """,
        normalized_ids,
    ).fetchall()
    return {
        str(row[0]): (str(row[1]), str(row[2]), str(row[3]), str(row[4]))
        for row in rows
    }
