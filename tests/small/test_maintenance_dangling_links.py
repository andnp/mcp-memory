from __future__ import annotations

import pytest

from mcp_memory.core.task_handlers.maintenance_housekeeping import reconcile_dangling_links
from mcp_memory.relational.repository import RelationalMemoryRepository


pytestmark = pytest.mark.small


def test_reconcile_dangling_links_is_bounded_deterministic_and_idempotent(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    source = repository.create_memory("Source", "Source content.", ["workspace-a"], memory_id="source")
    archived = repository.create_memory("Archived", "Archived content.", ["workspace-a"], memory_id="archived")
    assert source is not None
    assert archived is not None

    connection = db_manager.get_connection()
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.executemany(
        "INSERT INTO links (source_id, target_id, type) VALUES (?, ?, ?)",
        [
            ("source", "missing-b", "RELATES"),
            ("source", "missing-a", "SUPPORTS"),
            ("missing-source", "source", "DEPENDS_ON"),
            ("source", "archived", "AMENDS"),
            ("source", "ext:missing.txt", "REFERENCES"),
        ],
    )
    connection.execute("UPDATE memories SET status = 'archived' WHERE id = 'archived'")
    connection.commit()
    connection.execute("PRAGMA foreign_keys = ON")

    first = reconcile_dangling_links(connection, sqlite_mode=True, batch_size=1)
    second = reconcile_dangling_links(connection, sqlite_mode=True, batch_size=1)

    assert first == {
        "scanned": 3,
        "deleted": 3,
        "deleted_link_examples": [
            {"source_id": "missing-source", "target_id": "source", "type": "DEPENDS_ON"},
            {"source_id": "source", "target_id": "missing-a", "type": "SUPPORTS"},
            {"source_id": "source", "target_id": "missing-b", "type": "RELATES"},
        ],
    }
    assert second == {"scanned": 0, "deleted": 0, "deleted_link_examples": []}
    assert connection.execute("SELECT COUNT(*) FROM links").fetchone()[0] == 2
    assert connection.execute(
        "SELECT 1 FROM links WHERE target_id = 'ext:missing.txt'"
    ).fetchone() is not None


def test_reconcile_dangling_links_rejects_unbounded_batch_size(db_manager) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        reconcile_dangling_links(db_manager.get_connection(), sqlite_mode=True, batch_size=0)
