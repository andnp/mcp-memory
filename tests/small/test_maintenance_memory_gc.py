from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mcp_memory.config import Config, MaintenanceConfig
from mcp_memory.context import ApplicationContext
from mcp_memory.core.maintenance_schedule import SWEEPER_TASK_NAME
from mcp_memory.core.task_handlers.maintenance_housekeeping import (
    gc_archived_memories,
    handle_sweeper_task,
)
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.relational.repository import RelationalMemoryRepository

pytestmark = pytest.mark.small


def _archive_old(
    repository: RelationalMemoryRepository,
    connection,
    memory_id: str,
) -> None:
    updated = repository.update_memory(memory_id, status="archived")
    assert updated is not None
    old_timestamp = (datetime.now(UTC) - timedelta(days=120)).isoformat()
    connection.execute(
        "UPDATE memories SET archived_at = ? WHERE id = ?",
        (old_timestamp, memory_id),
    )
    connection.commit()


def _sweeper_task() -> TaskRecord:
    return TaskRecord(
        id="memory-gc-sweeper",
        task_name=SWEEPER_TASK_NAME,
        data={},
        workspace_id=None,
        status="running",
        priority=100,
        retries_count=0,
        max_retries=3,
        created_at=0.0,
        updated_at=0.0,
        available_at=0.0,
        claimed_at=0.0,
        started_at=0.0,
        completed_at=None,
        last_error=None,
    )


def test_gc_reports_only_safe_archived_memories(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    eligible = repository.create_memory("Eligible", "Old archived memory.", ["workspace-a"], memory_id="eligible")
    protected = repository.create_memory("Protected", "Keep this memory.", ["workspace-a"], memory_id="protected")
    linked = repository.create_memory("Linked", "Keep this relationship.", ["workspace-a"], memory_id="linked")
    superseded = repository.create_memory("Superseded", "Safe lineage candidate.", ["workspace-a"], memory_id="superseded")
    replacement = repository.create_memory("Replacement", "Newer replacement.", ["workspace-a"], memory_id="replacement")
    assert all(record is not None for record in (eligible, protected, linked, superseded, replacement))

    connection = db_manager.get_connection()
    for memory_id in ("eligible", "protected", "linked", "superseded"):
        _archive_old(repository, connection, memory_id)
    connection.execute(
        "INSERT INTO memory_protections (memory_id, mode, reason, created_at) VALUES (?, ?, ?, ?)",
        ("protected", "pinned_active", "test protection", datetime.now(UTC).isoformat()),
    )
    repository.add_link("linked", "replacement", "AMENDS")
    repository.add_link("replacement", "superseded", "SUPERSEDES")
    connection.commit()

    result = gc_archived_memories(
        connection,
        sqlite_mode=True,
        cutoff=datetime.now(UTC) - timedelta(days=90),
        batch_size=10,
    )

    assert result["mode"] == "report-only"
    assert result["scanned"] == 4
    assert result["eligible"] == 2
    assert result["skipped_protected"] == 1
    assert result["skipped_linked"] == 1
    assert result["deleted"] == 0
    assert result["eligible_memory_examples"] == ["eligible", "superseded"]
    assert connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 5


def test_gc_delete_cleans_projections_embeddings_links_and_protection(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    candidate = repository.create_memory("Delete me", "Archived content.", ["workspace-a"], memory_id="delete-me")
    replacement = repository.create_memory("Replacement", "Replacement content.", ["workspace-a"], memory_id="replacement")
    assert candidate is not None
    assert replacement is not None
    connection = db_manager.get_connection()
    _archive_old(repository, connection, candidate.id)
    repository.add_link(replacement.id, candidate.id, "SUPERSEDES")
    connection.execute(
        "INSERT INTO embeddings (source_kind, source_id, model_name, embedding_json, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("memory", candidate.id, "test-model", "[0.1, 0.2]", 1.0),
    )
    connection.execute(
        "INSERT INTO memory_protections (memory_id, mode, reason, created_at) VALUES (?, ?, ?, ?)",
        (
            candidate.id,
            "pinned_active",
            "expired protection",
            (datetime.now(UTC) - timedelta(days=2)).isoformat(),
        ),
    )
    connection.execute(
        "UPDATE memory_protections SET expires_at = ? WHERE memory_id = ?",
        ((datetime.now(UTC) - timedelta(days=1)).isoformat(), candidate.id),
    )
    connection.commit()

    result = gc_archived_memories(
        connection,
        sqlite_mode=True,
        cutoff=datetime.now(UTC) - timedelta(days=90),
        batch_size=10,
        mode="delete",
    )

    assert result["deleted"] == 1
    assert result["deleted_memory_examples"] == [candidate.id]
    assert connection.execute("SELECT 1 FROM memories WHERE id = ?", (candidate.id,)).fetchone() is None
    assert connection.execute("SELECT 1 FROM memories_fts WHERE memory_id = ?", (candidate.id,)).fetchone() is None
    assert connection.execute("SELECT 1 FROM embeddings WHERE source_id = ?", (candidate.id,)).fetchone() is None
    assert connection.execute("SELECT 1 FROM links WHERE source_id = ? OR target_id = ?", (candidate.id, candidate.id)).fetchone() is None
    assert connection.execute("SELECT 1 FROM memory_protections WHERE memory_id = ?", (candidate.id,)).fetchone() is None


def test_sweeper_uses_configured_delete_mode(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    candidate = repository.create_memory("Delete via sweeper", "Archived content.", ["workspace-a"])
    assert candidate is not None
    connection = db_manager.get_connection()
    _archive_old(repository, connection, candidate.id)

    result = handle_sweeper_task(
        ApplicationContext(
            config=Config(maintenance=MaintenanceConfig(memory_gc_mode="delete")),
            db_manager=db_manager,
            repository=repository,
        ),
        _sweeper_task(),
    )

    assert result["memory_gc"]["mode"] == "delete"
    assert result["memory_gc"]["deleted"] == 1
    assert repository.get_memory(candidate.id) is None
