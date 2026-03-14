from datetime import datetime
from uuid import UUID

import pytest

from mcp_memory.core.repository import RelationalMemoryRepository
from mcp_memory.utils.db import SCHEMA_VERSION


pytestmark = pytest.mark.small


def test_database_manager_initializes_relational_memory_schema(db_manager):
    conn = db_manager.get_connection()

    table_rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    table_names = {row[0] for row in table_rows}

    assert {"memories", "memory_workspaces", "tags", "memory_tags", "links"} <= table_names

    memory_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()
    }
    assert {
        "id",
        "title",
        "content",
        "summary",
        "type",
        "status",
        "created_at",
        "updated_at",
        "access_score",
        "last_accessed_at",
        "last_surfaced_at",
        "metadata",
    } <= memory_columns

    journal_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(system1_journal)").fetchall()
    }
    assert {"workspace_id", "author"} <= journal_columns
    assert db_manager.get_schema_version() == SCHEMA_VERSION


def test_relational_repository_create_read_update_and_list_memory(db_manager):
    repository = RelationalMemoryRepository(db_manager)

    created = repository.create_memory(
        title="Epic 01 bootstrap",
        content="Add the first relational schema slice.",
        summary="Tracks the first relational bootstrap step.",
        memory_type="plan",
        workspace_ids=["workspace-a", "workspace-a", "workspace-b"],
        tags=["sqlite", "testing", "sqlite"],
        metadata={"priority": "high"},
    )
    secondary = repository.create_memory(
        title="Standalone fact",
        content="Second record for list filtering.",
        memory_type="fact",
        workspace_ids=["workspace-c"],
        tags=["facts"],
    )

    assert created is not None
    assert secondary is not None

    UUID(created.id)
    assert created.title == "Epic 01 bootstrap"
    assert created.type == "plan"
    assert created.workspace_ids == ["workspace-a", "workspace-b"]
    assert created.tags == ["sqlite", "testing"]
    assert created.metadata == {"priority": "high"}

    fetched = repository.get_memory(created.id)

    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.summary == "Tracks the first relational bootstrap step."

    updated = repository.update_memory(
        created.id,
        title="Epic 01 relational bootstrap",
        content="Add schema and a repository slice.",
        summary="Updated after wiring the repository.",
        status="stale",
        metadata={"priority": "medium", "phase": 1},
        workspace_ids=["workspace-b"],
        tags=["repository", "sqlite"],
        access_score=2.5,
        last_accessed_at="2026-03-14T12:00:00+00:00",
        last_surfaced_at="2026-03-14T13:00:00+00:00",
    )

    assert updated is not None
    assert updated.title == "Epic 01 relational bootstrap"
    assert updated.content == "Add schema and a repository slice."
    assert updated.summary == "Updated after wiring the repository."
    assert updated.status == "stale"
    assert updated.workspace_ids == ["workspace-b"]
    assert updated.tags == ["repository", "sqlite"]
    assert updated.metadata == {"phase": 1, "priority": "medium"}
    assert updated.access_score == 2.5
    assert updated.last_accessed_at == "2026-03-14T12:00:00+00:00"
    assert updated.last_surfaced_at == "2026-03-14T13:00:00+00:00"
    assert datetime.fromisoformat(updated.updated_at) >= datetime.fromisoformat(
        created.updated_at
    )

    workspace_filtered = repository.list_memories(workspace_id="workspace-b")
    fact_filtered = repository.list_memories(memory_type="fact")
    stale_filtered = repository.list_memories(status="stale")

    assert [record.id for record in workspace_filtered] == [created.id]
    assert [record.id for record in fact_filtered] == [secondary.id]
    assert [record.id for record in stale_filtered] == [created.id]


def test_relational_repository_rejects_invalid_domain_values(db_manager):
    repository = RelationalMemoryRepository(db_manager)

    with pytest.raises(ValueError, match="workspace_ids must contain at least one non-empty value"):
        repository.create_memory(
            title="Bad memory",
            content="No workspace IDs should fail.",
            workspace_ids=["", "   "],
        )

    with pytest.raises(ValueError, match="invalid memory_type"):
        repository.create_memory(
            title="Bad memory",
            content="Unsupported type should fail.",
            workspace_ids=["workspace-a"],
            memory_type="todo",
        )

    created = repository.create_memory(
        title="Valid memory",
        content="This one is okay.",
        workspace_ids=["workspace-a"],
    )

    assert created is not None

    with pytest.raises(ValueError, match="invalid status"):
        repository.update_memory(created.id, status="unknown")