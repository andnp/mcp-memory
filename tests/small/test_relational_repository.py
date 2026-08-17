import sqlite3
from pathlib import Path
from datetime import datetime
from uuid import UUID

import pytest

from mcp_memory.core.ports.memory import (
    MemoryCreateRequest,
    MemoryLinkPort,
    MemoryMaintenanceReadPort,
    MemoryMutationPort,
    MemoryReadPort,
)
from mcp_memory.embeddings import SQLiteVectorStore
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.utils.db import DatabaseManager, SCHEMA_VERSION
from mcp_memory.utils.db_schema import create_current_schema, finalize_schema_setup


pytestmark = pytest.mark.small


class _TrackingConnection:
    def __init__(self, connection: sqlite3.Connection, *, fail: bool) -> None:
        self._connection = connection
        self._fail = fail
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, parameters=()):
        if self._fail:
            raise RuntimeError("surface update failed")
        return self._connection.execute(query, parameters)

    def close(self) -> None:
        self.closed = True
        self._connection.close()


def test_sqlite_repository_implements_memory_ports(db_manager):
    repository = RelationalMemoryRepository(db_manager)

    assert isinstance(repository, MemoryReadPort)
    assert isinstance(repository, MemoryMutationPort)
    assert isinstance(repository, MemoryLinkPort)
    assert isinstance(repository, MemoryMaintenanceReadPort)


@pytest.mark.parametrize("fail", [False, True])
def test_best_effort_surface_update_closes_transient_connection(db_manager, fail: bool) -> None:
    repository = RelationalMemoryRepository(db_manager)
    record = repository.create_memory("Surface target", "Surface content", ["workspace-1"])
    assert record is not None

    connection = _TrackingConnection(sqlite3.connect(str(db_manager.db_path)), fail=fail)
    db_manager.open_connection = lambda **kwargs: connection

    if fail:
        with pytest.raises(RuntimeError, match="surface update failed"):
            repository.touch_last_surfaced([record.id], "2026-08-07T00:00:00+00:00", best_effort=True)
    else:
        assert repository.touch_last_surfaced(
            [record.id], "2026-08-07T00:00:00+00:00", best_effort=True
        ) == 1

    assert connection.closed is True


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
    assert {"workspace_id", "author", "claim_task_id", "claimed_at"} <= journal_columns
    assert db_manager.get_schema_version() == SCHEMA_VERSION


def test_create_memories_commits_records_and_fts_together(db_manager) -> None:
    """A batch create persists records, mappings, and keyword search rows."""
    repository = RelationalMemoryRepository(db_manager)

    created = repository.create_memories(
        [
            MemoryCreateRequest(
                title="Batch alpha",
                content="SQLite batch indexing",
                workspace_ids=["workspace-a"],
                tags=["batch"],
                memory_id="batch-alpha",
            ),
            MemoryCreateRequest(
                title="Batch beta",
                content="SQLite transaction boundary",
                workspace_ids=["workspace-b"],
                memory_id="batch-beta",
            ),
        ]
    )

    assert [record.id for record in created] == ["batch-alpha", "batch-beta"]
    assert [record.memory_ref for record in created] == [1, 2]
    assert repository.search_keyword_memory_ids("indexing", limit=10) == ["batch-alpha"]
    alpha = repository.get_memory("batch-alpha")
    assert alpha is not None
    assert alpha.workspace_ids == ["workspace-a"]


def test_create_memories_rolls_back_the_entire_batch_on_insert_failure(db_manager) -> None:
    """A failed batch leaves no earlier records or FTS rows committed."""
    repository = RelationalMemoryRepository(db_manager)
    request = MemoryCreateRequest(
        title="Duplicate batch",
        content="Will roll back",
        workspace_ids=["workspace"],
        memory_id="duplicate-batch",
    )

    with pytest.raises(sqlite3.IntegrityError):
        repository.create_memories([request, request])

    assert repository.get_memory("duplicate-batch") is None
    assert repository.search_keyword_memory_ids("roll back", limit=10) == []


def test_current_schema_creation_bootstraps_fresh_db_without_legacy_migration(tmp_path: Path) -> None:
    db_path = tmp_path / "fresh-memory.db"
    conn = sqlite3.connect(db_path)
    try:
        create_current_schema(conn)
        finalize_schema_setup(conn)

        table_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')"
            ).fetchall()
        }
        journal_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(system1_journal)").fetchall()
        }
        schema_version = conn.execute(
            "SELECT value FROM schema_metadata WHERE key = ?",
            ("schema_version",),
        ).fetchone()

        assert {"memories", "memories_fts", "idx_system1_journal_claim_task_id"} <= table_names
        assert {"claim_task_id", "claimed_at", "recoverable_until"} <= journal_columns
        assert schema_version == (str(SCHEMA_VERSION),)
    finally:
        conn.close()


def test_search_epochs_are_persistent_and_lane_specific(db_manager):
    repository = RelationalMemoryRepository(db_manager)
    initial = repository.get_search_epochs()

    record = repository.create_memory(
        "Epoch title",
        "Epoch content",
        ["workspace-1"],
        memory_id="epoch-memory",
    )
    after_memory = repository.get_search_epochs()
    assert after_memory["keyword"] > initial["keyword"]
    assert after_memory["vector"] == initial["vector"]
    assert after_memory["graph"] == initial["graph"]

    repository.add_link("epoch-memory", "epoch-memory", "DEPENDS_ON")
    after_link = repository.get_search_epochs()
    assert after_link["graph"] > after_memory["graph"]

    assert record is not None
    SQLiteVectorStore(db_manager).upsert(
        source_kind="memory",
        source_id=record.id,
        workspace_id="workspace-1",
        model_name="test-model",
        embedding=[1.0, 0.0],
        source_updated_at=record.updated_at,
    )
    after_embedding = repository.get_search_epochs()
    assert after_embedding["vector"] > after_link["vector"]


def test_database_manager_migrates_legacy_journal_schema_without_claim_columns(tmp_path: Path) -> None:
    legacy_db_path = tmp_path / "legacy-memory.db"
    conn = sqlite3.connect(legacy_db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE system1_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                workspace_id TEXT,
                author TEXT,
                timestamp REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
            );
            CREATE INDEX idx_system1_journal_status ON system1_journal(status);
            CREATE TABLE schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT INTO schema_metadata (key, value) VALUES ('schema_version', '10');
            """
        )
        conn.commit()
    finally:
        conn.close()

    manager = DatabaseManager(legacy_db_path)
    try:
        migrated_conn = manager.get_connection()
        journal_columns = {
            row[1] for row in migrated_conn.execute("PRAGMA table_info(system1_journal)").fetchall()
        }
        journal_indexes = {
            row[1] for row in migrated_conn.execute("PRAGMA index_list(system1_journal)").fetchall()
        }

        assert {"claim_task_id", "claimed_at", "recoverable_until"} <= journal_columns
        assert "idx_system1_journal_claim_task_id" in journal_indexes
        assert "idx_system1_journal_recoverable_until" in journal_indexes
        assert manager.get_schema_version() == SCHEMA_VERSION
    finally:
        manager.close()


def test_database_manager_backfills_memory_references_in_creation_order(tmp_path: Path) -> None:
    legacy_db_path = tmp_path / "legacy-memory-refs.db"
    conn = sqlite3.connect(legacy_db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT,
                type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                read_count INTEGER NOT NULL DEFAULT 0,
                access_score REAL NOT NULL DEFAULT 0,
                last_accessed_at TEXT,
                last_surfaced_at TEXT,
                metadata TEXT NOT NULL DEFAULT '{}'
            );
            INSERT INTO memories (id, title, content, type, created_at, updated_at)
            VALUES
                ('later', 'Later', 'Later content', 'fact', '2026-03-15', '2026-03-15'),
                ('earlier', 'Earlier', 'Earlier content', 'fact', '2026-03-14', '2026-03-14');
            """
        )
        conn.commit()
    finally:
        conn.close()

    manager = DatabaseManager(legacy_db_path)
    try:
        rows = manager.get_connection().execute(
            "SELECT id, memory_ref FROM memories ORDER BY memory_ref"
        ).fetchall()
        assert [(row[0], row[1]) for row in rows] == [("earlier", 1), ("later", 2)]
    finally:
        manager.close()


def test_relational_repository_persists_and_resolves_memory_references(db_manager):
    repository = RelationalMemoryRepository(db_manager)

    first = repository.create_memory(
        title="First memory",
        content="First content.",
        workspace_ids=["workspace-a"],
    )
    second = repository.create_memory(
        title="Second memory",
        content="Second content.",
        workspace_ids=["workspace-a"],
    )

    assert first is not None and second is not None
    assert (first.memory_ref, second.memory_ref) == (1, 2)
    first_by_ref = repository.get_memory("mem-1")
    second_by_ref = repository.get_memory("2")
    assert first_by_ref is not None and first_by_ref.id == first.id
    assert second_by_ref is not None and second_by_ref.id == second.id

    updated = repository.update_memory("mem-2", title="Updated second memory")
    assert updated is not None
    assert updated.id == second.id
    assert updated.title == "Updated second memory"

    deleted = repository.delete_memory("mem-1")
    assert deleted is not None
    assert deleted.id == first.id
    assert repository.get_memory("mem-1") is None


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


def test_relational_repository_normalizes_link_types_and_collapses_semantic_duplicates(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)

    source = repository.create_memory(
        title="Source fact",
        content="Depends on the canonical auth architecture.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
    )
    target = repository.create_memory(
        title="Target fact",
        content="Canonical auth architecture.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
    )

    assert source is not None and target is not None

    first = repository.add_link(source.id, target.id, "depends_on", "first context")
    second = repository.add_link(source.id, target.id, "Depends-On", "updated context")
    outgoing = repository.get_links(source.id, direction="outgoing")
    filtered = repository.get_links(source.id, direction="outgoing", link_type="depends on")

    assert first.link_type == "DEPENDS_ON"
    assert second.link_type == "DEPENDS_ON"
    assert len(outgoing) == 1
    assert outgoing[0].link_type == "DEPENDS_ON"
    assert outgoing[0].context == "updated context"
    assert filtered[0].link_type == "DEPENDS_ON"
    assert repository.has_incoming_link(target.id, "depends-on") is True
    assert repository.remove_link(source.id, target.id, "depends on") is True
    assert repository.get_links(source.id, direction="outgoing") == []


def test_relational_repository_read_cache_validation_tokens_are_stable_for_unchanged_data(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)

    current = repository.create_memory(
        title="Current fact",
        content="Current canonical fact.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
    )
    superseded = repository.create_memory(
        title="Legacy fact",
        content="Legacy fact content.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
    )
    incoming = repository.create_memory(
        title="Supporting fact",
        content="Supports the current fact.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
    )

    assert current is not None and superseded is not None and incoming is not None

    repository.add_link(current.id, superseded.id, "SUPERSEDES", "replacement")
    repository.add_link(incoming.id, current.id, "DEPENDS_ON", "supporting evidence")

    first = repository.get_read_cache_validation_tokens([current.id, "missing-memory"])
    second = repository.get_read_cache_validation_tokens([current.id, "missing-memory"])

    assert list(first) == [current.id]
    assert second == first


def test_relational_repository_read_cache_validation_tokens_invalidate_for_link_and_superseded_changes(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)

    current = repository.create_memory(
        title="Current fact",
        content="Current canonical fact.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
    )
    superseded = repository.create_memory(
        title="Legacy fact",
        content="Legacy fact content.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
    )
    incoming = repository.create_memory(
        title="Supporting fact",
        content="Supports the current fact.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
    )

    assert current is not None and superseded is not None and incoming is not None

    repository.add_link(current.id, superseded.id, "SUPERSEDES", "replacement")
    repository.add_link(incoming.id, current.id, "DEPENDS_ON", "supporting evidence")

    initial_token = repository.get_read_cache_validation_tokens([current.id])[current.id]

    repository.add_link(incoming.id, current.id, "DEPENDS_ON", "updated supporting evidence")
    link_updated_token = repository.get_read_cache_validation_tokens([current.id])[current.id]

    repository.update_memory(
        superseded.id,
        content="Legacy fact content, revised.",
    )
    superseded_updated_token = repository.get_read_cache_validation_tokens([current.id])[current.id]

    assert link_updated_token != initial_token
    assert superseded_updated_token != link_updated_token
