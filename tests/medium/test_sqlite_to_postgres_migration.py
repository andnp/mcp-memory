from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from click.testing import CliRunner
import pytest

from mcp_memory.cli import main
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.storage.postgres import ensure_postgres_schema
from mcp_memory.storage.postgres_connection import PostgresConnectionManager
from mcp_memory.storage.postgres_repository import PostgresRelationalMemoryRepository
from mcp_memory.storage.sqlite_to_postgres_migration import migrate_sqlite_to_postgres
from mcp_memory.utils.db import DatabaseManager


pytestmark = pytest.mark.medium


def _seed_sqlite_source(sqlite_path: Path) -> None:
    manager = DatabaseManager(sqlite_path)
    try:
        repository = RelationalMemoryRepository(manager)
        repository.create_memory(
            title="Imported fact",
            content="Postgres dogfooding starts with a safe import.",
            workspace_ids=["workspace-a", "workspace-b"],
            tags=["postgres", "migration"],
            summary="Safe import fact",
            memory_type="fact",
            status="active",
            metadata={"source": "sqlite", "lineage": [1, 2, 3]},
            memory_id="memory-1",
            created_at="2026-03-28T00:00:00+00:00",
            updated_at="2026-03-28T00:10:00+00:00",
        )
        repository.create_memory(
            title="Imported plan",
            content="Wire the shared backend carefully.",
            workspace_ids=["workspace-b"],
            tags=["postgres", "shared-mode"],
            summary="Shared-mode plan",
            memory_type="plan",
            status="archived",
            metadata={"kind": "plan"},
            memory_id="memory-2",
            created_at="2026-03-28T01:00:00+00:00",
            updated_at="2026-03-28T01:30:00+00:00",
        )
        repository.add_link(
            source_id="memory-2",
            target_id="memory-1",
            link_type="DEPENDS_ON",
            context="Imported relationship",
        )

        connection = manager.get_connection()
        connection.execute(
            """
            UPDATE memories
            SET read_count = ?, access_score = ?, last_accessed_at = ?, last_surfaced_at = ?
            WHERE id = ?
            """,
            (3, 1.75, "2026-03-28T02:00:00+00:00", "2026-03-28T02:05:00+00:00", "memory-1"),
        )
        connection.commit()
    finally:
        manager.close()


def _insert_dangling_sqlite_link(sqlite_path: Path) -> None:
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            """
            INSERT INTO links (source_id, target_id, type, context)
            VALUES (?, ?, ?, ?)
            """,
            (
                "missing-memory",
                "memory-1",
                "DEPENDS_ON",
                "Dangling source link for migration regression coverage.",
            ),
        )
        connection.commit()


def test_sqlite_to_postgres_migration_imports_memory_graph_and_metadata(
    postgres_storage_config,
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "source.db"
    _seed_sqlite_source(sqlite_path)

    summary = migrate_sqlite_to_postgres(sqlite_path, postgres_storage_config)

    assert summary.memories == 2
    assert summary.workspace_mappings == 3
    assert summary.tag_names == 3
    assert summary.tag_mappings == 4
    assert summary.links == 1
    assert summary.skipped_links == 0
    assert summary.target_existing_memories == 0

    imported_fact = None
    imported_plan = None
    links = []
    with PostgresConnectionManager(postgres_storage_config) as manager:
        repository = PostgresRelationalMemoryRepository(manager)
        imported_fact = repository.get_memory("memory-1")
        imported_plan = repository.get_memory("memory-2")
        links = repository.get_links("memory-2", direction="outgoing")

    assert imported_fact is not None
    assert imported_fact.title == "Imported fact"
    assert imported_fact.workspace_ids == ["workspace-a", "workspace-b"]
    assert imported_fact.tags == ["migration", "postgres"]
    assert imported_fact.metadata == {"lineage": [1, 2, 3], "source": "sqlite"}
    assert imported_fact.read_count == 3
    assert imported_fact.access_score == pytest.approx(1.75)
    assert imported_fact.last_accessed_at == "2026-03-28T02:00:00+00:00"
    assert imported_fact.last_surfaced_at == "2026-03-28T02:05:00+00:00"

    assert imported_plan is not None
    assert imported_plan.status == "archived"
    assert imported_plan.tags == ["postgres", "shared-mode"]
    assert [(link.source_id, link.target_id, link.link_type, link.context) for link in links] == [
        ("memory-2", "memory-1", "DEPENDS_ON", "Imported relationship")
    ]


def test_sqlite_to_postgres_migration_dry_run_leaves_target_empty(
    postgres_storage_config,
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "source.db"
    _seed_sqlite_source(sqlite_path)

    summary = migrate_sqlite_to_postgres(
        sqlite_path,
        postgres_storage_config,
        dry_run=True,
    )

    assert summary.dry_run is True
    assert summary.memories == 2
    assert summary.skipped_links == 0

    memory_count = -1
    link_count = -1
    with PostgresConnectionManager(postgres_storage_config) as manager:
        with manager.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM memories")
                memory_row = cursor.fetchone()
                assert memory_row is not None and memory_row[0] is not None
                assert isinstance(memory_row[0], int | float | str)
                memory_count = int(memory_row[0])
                cursor.execute("SELECT COUNT(*) FROM links")
                link_row = cursor.fetchone()
                assert link_row is not None and link_row[0] is not None
                assert isinstance(link_row[0], int | float | str)
                link_count = int(link_row[0])

    assert memory_count == 0
    assert link_count == 0


def test_sqlite_to_postgres_migration_rejects_non_empty_target_without_flag(
    postgres_storage_config,
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "source.db"
    _seed_sqlite_source(sqlite_path)
    ensure_postgres_schema(postgres_storage_config)

    with PostgresConnectionManager(postgres_storage_config) as manager:
        repository = PostgresRelationalMemoryRepository(manager)
        repository.create_memory(
            title="Existing target fact",
            content="Do not clobber this by accident.",
            workspace_ids=["workspace-existing"],
            memory_type="fact",
            tags=["existing"],
            memory_id="existing-memory",
        )

    with pytest.raises(ValueError, match="non-empty Postgres target"):
        migrate_sqlite_to_postgres(sqlite_path, postgres_storage_config)


def test_sqlite_to_postgres_migration_skips_dangling_sqlite_links(
    postgres_storage_config,
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "source.db"
    _seed_sqlite_source(sqlite_path)
    _insert_dangling_sqlite_link(sqlite_path)

    summary = migrate_sqlite_to_postgres(sqlite_path, postgres_storage_config)

    assert summary.links == 1
    assert summary.skipped_links == 1

    rows = []
    with PostgresConnectionManager(postgres_storage_config) as manager:
        with manager.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT source_id, target_id, type FROM links ORDER BY source_id ASC")
                rows = cursor.fetchall()

    assert rows == [("memory-2", "memory-1", "DEPENDS_ON")]


def test_sqlite_to_postgres_cli_dry_run_smoke(
    postgres_storage_config,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "source.db"
    _seed_sqlite_source(sqlite_path)

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "admin",
            "migrate-sqlite-to-postgres",
            "--sqlite-path",
            str(sqlite_path),
            "--postgres-dsn",
            postgres_storage_config.dsn,
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["memories"] == 2
    assert payload["links"] == 1
