import sqlite3
from pathlib import Path

import pytest

from mcp_memory.mutation_history_store import SQLiteMutationHistoryStore
from mcp_memory.utils.db import SCHEMA_VERSION, DatabaseManager
from mcp_memory.utils.db_schema import (
    apply_legacy_additive_migrations,
    create_current_schema,
    finalize_schema_setup,
)
from tests.small.mutation_history_repository_contract import assert_mutation_history_repository_contract

pytestmark = pytest.mark.small

def test_fresh_schema_contains_additive_history_tables_and_indexes(tmp_path: Path) -> None:
    path = tmp_path / "fresh.db"
    conn = sqlite3.connect(path)
    try:
        create_current_schema(conn)
        finalize_schema_setup(conn)

        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        assert {
            "memory_mutation_events",
            "memory_record_revisions",
            "memory_link_revisions",
            "memory_protections",
            "memory_restore_requests",
        } <= tables
        indexes = {
            row[1]
            for row in conn.execute("PRAGMA index_list(memory_mutation_events)").fetchall()
        }
        assert "uq_memory_mutation_events_action" in indexes
        assert conn.execute(
            "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
        ).fetchone() == (str(SCHEMA_VERSION),)
    finally:
        conn.close()


def test_legacy_schema_migration_adds_history_tables_without_requiring_data(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    try:
        create_current_schema(conn)
        conn.executescript(
            """
            DROP TABLE memory_restore_requests;
            DROP TABLE memory_protections;
            DROP TABLE memory_link_revisions;
            DROP TABLE memory_record_revisions;
            DROP TABLE memory_mutation_events;
            """
        )
        apply_legacy_additive_migrations(conn)
        finalize_schema_setup(conn)
        assert conn.execute(
            "SELECT COUNT(*) FROM memory_mutation_events"
        ).fetchone() == (0,)
    finally:
        conn.close()


def test_sqlite_mutation_history_repository_contract(db_manager: DatabaseManager) -> None:
    assert_mutation_history_repository_contract(lambda: SQLiteMutationHistoryStore(db_manager))
