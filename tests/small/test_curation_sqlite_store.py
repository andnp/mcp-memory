import sqlite3
from pathlib import Path

import pytest

from mcp_memory.curation_store import SQLiteCurationStore
from mcp_memory.utils.db import DatabaseManager, SCHEMA_VERSION
from mcp_memory.utils.db_schema import apply_legacy_additive_migrations, create_current_schema, finalize_schema_setup
from tests.small.curation_repository_contract import assert_curation_repository_contract


pytestmark = pytest.mark.small


def test_fresh_schema_contains_curation_tables_indexes_and_version(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "fresh.db")
    try:
        create_current_schema(connection)
        finalize_schema_setup(connection)

        tables = {
            row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        assert {"curation_runs", "curation_action_receipts", "curation_candidate_state"} <= tables
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(curation_candidate_state)").fetchall()
        }
        assert {
            "last_considered_at",
            "last_considered_strategy",
            "last_mutation_family",
            "last_mutated_at",
            "coverage_evidence_json",
        } <= columns
        quality_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(curation_quality_evidence)").fetchall()
        }
        assert {
            "retrieval_utility_delta",
            "acceptance_met",
            "neutral_reason",
            "content_quality_score_before",
            "content_quality_score_after",
            "content_quality_delta",
            "content_quality_improved",
        } <= quality_columns
        indexes = {
            row[1] for row in connection.execute("PRAGMA index_list(curation_action_receipts)").fetchall()
        }
        assert "uq_curation_action_receipts_run_action" in indexes
        assert connection.execute(
            "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
        ).fetchone() == (str(SCHEMA_VERSION),)
    finally:
        connection.close()


def test_legacy_migration_adds_curation_tables_without_data(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "legacy.db")
    try:
        create_current_schema(connection)
        connection.executescript(
            """
            DROP TABLE curation_action_receipts;
            DROP TABLE curation_candidate_state;
            DROP TABLE curation_runs;
            """
        )
        apply_legacy_additive_migrations(connection)
        finalize_schema_setup(connection)
        assert connection.execute("SELECT COUNT(*) FROM curation_runs").fetchone() == (0,)
    finally:
        connection.close()


def test_sqlite_curation_repository_contract(db_manager: DatabaseManager) -> None:
    assert_curation_repository_contract(lambda: SQLiteCurationStore(db_manager))
