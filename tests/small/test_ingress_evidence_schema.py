import sqlite3
from pathlib import Path

import pytest

from mcp_memory.utils.db_schema import (
    apply_legacy_additive_migrations,
    create_current_schema,
    finalize_schema_setup,
)

pytestmark = pytest.mark.small


def test_fresh_schema_contains_ingress_evidence_records_and_indexes(tmp_path: Path) -> None:
    """Fresh SQLite initialization exposes replay, receipt, and coverage fields."""
    connection = sqlite3.connect(tmp_path / "fresh.db")
    try:
        create_current_schema(connection)
        finalize_schema_setup(connection)

        tables = {
            row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert {
            "ingress_batch_evidence",
            "ingress_action_receipts",
            "ingress_source_coverage",
            "ingress_quality_evidence",
        } <= tables
        assert {
            row[1] for row in connection.execute("PRAGMA table_info(ingress_batch_evidence)")
        } >= {
            "batch_id",
            "execution_epoch",
            "claimed_entry_ids_json",
            "source_entries_json",
            "claimed_at",
            "finalized_at",
        }
        assert {
            row[1] for row in connection.execute("PRAGMA table_info(ingress_action_receipts)")
        } >= {
            "action_id",
            "canonical_payload_digest",
            "status",
            "before_revision_tokens_json",
            "after_revision_tokens_json",
            "created_at",
            "terminalized_at",
        }
        assert {
            row[1] for row in connection.execute("PRAGMA table_info(ingress_source_coverage)")
        } >= {"entry_id", "outcome", "action_id", "reason"}
        assert {
            row[1] for row in connection.execute("PRAGMA table_info(ingress_quality_evidence)")
        } >= {"action_id", "disposition", "reason", "evaluated_at", "evaluator", "query_provenance_json"}
        assert {
            row[1] for row in connection.execute("PRAGMA index_list(ingress_batch_evidence)")
        } >= {"idx_ingress_batch_evidence_execution", "idx_ingress_batch_evidence_claimed_at"}
        assert {
            row[1] for row in connection.execute("PRAGMA index_list(ingress_action_receipts)")
        } >= {
            "idx_ingress_action_receipts_batch_status",
            "idx_ingress_action_receipts_mutation_evidence",
        }
        assert {
            row[1] for row in connection.execute("PRAGMA index_list(ingress_source_coverage)")
        } >= {"idx_ingress_source_coverage_action", "idx_ingress_source_coverage_outcome"}
        assert {
            row[1] for row in connection.execute("PRAGMA index_list(ingress_quality_evidence)")
        } >= {"idx_ingress_quality_evidence_disposition", "idx_ingress_quality_evidence_evaluated_at"}
    finally:
        connection.close()


def test_ingress_schema_rejects_duplicate_actions_and_source_assignments(tmp_path: Path) -> None:
    """SQLite uniqueness constraints prevent replay collisions and double coverage."""
    connection = sqlite3.connect(tmp_path / "constraints.db")
    try:
        create_current_schema(connection)
        connection.execute(
            "INSERT INTO ingress_action_receipts "
            "(action_id, batch_id, operation, canonical_payload_digest, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("action-1", "batch-1", "create", "digest-1", "applied_unverified", "now"),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO ingress_action_receipts "
                "(action_id, batch_id, operation, canonical_payload_digest, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("action-1", "batch-2", "create", "digest-2", "failed", "later"),
            )
        connection.execute(
            "INSERT INTO ingress_source_coverage (entry_id, outcome) VALUES (?, ?)",
            ("entry-1", "created"),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO ingress_source_coverage (entry_id, outcome) VALUES (?, ?)",
                ("entry-1", "ignored"),
            )
    finally:
        connection.close()


def test_legacy_migration_recreates_ingress_evidence_tables(tmp_path: Path) -> None:
    """Legacy SQLite initialization recreates absent ingress evidence tables."""
    connection = sqlite3.connect(tmp_path / "legacy.db")
    try:
        create_current_schema(connection)
        connection.executescript(
            """
            DROP TABLE ingress_source_coverage;
            DROP TABLE ingress_action_receipts;
            DROP TABLE ingress_batch_evidence;
            DROP TABLE ingress_quality_evidence;
            """
        )
        apply_legacy_additive_migrations(connection)
        assert connection.execute(
            "SELECT COUNT(*) FROM ingress_batch_evidence"
        ).fetchone() == (0,)
    finally:
        connection.close()
