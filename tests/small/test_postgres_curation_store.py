from __future__ import annotations

import sqlite3
from typing import Any, cast

import pytest

from mcp_memory.storage.postgres_curation_store import PostgresCurationStore
from mcp_memory.curation_store import CurationReceiptHydrationError
from mcp_memory.storage.postgres_migrations import POSTGRES_MIGRATIONS, POSTGRES_SCHEMA_VERSION
from tests.small.curation_repository_contract import assert_curation_repository_contract


pytestmark = pytest.mark.small


class FakeCursor:
    def __init__(self, connection: FakeConnection) -> None:
        self._cursor = connection.database.cursor()

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._cursor.close()
        return False

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        self._cursor.execute(_sqlite_query(query), tuple(() if params is None else params))

    def fetchone(self) -> tuple[object, ...] | None:
        return self._cursor.fetchone()

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._cursor.fetchall()


class FakeConnection:
    def __init__(self) -> None:
        self.database = sqlite3.connect(":memory:")
        self.database.execute("PRAGMA foreign_keys = ON")
        self.database.executescript(
            """
            CREATE TABLE memory_mutation_events (id TEXT PRIMARY KEY);
            CREATE TABLE curation_runs (
                run_id TEXT PRIMARY KEY, task_id TEXT, work_item_id TEXT,
                frontier_key TEXT NOT NULL, selector_strategy TEXT,
                context_fingerprint TEXT NOT NULL, planner_id TEXT, provider_id TEXT,
                model_id TEXT, policy_version TEXT NOT NULL, schema_version INTEGER NOT NULL,
                state TEXT NOT NULL, outcome TEXT, plan_id TEXT,
                rejection_codes_json TEXT NOT NULL, retry_reason TEXT,
                budget_usage_json TEXT NOT NULL, disclosure_audit_json TEXT NOT NULL,
                created_at TEXT NOT NULL, terminalized_at TEXT
            );
            CREATE TABLE curation_action_receipts (
                run_id TEXT NOT NULL, action_id TEXT NOT NULL, operation TEXT NOT NULL,
                affected_ids_json TEXT NOT NULL, status TEXT NOT NULL, before_token TEXT,
                after_token TEXT, mutation_event_id TEXT, intent_hash TEXT, error_code TEXT, applied_at TEXT,
                verified_at TEXT, verification_descriptor_json TEXT, PRIMARY KEY (run_id, action_id),
                FOREIGN KEY (run_id) REFERENCES curation_runs(run_id) ON DELETE CASCADE,
                FOREIGN KEY (mutation_event_id) REFERENCES memory_mutation_events(id)
            );
            CREATE TABLE curation_candidate_state (
                memory_id TEXT PRIMARY KEY, last_observed_revision_token TEXT,
                disposition TEXT NOT NULL, consecutive_no_op_count INTEGER NOT NULL,
                cooldown_until TEXT, last_disposition_reason TEXT, last_frontier_key TEXT,
                last_run_id TEXT, escalation_count INTEGER NOT NULL, last_escalated_strategy TEXT,
                last_considered_at TEXT, last_considered_strategy TEXT, last_mutation_family TEXT,
                last_mutated_at TEXT, coverage_evidence_json TEXT NOT NULL DEFAULT '{}'
            );
            """
        )

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.database.commit()

    def rollback(self) -> None:
        self.database.rollback()


class FakeLease:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    def __enter__(self) -> FakeConnection:
        return self.connection

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeSessionManager:
    def __init__(self) -> None:
        self.connection = FakeConnection()

    def open_connection(self) -> FakeLease:
        return FakeLease(self.connection)


def _sqlite_query(query: str) -> str:
    return query.replace("%s::jsonb", "?").replace("%s", "?")


def test_postgres_migration_adds_curation_ledger_as_additive_version() -> None:
    assert POSTGRES_SCHEMA_VERSION == 23
    migration = next(migration for migration in POSTGRES_MIGRATIONS if migration.version == 12)
    assert migration.version == 12
    assert migration.name == "add_curation_ledger"
    assert any("CREATE TABLE IF NOT EXISTS curation_runs" in statement for statement in migration.statements)
    assert any("PRIMARY KEY (run_id, action_id)" in statement for statement in migration.statements)


def test_postgres_migration_adds_quality_acceptance_evidence() -> None:
    migration = next(migration for migration in POSTGRES_MIGRATIONS if migration.version == 22)

    assert migration.name == "add_curation_quality_acceptance_evidence"
    statements = " ".join(migration.statements)
    assert "ADD COLUMN IF NOT EXISTS retrieval_utility_delta DOUBLE PRECISION" in statements
    assert "ADD COLUMN IF NOT EXISTS acceptance_met INTEGER" in statements
    assert "ADD COLUMN IF NOT EXISTS neutral_reason TEXT" in statements

    content_migration = next(migration for migration in POSTGRES_MIGRATIONS if migration.version == 23)
    assert content_migration.name == "add_curation_content_quality_evidence"
    content_statements = " ".join(content_migration.statements)
    assert "ADD COLUMN IF NOT EXISTS content_quality_delta DOUBLE PRECISION" in content_statements


def test_postgres_migration_adds_memory_references() -> None:
    migration = next(migration for migration in POSTGRES_MIGRATIONS if migration.version == 18)

    assert migration.name == "add_memory_references"
    assert any("ADD COLUMN IF NOT EXISTS memory_ref BIGINT" in statement for statement in migration.statements)
    assert any("CREATE SEQUENCE IF NOT EXISTS memory_ref_seq" in statement for statement in migration.statements)
    assert any("CREATE UNIQUE INDEX IF NOT EXISTS uq_memories_memory_ref" in statement for statement in migration.statements)


def test_postgres_curation_repository_contract() -> None:
    session_manager = FakeSessionManager()
    assert_curation_repository_contract(
        lambda: PostgresCurationStore(cast(Any, session_manager))
    )


def test_postgres_migration_adds_persistent_search_epochs() -> None:
    migration = next(migration for migration in POSTGRES_MIGRATIONS if migration.version == 19)
    assert migration.name == "add_search_mutation_epochs"
    statements = " ".join(migration.statements)
    assert "search_epoch_keyword" in statements
    assert "search_epoch_vector" in statements
    assert "search_epoch_graph" in statements
    assert "CREATE TRIGGER embeddings_search_epoch" in statements


def test_postgres_json_hydration_accepts_native_jsonb_values() -> None:
    from uuid import uuid4

    from mcp_memory.storage.postgres_curation_store import _json_value, _receipt_from_row
    from mcp_memory.curation_store import CurationReceiptState

    run_id, action_id, memory_id = uuid4(), uuid4(), uuid4()
    descriptor = {
        "schema_version": 1,
        "operation": "archive_memory",
        "target_ids": [str(memory_id)],
        "target_status": "archived",
    }
    receipt = _receipt_from_row(
        (
            run_id,
            action_id,
            "archive_memory",
            [str(memory_id)],
            CurationReceiptState.APPLIED_UNVERIFIED,
            None,
            None,
            None,
            "intent",
            descriptor,
            None,
            None,
            None,
        )
    )
    assert receipt.verification_descriptor is not None
    assert receipt.verification_descriptor.operation == "archive_memory"
    assert _json_value({"key": ["value"]}, default={}) == {"key": ["value"]}


def test_postgres_descriptor_hydration_wraps_malformed_values() -> None:
    from uuid import uuid4

    from mcp_memory.curation_store import CurationReceiptState
    from mcp_memory.storage.postgres_curation_store import _receipt_from_row

    with pytest.raises(CurationReceiptHydrationError, match="descriptor_hydration_failed"):
        _receipt_from_row(
            (
                uuid4(),
                uuid4(),
                "archive_memory",
                [str(uuid4())],
                CurationReceiptState.APPLIED_UNVERIFIED,
                None,
                None,
                None,
                "intent",
                {"operation": "archive_memory", "target_ids": ["not-a-uuid"]},
                None,
                None,
                None,
            )
        )


def test_postgres_receipt_identity_includes_intent_hash_with_legacy_null_wildcard() -> None:
    from uuid import uuid4

    from mcp_memory.curation_store import CurationActionReceipt, CurationReceiptState
    from mcp_memory.storage.postgres_curation_store import _receipt_identity_matches

    common = {
        "run_id": uuid4(),
        "action_id": uuid4(),
        "operation": "archive_memory",
        "affected_ids": [uuid4()],
        "status": CurationReceiptState.APPLIED_UNVERIFIED,
    }
    left = CurationActionReceipt(**common, intent_hash="left")
    right = left.model_copy(update={"intent_hash": "right"})
    assert not _receipt_identity_matches(left, right)
    assert _receipt_identity_matches(left, right.model_copy(update={"intent_hash": None}))
