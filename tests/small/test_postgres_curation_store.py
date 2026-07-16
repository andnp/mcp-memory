from __future__ import annotations

import sqlite3
from typing import Any, cast

import pytest

from mcp_memory.storage.postgres_curation_store import PostgresCurationStore
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
                after_token TEXT, mutation_event_id TEXT, error_code TEXT, applied_at TEXT,
                verified_at TEXT, PRIMARY KEY (run_id, action_id),
                FOREIGN KEY (run_id) REFERENCES curation_runs(run_id) ON DELETE CASCADE,
                FOREIGN KEY (mutation_event_id) REFERENCES memory_mutation_events(id)
            );
            CREATE TABLE curation_candidate_state (
                memory_id TEXT PRIMARY KEY, last_observed_revision_token TEXT,
                disposition TEXT NOT NULL, consecutive_no_op_count INTEGER NOT NULL,
                cooldown_until TEXT, last_disposition_reason TEXT, last_frontier_key TEXT,
                last_run_id TEXT, escalation_count INTEGER NOT NULL, last_escalated_strategy TEXT
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
    assert POSTGRES_SCHEMA_VERSION == 14
    migration = next(migration for migration in POSTGRES_MIGRATIONS if migration.version == 12)
    assert migration.version == 12
    assert migration.name == "add_curation_ledger"
    assert any("CREATE TABLE IF NOT EXISTS curation_runs" in statement for statement in migration.statements)
    assert any("PRIMARY KEY (run_id, action_id)" in statement for statement in migration.statements)


def test_postgres_curation_repository_contract() -> None:
    session_manager = FakeSessionManager()
    assert_curation_repository_contract(
        lambda: PostgresCurationStore(cast(Any, session_manager))
    )
