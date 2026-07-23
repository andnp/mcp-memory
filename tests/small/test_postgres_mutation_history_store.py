from __future__ import annotations

import sqlite3
from typing import Any, cast
from uuid import UUID

import pytest

from mcp_memory.mutation_history import Protection, ProtectionMode
from mcp_memory.storage.postgres_migrations import POSTGRES_MIGRATIONS, POSTGRES_SCHEMA_VERSION
from mcp_memory.storage.postgres_mutation_history_store import PostgresMutationHistoryStore
from tests.small.mutation_history_repository_contract import assert_mutation_history_repository_contract


pytestmark = pytest.mark.small

class FakePostgresCursor:
    def __init__(self, connection: FakePostgresConnection) -> None:
        self._connection = connection
        self._cursor = connection._database.cursor()

    def __enter__(self) -> FakePostgresCursor:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._cursor.close()
        return False

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        self._connection.statements.append(query)
        self._cursor.execute(_sqlite_query(query), tuple(() if params is None else params))

    def executemany(self, query: str, rows: list[tuple[object, ...]]) -> None:
        self._cursor.executemany(_sqlite_query(query), rows)

    def fetchone(self) -> tuple[object, ...] | None:
        return self._cursor.fetchone()

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._cursor.fetchall()


class FakePostgresConnection:
    def __init__(self) -> None:
        self._database = sqlite3.connect(":memory:")
        self.statements: list[str] = []
        self._database.execute("PRAGMA foreign_keys = ON")
        self._database.executescript(
            """
            CREATE TABLE memory_mutation_events (
                id TEXT PRIMARY KEY, operation TEXT NOT NULL, actor_kind TEXT NOT NULL,
                actor_id TEXT, family TEXT, task_id TEXT, curation_run_id TEXT,
                plan_id TEXT, action_id TEXT, provider_id TEXT, reason_code TEXT,
                rationale TEXT, policy_version TEXT, schema_version INTEGER NOT NULL,
                status TEXT NOT NULL, restores_event_id TEXT REFERENCES memory_mutation_events(id),
                idempotency_key TEXT, created_at TEXT NOT NULL, terminalized_at TEXT
            );
            CREATE TABLE memories (id TEXT PRIMARY KEY);
            CREATE UNIQUE INDEX uq_events_action ON memory_mutation_events(curation_run_id, action_id);
            CREATE UNIQUE INDEX uq_events_idempotency ON memory_mutation_events(idempotency_key);
            CREATE TABLE memory_record_revisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL,
                memory_id TEXT NOT NULL, role TEXT NOT NULL, before_exists BOOLEAN NOT NULL,
                before_snapshot TEXT, after_exists BOOLEAN NOT NULL, after_snapshot TEXT,
                before_token TEXT, after_token TEXT,
                UNIQUE(event_id, memory_id, role),
                FOREIGN KEY(event_id) REFERENCES memory_mutation_events(id)
            );
            CREATE TABLE memory_link_revisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL,
                source_id TEXT NOT NULL, target_id TEXT NOT NULL, link_type TEXT NOT NULL,
                context TEXT, before_exists BOOLEAN NOT NULL, after_exists BOOLEAN NOT NULL,
                UNIQUE(event_id, source_id, target_id, link_type),
                FOREIGN KEY(event_id) REFERENCES memory_mutation_events(id)
            );
            CREATE TABLE memory_protections (
                memory_id TEXT NOT NULL, mode TEXT NOT NULL, reason TEXT NOT NULL,
                actor_id TEXT, created_at TEXT NOT NULL, expires_at TEXT,
                PRIMARY KEY(memory_id, mode)
            );
            CREATE TABLE memory_restore_requests (
                id TEXT PRIMARY KEY, target_event_id TEXT NOT NULL, scope TEXT NOT NULL,
                expected_record_tokens TEXT NOT NULL, expected_link_tokens TEXT NOT NULL,
                actor_id TEXT, reason TEXT NOT NULL, idempotency_key TEXT NOT NULL UNIQUE,
                confirmation BOOLEAN NOT NULL, status TEXT NOT NULL, event_id TEXT,
                conflict_reason TEXT, conflict_details TEXT NOT NULL, created_at TEXT NOT NULL,
                terminalized_at TEXT, FOREIGN KEY(target_event_id) REFERENCES memory_mutation_events(id)
            );
            """
        )

    def cursor(self) -> FakePostgresCursor:
        return FakePostgresCursor(self)

    def commit(self) -> None:
        self._database.commit()

    def rollback(self) -> None:
        self._database.rollback()


class FakePostgresLease:
    def __init__(self, connection: FakePostgresConnection) -> None:
        self._connection = connection

    def __enter__(self) -> FakePostgresConnection:
        return self._connection

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class FakePostgresSessionManager:
    def __init__(self) -> None:
        self.connection = FakePostgresConnection()

    def open_connection(self) -> FakePostgresLease:
        return FakePostgresLease(self.connection)


def _sqlite_query(query: str) -> str:
    return (
        query.replace(" FOR UPDATE", "")
        .replace("'{}'::jsonb", "'{}'")
        .replace("%s::jsonb", "?")
        .replace("%s", "?")
    )


def test_postgres_mutation_history_repository_contract() -> None:
    session_manager = FakePostgresSessionManager()
    assert_mutation_history_repository_contract(
        lambda: PostgresMutationHistoryStore(cast(Any, session_manager))
    )


def test_postgres_protection_writers_lock_existing_targets_and_preserve_missing_target_contract() -> None:
    session_manager = FakePostgresSessionManager()
    repository = PostgresMutationHistoryStore(cast(Any, session_manager))
    existing_id = "00000000-0000-0000-0000-000000000010"
    missing_id = "00000000-0000-0000-0000-000000000011"
    session_manager.connection._database.execute("INSERT INTO memories (id) VALUES (?)", (existing_id,))
    session_manager.connection.commit()

    repository.set_protection(
        Protection(
            memory_id=UUID(existing_id),
            mode=ProtectionMode.PINNED_ACTIVE,
            reason="lock test",
        )
    )
    repository.remove_protection(UUID(existing_id), ProtectionMode.PINNED_ACTIVE)
    repository.set_protection(
        Protection(
            memory_id=UUID(missing_id),
            mode=ProtectionMode.PINNED_ACTIVE,
            reason="retention test",
        )
    )
    repository.remove_protection(UUID(missing_id), ProtectionMode.PINNED_ACTIVE)

    lock_statements = [statement for statement in session_manager.connection.statements if "FOR UPDATE" in statement]
    assert len(lock_statements) == 4
    assert all("SELECT id FROM memories WHERE id = %s FOR UPDATE" in statement for statement in lock_statements)


def test_postgres_mutation_history_migration_remains_additive() -> None:
    migration = next(migration for migration in POSTGRES_MIGRATIONS if migration.version == 11)
    assert POSTGRES_SCHEMA_VERSION == 16
    assert migration.version == 11
    assert migration.name == "add_mutation_history_and_protections"
    statements = " ".join(migration.statements)
    for table in (
        "memory_mutation_events",
        "memory_record_revisions",
        "memory_link_revisions",
        "memory_protections",
        "memory_restore_requests",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in statements
