from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, cast
from uuid import uuid4

import pytest

from mcp_memory.core.curation_models import CurationRunOutcome
from mcp_memory.curation_store import (
    CandidateDisposition,
    CurationActionReceipt,
    CurationCandidateState,
    CurationReceiptIdentityConflictError,
    CurationReceiptState,
    CurationRun,
    CurationRunState,
)
from mcp_memory.storage.postgres_curation_store import PostgresCurationStore
from mcp_memory.storage.postgres_migrations import POSTGRES_MIGRATIONS, POSTGRES_SCHEMA_VERSION


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
                budget_usage_json TEXT NOT NULL, created_at TEXT NOT NULL, terminalized_at TEXT
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


def _run() -> CurationRun:
    return CurationRun(run_id=uuid4(), frontier_key="frontier", context_fingerprint="context")


def _receipt(run_id, *, operation: str = "normalize_memory") -> CurationActionReceipt:
    return CurationActionReceipt(
        run_id=run_id,
        action_id=uuid4(),
        operation=operation,
        affected_ids=[uuid4()],
        status=CurationReceiptState.APPLIED_UNVERIFIED,
        before_token="before",
    )


def test_postgres_migration_adds_curation_ledger_as_latest_additive_version() -> None:
    assert POSTGRES_SCHEMA_VERSION == 12
    migration = POSTGRES_MIGRATIONS[-1]
    assert migration.version == POSTGRES_SCHEMA_VERSION
    assert migration.name == "add_curation_ledger"
    assert any("CREATE TABLE IF NOT EXISTS curation_runs" in statement for statement in migration.statements)
    assert any("PRIMARY KEY (run_id, action_id)" in statement for statement in migration.statements)


def test_postgres_store_matches_curation_lifecycle_contract() -> None:
    store = PostgresCurationStore(cast(Any, FakeSessionManager()))
    run = store.create_run(_run())
    planning = run.model_copy(update={"state": CurationRunState.PLANNING})

    assert store.transition_run(run.run_id, CurationRunState.CREATED, planning) == planning
    terminal = store.terminalize_run(run.run_id, CurationRunState.PLANNING, CurationRunOutcome.APPLIED)
    assert terminal is not None
    assert terminal.outcome is CurationRunOutcome.APPLIED
    assert store.terminalize_run(run.run_id, CurationRunState.PLANNING, CurationRunOutcome.CANCELLED) == terminal


def test_postgres_store_enforces_receipt_identity_and_candidate_cooldown() -> None:
    store = PostgresCurationStore(cast(Any, FakeSessionManager()))
    run = store.create_run(_run())
    receipt = store.put_receipt(_receipt(run.run_id))
    assert store.put_receipt(receipt) == receipt
    with pytest.raises(CurationReceiptIdentityConflictError):
        store.put_receipt(receipt.model_copy(update={"operation": "create_link"}))

    verified = receipt.model_copy(
        update={"status": CurationReceiptState.VERIFIED, "verified_at": datetime.now(timezone.utc)}
    )
    assert store.transition_receipt(
        run.run_id, receipt.action_id, CurationReceiptState.APPLIED_UNVERIFIED, verified
    ) == verified

    memory_id = uuid4()
    state = CurationCandidateState(
        memory_id=memory_id,
        disposition=CandidateDisposition.COOLDOWN,
        consecutive_no_op_count=2,
        cooldown_until=datetime.now(timezone.utc) + timedelta(minutes=5),
        last_run_id=run.run_id,
    )
    assert store.put_candidate_state(state) == state
    assert store.get_candidate_state(memory_id) == state
    assert store.is_candidate_in_cooldown(memory_id)


def test_postgres_store_bounds_reads() -> None:
    store = PostgresCurationStore(cast(Any, FakeSessionManager()))
    for _ in range(3):
        store.create_run(_run())
    assert len(store.list_runs(limit=2)) == 2
    with pytest.raises(ValueError):
        store.list_runs(limit=0)
