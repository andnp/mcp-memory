import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
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
    SQLiteCurationStore,
)
from mcp_memory.utils.db import DatabaseManager, SCHEMA_VERSION
from mcp_memory.utils.db_schema import apply_legacy_additive_migrations, create_current_schema, finalize_schema_setup


pytestmark = pytest.mark.small


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


def test_fresh_schema_contains_curation_tables_indexes_and_version(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "fresh.db")
    try:
        create_current_schema(connection)
        finalize_schema_setup(connection)

        tables = {
            row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        assert {"curation_runs", "curation_action_receipts", "curation_candidate_state"} <= tables
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


def test_sqlite_store_persists_lifecycle_and_first_terminal_write_wins(db_manager: DatabaseManager) -> None:
    store = SQLiteCurationStore(db_manager)
    run = store.create_run(_run())

    planning = run.model_copy(update={"state": CurationRunState.PLANNING})
    assert store.transition_run(run.run_id, CurationRunState.CREATED, planning) == planning
    terminal = store.terminalize_run(run.run_id, CurationRunState.PLANNING, CurationRunOutcome.APPLIED)
    assert terminal is not None
    assert terminal.outcome is CurationRunOutcome.APPLIED

    late = store.terminalize_run(run.run_id, CurationRunState.PLANNING, CurationRunOutcome.CANCELLED)
    assert late == terminal
    assert store.transition_run(run.run_id, CurationRunState.TERMINAL, run) == terminal


def test_sqlite_store_enforces_receipt_identity_and_terminalization(db_manager: DatabaseManager) -> None:
    store = SQLiteCurationStore(db_manager)
    run = store.create_run(_run())
    receipt = store.put_receipt(_receipt(run.run_id))

    assert store.put_receipt(receipt) == receipt
    with pytest.raises(CurationReceiptIdentityConflictError):
        store.put_receipt(receipt.model_copy(update={"operation": "create_link"}))

    verified = receipt.model_copy(update={"status": CurationReceiptState.VERIFIED, "verified_at": datetime.now(timezone.utc)})
    assert store.transition_receipt(
        run.run_id, receipt.action_id, CurationReceiptState.APPLIED_UNVERIFIED, verified
    ) == verified
    failed = verified.model_copy(update={"status": CurationReceiptState.FAILED, "error_code": "late"})
    assert store.transition_receipt(run.run_id, receipt.action_id, CurationReceiptState.APPLIED_UNVERIFIED, failed) == verified
    assert len(store.list_receipts(run.run_id, limit=1)) == 1


def test_sqlite_store_bounds_reads_and_round_trips_candidate_cooldown_state(db_manager: DatabaseManager) -> None:
    store = SQLiteCurationStore(db_manager)
    runs = [store.create_run(_run()) for _ in range(3)]
    assert len(store.list_runs(limit=2)) == 2
    with pytest.raises(ValueError):
        store.list_runs(limit=0)

    memory_id = uuid4()
    cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=5)
    state = CurationCandidateState(
        memory_id=memory_id,
        disposition=CandidateDisposition.COOLDOWN,
        consecutive_no_op_count=2,
        cooldown_until=cooldown_until,
        last_disposition_reason="unchanged",
        last_frontier_key="frontier",
        last_run_id=runs[0].run_id,
        escalation_count=1,
        last_escalated_strategy="cold",
    )
    assert store.put_candidate_state(state) == state
    assert store.get_candidate_state(memory_id) == state
    assert store.is_candidate_in_cooldown(memory_id)
    assert not store.is_candidate_in_cooldown(memory_id, now=cooldown_until + timedelta(seconds=1))
