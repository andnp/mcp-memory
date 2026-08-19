from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mcp_memory.core.ports.maintenance import MaintenanceHousekeepingTransaction
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.storage.sqlite_maintenance_housekeeping import SQLiteMaintenanceHousekeeping

pytestmark = pytest.mark.small


def test_sqlite_housekeeping_commits_provider_neutral_transaction(db_manager) -> None:
    """Commits policy-driven stale-plan updates through the SQLite adapter."""
    repository = RelationalMemoryRepository(db_manager)
    plan = repository.create_memory(
        "Old plan",
        "Plan content.",
        ["workspace-a"],
        memory_type="plan",
        updated_at=(datetime.now(UTC) - timedelta(days=30)).isoformat(),
    )
    assert plan is not None
    housekeeping = SQLiteMaintenanceHousekeeping(db_manager)

    updated = housekeeping.execute(
        lambda transaction: transaction.mark_stale_plans(
            (datetime.now(UTC) - timedelta(days=7)).isoformat(),
            None,
        )
    )

    assert updated == 1
    current = repository.get_memory(plan.id)
    assert current is not None
    assert current.status == "stale"


def test_sqlite_housekeeping_rolls_back_failed_transaction(db_manager) -> None:
    """Rolls back writes when a housekeeping transaction raises."""
    repository = RelationalMemoryRepository(db_manager)
    plan = repository.create_memory(
        "Old plan",
        "Plan content.",
        ["workspace-a"],
        memory_type="plan",
        updated_at=(datetime.now(UTC) - timedelta(days=30)).isoformat(),
    )
    assert plan is not None
    housekeeping = SQLiteMaintenanceHousekeeping(db_manager)

    def fail_after_update(transaction: MaintenanceHousekeepingTransaction) -> None:
        transaction.mark_stale_plans(
            (datetime.now(UTC) - timedelta(days=7)).isoformat(),
            None,
        )
        raise RuntimeError("test failure")

    with pytest.raises(RuntimeError, match="test failure"):
        housekeeping.execute(fail_after_update)

    current = repository.get_memory(plan.id)
    assert current is not None
    assert current.status == "active"
