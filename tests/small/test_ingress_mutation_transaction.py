from __future__ import annotations

from uuid import uuid4

import pytest

from mcp_memory.core.ports.ingress import IngressActionReceiptIdentityConflictError
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.storage.ingress_mutation_transaction import (
    IngressMutationInjectedFailure,
    IngressMutationResult,
    SQLiteIngressMutationStore,
)
from mcp_memory.utils.db import DatabaseManager


pytestmark = pytest.mark.small


def test_create_commits_memory_receipt_coverage_history_and_repair_intent(
    db_manager: DatabaseManager,
) -> None:
    """A create commits the domain row and all first-slice evidence together."""
    store = SQLiteIngressMutationStore(db_manager, embedding_model="test-model")
    memory_id = str(uuid4())
    arguments = {
        "batch_id": "batch-1",
        "operation": "create",
        "entry_ids": ["entry-1", "entry-2"],
        "target_ids": [],
        "payload": {"title": "A durable note", "content": "Important context"},
    }

    receipt = store.execute(
        **arguments,
        apply=lambda transaction: IngressMutationResult(
            "create",
            [transaction.create_memory(
                memory_id=memory_id,
                title="A durable note",
                content="Important context",
                workspace_ids=["workspace"],
            ).id],
        ),
    )

    assert receipt.status.value == "applied_unverified"
    assert receipt.mutation_evidence_id is not None
    repository = RelationalMemoryRepository(db_manager)
    assert repository.get_memory(memory_id) is not None
    connection = db_manager.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM ingress_action_receipts").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM ingress_source_coverage").fetchone()[0] == 2
    assert connection.execute("SELECT COUNT(*) FROM memory_mutation_events").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM memory_record_revisions").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM embedding_repair_queue").fetchone()[0] == 1


def test_replay_returns_the_stored_receipt_without_reapplying_create(
    db_manager: DatabaseManager,
) -> None:
    """A same-identity retry returns the committed receipt and skips the callback."""
    store = SQLiteIngressMutationStore(db_manager)
    calls = 0

    def apply(transaction):
        nonlocal calls
        calls += 1
        record = transaction.create_memory(
            memory_id=str(uuid4()),
            title="Replayable note",
            content="Only once",
            workspace_ids=["workspace"],
        )
        return IngressMutationResult("create", [record.id])

    arguments = {
        "batch_id": "batch-1",
        "operation": "create",
        "entry_ids": ["entry-1"],
        "target_ids": [],
        "payload": {"content": "Only once"},
        "apply": apply,
    }
    first = store.execute(**arguments)
    replay = store.execute(**arguments)

    assert replay == first
    assert calls == 1


def test_divergent_replay_payload_raises_collision_before_domain_mutation(
    db_manager: DatabaseManager,
) -> None:
    """A reused action identity with a changed payload fails closed."""
    store = SQLiteIngressMutationStore(db_manager)
    applied = store.execute(
        batch_id="batch-1",
        operation="create",
        entry_ids=["entry-1"],
        target_ids=[],
        payload={"content": "original"},
        apply=lambda transaction: IngressMutationResult(
            "create",
            [transaction.create_memory(
                title="Original",
                content="original",
                workspace_ids=["workspace"],
            ).id],
        ),
    )
    called = False

    def unexpected_apply(transaction):
        nonlocal called
        called = True
        raise AssertionError("replay callback must not run")

    with pytest.raises(IngressActionReceiptIdentityConflictError):
        store.execute(
            batch_id="batch-1",
            operation="create",
            entry_ids=["entry-1"],
            target_ids=[],
            payload={"content": "changed"},
            apply=unexpected_apply,
        )

    assert not called
    assert store.execute(
        batch_id="batch-1",
        operation="create",
        entry_ids=["entry-1"],
        target_ids=[],
        payload={"content": "original"},
        apply=unexpected_apply,
    ) == applied


@pytest.mark.parametrize("fault_stage", ["after_domain_mutation", "before_receipt_coverage_commit"])
def test_fault_stages_roll_back_domain_and_all_ingress_artifacts(
    db_manager: DatabaseManager,
    fault_stage: str,
) -> None:
    """Injected crash-window failures leave no partial create or evidence."""
    memory_id = str(uuid4())
    store = SQLiteIngressMutationStore(db_manager, fault_stage=fault_stage)

    with pytest.raises(IngressMutationInjectedFailure):
        store.execute(
            batch_id="batch-1",
            operation="create",
            entry_ids=["entry-1"],
            target_ids=[],
            payload={"content": "rolled back"},
            apply=lambda transaction: IngressMutationResult(
                "create",
                [transaction.create_memory(
                    memory_id=memory_id,
                    title="Rolled back",
                    content="rolled back",
                    workspace_ids=["workspace"],
                ).id],
            ),
        )

    connection = db_manager.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM memories WHERE id = ?", (memory_id,)).fetchone()[0] == 0
    for table in (
        "ingress_action_receipts",
        "ingress_source_coverage",
        "memory_mutation_events",
        "memory_record_revisions",
        "embedding_repair_queue",
    ):
        assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_append_commits_terminal_appended_coverage(db_manager: DatabaseManager) -> None:
    """An append updates one memory and assigns appended coverage atomically."""
    repository = RelationalMemoryRepository(db_manager)
    record = repository.create_memory("Existing", "Before", ["workspace"], memory_id=str(uuid4()))
    assert record is not None
    store = SQLiteIngressMutationStore(db_manager)

    receipt = store.execute(
        batch_id="batch-1",
        operation="append",
        entry_ids=["entry-1"],
        target_ids=[record.id],
        payload={"content": "After"},
        apply=lambda transaction: IngressMutationResult(
            "append",
            [transaction.append_memory(record.id, "After").id],
        ),
    )

    updated = repository.get_memory(record.id)
    assert updated is not None
    assert updated.content == "Before\n\nAfter"
    coverage = db_manager.get_connection().execute(
        "SELECT outcome, action_id FROM ingress_source_coverage WHERE entry_id = ?",
        ("entry-1",),
    ).fetchone()
    assert tuple(coverage) == ("appended", receipt.action_id)
