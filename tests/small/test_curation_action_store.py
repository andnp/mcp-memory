from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest

from mcp_memory.core.curation_identity import record_token
from mcp_memory.curation_action_store import (
    CurationActionInjectedFailure,
    CurationActionStaleError,
    MutationResult,
    SQLiteCurationActionStore,
)
from mcp_memory.curation_store import CurationRun, CurationRunState, SQLiteCurationStore
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.storage.shared_read_cache import (
    SharedReadCache,
    SharedReadCacheProjectionUpsert,
    SharedReadCacheSearchRequest,
)
from mcp_memory.utils.db import DatabaseManager


pytestmark = pytest.mark.small


def _run() -> CurationRun:
    return CurationRun(
        run_id=uuid4(),
        frontier_key="test-frontier",
        context_fingerprint="test-context",
        state=CurationRunState.EXECUTING,
    )


def _seed(db_manager: DatabaseManager) -> tuple[RelationalMemoryRepository, CurationRun, UUID, UUID]:
    repository = RelationalMemoryRepository(db_manager)
    first_id = uuid4()
    second_id = uuid4()
    repository.create_memory(
        "First title",
        "First content",
        ["workspace"],
        memory_id=str(first_id),
        tags=["old"],
    )
    repository.create_memory(
        "Second title",
        "Second content",
        ["workspace"],
        memory_id=str(second_id),
    )
    run = _run()
    SQLiteCurationStore(db_manager).create_run(run)
    return repository, run, first_id, second_id


def test_sqlite_action_transaction_commits_all_authoritative_artifacts(db_manager: DatabaseManager) -> None:
    repository, run, first_id, second_id = _seed(db_manager)
    action_id = uuid4()
    first = repository.get_memory(str(first_id))
    second = repository.get_memory(str(second_id))
    assert first is not None and second is not None
    action_store = SQLiteCurationActionStore(db_manager, embedding_model="test-embedder")

    def apply(transaction):
        transaction.update_memory(str(first_id), content="Updated content", tags=["new"])
        transaction.add_link(str(first_id), str(second_id), "supports", "action link")
        return MutationResult("rewrite_memory", [first_id, second_id])

    receipt = action_store.execute_action(
        run_id=run.run_id,
        action_id=action_id,
        target_ids=[str(second_id), str(first_id), str(first_id)],
        expected_tokens={str(first_id): record_token(first), str(second_id): record_token(second)},
        apply=apply,
    )

    assert receipt.status.value == "applied_unverified"
    assert receipt.mutation_event_id is not None
    updated = repository.get_memory(str(first_id))
    assert updated is not None
    assert updated.content == "Updated content"
    assert repository.get_links(str(first_id))[0].link_type == "SUPPORTS"
    assert repository.search_keyword_memory_ids("new") == [str(first_id)]
    connection = db_manager.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM embedding_repair_queue").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM memory_mutation_events").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM memory_record_revisions").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM memory_link_revisions").fetchone()[0] == 1
    assert SQLiteCurationStore(db_manager).get_receipt(run.run_id, action_id) == receipt


def test_stale_revision_token_writes_nothing_and_does_not_invoke_callback(db_manager: DatabaseManager) -> None:
    repository, run, first_id, _ = _seed(db_manager)
    record = repository.get_memory(str(first_id))
    assert record is not None
    called = False

    def apply(transaction):
        nonlocal called
        called = True
        transaction.update_memory(str(first_id), content="must not be written")
        return MutationResult("rewrite_memory", [first_id])

    with pytest.raises(CurationActionStaleError):
        SQLiteCurationActionStore(db_manager).execute_action(
            run_id=run.run_id,
            action_id=uuid4(),
            target_ids=[str(first_id)],
            expected_tokens={str(first_id): "v1:stale"},
            apply=apply,
        )

    assert not called
    unchanged = repository.get_memory(str(first_id))
    assert unchanged is not None
    assert unchanged.content == record.content
    connection = db_manager.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM curation_action_receipts").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM memory_mutation_events").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM embedding_repair_queue").fetchone()[0] == 0


def test_replay_returns_original_receipt_without_reapplying_callback(db_manager: DatabaseManager) -> None:
    repository, run, first_id, _ = _seed(db_manager)
    record = repository.get_memory(str(first_id))
    assert record is not None
    action_id = uuid4()
    calls = 0

    def apply(transaction):
        nonlocal calls
        calls += 1
        transaction.update_memory(str(first_id), content="once")
        return MutationResult("rewrite_memory", [first_id])

    store = SQLiteCurationActionStore(db_manager)
    arguments = {
        "run_id": run.run_id,
        "action_id": action_id,
        "target_ids": [str(first_id)],
        "expected_tokens": {str(first_id): record_token(record)},
        "apply": apply,
    }
    first_receipt = store.execute_action(**arguments)
    replayed_receipt = store.execute_action(**arguments)

    assert replayed_receipt == first_receipt
    assert calls == 1
    connection = db_manager.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM curation_action_receipts").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM memory_mutation_events").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM embedding_repair_queue").fetchone()[0] == 1


def test_committed_action_invalidates_affected_derivative_caches(
    db_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    repository, run, first_id, _ = _seed(db_manager)
    first = repository.get_memory(str(first_id))
    assert first is not None
    cache = SharedReadCache(tmp_path / "shared-read-cache.sqlite3")
    request = SharedReadCacheSearchRequest(
        query="old content",
        workspace_id="workspace",
        limit=5,
        adaptive_limit=True,
        memory_type=None,
        status=None,
        include_superseded=False,
    )
    cache.store_read_response(str(first_id), {"status": "ok", "record": {"content": "old content"}}, validation_token="old")
    cache.store_projection_entries(
        [
            SharedReadCacheProjectionUpsert(
                memory_id=str(first_id),
                payload={"memory_id": str(first_id), "title": "old"},
                validation_token="old",
            )
        ]
    )
    cache.store_search_response(request, {"status": "ok", "results": [{"memory_id": str(first_id)}]})

    def apply(transaction):
        transaction.update_memory(str(first_id), content="new content")
        return MutationResult("normalize_memory", [first_id])

    SQLiteCurationActionStore(db_manager, read_cache=cache).execute_action(
        run_id=run.run_id,
        action_id=uuid4(),
        target_ids=[str(first_id)],
        expected_tokens={str(first_id): record_token(first)},
        apply=apply,
    )

    assert cache.load_read_response(str(first_id)) is None
    assert cache.load_projection_entry(str(first_id)) is None
    assert cache.load_search_response(request) is None


@pytest.mark.parametrize(
    "fault_stage",
    ["domain_mutation", "projection_update", "repair_intent", "history", "receipt_preparation"],
)
def test_fault_at_each_stage_rolls_back_every_action_write(
    db_manager: DatabaseManager,
    fault_stage: str,
) -> None:
    repository, run, first_id, _ = _seed(db_manager)
    before = repository.get_memory(str(first_id))
    assert before is not None
    action_id = uuid4()

    def apply(transaction):
        transaction.update_memory(str(first_id), content="rolled back")
        return MutationResult("rewrite_memory", [first_id])

    with pytest.raises(CurationActionInjectedFailure):
        SQLiteCurationActionStore(db_manager, fault_stage=fault_stage).execute_action(
            run_id=run.run_id,
            action_id=action_id,
            target_ids=[str(first_id)],
            expected_tokens={str(first_id): record_token(before)},
            apply=apply,
        )

    unchanged = repository.get_memory(str(first_id))
    assert unchanged is not None
    assert unchanged.content == before.content
    connection = db_manager.get_connection()
    for table in (
        "embedding_repair_queue",
        "memory_mutation_events",
        "memory_record_revisions",
        "memory_link_revisions",
        "curation_action_receipts",
    ):
        assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
