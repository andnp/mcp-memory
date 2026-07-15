from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest

from mcp_memory.core.curation_executor import CurationExecutor
from mcp_memory.core.curation_identity import record_token
from mcp_memory.core.curation_models import ActionPreconditions, NormalizeMemoryAction
from mcp_memory.core.curation_reconciliation import (
    CurationReconciliationDisposition,
    CurationReconciler,
)
from mcp_memory.core.task_handlers import CURATOR_TASK_NAME
from mcp_memory.core.task_worker import RuntimeTaskWorker
from mcp_memory.curation_action_store import SQLiteCurationActionStore
from mcp_memory.curation_store import (
    CurationReceiptState,
    CurationRun,
    CurationRunOutcome,
    CurationRunState,
    CurationRepository,
)
from mcp_memory.mcp.runtime import create_runtime
from mcp_memory.relational.search import MaintenanceReadRepositoryLike


pytestmark = pytest.mark.medium


def _make_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    return create_runtime(cwd=workspace)


def _applied_unverified_run(runtime):
    assert runtime.repository is not None
    record = runtime.repository.create_memory(
        title="Authentication target",
        content="JWT coverage is required for client authentication.",
        summary="Generic summary.",
        workspace_ids=[runtime.workspace_id or "global"],
        memory_type="fact",
        tags=["auth"],
    )
    assert record is not None
    run = CurationRun(
        run_id=uuid4(),
        frontier_key="recovery-frontier",
        context_fingerprint="recovery-context",
        state=CurationRunState.EXECUTING,
    )
    runtime.curation.create_run(run)
    action = NormalizeMemoryAction(
        action_id=uuid4(),
        target_id=UUID(record.id),
        confidence=1.0,
        rationale="make the summary specific",
        preconditions=ActionPreconditions(record_tokens={UUID(record.id): record_token(record)}),
        summary="The record states a durable authentication conclusion.",
    )
    receipt = CurationExecutor(SQLiteCurationActionStore(runtime.db_manager)).execute_normalize(
        action,
        run_id=run.run_id,
        memory_type=record.type,
    )
    return run, action, receipt


def test_recovery_verifies_once_and_finalizes_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = _make_runtime(monkeypatch, tmp_path)
    try:
        run, _action, receipt = _applied_unverified_run(runtime)
        calls = 0
        real_reads = runtime.relational_search

        class _CountingReads:
            def peek_memory(self, memory_id: str):
                nonlocal calls
                calls += 1
                return real_reads.peek_memory(memory_id)

        reconciler = CurationReconciler(
            runtime.curation,
            cast(MaintenanceReadRepositoryLike, _CountingReads()),
            sleep=lambda _delay: None,
        )

        outcomes = reconciler.reconcile()
        assert outcomes[0].disposition is CurationReconciliationDisposition.CONTINUE
        assert outcomes[0].reason_code == "run_finalized"
        assert calls == 1
        stored_receipt = runtime.curation.get_receipt(run.run_id, receipt.action_id)
        stored_run = runtime.curation.get_run(run.run_id)
        assert stored_receipt is not None
        assert stored_receipt.status is CurationReceiptState.VERIFIED
        assert stored_run is not None
        assert stored_run.outcome is CurationRunOutcome.APPLIED

        assert reconciler.reconcile() == []
        assert calls == 1
    finally:
        runtime.close()


def test_failed_reconciliation_blocks_and_late_terminal_writes_are_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = _make_runtime(monkeypatch, tmp_path)
    try:
        run, _action, receipt = _applied_unverified_run(runtime)
        assert runtime.repository.update_memory(str(receipt.affected_ids[0]), title="Edited after the action") is not None

        outcome = CurationReconciler(runtime.curation, runtime.relational_search, sleep=lambda _delay: None).reconcile()[0]
        assert outcome.disposition is CurationReconciliationDisposition.BLOCK
        assert outcome.reason_code == "verification_failed"
        stored_run = runtime.curation.get_run(run.run_id)
        assert stored_run is not None
        assert stored_run.outcome is CurationRunOutcome.VERIFICATION_FAILED

        late = runtime.curation.terminalize_run(
            run.run_id,
            CurationRunState.TERMINAL,
            CurationRunOutcome.APPLIED,
        )
        stored_receipt = runtime.curation.get_receipt(run.run_id, receipt.action_id)
        assert late is not None
        assert late.outcome is CurationRunOutcome.VERIFICATION_FAILED
        assert stored_receipt is not None
        assert stored_receipt.status is CurationReceiptState.VERIFICATION_FAILED
    finally:
        runtime.close()


def test_sqlite_lock_retries_are_bounded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = _make_runtime(monkeypatch, tmp_path)
    try:
        attempts = 0
        sleeps: list[float] = []

        class _LockedStore:
            def list_runs(self, *, limit: int):
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise sqlite3.OperationalError("database is locked")
                return []

        reconciler = CurationReconciler(
            cast(CurationRepository, _LockedStore()),
            runtime.relational_search,
            sleep=sleeps.append,
        )
        assert reconciler.reconcile() == []
        assert attempts == 3
        assert sleeps == [0.05, 0.1]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_curator_work_is_gated_by_typed_block_outcome(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = _make_runtime(monkeypatch, tmp_path)
    try:
        task = runtime.task_queue.enqueue(
            CURATOR_TASK_NAME,
            workspace_id=runtime.workspace_id,
            task_id="curation-gate-task",
        )
        claimed = runtime.task_queue.claim_next(workspace_id=runtime.workspace_id)
        assert claimed is not None
        calls = 0

        class _BlockingReconciler:
            def reconcile(self):
                return []

            def before_curator_work(self):
                from mcp_memory.core.curation_reconciliation import CurationReconciliationOutcome

                return CurationReconciliationOutcome(CurationReconciliationDisposition.BLOCK, "verification_failed")

        async def handler(_ctx, _task):
            nonlocal calls
            calls += 1
            return {}

        worker = RuntimeTaskWorker(
            runtime.task_runtime_view(),
            handlers={CURATOR_TASK_NAME: handler},
            curation_reconciler=cast(CurationReconciler, _BlockingReconciler()),
        )
        await worker._process_task(claimed)
        stored = runtime.task_queue.get_task(task.id)
        assert calls == 0
        assert stored.status == "failed"
        assert stored.last_error is not None
        assert "verification_failed" in stored.last_error
    finally:
        runtime.close()
