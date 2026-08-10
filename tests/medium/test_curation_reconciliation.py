from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest

from mcp_memory.core.curation_executor import CurationExecutor
from mcp_memory.core.curation_identity import record_token
from mcp_memory.core.curation_models import (
    ActionPreconditions,
    ArchiveMemoryAction,
    ClaimManifest,
    CurationVerificationDescriptor,
    EvidenceRef,
    NormalizeMemoryAction,
)
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


def _relational_search(runtime):
    assert runtime.relational_search is not None
    return runtime.relational_search


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
        real_reads = _relational_search(runtime)

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

        outcome = CurationReconciler(runtime.curation, _relational_search(runtime), sleep=lambda _delay: None).reconcile()[0]
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


def test_recovery_prefers_persisted_descriptor_over_action_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = _make_runtime(monkeypatch, tmp_path)
    try:
        run, action, receipt = _applied_unverified_run(runtime)
        descriptor = CurationVerificationDescriptor(
            operation="normalize_memory",
            target_ids=[action.target_id],
            target_status="active",
        )
        updated = receipt.model_copy(update={"verification_descriptor": descriptor})
        assert runtime.curation.transition_receipt(
            run.run_id,
            receipt.action_id,
            CurationReceiptState.APPLIED_UNVERIFIED,
            updated,
        ) is not None

        reconciler = CurationReconciler(
            runtime.curation,
            _relational_search(runtime),
            action_resolver=lambda _run, _receipt: None,
            sleep=lambda _delay: None,
        )
        outcome = reconciler.reconcile()[0]
        assert outcome.reason_code == "run_finalized"
        stored = runtime.curation.get_receipt(run.run_id, receipt.action_id)
        assert stored is not None
        assert stored.status is CurationReceiptState.VERIFIED
    finally:
        runtime.close()


def test_recovery_rejects_descriptor_target_status_before_verifying(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = _make_runtime(monkeypatch, tmp_path)
    try:
        run, action, receipt = _applied_unverified_run(runtime)
        descriptor = CurationVerificationDescriptor(
            operation="normalize_memory",
            target_ids=[action.target_id],
            target_status="archived",
        )
        updated = receipt.model_copy(update={"verification_descriptor": descriptor})
        assert runtime.curation.transition_receipt(
            run.run_id,
            receipt.action_id,
            CurationReceiptState.APPLIED_UNVERIFIED,
            updated,
        ) is not None

        outcome = CurationReconciler(runtime.curation, _relational_search(runtime), sleep=lambda _delay: None).reconcile()[0]
        assert outcome.disposition is CurationReconciliationDisposition.BLOCK
        assert outcome.reason_code == "verification_failed"
        stored = runtime.curation.get_receipt(run.run_id, receipt.action_id)
        assert stored is not None
        assert stored.status is CurationReceiptState.VERIFICATION_FAILED
        assert stored.error_code == "status_mismatch"
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "descriptor_json",
    ["not-json", '{"schema_version":999,"operation":"unsupported"}'],
)
def test_recovery_terminalizes_run_when_descriptor_hydration_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    descriptor_json: str,
) -> None:
    runtime = _make_runtime(monkeypatch, tmp_path)
    try:
        run, _action, receipt = _applied_unverified_run(runtime)
        connection = runtime.db_manager.get_connection()
        with connection:
            connection.execute(
                """
                UPDATE curation_action_receipts
                SET verification_descriptor_json = ?
                WHERE run_id = ? AND action_id = ?
                """,
                (descriptor_json, str(run.run_id), str(receipt.action_id)),
            )

        outcome = CurationReconciler(runtime.curation, _relational_search(runtime), sleep=lambda _delay: None).reconcile()[0]
        assert outcome.disposition is CurationReconciliationDisposition.BLOCK
        assert outcome.reason_code == "descriptor_hydration_failed"
        stored_run = runtime.curation.get_run(run.run_id)
        assert stored_run is not None
        assert stored_run.state is CurationRunState.TERMINAL
        assert stored_run.outcome is CurationRunOutcome.VERIFICATION_FAILED
    finally:
        runtime.close()


def test_legacy_non_normalize_receipt_is_action_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = _make_runtime(monkeypatch, tmp_path)
    try:
        assert runtime.repository is not None
        record = runtime.repository.create_memory(
            title="Archive target",
            content="Archive this record.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
        )
        assert record is not None
        run = CurationRun(
            run_id=uuid4(),
            frontier_key="legacy-frontier",
            context_fingerprint="legacy-context",
            state=CurationRunState.EXECUTING,
        )
        runtime.curation.create_run(run)
        action = ArchiveMemoryAction(
            action_id=uuid4(),
            target_id=UUID(record.id),
            confidence=1.0,
            rationale="archive",
            evidence=[EvidenceRef(memory_id=UUID(record.id))],
            preconditions=ActionPreconditions(
                record_tokens={UUID(record.id): record_token(record)}
            ),
            claim_manifest=ClaimManifest(preserved_claims=["Archive this record."]),
        )
        receipt = CurationExecutor(SQLiteCurationActionStore(runtime.db_manager)).execute_archive(
            action,
            run_id=run.run_id,
            memory_type=record.type,
        )
        legacy = receipt.model_copy(update={"verification_descriptor": None})
        assert runtime.curation.transition_receipt(
            run.run_id,
            receipt.action_id,
            CurationReceiptState.APPLIED_UNVERIFIED,
            legacy,
        ) is not None

        outcome = CurationReconciler(
            runtime.curation,
            _relational_search(runtime),
            sleep=lambda _delay: None,
        ).reconcile()[0]
        assert outcome.disposition is CurationReconciliationDisposition.BLOCK
        assert outcome.reason_code == "action_unavailable"
    finally:
        runtime.close()


def _empty_run(runtime, *, task_id: UUID | None, created_at: datetime) -> CurationRun:
    run = CurationRun(
        run_id=uuid4(),
        task_id=task_id,
        frontier_key="empty-run-frontier",
        context_fingerprint="empty-run-context",
        state=CurationRunState.PLANNING,
        created_at=created_at,
    )
    return runtime.curation.create_run(run)


def _empty_run_reconciler(runtime, now: float) -> CurationReconciler:
    return CurationReconciler(
        runtime.curation,
        _relational_search(runtime),
        task_queue=runtime.task_queue,
        clock=lambda: now,
        stale_after_seconds=60.0,
        sleep=lambda _delay: None,
    )


@pytest.mark.parametrize("running", [False, True])
def test_empty_run_stays_active_while_task_is_pending_or_running(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    running: bool,
) -> None:
    runtime = _make_runtime(monkeypatch, tmp_path)
    try:
        now = 1_000.0
        task_id = uuid4()
        task = runtime.task_queue.enqueue(CURATOR_TASK_NAME, task_id=str(task_id))
        if running:
            assert runtime.task_queue.claim_next() is not None
        run = _empty_run(
            runtime,
            task_id=task_id,
            created_at=datetime.fromtimestamp(now - 600.0, UTC),
        )

        assert _empty_run_reconciler(runtime, now).reconcile() == []
        stored = runtime.curation.get_run(run.run_id)
        assert stored is not None
        assert stored.state is CurationRunState.PLANNING
        assert runtime.task_queue.get_task(task.id).status == ("running" if running else "pending")
    finally:
        runtime.close()


def test_empty_run_follows_cancelled_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = _make_runtime(monkeypatch, tmp_path)
    try:
        now = 1_000.0
        task_id = uuid4()
        runtime.task_queue.enqueue(CURATOR_TASK_NAME, task_id=str(task_id))
        runtime.task_queue.request_cancel(
            str(task_id), cancelled_by="test", reason="abandoned trial"
        )
        run = _empty_run(
            runtime,
            task_id=task_id,
            created_at=datetime.fromtimestamp(now - 10.0, UTC),
        )

        outcomes = _empty_run_reconciler(runtime, now).reconcile()
        stored = runtime.curation.get_run(run.run_id)
        assert len(outcomes) == 1
        assert outcomes[0].reason_code == "task_cancelled"
        assert stored is not None
        assert stored.outcome is CurationRunOutcome.CANCELLED
    finally:
        runtime.close()


@pytest.mark.parametrize("task_status", ["failed", "completed"])
def test_empty_run_follows_finished_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, task_status: str
) -> None:
    runtime = _make_runtime(monkeypatch, tmp_path)
    try:
        now = 1_000.0
        task_id = uuid4()
        task = runtime.task_queue.enqueue(CURATOR_TASK_NAME, task_id=str(task_id))
        claimed = runtime.task_queue.claim_next()
        assert claimed is not None
        if task_status == "failed":
            runtime.task_queue.fail_permanently(task.id, "provider stopped", execution_epoch=claimed.execution_epoch)
        else:
            runtime.task_queue.complete(task.id, completed_at=now, execution_epoch=claimed.execution_epoch)
        run = _empty_run(
            runtime,
            task_id=task_id,
            created_at=datetime.fromtimestamp(now - 10.0, UTC),
        )

        outcomes = _empty_run_reconciler(runtime, now).reconcile()
        stored = runtime.curation.get_run(run.run_id)
        assert len(outcomes) == 1
        assert outcomes[0].reason_code == "task_finished_without_run"
        assert stored is not None
        assert stored.outcome is CurationRunOutcome.PROVIDER_FAILED
    finally:
        runtime.close()


@pytest.mark.parametrize("task_id", [None, "missing-task"])
def test_empty_run_missing_task_is_terminalized_only_when_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, task_id: str | None
) -> None:
    runtime = _make_runtime(monkeypatch, tmp_path)
    try:
        now = 1_000.0
        run = _empty_run(
            runtime,
            task_id=None if task_id is None else UUID(int=0),
            created_at=datetime.fromtimestamp(now - 61.0, UTC),
        )

        outcomes = _empty_run_reconciler(runtime, now).reconcile()
        stored = runtime.curation.get_run(run.run_id)
        assert len(outcomes) == 1
        assert outcomes[0].reason_code == "stale_task_missing"
        assert stored is not None
        assert stored.outcome is CurationRunOutcome.PROVIDER_FAILED
    finally:
        runtime.close()


def test_empty_run_missing_task_recent_is_left_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = _make_runtime(monkeypatch, tmp_path)
    try:
        now = 1_000.0
        run = _empty_run(
            runtime,
            task_id=UUID(int=0),
            created_at=datetime.fromtimestamp(now - 59.0, UTC),
        )

        assert _empty_run_reconciler(runtime, now).reconcile() == []
        stored = runtime.curation.get_run(run.run_id)
        assert stored is not None
        assert stored.state is CurationRunState.PLANNING
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
            _relational_search(runtime),
            sleep=sleeps.append,
        )
        assert reconciler.reconcile() == []
        assert attempts == 3
        assert sleeps == [0.05, 0.1]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_curator_work_runs_past_typed_block_outcome(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reconciliation outcomes do not gate direct curator task execution."""
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
        assert calls == 1
        assert stored.status == "completed"
        assert stored.last_error is None
    finally:
        runtime.close()
