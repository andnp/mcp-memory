from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
import time
from typing import Any, cast
from uuid import UUID

import pytest

from mcp_memory.core.agent_runtime import build_runtime_task_worker
from mcp_memory.core.curation_context import AcceptedMaintenanceRead
from mcp_memory.core.curation_executor import CurationExecutor
from mcp_memory.core.curation_harness import CurationDryRunHarness, CurationFrontier, CurationHarnessConfig
from mcp_memory.core.curation_identity import record_token
from mcp_memory.core.curation_models import ActionPreconditions, CurationPlan, NormalizeMemoryAction
from mcp_memory.core.curation_planner import FakeCurationPlanner, FakePlannerScenario
from mcp_memory.core.curation_verifier import CurationVerifier
from mcp_memory.core.mutation_restore import RestoreExecutor
from mcp_memory.core.task_handlers import SYSTEM1_INGEST_TASK_NAME
from mcp_memory.curation_action_store import SQLiteCurationActionStore
from mcp_memory.curation_store import CurationReceiptState, CurationRunOutcome
from mcp_memory.mcp.handlers import call_memory_tool
from mcp_memory.mcp.runtime import create_runtime
from mcp_memory.mutation_history import RestoreRequest, RestoreResultStatus, RestoreScope


pytestmark = pytest.mark.large


_FIXED_PLANNER_TIME = datetime(2026, 1, 1, tzinfo=UTC)
_THOUGHT = "Token rotation requires a durable security review checklist."
_CURATED_TITLE = "Curated authentication lifecycle note"
_CURATED_SUMMARY = "Curated authentication lifecycle summary."


async def _wait_until(predicate, timeout_seconds: float = 5.0, interval_seconds: float = 0.02) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if await predicate():
            return
        await asyncio.sleep(interval_seconds)
    raise AssertionError("timed out waiting for expected lifecycle state")


def _payload(response) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(response[0].text))


def _record_read(record) -> AcceptedMaintenanceRead:
    return AcceptedMaintenanceRead(
        record={
            "id": record.id,
            "title": record.title,
            "content": record.content,
            "summary": record.summary,
            "type": record.type,
            "status": record.status,
            "tags": list(record.tags),
            "workspace_ids": list(record.workspace_ids),
            "metadata": dict(record.metadata),
        }
    )


class _LifecycleFakePlanner:
    """Bind a deterministic fake planner scenario to the harness request."""

    def __init__(self, seed_id: UUID, before_token: str) -> None:
        self._seed_id = seed_id
        self._before_token = before_token
        self.calls = 0

    async def create_plan(self, request, tools):
        self.calls += 1
        plan = CurationPlan(
            run_id=request.run_id,
            plan_id=request.plan_id,
            frontier_key=request.frontier_key,
            context_fingerprint=request.context_fingerprint,
            seed_memory_ids=[self._seed_id],
            actions=[
                NormalizeMemoryAction(
                    action_id=UUID("00000000-0000-0000-0000-000000000074"),
                    target_id=self._seed_id,
                    confidence=1.0,
                    rationale="make the security summary specific",
                    preconditions=ActionPreconditions(record_tokens={self._seed_id: self._before_token}),
                    title=_CURATED_TITLE,
                    summary=_CURATED_SUMMARY,
                )
            ],
            rationale="curate the security lifecycle note",
        )
        return await FakeCurationPlanner(
            [FakePlannerScenario(plan=plan, provider_key="local-fake", model_name="local-fake-model")],
            clock=lambda: _FIXED_PLANNER_TIME,
        ).create_plan(request, tools)


@pytest.mark.asyncio
async def test_curation_lifecycle_persists_reacquires_restores_and_reindexes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("MCP_MEMORY_TEST_MODE", "1")
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime_one = create_runtime(cwd=workspace)
    assert runtime_one.repository is not None
    assert runtime_one.task_queue is not None
    assert runtime_one.workspace_id is not None
    assert runtime_one.config is not None
    assert not vars(runtime_one.config.curation)

    try:
        assert _payload(await call_memory_tool(runtime_one, "record_thought", {"content": _THOUGHT})) == {
            "status": "recorded"
        }
        ingest_task = runtime_one.task_queue.find_open_task_any_workspace(SYSTEM1_INGEST_TASK_NAME)
        assert ingest_task is not None
        runtime_one.task_queue.update_pending_task(ingest_task.id, available_at=time.time())

        ingest_worker = build_runtime_task_worker(runtime_one)
        await ingest_worker.start()
        try:
            async def thought_ingested() -> bool:
                return bool(runtime_one.repository.list_memories(workspace_id=runtime_one.workspace_id))

            await _wait_until(thought_ingested)
        finally:
            await ingest_worker.stop(0.1)

        records = runtime_one.repository.list_memories(workspace_id=runtime_one.workspace_id)
        assert len(records) == 1
        original = records[0]
        original_title = original.title
        original_content = original.content
        original_summary = original.summary
        seed_id = UUID(original.id)

        planner = _LifecycleFakePlanner(seed_id, record_token(original))
        harness = CurationDryRunHarness(
            curation_store=runtime_one.curation,
            planner=planner,
            work_items=runtime_one.work_items,
            config=CurationHarnessConfig(execute_accepted_actions=True),
            memory_types={seed_id: original.type},
            executor=CurationExecutor(SQLiteCurationActionStore(runtime_one.db_manager)),
            verifier=CurationVerifier(runtime_one.curation, runtime_one.relational_search),
        )
        curation_result = await harness.run(
            CurationFrontier.direct(
                family="curator",
                strategy="focused",
                seed_reads=[_record_read(original)],
            )
        )
        assert planner.calls == 1
        assert curation_result.run.outcome is CurationRunOutcome.APPLIED
        assert curation_result.run.run_id == curation_result.result.run_id
        assert original.memory_ref is not None

        search_after_curation = _payload(
            await call_memory_tool(
                runtime_one,
                "search_memory_records",
                {"query": "curated authentication lifecycle"},
            )
        )
        assert [item["memory_ref"] for item in search_after_curation["results"]] == [
            f"mem-{original.memory_ref}"
        ]
        read_after_curation = _payload(
            await call_memory_tool(runtime_one, "read_memory_record", {"memory_id": original.id})
        )
        assert read_after_curation["record"]["title"] == _CURATED_TITLE
        assert read_after_curation["record"]["content"] == original_content

        run_id = curation_result.run.run_id
        receipt = runtime_one.curation.list_receipts(run_id)[0]
        assert receipt.status is CurationReceiptState.VERIFIED
        assert receipt.mutation_event_id is not None
        assert receipt.verified_at is not None
        mutation_event_id = receipt.mutation_event_id
    finally:
        runtime_one.close()

    runtime_two = create_runtime(cwd=workspace)
    assert runtime_two.repository is not None
    assert runtime_two.workspace_id is not None
    try:
        persisted_run = runtime_two.curation.get_run(run_id)
        assert persisted_run is not None
        assert persisted_run.outcome is CurationRunOutcome.APPLIED
        persisted_receipt = runtime_two.curation.get_receipt(run_id, receipt.action_id)
        assert persisted_receipt is not None
        assert persisted_receipt.status is CurationReceiptState.VERIFIED
        assert persisted_receipt.mutation_event_id == mutation_event_id

        history_event = runtime_two.mutation_history.get_event(mutation_event_id)
        assert history_event is not None
        assert history_event.curation_run_id == run_id
        assert history_event.action_id == persisted_receipt.action_id
        assert history_event.status.value == "applied"
        revisions = runtime_two.mutation_history.get_record_revisions(mutation_event_id)
        assert len(revisions) == 1
        revision = revisions[0]
        assert revision.before_snapshot is not None
        assert revision.after_snapshot is not None
        assert revision.before_snapshot["record"]["summary"] == original_summary
        assert revision.after_snapshot["record"]["summary"] == _CURATED_SUMMARY
        curated_current = runtime_two.repository.get_memory(original.id)
        assert curated_current is not None
        assert revision.after_token == record_token(curated_current)
        assert persisted_receipt.after_token is not None

        search_before_restore = _payload(
            await call_memory_tool(
                runtime_two,
                "search_memory_records",
                {"query": "curated authentication lifecycle"},
            )
        )
        assert [item["memory_ref"] for item in search_before_restore["results"]] == [
            f"mem-{original.memory_ref}"
        ]
        read_before_restore = _payload(
            await call_memory_tool(runtime_two, "read_memory_record", {"memory_id": original.id})
        )
        assert read_before_restore["record"]["title"] == _CURATED_TITLE
        assert read_before_restore["record"]["content"] == original_content

        current = runtime_two.repository.get_memory(original.id)
        assert current is not None
        restore_request = RestoreRequest(
            target_event_id=mutation_event_id,
            scope=RestoreScope.RECORDS,
            expected_record_tokens={seed_id: record_token(current)},
            reason="restore the verified low-risk normalization",
            idempotency_key="curation-lifecycle-restore",
        )
        restore_result = RestoreExecutor(
            SQLiteCurationActionStore(runtime_two.db_manager),
            runtime_two.mutation_history,
            curation_store=runtime_two.curation,
        ).execute(restore_request)
        assert restore_result.status is RestoreResultStatus.APPLIED
        assert restore_result.event_id is not None

        restored_event = runtime_two.mutation_history.get_event(restore_result.event_id)
        assert restored_event is not None
        assert restored_event.restores_event_id == mutation_event_id
        assert restored_event.status.value == "applied"
        restored_history = runtime_two.mutation_history.get_record_revisions(restore_result.event_id)
        assert len(restored_history) == 1

        search_after_restore = _payload(
            await call_memory_tool(
                runtime_two,
                "search_memory_records",
                {"query": "curated authentication lifecycle"},
            )
        )
        assert f"mem-{original.memory_ref}" not in [
            item["memory_ref"] for item in search_after_restore["results"]
        ]
        search_original = _payload(
            await call_memory_tool(
                runtime_two,
                "search_memory_records",
                {"query": "token rotation durable security review"},
            )
        )
        assert [item["memory_ref"] for item in search_original["results"]] == [
            f"mem-{original.memory_ref}"
        ]
        read_after_restore = _payload(
            await call_memory_tool(runtime_two, "read_memory_record", {"memory_id": original.id})
        )
        assert read_after_restore["record"]["title"] == original_title
        assert read_after_restore["record"]["content"] == original_content
    finally:
        runtime_two.close()
