from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.ports.work_items import EXECUTION_LANE_AGENTIC, WORK_FAMILY_MEMORY_CURATION_REVIEW
from mcp_memory.core.providers.interfaces import AgenticRunResult
from mcp_memory.core.task_handlers import CURATOR_TASK_NAME
from mcp_memory.core.task_handlers.curator_handlers import handle_memory_curator_task
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.mcp.runtime import create_runtime
from mcp_memory.mcp.transport import dispatch_internal_memory_tool

pytestmark = pytest.mark.medium


class _DirectSession:
    def __init__(self, ctx: ApplicationContext, task_id: str, memory_id: str, *, swallow_mutation_errors: bool = False) -> None:
        self._ctx = ctx
        self._task_id = task_id
        self._memory_id = memory_id
        self._swallow_mutation_errors = swallow_mutation_errors

    async def run_agent(self, prompt: str) -> AgenticRunResult:
        assert "no planning or submission stage" in prompt
        try:
            result = await dispatch_internal_memory_tool(
                self._ctx,
                "internal_update_memory_record",
                {
                    "memory_id": self._memory_id,
                    "summary": "A direct MCP curator conclusion.",
                    "task_id": self._task_id,
                },
            )
        except RuntimeError:
            if not self._swallow_mutation_errors:
                raise
            return AgenticRunResult(status="success", parsed={"summary": "provider continued"}, raw_text="provider continued")
        return AgenticRunResult(
            status="success",
            parsed={"summary": result[0].text, "mutations_attempted": 1},
            raw_text=result[0].text,
        )

    async def close(self) -> None:
        return None


class _DirectProvider:
    def __init__(self, ctx: ApplicationContext, memory_id: str, *, swallow_mutation_errors: bool = False) -> None:
        self._ctx = ctx
        self._memory_id = memory_id
        self._swallow_mutation_errors = swallow_mutation_errors
        self.allowed_tools: tuple[str, ...] | None = None

    def with_usage_context(self, **context: object) -> _DirectProvider:
        del context
        return self

    async def open_agent_session(
        self,
        *,
        allowed_tool_names: tuple[str, ...] | None = None,
    ) -> _DirectSession:
        self.allowed_tools = allowed_tool_names
        return _DirectSession(
            self._ctx, "direct-curator-task", self._memory_id,
            swallow_mutation_errors=self._swallow_mutation_errors,
        )

    async def run_agent(self, prompt: str) -> AgenticRunResult:
        del prompt
        return AgenticRunResult(status="success", raw_text="{}")


class _ConversationUsage:
    def __init__(self, conversations: list[object]) -> None:
        self._conversations = conversations

    def list_conversations(self, **kwargs: object) -> list[object]:
        assert kwargs["task_id"] == "direct-curator-task"
        return self._conversations


def _task(runtime: Any) -> TaskRecord:
    return TaskRecord(
        id="direct-curator-task",
        task_name=CURATOR_TASK_NAME,
        data={"workspace_id": runtime.workspace_id},
        workspace_id=runtime.workspace_id,
        status="running",
        priority=100,
        retries_count=0,
        max_retries=3,
        created_at=0.0,
        updated_at=0.0,
        available_at=0.0,
        claimed_at=0.0,
        started_at=0.0,
        completed_at=None,
        last_error=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_ledger", [False, True])
async def test_direct_campaign_invokes_mcp_mutation_and_completes_work_item(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    invalid_ledger: bool,
) -> None:
    """Record direct curator mutations under the worker execution epoch."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = create_runtime(cwd=workspace)
    assert runtime.repository is not None and runtime.work_items is not None

    try:
        record = runtime.repository.create_memory(
            title="Authentication target",
            content="JWT coverage is required for client authentication.",
            summary="Generic summary.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
        )
        assert record is not None
        item, _ = runtime.work_items.enqueue_unique(
            family_key=WORK_FAMILY_MEMORY_CURATION_REVIEW,
            execution_lane=EXECUTION_LANE_AGENTIC,
            workspace_id=runtime.workspace_id,
            payload={"seed_memory_ids": [record.id]},
            idempotency_key="direct-curator-normalize",
        )
        provider = _DirectProvider(runtime, record.id)
        runtime.ai_agent_provider = provider
        if invalid_ledger:
            monkeypatch.setattr(
                "mcp_memory.core.task_handlers.agentic_tool_tracking.finalize_agentic_tool_tracking",
                lambda ctx, task_id: SimpleNamespace(
                    total_calls=1,
                    mutating_calls=1,
                    tool_names_used=["internal_unexpected_tool"],
                    tool_call_ledger=[
                        {
                            "sequence": 1,
                            "tool_name": "internal_unexpected_tool",
                            "kind": "mutation",
                            "status": "success",
                            "argument_keys": [],
                            "memory_ids": [],
                        }
                    ],
                ),
            )
        runtime.provider_usage = _ConversationUsage(
            [
                SimpleNamespace(
                    id=1,
                    request_id="provider-request",
                    attempt=1,
                    task_id="direct-curator-task",
                    completed_at=1.0,
                    input_tokens=10,
                    output_tokens=2,
                    total_tokens=12,
                    token_usage_source="test",
                ),
                SimpleNamespace(
                    id=2,
                    request_id="provider-request",
                    attempt=2,
                    task_id="direct-curator-task",
                    completed_at=2.0,
                    input_tokens=20,
                    output_tokens=4,
                    total_tokens=24,
                    token_usage_source="test",
                ),
            ]
        )

        result = await handle_memory_curator_task(runtime, _task(runtime), object())

        assert provider.allowed_tools is not None
        assert "internal_update_memory_record" in provider.allowed_tools
        assert result["execution_mode"] == "curation_direct_mcp"
        assert result["curation_outcome"] == "applied_verified"
        assert result["tool_calls_executed"] == 1
        assert result["mutations"] == 1
        assert result["actual_mutation_count"] == 1
        assert result["tool_call_ledger_validation"]["valid"] is not invalid_ledger
        assert result["curation_campaign_result"]["productive_mutation_count"] == 0
        assert result["provider_call_count"] == 1
        assert result["provider_calls_used"] == 1
        assert result["input_tokens"] == 20
        assert result["output_tokens"] == 4
        assert result["total_tokens"] == 24
        persisted_runs = runtime.curation.list_runs()
        assert any(run.frontier_key == "direct:direct-curator-task" for run in persisted_runs)
        expected_ledger = [
            {
                "sequence": 1,
                "tool_name": "internal_update_memory_record",
                "kind": "mutation",
                "status": "success",
                "argument_keys": ["memory_id", "summary", "task_id"],
                "memory_ids": [str(record.id)],
            }
        ]
        if invalid_ledger:
            expected_ledger[0]["tool_name"] = "internal_unexpected_tool"
            expected_ledger[0]["argument_keys"] = []
            expected_ledger[0]["memory_ids"] = []
        if invalid_ledger:
            assert result["tool_call_ledger"] == expected_ledger
        else:
            actual_entry = result["tool_call_ledger"][0]
            assert {
                key: actual_entry[key]
                for key in expected_ledger[0]
            } == expected_ledger[0]
            assert actual_entry["call_id"]
            assert actual_entry["task_id"] == "direct-curator-task"
            assert actual_entry["execution_epoch"] == 0
            assert actual_entry["session_id"] is None
        assert "A direct MCP curator conclusion." not in str(result["tool_call_ledger"])
        refreshed = runtime.repository.get_memory(record.id)
        assert refreshed is not None and refreshed.summary == "A direct MCP curator conclusion."
        assert runtime.work_items.get_item(item.id).status == "completed"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_direct_campaign_does_not_report_unmatched_mutations_as_verified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep raw mutations unverified when matching evidence is unavailable."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = create_runtime(cwd=workspace)
    assert runtime.repository is not None and runtime.work_items is not None

    try:
        record = runtime.repository.create_memory(
            title="Unmatched evidence target",
            content="This record exercises the unverified mutation contract.",
            summary="Telemetry target.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
        )
        assert record is not None
        runtime.work_items.enqueue_unique(
            family_key=WORK_FAMILY_MEMORY_CURATION_REVIEW,
            execution_lane=EXECUTION_LANE_AGENTIC,
            workspace_id=runtime.workspace_id,
            payload={"seed_memory_ids": [record.id]},
            idempotency_key="direct-curator-unmatched-evidence",
        )
        runtime.ai_agent_provider = _DirectProvider(runtime, record.id)
        monkeypatch.setattr(runtime.direct_mutation_evidence, "list_for_execution", lambda *_args: [])

        result = await handle_memory_curator_task(runtime, _task(runtime), object())

        assert result["actual_mutation_count"] == 1
        assert result["verified_mutation_count"] == 0
        assert result["mutations"] == 0
        assert result["curation_outcome"] == "applied_unverified"
        assert result["quality_evidence_status"] == "not_observed"
        assert result["quality_evidence_reason"] == "direct_mutation_evidence_missing"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_direct_campaign_surfaces_evidence_persistence_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fail the curator task when the provider hides an evidence write failure."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = create_runtime(cwd=workspace)
    assert runtime.repository is not None and runtime.work_items is not None

    try:
        record = runtime.repository.create_memory(
            title="Persistence failure target",
            content="This record exists to exercise telemetry failure propagation.",
            summary="Telemetry target.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
        )
        assert record is not None
        runtime.work_items.enqueue_unique(
            family_key=WORK_FAMILY_MEMORY_CURATION_REVIEW,
            execution_lane=EXECUTION_LANE_AGENTIC,
            workspace_id=runtime.workspace_id,
            payload={"seed_memory_ids": [record.id]},
            idempotency_key="direct-curator-persistence-failure",
        )
        provider = _DirectProvider(runtime, record.id, swallow_mutation_errors=True)
        runtime.ai_agent_provider = provider

        def fail_save(_evidence) -> None:
            raise RuntimeError("database write failed")

        monkeypatch.setattr(runtime.direct_mutation_evidence, "save", fail_save)

        with pytest.raises(RuntimeError, match="direct_mutation_evidence_persistence_failed"):
            await handle_memory_curator_task(runtime, _task(runtime), object())
    finally:
        runtime.close()
