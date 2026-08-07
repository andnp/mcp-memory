from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.providers.interfaces import AgenticRunResult
from mcp_memory.core.task_handlers import CURATOR_TASK_NAME
from mcp_memory.core.task_handlers.curator_handlers import handle_memory_curator_task
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.mcp.runtime import create_runtime
from mcp_memory.mcp.internal_mutation_services import internal_update_memory_record_service
from mcp_memory.work_item_store import EXECUTION_LANE_AGENTIC, WORK_FAMILY_MEMORY_CURATION_REVIEW

pytestmark = pytest.mark.medium


class _DirectSession:
    def __init__(self, ctx: ApplicationContext, task_id: str, memory_id: str) -> None:
        self._ctx = ctx
        self._task_id = task_id
        self._memory_id = memory_id

    async def run_agent(self, prompt: str) -> AgenticRunResult:
        assert "no planning or submission stage" in prompt
        tracker = self._ctx.internal_tool_call_tracker
        assert tracker is not None
        tracker.record_call("internal_update_memory_record", task_id=self._task_id)
        result = internal_update_memory_record_service(
            self._ctx,
            {
                "memory_id": self._memory_id,
                "summary": "A direct MCP curator conclusion.",
                "task_id": self._task_id,
            },
        )
        return AgenticRunResult(
            status="success",
            parsed={"summary": json.dumps(result), "mutations_attempted": 1},
            raw_text=json.dumps(result),
        )

    async def close(self) -> None:
        return None


class _DirectProvider:
    def __init__(self, ctx: ApplicationContext, memory_id: str) -> None:
        self._ctx = ctx
        self._memory_id = memory_id
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
        return _DirectSession(self._ctx, "direct-curator-task", self._memory_id)

    async def run_agent(self, prompt: str) -> AgenticRunResult:
        del prompt
        return AgenticRunResult(status="success", raw_text="{}")


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
async def test_direct_campaign_invokes_mcp_mutation_and_completes_work_item(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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

        result = await handle_memory_curator_task(runtime, _task(runtime), object())

        assert provider.allowed_tools is not None
        assert "internal_update_memory_record" in provider.allowed_tools
        assert result["execution_mode"] == "curation_direct_mcp"
        assert result["curation_outcome"] == "applied"
        assert result["tool_calls_executed"] == 1
        assert result["mutations"] == 1
        refreshed = runtime.repository.get_memory(record.id)
        assert refreshed is not None and refreshed.summary == "A direct MCP curator conclusion."
        assert runtime.work_items.get_item(item.id).status == "completed"
    finally:
        runtime.close()
