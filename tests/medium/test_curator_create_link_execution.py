from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.curation_models import CurationRunOutcome
from mcp_memory.core.ports.work_items import EXECUTION_LANE_AGENTIC, WORK_FAMILY_MEMORY_CURATION_REVIEW
from mcp_memory.core.providers.interfaces import AgenticRunResult
from mcp_memory.core.task_handlers import CURATOR_TASK_NAME
from mcp_memory.core.task_handlers.curator_handlers import handle_memory_curator_task
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.mcp.runtime import create_runtime
from mcp_memory.mcp.transport import dispatch_internal_memory_tool

pytestmark = pytest.mark.medium


class _DirectLinkSession:
    def __init__(self, ctx: ApplicationContext, task_id: str, source_id: str, target_id: str) -> None:
        self._ctx = ctx
        self._task_id = task_id
        self._source_id = source_id
        self._target_id = target_id

    async def run_agent(self, prompt: str) -> AgenticRunResult:
        assert "no planning or submission stage" in prompt
        result = await dispatch_internal_memory_tool(
            self._ctx,
            "internal_create_memory_link",
            {
                "source_id": self._source_id,
                "target_id": self._target_id,
                "link_type": "SUPPORTS",
                "context": "The source directly supports the target.",
                "task_id": self._task_id,
            },
        )
        return AgenticRunResult(status="success", parsed={"summary": result[0].text}, raw_text=result[0].text)

    async def close(self) -> None:
        return None


class _DirectLinkProvider:
    def __init__(self, ctx: ApplicationContext, source_id: str, target_id: str) -> None:
        self._ctx = ctx
        self._source_id = source_id
        self._target_id = target_id

    def with_usage_context(self, **context: object) -> _DirectLinkProvider:
        del context
        return self

    async def open_agent_session(
        self,
        *,
        allowed_tool_names: tuple[str, ...] | None = None,
    ) -> _DirectLinkSession:
        assert allowed_tool_names is not None
        assert "internal_create_memory_link" in allowed_tool_names
        return _DirectLinkSession(self._ctx, "direct-link-task", self._source_id, self._target_id)

    async def run_agent(self, prompt: str) -> AgenticRunResult:
        del prompt
        return AgenticRunResult(status="success", raw_text="{}")


def _task(runtime: Any) -> TaskRecord:
    return TaskRecord(
        id="direct-link-task",
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
async def test_direct_campaign_applies_create_link_mcp_tool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep the link mutation while rejecting its measured quality wave."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = create_runtime(cwd=workspace)
    assert runtime.repository is not None and runtime.work_items is not None

    try:
        source = runtime.repository.create_memory(
            title="Supporting source",
            content="The source describes authentication evidence.",
            summary="Authentication evidence source.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
        )
        target = runtime.repository.create_memory(
            title="Supported target",
            content="The target is the durable authentication conclusion.",
            summary="Authentication conclusion.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
        )
        assert source is not None and target is not None
        item, _ = runtime.work_items.enqueue_unique(
            family_key=WORK_FAMILY_MEMORY_CURATION_REVIEW,
            execution_lane=EXECUTION_LANE_AGENTIC,
            workspace_id=runtime.workspace_id,
            payload={"seed_memory_ids": [source.id, target.id]},
            idempotency_key="direct-curator-link",
        )
        runtime.ai_agent_provider = _DirectLinkProvider(runtime, source.id, target.id)

        result = await handle_memory_curator_task(runtime, _task(runtime), object())

        assert result["curation_outcome"] == CurationRunOutcome.QUALITY_REJECTED.value
        assert result["mutations"] == 1
        assert runtime.repository.get_links(source.id, direction="outgoing")
        assert runtime.work_items.get_item(item.id).status == "completed"
    finally:
        runtime.close()
