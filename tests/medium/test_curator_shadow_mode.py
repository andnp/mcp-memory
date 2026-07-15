from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mcp_memory.config import CurationConfig
from mcp_memory.core.task_handlers import CURATOR_TASK_NAME
from mcp_memory.core.task_handlers.curator_handlers import handle_memory_curator_task
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.mcp.runtime import create_runtime
from mcp_memory.work_item_store import EXECUTION_LANE_AGENTIC, WORK_FAMILY_MEMORY_CURATION_REVIEW


pytestmark = pytest.mark.medium


class _ShadowJSONProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def ask_json(self, prompt: str) -> dict[str, object]:
        self.calls += 1
        payload = json.loads(prompt.split("\n", 1)[1])
        request = payload["request"]
        context = payload["context"]
        seed = context["seeds"][0]["memory_id"]
        return {
            "run_id": request["run_id"],
            "plan_id": request["plan_id"],
            "frontier_key": request["frontier_key"],
            "context_fingerprint": request["context_fingerprint"],
            "seed_memory_ids": [seed],
            "actions": [
                {
                    "operation": "normalize_memory",
                    "action_id": "00000000-0000-0000-0000-000000000003",
                    "target_id": seed,
                    "confidence": 1.0,
                    "rationale": "the summary can be made more specific",
                    "summary": "A specific durable authentication conclusion.",
                }
            ],
            "retained": [],
            "rationale": "shadow plan for a safe normalization",
        }


@pytest.mark.asyncio
async def test_curator_shadow_mode_persists_plan_and_defers_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = create_runtime(cwd=workspace)
    assert runtime.repository is not None
    assert runtime.work_items is not None
    assert runtime.curation is not None
    assert runtime.config is not None

    provider = _ShadowJSONProvider()
    runtime.ai_json_provider = provider
    runtime.config = replace(runtime.config, curation=CurationConfig(shadow_mode_enabled=True))
    record = runtime.repository.create_memory(
        title="Authentication target",
        content="JWT coverage is required for client authentication.",
        summary="Generic summary.",
        workspace_ids=[runtime.workspace_id or "global"],
        memory_type="fact",
        tags=["auth"],
    )
    task = TaskRecord(
        id="curator-shadow-task",
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
    item, _ = runtime.work_items.enqueue_unique(
        family_key=WORK_FAMILY_MEMORY_CURATION_REVIEW,
        execution_lane=EXECUTION_LANE_AGENTIC,
        workspace_id=runtime.workspace_id,
        payload={"seed_memory_ids": [record.id]},
        idempotency_key="curator-shadow-test",
    )

    try:
        result = await handle_memory_curator_task(runtime, task, provider)

        assert provider.calls == 1
        assert result["shadow_mode"] is True
        assert result["execution_mode"] == "curation_shadow_harness"
        assert result["curation_outcome"] == "deferred"
        assert result["curation_plan"] is not None
        assert result["curation_decisions"][0]["decision"] == "accepted"
        assert runtime.work_items.get_item(item.id).status == "deferred"
        assert runtime.repository.get_memory(record.id).summary == "Generic summary."
        assert runtime.db_manager.get_connection().execute(
            "SELECT COUNT(*) FROM memory_mutation_events"
        ).fetchone()[0] == 0
        assert runtime.curation.get_run(result["curation_run_id"]) is not None
    finally:
        runtime.close()
