from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mcp_memory.core.task_handlers import CURATOR_TASK_NAME
from mcp_memory.core.task_handlers.curator_handlers import handle_memory_curator_task
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.mcp.runtime import create_runtime
from mcp_memory.work_item_store import EXECUTION_LANE_AGENTIC, WORK_FAMILY_MEMORY_CURATION_REVIEW


pytestmark = pytest.mark.medium


class _CreateLinkJSONProvider:
    provider_trust_class = "local"

    async def ask_json(self, prompt: str) -> dict[str, object]:
        payload = json.loads(prompt.split("\n", 1)[1])
        request = payload["request"]
        context = payload["context"]
        source, target = context["seeds"][:2]
        link = {
            "source_id": source["memory_id"],
            "target_id": target["memory_id"],
            "link_type": "SUPPORTS",
            "context": "The source records the target as supporting evidence.",
        }
        absent_link = {
            "source_id": source["memory_id"],
            "target_id": target["memory_id"],
            "link_type": "SUPPORTS",
        }
        return {
            "run_id": request["run_id"],
            "plan_id": request["plan_id"],
            "frontier_key": request["frontier_key"],
            "context_fingerprint": request["context_fingerprint"],
            "seed_memory_ids": [source["memory_id"], target["memory_id"]],
            "actions": [
                {
                    "operation": "create_link",
                    "action_id": "00000000-0000-0000-0000-000000000004",
                    "source_id": source["memory_id"],
                    "target_id": target["memory_id"],
                    "link_type": "SUPPORTS",
                    "context": link["context"],
                    "confidence": 1.0,
                    "rationale": "the source directly supports the target",
                    "evidence": [{"link": link}],
                    "preconditions": {
                        "record_tokens": {
                            source["memory_id"]: context["record_tokens"][source["memory_id"]],
                            target["memory_id"]: context["record_tokens"][target["memory_id"]],
                        },
                        "absent_links": [absent_link],
                    },
                }
            ],
            "retained": [],
            "rationale": "record the exact supporting edge",
        }


def _task(runtime, task_id: str) -> TaskRecord:
    return TaskRecord(
        id=task_id,
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


def _seed_runtime(tmp_path: Path) -> tuple[Any, Any, Any, Any]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = create_runtime(cwd=workspace)
    assert runtime.repository is not None and runtime.work_items is not None and runtime.config is not None
    source = runtime.repository.create_memory(
        title="Supporting source",
        content="The source describes the authentication evidence.",
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
        idempotency_key="curator-create-link-test",
    )
    return runtime, source, target, item


@pytest.mark.asyncio
async def test_default_campaign_applies_create_link_with_verification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    runtime, source, target, item = _seed_runtime(tmp_path)
    try:
        runtime.ai_json_provider = _CreateLinkJSONProvider()
        result = await handle_memory_curator_task(runtime, _task(runtime, "curator-create-link-task"), object())

        assert result["execution_mode"] == "curation_verified_campaign"
        assert result["curation_outcome"] == "applied"
        assert result["mutations"] == 1
        assert runtime.repository.get_links(source.id, direction="outgoing")
        receipt = runtime.curation.list_receipts(result["curation_run_id"])[0]
        assert receipt.status.value == "verified"
        assert runtime.work_items.get_item(item.id).status == "completed"
    finally:
        runtime.close()
