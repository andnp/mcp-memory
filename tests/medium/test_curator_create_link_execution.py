from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from mcp_memory.config import CurationConfig
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
                        "absent_links": [link],
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
async def test_enabled_create_link_routes_to_graph_specialist_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    runtime, source, target, item = _seed_runtime(tmp_path)
    try:
        runtime.ai_json_provider = _CreateLinkJSONProvider()
        runtime.config = replace(
            runtime.config,
            curation=CurationConfig(create_link_execution_enabled=True),
        )
        result = await handle_memory_curator_task(runtime, _task(runtime, "curator-create-link-task"), object())

        assert runtime.repository.get_links(source.id, direction="outgoing") == []
        assert result["execution_mode"] == "curation_verified_executor"
        assert result["curation_outcome"] == "deferred"
        assert result["mutations"] == 0
        assert runtime.work_items.get_item(item.id).status == "deferred"
        specialist_items = runtime.work_items.list_items(family_key="graph_link_review")
        assert len(specialist_items) == 1
        assert specialist_items[0].payload["operation"] == "create_link"
        assert specialist_items[0].payload["primary_family"] == "graph_linker"
        assert runtime.db_manager.get_connection().execute(
            "SELECT COUNT(*) FROM memory_link_revisions"
        ).fetchone()[0] == 0
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_disabled_create_link_execution_stays_in_shadow_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    runtime, source, target, item = _seed_runtime(tmp_path)
    try:
        runtime.ai_json_provider = _CreateLinkJSONProvider()
        runtime.config = replace(
            runtime.config,
            curation=CurationConfig(shadow_mode_enabled=True, create_link_execution_enabled=False),
        )
        result = await handle_memory_curator_task(runtime, _task(runtime, "curator-disabled-create-link-task"), object())

        assert runtime.repository.get_links(source.id, direction="outgoing") == []
        assert result["execution_mode"] == "curation_shadow_harness"
        assert result["curation_outcome"] == "deferred"
        assert runtime.work_items.get_item(item.id).status == "deferred"
        assert runtime.db_manager.get_connection().execute(
            "SELECT COUNT(*) FROM memory_link_revisions"
        ).fetchone()[0] == 0
        assert runtime.db_manager.get_connection().execute(
            "SELECT COUNT(*) FROM memory_mutation_events"
        ).fetchone()[0] == 0
    finally:
        runtime.close()
