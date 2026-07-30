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
from mcp_memory.mutation_history import Protection, ProtectionMode
from mcp_memory.work_item_store import EXECUTION_LANE_AGENTIC, WORK_FAMILY_MEMORY_CURATION_REVIEW


pytestmark = pytest.mark.medium


class _NormalizeJSONProvider:
    provider_trust_class = "local"

    async def ask_json(self, prompt: str) -> dict[str, object]:
        payload = json.loads(prompt.split("\n", 1)[1])
        request = payload["request"]
        seed = payload["context"]["seeds"][0]["memory_id"]
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
                    "rationale": "make the summary specific",
                    "summary": "A durable authentication conclusion.",
                }
            ],
            "retained": [],
            "rationale": "normalize the target summary",
        }


class _CampaignJSONProvider:
    provider_trust_class = "local"

    async def ask_json(self, prompt: str) -> dict[str, object]:
        payload = json.loads(prompt.split("\n", 1)[1])
        request = payload["request"]
        context = payload["context"]
        normalized = context["seeds"][0]
        source = context["seeds"][1]
        target = context["seeds"][2]
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
            "seed_memory_ids": [normalized["memory_id"], source["memory_id"], target["memory_id"]],
            "actions": [
                {
                    "operation": "normalize_memory",
                    "action_id": "00000000-0000-0000-0000-000000000003",
                    "target_id": normalized["memory_id"],
                    "confidence": 1.0,
                    "rationale": "make the summary specific",
                    "summary": "A durable authentication conclusion.",
                    "preconditions": {
                        "record_tokens": {
                            normalized["memory_id"]: context["record_tokens"][normalized["memory_id"]],
                        },
                    },
                },
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
                },
            ],
            "retained": [],
            "rationale": "normalize the target and record the supporting edge",
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


@pytest.mark.asyncio
async def test_enabled_normalize_uses_verified_executor_and_completes_claimed_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = create_runtime(cwd=workspace)
    assert runtime.repository is not None and runtime.work_items is not None and runtime.config is not None

    try:
        runtime.ai_json_provider = _NormalizeJSONProvider()
        runtime.config = replace(
            runtime.config,
            curation=CurationConfig(normalize_execution_enabled=True),
        )
        record = runtime.repository.create_memory(
            title="Authentication target",
            content="JWT coverage is required for client authentication.",
            summary="Generic summary.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        assert record is not None
        item, _ = runtime.work_items.enqueue_unique(
            family_key=WORK_FAMILY_MEMORY_CURATION_REVIEW,
            execution_lane=EXECUTION_LANE_AGENTIC,
            workspace_id=runtime.workspace_id,
            payload={"seed_memory_ids": [record.id]},
            idempotency_key="curator-verified-normalize",
        )
        result = await handle_memory_curator_task(runtime, _task(runtime, "curator-verified-task"), object())

        refreshed = runtime.repository.get_memory(record.id)
        receipt = runtime.curation.list_receipts(result["curation_run_id"])[0]
        assert result["execution_mode"] == "curation_verified_executor"
        assert result["curation_outcome"] == "applied"
        assert result["mutations"] == 1
        assert refreshed is not None and refreshed.summary == "A durable authentication conclusion."
        assert receipt.status.value == "verified"
        assert receipt.mutation_event_id is not None
        assert runtime.work_items.get_item(item.id).status == "completed"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_default_verified_campaign_reports_authoritative_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = create_runtime(cwd=workspace)
    assert runtime.repository is not None and runtime.work_items is not None and runtime.config is not None

    try:
        runtime.ai_json_provider = _CampaignJSONProvider()
        normalized = runtime.repository.create_memory(
            title="Authentication target",
            content="JWT coverage is required for client authentication.",
            summary="Generic summary.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
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
        assert normalized is not None and source is not None and target is not None
        item, _ = runtime.work_items.enqueue_unique(
            family_key=WORK_FAMILY_MEMORY_CURATION_REVIEW,
            execution_lane=EXECUTION_LANE_AGENTIC,
            workspace_id=runtime.workspace_id,
            payload={"seed_memory_ids": [normalized.id, source.id, target.id]},
            idempotency_key="curator-default-campaign",
        )

        result = await handle_memory_curator_task(runtime, _task(runtime, "curator-default-campaign-task"), object())

        refreshed = runtime.repository.get_memory(normalized.id)
        campaign = result["curation_campaign_result"]
        assert result["execution_mode"] == "curation_verified_campaign"
        assert result["curation_outcome"] == "applied"
        assert result["mutations"] == 2
        assert campaign["mutation_count"] == 2
        assert campaign["verified_action_count"] == 2
        assert campaign["verification_failure_count"] == 0
        assert campaign["retention_count"] == 0
        assert campaign["specialist_family_routing"]["route_count"] == 0
        assert campaign["specialist_family_routing"]["family_counts"] == {}
        assert campaign["receipts"][0]["status"] == "verified"
        assert campaign["receipts"][0]["operation"] == "normalize_memory"
        assert campaign["receipts"][1]["operation"] == "create_link"
        assert refreshed is not None and refreshed.summary == "A durable authentication conclusion."
        assert runtime.work_items.get_item(item.id).status == "completed"
        assert runtime.work_items.list_items(family_key="graph_link_review") == []
        assert runtime.repository.get_links(source.id, direction="outgoing")
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_local_manual_review_protection_denies_normalize_before_any_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = create_runtime(cwd=workspace)
    assert runtime.repository is not None and runtime.work_items is not None and runtime.config is not None

    try:
        runtime.ai_json_provider = _NormalizeJSONProvider()
        runtime.config = replace(runtime.config, curation=CurationConfig(normalize_execution_enabled=True))
        record = runtime.repository.create_memory(
            title="Protected authentication target",
            content="JWT coverage is required for client authentication.",
            summary="Generic summary.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
        )
        assert record is not None
        runtime.mutation_history.set_protection(
            Protection(
                memory_id=record.id,
                mode=ProtectionMode.MANUAL_REVIEW_REQUIRED,
                reason="operator review required",
            )
        )
        item, _ = runtime.work_items.enqueue_unique(
            family_key=WORK_FAMILY_MEMORY_CURATION_REVIEW,
            execution_lane=EXECUTION_LANE_AGENTIC,
            workspace_id=runtime.workspace_id,
            payload={"seed_memory_ids": [record.id]},
            idempotency_key="curator-protected-normalize",
        )

        result = await handle_memory_curator_task(runtime, _task(runtime, "curator-protected-task"), object())

        refreshed = runtime.repository.get_memory(record.id)
        assert result["curation_outcome"] == "deferred"
        assert "manual_review_required" in result["curation_rejection_codes"]
        assert refreshed is not None and refreshed.summary == record.summary
        assert runtime.work_items.get_item(item.id).status == "deferred"
        connection = runtime.db_manager.get_connection()
        for table in ("memory_mutation_events", "memory_record_revisions", "embedding_repair_queue", "curation_action_receipts"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_disabled_normalize_execution_stays_in_shadow_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = create_runtime(cwd=workspace)
    assert runtime.repository is not None and runtime.work_items is not None and runtime.config is not None

    try:
        runtime.ai_json_provider = _NormalizeJSONProvider()
        runtime.config = replace(
            runtime.config,
            curation=CurationConfig(shadow_mode_enabled=True, normalize_execution_enabled=False),
        )
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
            idempotency_key="curator-disabled-normalize",
        )
        result = await handle_memory_curator_task(runtime, _task(runtime, "curator-disabled-task"), object())

        assert result["execution_mode"] == "curation_shadow_harness"
        assert result["curation_outcome"] == "deferred"
        assert runtime.repository.get_memory(record.id).summary == "Generic summary."
        assert runtime.work_items.get_item(item.id).status == "deferred"
        assert runtime.db_manager.get_connection().execute(
            "SELECT COUNT(*) FROM memory_mutation_events"
        ).fetchone()[0] == 0
    finally:
        runtime.close()
