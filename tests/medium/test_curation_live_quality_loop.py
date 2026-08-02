from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from mcp_memory.core.curation_quality import CurationQualitySampler
from mcp_memory.core.sampling import SamplingBatch
from mcp_memory.core.task_handlers import CURATOR_TASK_NAME
from mcp_memory.core.task_handlers.curator_handlers import handle_memory_curator_task
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.mcp.runtime import create_runtime
from mcp_memory.mutation_history import Protection, ProtectionMode


pytestmark = pytest.mark.medium


class _LiveJSONProvider:
    provider_trust_class = "local"

    def __init__(self, *, invalid: bool = False) -> None:
        self.invalid = invalid
        self.payloads: list[dict[str, Any]] = []

    async def ask_json(self, prompt: str) -> dict[str, object]:
        payload = json.loads(prompt.split("\n", 1)[1])
        self.payloads.append(payload)
        request = payload["request"]
        seeds = [item["memory_id"] for item in payload["context"]["seeds"]]
        if self.invalid:
            return {
                "run_id": request["run_id"],
                "plan_id": request["plan_id"],
                "frontier_key": request["frontier_key"],
                "context_fingerprint": request["context_fingerprint"],
                "seed_memory_ids": seeds,
                "actions": [
                    {
                        "operation": "delete_memory",
                        "action_id": str(uuid4()),
                        "target_id": seeds[0],
                        "confidence": 1.0,
                        "rationale": "invalid destructive operation",
                    }
                ],
                "retained": [],
                "rationale": "invalid plan",
            }

        if len(seeds) == 1:
            actions = [
                {
                    "operation": "normalize_memory",
                    "action_id": str(uuid4()),
                    "target_id": seeds[0],
                    "confidence": 1.0,
                    "rationale": "make the summary specific",
                    "summary": "A specific live curation summary.",
                }
            ]
            retained = []
        else:
            applied_id, protected_id, no_op_id = seeds
            actions = [
                {
                    "operation": "normalize_memory",
                    "action_id": str(uuid4()),
                    "target_id": applied_id,
                    "confidence": 1.0,
                    "rationale": "make the summary specific",
                    "summary": "A specific live curation summary.",
                },
                {
                    "operation": "normalize_memory",
                    "action_id": str(uuid4()),
                    "target_id": protected_id,
                    "confidence": 1.0,
                    "rationale": "attempt a protected normalization",
                    "summary": "A protected live curation summary.",
                },
            ]
            retained = [
                {
                    "memory_id": no_op_id,
                    "reason": "already_focused",
                    "rationale": "the candidate is already focused",
                }
            ]

        return {
            "run_id": request["run_id"],
            "plan_id": request["plan_id"],
            "frontier_key": request["frontier_key"],
            "context_fingerprint": request["context_fingerprint"],
            "seed_memory_ids": seeds,
            "actions": actions,
            "retained": retained,
            "rationale": "apply one safe action while retaining one candidate",
        }


def _task(runtime: Any, task_id: str) -> TaskRecord:
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


def _seed_batch(records: list[Any]) -> SamplingBatch:
    return SamplingBatch(
        requested_strategy="quality-signal",
        strategy_used="quality-signal",
        strategy_fallback_reason=None,
        candidate_count=len(records),
        records=records,
        strategy_selection_mode="deterministic",
        strategy_selection_reason="selected=quality-signal",
        strategy_selection_scores={"quality-signal": 0.9},
        selector_feature_snapshot={"strategy_signals": {"quality_signal_share": 0.75}},
    )


def _force_quality_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    from mcp_memory.core import curation_shadow

    real_sampler = CurationQualitySampler

    def factory(**kwargs: Any) -> CurationQualitySampler:
        return real_sampler(**kwargs, sample_rate=1.0)

    monkeypatch.setattr(curation_shadow, "CurationQualitySampler", factory)


def _patch_candidates(monkeypatch: pytest.MonkeyPatch, records: list[Any]) -> None:
    from mcp_memory.core.task_handlers import curator_handlers

    batch = _seed_batch(records)
    monkeypatch.setattr(
        curator_handlers,
        "acquire_curator_candidates",
        lambda _ctx, _request: batch,
    )


def _create_record(runtime: Any, title: str, summary: str) -> Any:
    assert runtime.repository is not None
    record = runtime.repository.create_memory(
        title=title,
        content=f"Durable content for {title}.",
        summary=summary,
        workspace_ids=[runtime.workspace_id or "global"],
        memory_type="fact",
    )
    assert record is not None
    return record


@pytest.mark.asyncio
async def test_live_campaign_applies_policy_allowed_action_blocks_protected_action_and_records_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    runtime = create_runtime(cwd=tmp_path / "workspace")
    assert runtime.repository is not None
    try:
        applied = _create_record(runtime, "Live applied target", "Generic summary.")
        protected = _create_record(runtime, "Live protected target", "Generic summary.")
        no_op = _create_record(runtime, "Live retained target", "Already focused summary.")
        records = [applied, protected, no_op]
        runtime.mutation_history.set_protection(
            Protection(
                memory_id=UUID(protected.id),
                mode=ProtectionMode.MANUAL_REVIEW_REQUIRED,
                reason="operator review required",
            )
        )
        provider = _LiveJSONProvider()
        runtime.ai_json_provider = provider
        _patch_candidates(monkeypatch, records)
        _force_quality_sampling(monkeypatch)

        result = await handle_memory_curator_task(
            runtime,
            _task(runtime, "live-quality-loop-task"),
            object(),
        )

        assert result["curation_outcome"] == "partially_applied"
        assert result["mutations"] == 1
        assert "manual_review_required" in result["curation_rejection_codes"]
        assert runtime.repository.get_memory(applied.id).summary == "A specific live curation summary."
        assert runtime.repository.get_memory(protected.id).summary == protected.summary

        context_seed = provider.payloads[0]["context"]["seeds"][0]
        assert context_seed["selection_reason"] == "selected=quality-signal"
        assert context_seed["selection_signals"] == {"quality_signal_share": 0.75}
        assert context_seed["selection_scores"] == {"quality-signal": 0.9}

        store = runtime.curation
        applied_state = store.get_candidate_state(UUID(applied.id))
        protected_state = store.get_candidate_state(UUID(protected.id))
        no_op_state = store.get_candidate_state(UUID(no_op.id))
        assert applied_state is not None and applied_state.disposition.value == "actioned"
        assert applied_state.last_disposition_reason == "verified_receipt"
        assert protected_state is not None and protected_state.disposition.value == "escalated"
        assert protected_state.last_disposition_reason == "manual_review_required"
        assert no_op_state is not None and no_op_state.disposition.value == "cooldown"
        assert no_op_state.last_disposition_reason == "already_focused"

        evidence = result["curation_campaign_result"]["quality_evidence"]
        assert evidence
        assert {item["status"] for item in evidence} == {"no_query"}
        assert runtime.db_manager.get_connection().execute(
            "SELECT COUNT(*) FROM memory_tool_events"
        ).fetchone()[0] == 0
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_live_campaign_quality_sampling_evaluates_historical_query_without_user_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    runtime = create_runtime(cwd=tmp_path / "workspace")
    assert runtime.repository is not None
    try:
        target = _create_record(runtime, "Historical query target", "Generic summary.")
        now = datetime.now(UTC)
        connection = runtime.db_manager.get_connection()
        connection.execute(
            """
            INSERT INTO memory_tool_events (
                invocation_id, caller_kind, event_kind, memory_id, query_text,
                result_rank, result_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "live-quality-query",
                "user",
                "search",
                target.id,
                "historical authentication query",
                1,
                1,
                (now - timedelta(minutes=1)).isoformat(),
            ),
        )
        connection.commit()
        before_event_count = connection.execute(
            "SELECT COUNT(*) FROM memory_tool_events"
        ).fetchone()[0]

        provider = _LiveJSONProvider()
        runtime.ai_json_provider = provider
        _patch_candidates(monkeypatch, [target])
        _force_quality_sampling(monkeypatch)

        result = await handle_memory_curator_task(
            runtime,
            _task(runtime, "live-quality-evaluated-task"),
            object(),
        )

        evidence = result["curation_campaign_result"]["quality_evidence"]
        assert len(evidence) == 1
        assert evidence[0]["status"] == "evaluated"
        assert evidence[0]["query_id"] == "live-quality-query"
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_tool_events"
        ).fetchone()[0] == before_event_count
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_live_campaign_rejects_invalid_operation_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    runtime = create_runtime(cwd=tmp_path / "workspace")
    assert runtime.repository is not None
    try:
        target = _create_record(runtime, "Invalid operation target", "Original summary.")
        provider = _LiveJSONProvider(invalid=True)
        runtime.ai_json_provider = provider
        _patch_candidates(monkeypatch, [target, _create_record(runtime, "Unused target", "Unused summary.")])

        result = await handle_memory_curator_task(
            runtime,
            _task(runtime, "live-quality-invalid-task"),
            object(),
        )

        assert result["curation_outcome"] == "invalid_plan"
        assert result["mutations"] == 0
        assert "schema_invalid" in result["curation_rejection_codes"]
        assert runtime.repository.get_memory(target.id).summary == "Original summary."
        state = runtime.curation.get_candidate_state(UUID(target.id))
        assert state is not None
        assert state.disposition.value == "escalated"
        assert state.last_disposition_reason == "invalid_plan"
        assert runtime.db_manager.get_connection().execute(
            "SELECT COUNT(*) FROM memory_mutation_events"
        ).fetchone()[0] == 0
    finally:
        runtime.close()
