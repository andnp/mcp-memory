from __future__ import annotations

from mcp_memory.core.task_handlers.ingest_run_support import (
    IngestRunAccumulator,
    should_continue_ingest_run,
)


def test_ingest_run_accumulator_absorbs_batch_results_and_builds_handler_payload() -> None:
    state = IngestRunAccumulator()
    state.absorb_batch_result(
        {
            "created_ids": ["memory-1"],
            "claimed_ids": [1, 2],
            "recoverable_ids": [2],
            "released_ids": [3],
            "entry_dispositions": [{"entry_id": 1, "disposition": "created", "memory_id": "memory-1"}],
            "meaningful_actions": 2,
        }
    )

    payload = state.build_handler_result(
        requested_grouping_strategy="semantic-seeded",
        grouping_strategy_used="lexical-seeded",
        grouping_fallback_reason="semantic_grouping_unavailable",
        pending_remaining=4,
    )

    assert state.batches_processed == 1
    assert payload["created_memory_ids"] == ["memory-1"]
    assert payload["claimed_entry_ids"] == [1, 2]
    assert payload["recoverable_entry_ids"] == [2]
    assert payload["released_entry_ids"] == [3]
    assert payload["meaningful_actions"] == 2
    assert payload["requested_grouping_strategy"] == "semantic-seeded"
    assert payload["grouping_strategy_used"] == "lexical-seeded"
    assert payload["grouping_fallback_reason"] == "semantic_grouping_unavailable"
    assert payload["pending_remaining"] == 4


def test_should_continue_ingest_run_requires_meaningful_work_and_pending_entries() -> None:
    assert should_continue_ingest_run(batch_meaningful_actions=1, pending_remaining=2) is True
    assert should_continue_ingest_run(batch_meaningful_actions=0, pending_remaining=2) is False
    assert should_continue_ingest_run(batch_meaningful_actions=1, pending_remaining=0) is False