from mcp_memory.management.agent_run_reporting import build_agent_run_history_payload, decode_run_result
from mcp_memory.management.models import AgentRunHistoryPayload, RunResultMetadataPayload
from mcp_memory.management.task_sampling_summary import build_selection_strategy_utility_priors, build_task_sampling_summary


def test_decode_run_result_accepts_postgres_jsonb_mapping() -> None:
    raw_result = {
        "claimed_entry_ids": [101, 102],
        "processed_entry_ids": [101],
        "released_entry_ids": [102],
        "meaningful_actions": 1,
        "mutations": 1,
        "entry_dispositions": [
            {
                "entry_id": 101,
                "disposition": "created",
                "finalization_status": "recoverable",
                "memory_id": "memory-101",
            }
        ],
    }

    result = decode_run_result(raw_result)
    payload = build_agent_run_history_payload(
        task_id="task-1",
        task_name="ingest-system1",
        status="completed",
        started_at=10.0,
        completed_at=12.0,
        duration_seconds=2.0,
        error_text=None,
        result=result,
        detail_level="full",
    )

    assert payload.result == raw_result
    assert payload.ingest_audit.claimed_count == 2
    assert payload.ingest_audit.handled_count == 1
    assert payload.ingest_audit.released_count == 1
    assert payload.ingest_audit.meaningful_actions == 1
    assert payload.ingest_audit.mutations == 1
    assert payload.ingest_audit.entry_dispositions[0].memory_id == "memory-101"


def test_build_agent_run_history_payload_extracts_selector_diagnostics() -> None:
    payload = build_agent_run_history_payload(
        task_id="task-1",
        task_name="memory-curator",
        status="completed",
        started_at=10.0,
        completed_at=12.0,
        duration_seconds=2.0,
        error_text=None,
        result={
            "requested_strategy": "semantic",
            "strategy_used": "semantic",
            "strategy_selection_mode": "deterministic_scores",
            "strategy_selection_reason": "selected=semantic",
            "strategy_selection_scores": {
                "semantic": 0.75,
                "anomaly": 0.2,
            },
            "candidate_count": 8,
        },
        detail_level="full",
    )

    assert payload.result_metadata.requested_strategy == "semantic"
    assert payload.result_metadata.strategy_used == "semantic"
    assert payload.result_metadata.strategy_selection_mode == "deterministic_scores"
    assert payload.result_metadata.strategy_selection_reason == "selected=semantic"
    assert payload.result_metadata.strategy_selection_scores == {"semantic": 0.75, "anomaly": 0.2}
    assert payload.result_metadata.candidate_count == 8


def test_build_task_sampling_summary_aggregates_selection_utility_metrics() -> None:
    summary = build_task_sampling_summary(
        [
            AgentRunHistoryPayload(
                task_id="task-1",
                task_name="graph-linker",
                status="completed",
                started_at=1.0,
                completed_at=2.0,
                duration_seconds=1.0,
                result_metadata=RunResultMetadataPayload(
                    strategy_used="semantic",
                    candidate_count=12,
                    tool_calls_executed=4,
                    mutations=3,
                    grouping_strategy_used="compatibility",
                ),
            ),
            AgentRunHistoryPayload(
                task_id="task-2",
                task_name="graph-linker",
                status="completed",
                started_at=3.0,
                completed_at=4.0,
                duration_seconds=1.0,
                result_metadata=RunResultMetadataPayload(
                    strategy_used="semantic",
                    strategy_fallback_reason="insufficient_candidates",
                    candidate_count=8,
                    tool_calls_executed=2,
                    mutations=0,
                    grouping_strategy_used="compatibility",
                    grouping_fallback_reason="small_batch",
                ),
            ),
            AgentRunHistoryPayload(
                task_id="task-3",
                task_name="memory-curator",
                status="completed",
                started_at=5.0,
                completed_at=6.0,
                duration_seconds=1.0,
                result_metadata=RunResultMetadataPayload(
                    strategy_used="lexical",
                    candidate_count=5,
                    tool_calls_executed=1,
                    mutations=1,
                ),
            ),
        ]
    )

    assert [(row.name, row.runs, row.fallbacks, row.tasks) for row in summary.selection] == [
        ("semantic", 2, 1, ["graph-linker"]),
        ("lexical", 1, 0, ["memory-curator"]),
    ]
    assert [(row.name, row.runs, row.fallbacks, row.tasks) for row in summary.grouping] == [
        ("compatibility", 2, 1, ["graph-linker"]),
    ]
    assert [
        (
            row.task_name,
            row.strategy_used,
            row.runs,
            row.fallback_count,
            row.mutation_runs,
            row.total_mutations,
            row.total_tool_calls,
            row.average_candidate_count,
            row.mutation_rate,
            row.mutations_per_run,
            row.mutations_per_tool_call,
            row.no_op_runs,
            row.no_op_rate,
        )
        for row in summary.selection_utility
    ] == [
        ("graph-linker", "semantic", 2, 1, 1, 3, 6, 10.0, 0.5, 1.5, 0.5, 1, 0.5),
        ("memory-curator", "lexical", 1, 0, 1, 1, 1, 5.0, 1.0, 1.0, 1.0, 0, 0.0),
    ]


def test_selection_strategy_utility_priors_ignore_strategies_below_minimum_runs() -> None:
    priors = build_selection_strategy_utility_priors(
        [
            AgentRunHistoryPayload(
                task_id="task-1",
                task_name="memory-curator",
                status="completed",
                started_at=1.0,
                completed_at=2.0,
                duration_seconds=1.0,
                result_metadata=RunResultMetadataPayload(
                    strategy_used="semantic",
                    candidate_count=12,
                    tool_calls_executed=3,
                    mutations=2,
                ),
            ),
            AgentRunHistoryPayload(
                task_id="task-2",
                task_name="memory-curator",
                status="completed",
                started_at=3.0,
                completed_at=4.0,
                duration_seconds=1.0,
                result_metadata=RunResultMetadataPayload(
                    strategy_used="semantic",
                    candidate_count=10,
                    tool_calls_executed=2,
                    mutations=1,
                ),
            ),
        ],
        task_name="memory-curator",
        allowed_strategies=("semantic",),
    )

    assert priors == {}


def test_selection_strategy_utility_priors_include_curator_strategies_with_enough_runs() -> None:
    priors = build_selection_strategy_utility_priors(
        [
            AgentRunHistoryPayload(
                task_id="task-1",
                task_name="memory-curator",
                status="completed",
                started_at=1.0,
                completed_at=2.0,
                duration_seconds=1.0,
                result_metadata=RunResultMetadataPayload(
                    strategy_used="cold-storage",
                    candidate_count=10,
                    tool_calls_executed=2,
                    mutations=2,
                ),
            ),
            AgentRunHistoryPayload(
                task_id="task-2",
                task_name="memory-curator",
                status="completed",
                started_at=3.0,
                completed_at=4.0,
                duration_seconds=1.0,
                result_metadata=RunResultMetadataPayload(
                    strategy_used="cold-storage",
                    candidate_count=11,
                    tool_calls_executed=2,
                    mutations=1,
                ),
            ),
            AgentRunHistoryPayload(
                task_id="task-3",
                task_name="memory-curator",
                status="completed",
                started_at=5.0,
                completed_at=6.0,
                duration_seconds=1.0,
                result_metadata=RunResultMetadataPayload(
                    strategy_used="cold-storage",
                    candidate_count=9,
                    tool_calls_executed=1,
                    mutations=1,
                ),
            ),
            AgentRunHistoryPayload(
                task_id="task-4",
                task_name="graph-linker",
                status="completed",
                started_at=7.0,
                completed_at=8.0,
                duration_seconds=1.0,
                result_metadata=RunResultMetadataPayload(
                    strategy_used="cold-storage",
                    candidate_count=9,
                    tool_calls_executed=5,
                    mutations=0,
                ),
            ),
        ],
        task_name="memory-curator",
        allowed_strategies=("cold-storage", "semantic"),
    )

    assert set(priors) == {"cold-storage"}
    assert 0.0 < priors["cold-storage"] <= 1.0
