from typing import Any

from mcp_memory.core.task_results import TaskRunResult, coerce_task_run_result
from mcp_memory.management.agent_run_reporting import (
    CURATOR_NARRATIVE_ONLY,
    CURATOR_OBSERVED_MUTATION,
    CURATOR_PROVIDER_FAILURE,
    CURATOR_UNKNOWN_LEGACY,
    CURATOR_VALID_NO_OP,
    build_agent_run_history_payload,
    build_recent_agent_runs,
    classify_curator_run,
    decode_run_result,
    extract_ingest_audit,
    extract_run_result_metadata,
)
from mcp_memory.management.models import AgentRunHistoryPayload, MutationOutcomePayload, RunResultMetadataPayload
from mcp_memory.utils.sql_portable_runner import ManagementQueryAdapter
from mcp_memory.management.reporting_rows import coerce_task_result_view
from mcp_memory.management.task_sampling_summary import (
    SAMPLER_OUTCOME_NEUTRAL,
    SAMPLER_OUTCOME_PROVIDER_FAILURE,
    SAMPLER_OUTCOME_QUALITY_FAILURE,
    SAMPLER_OUTCOME_QUALITY_PASS,
    SAMPLER_OUTCOME_ROLLED_BACK,
    SAMPLER_OUTCOME_UNOBSERVED,
    SAMPLER_UNOBSERVED_REASON_MISSING_QUALITY_EVIDENCE,
    build_selection_strategy_utility_priors,
    build_task_sampling_summary,
    project_sampler_outcome,
)


def test_build_recent_agent_runs_uses_explicit_query_adapter() -> None:
    seen: dict[str, object] = {}

    class ExplicitAdapter:
        def adapt_query(self, query: str) -> str:
            return query

        def uses_sqlite_connection_api(self) -> bool:
            return False

        def fetchall(self, db_manager: object, query: str, params=None) -> list[dict[str, object]]:
            seen.update(db_manager=db_manager, query=query, params=params)
            return []

    db_manager = object()
    adapter: ManagementQueryAdapter = ExplicitAdapter()
    build_recent_agent_runs(db_manager, None, query_adapter=adapter)

    assert seen["db_manager"] is db_manager
    assert "FROM task_runs" in str(seen["query"])


def test_classify_curator_run_uses_persisted_outcomes_not_provider_narrative() -> None:
    cases = [
        (
            CURATOR_PROVIDER_FAILURE,
            "failed",
            "provider_timeout",
            {},
        ),
        (
            CURATOR_NARRATIVE_ONLY,
            "completed",
            None,
            {"summary": "Rewrote the memory", "tool_calls_executed": 0, "mutations": 0},
        ),
        (
            CURATOR_OBSERVED_MUTATION,
            "completed",
            None,
            {"summary": "No changes were made", "tool_calls_executed": 2, "mutations": 1},
        ),
        (
            CURATOR_VALID_NO_OP,
            "completed",
            None,
            {"summary": "Reviewed and retained the records", "tool_calls_executed": 3, "mutations": 0},
        ),
        (
            CURATOR_UNKNOWN_LEGACY,
            "completed",
            None,
            {"summary": "Rewrote the memory"},
        ),
    ]

    for expected, status, error_text, result in cases:
        classification, _reason = classify_curator_run(
            task_name="memory-curator",
            status=status,
            error_text=error_text,
            result=result,
        )
        assert classification == expected


def test_build_agent_run_history_payload_adds_curator_classification_without_dropping_fields() -> None:
    result = {
        "summary": "Updated the record",  # Provider prose must not establish mutation.
        "tool_calls_executed": 0,
        "mutations": 0,
        "candidate_count": 4,
    }

    payload = build_agent_run_history_payload(
        task_id="curator-task",
        task_name="memory-curator",
        status="completed",
        started_at=10.0,
        completed_at=12.0,
        duration_seconds=2.0,
        error_text=None,
        result=result,
        detail_level="full",
    )

    assert payload.result == result
    assert payload.result_metadata.candidate_count == 4
    assert payload.run_classification == CURATOR_NARRATIVE_ONLY
    assert payload.classification_reason == "persisted tool_calls_executed=0; narrative present"


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


def test_build_agent_run_history_payload_compact_mode_keeps_entry_lists_out_of_ingest_audit() -> None:
    payload = build_agent_run_history_payload(
        task_id="task-1",
        task_name="ingest-system1",
        status="completed",
        started_at=10.0,
        completed_at=12.0,
        duration_seconds=2.0,
        error_text=None,
        result={
            "claimed_entry_ids": [101],
            "entry_dispositions": [{"entry_id": 101, "disposition": "created"}],
            "provider_reported_entry_outcomes": [{"entry_id": 101, "disposition": "created"}],
        },
        detail_level="compact",
    )

    assert payload.ingest_audit.claimed_count == 1
    assert payload.ingest_audit.entry_dispositions == []
    assert payload.ingest_audit.provider_reported_entry_outcomes == []


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
            "selector_feature_snapshot": {
                "strategy_signals": {"semantic_overlap_share": 0.81},
                "candidate_population": {
                    "count": 8,
                    "metrics": {
                        "content_chars": {"count": 8, "min": 10.0, "p50": 22.0, "p90": 40.0, "max": 50.0, "mean": 24.5},
                    },
                    "shares": {"never_surfaced_share": 0.5},
                },
                "selected_population": {
                    "count": 3,
                    "metrics": {
                        "content_chars": {"count": 3, "min": 20.0, "p50": 25.0, "p90": 30.0, "max": 30.0, "mean": 25.0},
                    },
                    "shares": {"never_surfaced_share": 0.3333},
                },
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
    assert payload.result_metadata.selector_feature_snapshot.strategy_signals == {"semantic_overlap_share": 0.81}
    assert payload.result_metadata.selector_feature_snapshot.candidate_population.count == 8
    assert payload.result_metadata.selector_feature_snapshot.candidate_population.metrics["content_chars"].p50 == 22.0
    assert payload.result_metadata.selector_feature_snapshot.selected_population.count == 3
    assert payload.result_metadata.candidate_count == 8


def test_build_agent_run_history_payload_accepts_core_task_run_result_wrapper() -> None:
    result = coerce_task_run_result(
        {
            "strategy_used": "semantic",
            "candidate_count": 3,
            "claimed_entry_ids": [101],
        }
    )

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

    assert isinstance(result, TaskRunResult)
    assert payload.result_summary == result.summary
    assert payload.result_metadata.strategy_used == "semantic"
    assert payload.result_metadata.candidate_count == 3
    assert payload.result == {"strategy_used": "semantic", "candidate_count": 3, "claimed_entry_ids": [101]}


def test_extract_run_result_metadata_builds_structured_mutation_outcome_from_legacy_delta_keys() -> None:
    metadata = extract_run_result_metadata(
        {
            "created": 1,
            "merged": 2,
            "updated": 3,
            "archived": 4,
            "degraded": 5,
            "restored": 6,
        }
    )

    assert metadata.mutation_outcome == MutationOutcomePayload(
        created=1,
        merged=2,
        updated=3,
        archived=4,
        degraded=5,
        restored=6,
    )
    assert metadata.mutations == 21


def test_extract_run_result_metadata_preserves_flat_mutations_without_structured_deltas() -> None:
    metadata = extract_run_result_metadata({"mutations": 7})

    assert metadata.mutations == 7
    assert metadata.mutation_outcome == MutationOutcomePayload()


def test_extract_run_result_metadata_preserves_provider_call_ratios_with_derived_mutations() -> None:
    metadata = extract_run_result_metadata(
        {
            "created": 2,
            "updated": 4,
            "provider_calls_used": 3,
            "claimed_work_item_count": 9,
            "tool_calls_executed": 12,
        }
    )

    assert metadata.mutations == 6
    assert metadata.mutations_per_provider_call == 2.0
    assert metadata.work_items_per_provider_call == 3.0
    assert metadata.tool_calls_per_provider_call == 4.0


def test_task_result_view_helpers_and_payloads_do_not_share_cached_models() -> None:
    result_view = coerce_task_result_view(
        {
            "claimed_entry_ids": [101],
            "provider_calls_used": 1,
            "entry_dispositions": [
                {
                    "entry_id": 101,
                    "disposition": "created",
                    "memory_id": "memory-101",
                }
            ],
        }
    )

    metadata = extract_run_result_metadata(result_view)
    metadata.provider_calls_used = 99
    full_audit = extract_ingest_audit(result_view, include_entries=True)
    full_audit.entry_dispositions[0].memory_id = "mutated-memory"
    compact_payload = build_agent_run_history_payload(
        task_id="task-1",
        task_name="ingest-system1",
        status="completed",
        started_at=10.0,
        completed_at=12.0,
        duration_seconds=2.0,
        error_text=None,
        result=result_view,
        detail_level="compact",
    )
    compact_payload.ingest_audit.claimed_count = 42

    assert result_view.metadata.provider_calls_used == 1
    assert result_view.ingest_audit.claimed_count == 1
    assert result_view.ingest_audit.entry_dispositions[0].memory_id == "memory-101"


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
                    strategy_selection_mode="deterministic_scores",
                    strategy_selection_reason="selected=semantic from deterministic scores",
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
                    strategy_selection_mode="utility_priors",
                    strategy_selection_reason="utility prior tie-break",
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
    assert [
        (
            row.task_name,
            row.strategy_selection_mode,
            row.strategy_used,
            row.reason_family,
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
        for row in summary.selector_behavior
    ] == [
        (
            "graph-linker",
            "deterministic_scores",
            "semantic",
            "deterministic_signals",
            1,
            0,
            1,
            3,
            4,
            12.0,
            1.0,
            3.0,
            0.75,
            0,
            0.0,
        ),
        (
            "graph-linker",
            "utility_priors",
            "semantic",
            "fallback",
            1,
            1,
            0,
            0,
            2,
            8.0,
            0.0,
            0.0,
            0.0,
            1,
            1.0,
        ),
        (
            "memory-curator",
            "unspecified",
            "lexical",
            "unknown",
            1,
            0,
            1,
            1,
            1,
            5.0,
            1.0,
            1.0,
            1.0,
            0,
            0.0,
        ),
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


def test_run_metadata_extracts_persisted_curator_quality_evidence() -> None:
    metadata = extract_run_result_metadata(
        {
            "strategy_used": "semantic",
            "curation_campaign_result": {
                "quality_evidence": [
                    {"useful_work": True, "retrieval_regression_count": 0, "zero_result_change": -1},
                    {"useful_work": False, "retrieval_regression_count": 1, "zero_result_change": 0},
                ]
            },
        }
    )

    assert metadata.quality_evidence_runs == 2
    assert metadata.useful_work_count == 1
    assert metadata.retrieval_regression_count == 1
    assert metadata.zero_result_change == -1


def test_quality_evidence_overrides_mutation_volume_in_curator_priors() -> None:
    runs = []
    for strategy, useful, regressions in (("semantic", True, 0), ("anomaly", False, 1)):
        for index in range(3):
            runs.append(
                AgentRunHistoryPayload(
                    task_id=f"{strategy}-{index}",
                    task_name="memory-curator",
                    status="completed",
                    started_at=float(index),
                    completed_at=float(index + 1),
                    duration_seconds=1.0,
                    result_metadata=RunResultMetadataPayload(
                        strategy_used=strategy,
                        mutations=5 if strategy == "anomaly" else 1,
                        quality_evidence_runs=1,
                        useful_work_count=int(useful),
                        retrieval_regression_count=regressions,
                    ),
                )
            )

    priors = build_selection_strategy_utility_priors(
        runs,
        task_name="memory-curator",
        allowed_strategies=("semantic", "anomaly"),
    )

    assert priors["semantic"] > priors["anomaly"]


def _sampler_run(**metadata: Any) -> AgentRunHistoryPayload:
    return AgentRunHistoryPayload(
        task_id="sampler-test",
        task_name="memory-curator",
        status=str(metadata.pop("status", "completed")),
        started_at=1.0,
        completed_at=2.0,
        duration_seconds=1.0,
        result_metadata=RunResultMetadataPayload(
            strategy_used="semantic",
            **metadata,
        ),
    )


def test_project_sampler_outcome_accepts_only_a_complete_quality_wave() -> None:
    passed = project_sampler_outcome(
        _sampler_run(
            mutations=4,
            quality_evidence_runs=2,
            quality_acceptance_met=True,
            useful_work_count=2,
        )
    )
    rejected = project_sampler_outcome(
        _sampler_run(
            mutations=9,
            quality_evidence_runs=2,
            quality_acceptance_met=False,
            useful_work_count=2,
        )
    )

    assert (passed.outcome, passed.productive_mutations) == (SAMPLER_OUTCOME_QUALITY_PASS, 4)
    assert (rejected.outcome, rejected.productive_mutations) == (SAMPLER_OUTCOME_QUALITY_FAILURE, 0)


def test_project_sampler_outcome_accepts_persisted_heuristic_wave() -> None:
    run = _sampler_run(
        mutations=4,
        quality_evidence_runs=2,
        quality_acceptance_met=None,
        quality_neutral_count=0,
        quality_rejected_count=0,
        retrieval_regression_count=2,
        curation_outcome="applied",
    )

    projected = project_sampler_outcome(run)

    assert (projected.outcome, projected.productive_mutations) == (SAMPLER_OUTCOME_QUALITY_PASS, 4)


def test_project_sampler_outcome_excludes_neutral_rollbacks_and_provider_failures() -> None:
    neutral = project_sampler_outcome(
        _sampler_run(mutations=3, quality_evidence_runs=1, quality_neutral_count=1, quality_acceptance_met=True)
    )
    rollback = project_sampler_outcome(
        _sampler_run(mutations=3, curation_outcome="verification_failed")
    )
    provider = project_sampler_outcome(
        _sampler_run(mutations=3, status="failed", provider_failure_classification="rate_limit")
    )

    assert (neutral.outcome, neutral.productive_mutations) == (SAMPLER_OUTCOME_NEUTRAL, 0)
    assert (rollback.outcome, rollback.productive_mutations) == (SAMPLER_OUTCOME_ROLLED_BACK, 0)
    assert (provider.outcome, provider.productive_mutations) == (SAMPLER_OUTCOME_PROVIDER_FAILURE, 0)
    assert provider.quality_failure is False


def test_project_sampler_outcome_marks_mutations_without_quality_evidence_unobserved() -> None:
    projected = project_sampler_outcome(_sampler_run(mutations=3))

    assert projected.outcome == SAMPLER_OUTCOME_UNOBSERVED
    assert projected.reason == SAMPLER_UNOBSERVED_REASON_MISSING_QUALITY_EVIDENCE
    assert projected.productive_mutations == 0
    assert projected.quality_failure is False


def test_sampling_summary_reports_productive_mutations_separately() -> None:
    summary = build_task_sampling_summary(
        [
            _sampler_run(mutations=4, quality_evidence_runs=1, quality_acceptance_met=True),
            _sampler_run(mutations=8, quality_evidence_runs=1, quality_acceptance_met=False),
            _sampler_run(mutations=2),
            _sampler_run(mutations=6, status="failed", provider_failure_classification="timeout"),
        ]
    )

    row = summary.selection_utility[0]
    assert row.total_mutations == 20
    assert row.productive_mutations == 4
    assert row.quality_pass_runs == 1
    assert row.quality_failure_runs == 1
    assert row.unobserved_runs == 1
    assert row.provider_failure_runs == 1


def test_selection_strategy_utility_priors_include_deduplicator_strategies_with_enough_runs() -> None:
    priors = build_selection_strategy_utility_priors(
        [
            AgentRunHistoryPayload(
                task_id="task-1",
                task_name="deduplicator",
                status="completed",
                started_at=1.0,
                completed_at=2.0,
                duration_seconds=1.0,
                result_metadata=RunResultMetadataPayload(
                    strategy_used="semantic",
                    candidate_count=6,
                    tool_calls_executed=2,
                    mutations=2,
                ),
            ),
            AgentRunHistoryPayload(
                task_id="task-2",
                task_name="deduplicator",
                status="completed",
                started_at=3.0,
                completed_at=4.0,
                duration_seconds=1.0,
                result_metadata=RunResultMetadataPayload(
                    strategy_used="semantic",
                    candidate_count=7,
                    tool_calls_executed=2,
                    mutations=1,
                ),
            ),
            AgentRunHistoryPayload(
                task_id="task-3",
                task_name="deduplicator",
                status="completed",
                started_at=5.0,
                completed_at=6.0,
                duration_seconds=1.0,
                result_metadata=RunResultMetadataPayload(
                    strategy_used="semantic",
                    candidate_count=5,
                    tool_calls_executed=1,
                    mutations=1,
                ),
            ),
            AgentRunHistoryPayload(
                task_id="task-4",
                task_name="deduplicator",
                status="completed",
                started_at=7.0,
                completed_at=8.0,
                duration_seconds=1.0,
                result_metadata=RunResultMetadataPayload(
                    strategy_used="anomaly",
                    candidate_count=4,
                    tool_calls_executed=3,
                    mutations=0,
                ),
            ),
        ],
        task_name="deduplicator",
        allowed_strategies=("semantic", "anomaly"),
    )

    assert set(priors) == {"semantic"}
    assert 0.0 < priors["semantic"] <= 1.0


def test_selection_strategy_utility_priors_include_taxonomist_strategies_with_enough_runs() -> None:
    priors = build_selection_strategy_utility_priors(
        [
            AgentRunHistoryPayload(
                task_id="task-1",
                task_name="taxonomist",
                status="completed",
                started_at=1.0,
                completed_at=2.0,
                duration_seconds=1.0,
                result_metadata=RunResultMetadataPayload(
                    strategy_used="never-surfaced",
                    candidate_count=8,
                    tool_calls_executed=2,
                    mutations=2,
                ),
            ),
            AgentRunHistoryPayload(
                task_id="task-2",
                task_name="taxonomist",
                status="completed",
                started_at=3.0,
                completed_at=4.0,
                duration_seconds=1.0,
                result_metadata=RunResultMetadataPayload(
                    strategy_used="never-surfaced",
                    candidate_count=9,
                    tool_calls_executed=2,
                    mutations=1,
                ),
            ),
            AgentRunHistoryPayload(
                task_id="task-3",
                task_name="taxonomist",
                status="completed",
                started_at=5.0,
                completed_at=6.0,
                duration_seconds=1.0,
                result_metadata=RunResultMetadataPayload(
                    strategy_used="never-surfaced",
                    candidate_count=7,
                    tool_calls_executed=1,
                    mutations=1,
                ),
            ),
            AgentRunHistoryPayload(
                task_id="task-4",
                task_name="taxonomist",
                status="completed",
                started_at=7.0,
                completed_at=8.0,
                duration_seconds=1.0,
                result_metadata=RunResultMetadataPayload(
                    strategy_used="cold-storage",
                    candidate_count=6,
                    tool_calls_executed=3,
                    mutations=0,
                ),
            ),
        ],
        task_name="taxonomist",
        allowed_strategies=("never-surfaced", "cold-storage", "bounded-noise"),
    )

    assert set(priors) == {"never-surfaced"}
    assert 0.0 < priors["never-surfaced"] <= 1.0
