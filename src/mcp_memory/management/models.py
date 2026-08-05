from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class EmbeddingStatusPayload(BaseModel):
    model_name: str | None = None
    configured_model_name: str | None = None
    backend: str | None = None
    model_cached: bool = False
    fallback_persistence_policy: str = "allowed"
    blocked_fallback_write_count: int = 0
    last_blocked_fallback_model_name: str | None = None
    integrity_events: "EmbeddingIntegrityEventSummaryPayload" = Field(default_factory=lambda: EmbeddingIntegrityEventSummaryPayload())


class EmbeddingIntegrityEventSnapshotPayload(BaseModel):
    created_at: float = 0.0
    model_name: str | None = None
    source_kind: str | None = None
    source_id: str | None = None
    scanned_row_count: int | None = None
    invalid_row_count: int | None = None
    mixed_dimension_group_count: int | None = None


class EmbeddingIntegrityEventSummaryPayload(BaseModel):
    total: int = 0
    by_kind: dict[str, int] = Field(default_factory=dict)
    last_scan: EmbeddingIntegrityEventSnapshotPayload | None = None
    last_blocked_fallback_write: EmbeddingIntegrityEventSnapshotPayload | None = None


class SearchHealthPayload(BaseModel):
    semantic_enabled: bool = False
    available: bool = False
    degraded: bool = False
    fallback_count: int = 0
    rebuild_count: int = 0
    background_repair_enabled: bool = False
    background_repair_wait_seconds: float = 0.0
    queued_repair_backlog_count: int = 0
    running_repair_count: int = 0
    oldest_queued_repair_age_seconds: float | None = None
    repair_wait_count: int = 0
    partial_semantic_search_count: int = 0
    last_partial_semantic_at: str | None = None
    last_repair_wait_seconds: float = 0.0
    last_repair_candidate_count: int = 0
    last_repair_pending_count: int = 0
    last_error: str | None = None
    last_failure_at: str | None = None
    last_recovery_at: str | None = None
    last_integrity_check_at: str | None = None
    integrity_check_error: str | None = None


class ExecutionAttemptHealthPayload(BaseModel):
    stale_after_seconds: float = 0.0
    running_task_count: int = 0
    running_attempt_count: int = 0
    fresh_attempt_count: int = 0
    stale_attempt_count: int = 0
    missing_attempt_count: int = 0
    live_subprocess_count: int = 0
    dead_subprocess_count: int = 0


class CacheRecentMetricsPayload(BaseModel):
    window_minutes: int = 15
    search_requests: int = 0
    fresh_exact_search_hits: int = 0
    stale_exact_search_fallbacks: int = 0
    projection_fallbacks: int = 0
    read_requests: int = 0
    validated_read_hits: int = 0
    read_validation_mismatches: int = 0
    read_validation_failures: int = 0
    warmed_projection_rows: int = 0
    fresh_exact_search_hit_rate: float = 0.0
    stale_exact_search_fallback_rate: float = 0.0
    projection_fallback_rate: float = 0.0
    validated_read_hit_rate: float = 0.0
    read_validation_mismatch_rate: float = 0.0
    read_validation_failure_rate: float = 0.0


class CacheMetricsPayload(BaseModel):
    search_requests: int = 0
    fresh_exact_search_hits: int = 0
    stale_exact_search_fallbacks: int = 0
    projection_fallbacks: int = 0
    read_requests: int = 0
    validated_read_hits: int = 0
    read_validation_mismatches: int = 0
    read_validation_failures: int = 0
    warmed_projection_rows: int = 0
    cached_search_result_count: int = 0
    cached_read_record_count: int = 0
    cached_projection_count: int = 0
    fresh_exact_search_hit_rate: float = 0.0
    stale_exact_search_fallback_rate: float = 0.0
    projection_fallback_rate: float = 0.0
    validated_read_hit_rate: float = 0.0
    read_validation_mismatch_rate: float = 0.0
    read_validation_failure_rate: float = 0.0
    recent: CacheRecentMetricsPayload = Field(default_factory=CacheRecentMetricsPayload)


class CacheHealthPayload(BaseModel):
    enabled: bool = False
    mode: str | None = None
    state: str = "disabled"
    path: str | None = None
    metrics: CacheMetricsPayload = Field(default_factory=CacheMetricsPayload)


class TransportRequestDiagnosticPayload(BaseModel):
    request_id: int = 0
    path: str
    phase: str
    age_ms: float = 0.0
    queue_wait_ms: float = 0.0
    execution_ms: float = 0.0
    total_ms: float = 0.0
    response_status: str | None = None
    error: str | None = None


class TransportDiagnosticsPayload(BaseModel):
    current_in_flight_count: int = 0
    current_constrained_in_flight_count: int = 0
    max_concurrent_requests: int = 0
    request_slots_available: int = 0
    queued_waiter_count: int = 0
    recent_completed_request_count: int = 0
    recent_queue_wait_avg_ms: float = 0.0
    recent_queue_wait_max_ms: float = 0.0
    recent_execution_avg_ms: float = 0.0
    recent_execution_max_ms: float = 0.0
    active_requests: list[TransportRequestDiagnosticPayload] = Field(default_factory=list)
    recent_requests: list[TransportRequestDiagnosticPayload] = Field(default_factory=list)


class HealthPayload(BaseModel):
    status: str
    storage_backend: str | None = None
    workspace_root: str | None = None
    memory_path: str | None = None
    db_path: str | None = None
    runtime_active: bool
    client_count: int
    task_queue_enabled: bool
    embeddings: EmbeddingStatusPayload = Field(default_factory=EmbeddingStatusPayload)
    search: SearchHealthPayload = Field(default_factory=SearchHealthPayload)
    cache: CacheHealthPayload = Field(default_factory=CacheHealthPayload)
    transport_diagnostics: TransportDiagnosticsPayload = Field(default_factory=TransportDiagnosticsPayload)
    execution_attempts: ExecutionAttemptHealthPayload = Field(default_factory=ExecutionAttemptHealthPayload)


class OperatorLogDigestPayload(BaseModel):
    window_minutes: int = 15
    total: int = 0
    by_level: dict[str, int] = Field(default_factory=dict)
    recent_errors: list["RuntimeLogPayload"] = Field(default_factory=list)


class OperatorWarningDigestPayload(BaseModel):
    window_minutes: int = 15
    total: int = 0
    recent: list["RuntimeLogPayload"] = Field(default_factory=list)


class OperatorTaskDigestPayload(BaseModel):
    recent_status_counts: dict[str, int] = Field(default_factory=dict)
    recent: list["AgentRunHistoryPayload"] = Field(default_factory=list)
    recent_failure_count: int = 0
    recent_retry_count: int = 0
    recent_failures: list["AgentRunHistoryPayload"] = Field(default_factory=list)
    recent_retries: list["AgentRunHistoryPayload"] = Field(default_factory=list)


class OperatorConversationDigestPayload(BaseModel):
    window_hours: int = 24
    total: int = 0
    by_status: dict[str, int] = Field(default_factory=dict)
    recent: list["AIConversationPayload"] = Field(default_factory=list)


class OperatorMemoryActivityPayload(BaseModel):
    updated_last_15_minutes: int = 0
    updated_last_hour: int = 0
    updated_last_day: int = 0
    recent: list["CompactMemoryRecord"] = Field(default_factory=list)


class MemoryToolLatencyMetricPayload(BaseModel):
    event_kind: str
    count: int = 0
    avg_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    slow_count: int = 0


class MemoryToolLatencyPayload(BaseModel):
    window_minutes: int = 15
    slow_threshold_ms: float = 0.0
    by_event_kind: list[MemoryToolLatencyMetricPayload] = Field(default_factory=list)


class OperatorHealthSnapshotPayload(BaseModel):
    generated_at: float
    status: str = "ok"
    alerts: list[str] = Field(default_factory=list)
    health: HealthPayload
    logs: OperatorLogDigestPayload = Field(default_factory=OperatorLogDigestPayload)
    warnings: OperatorWarningDigestPayload = Field(default_factory=OperatorWarningDigestPayload)
    tasks: OperatorTaskDigestPayload = Field(default_factory=OperatorTaskDigestPayload)
    conversations: OperatorConversationDigestPayload = Field(default_factory=OperatorConversationDigestPayload)
    memory_activity: OperatorMemoryActivityPayload = Field(default_factory=OperatorMemoryActivityPayload)
    tool_latency: MemoryToolLatencyPayload = Field(default_factory=MemoryToolLatencyPayload)
    provider_policy: "NerdProviderPolicyPayload" = Field(default_factory=lambda: NerdProviderPolicyPayload())


class OverviewCounts(BaseModel):
    total: int
    by_type: dict[str, int] = Field(default_factory=dict)
    by_status: dict[str, int] = Field(default_factory=dict)


class CompactMemoryRecord(BaseModel):
    id: str
    title: str
    summary: str | None = None
    type: str
    status: str
    updated_at: str
    read_count: int = 0
    workspace_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class MemorySearchResultPayload(BaseModel):
    memory_id: str
    title: str
    summary: str | None = None
    memory_type: str
    status: str
    tags: list[str] = Field(default_factory=list)
    workspace_ids: list[str] = Field(default_factory=list)
    score: float


class MemorySearchPayload(BaseModel):
    results: list[MemorySearchResultPayload] = Field(default_factory=list)


class TaskStatusSummary(BaseModel):
    by_status: dict[str, int] = Field(default_factory=dict)
    failed_count: int


class StorageSummary(BaseModel):
    sqlite_bytes: int
    sqlite_path: str | None = None


class JournalSummary(BaseModel):
    pending_count: int


class TokenUsageSummaryPayload(BaseModel):
    provider_calls_last_hour: int = 0
    provider_calls_last_day: int = 0
    input_tokens_last_hour: int = 0
    input_tokens_last_day: int = 0
    output_tokens_last_hour: int = 0
    output_tokens_last_day: int = 0
    cached_input_tokens_last_hour: int = 0
    cached_input_tokens_last_day: int = 0
    cache_write_tokens_last_hour: int = 0
    cache_write_tokens_last_day: int = 0
    reasoning_tokens_last_hour: int = 0
    reasoning_tokens_last_day: int = 0
    total_tokens_last_hour: int = 0
    total_tokens_last_day: int = 0
    token_usage_source: str | None = None


class MemoryMetricsPayload(BaseModel):
    total_memories: int
    total_memory_lines: int
    total_summary_lines: int
    total_lines_compressed: int
    thought_buffer_entries: int
    thought_buffer_lines: int


class QueueDiagnosticPayload(BaseModel):
    task_id: str
    task_name: str
    workspace_id: str | None = None
    priority: int
    pending_state: str
    trigger: str | None = None
    created_at: float
    available_at: float
    age_seconds: float
    ready_in_seconds: float = 0.0
    overdue_seconds: float = 0.0


class SelectorMetricSnapshotPayload(BaseModel):
    count: int = 0
    min: float | None = None
    p50: float | None = None
    p90: float | None = None
    max: float | None = None
    mean: float | None = None


class SelectorPopulationSnapshotPayload(BaseModel):
    count: int = 0
    metrics: dict[str, SelectorMetricSnapshotPayload] = Field(default_factory=dict)
    shares: dict[str, float] = Field(default_factory=dict)


class SelectorFeatureSnapshotPayload(BaseModel):
    strategy_signals: dict[str, float] = Field(default_factory=dict)
    candidate_population: SelectorPopulationSnapshotPayload = Field(default_factory=SelectorPopulationSnapshotPayload)
    selected_population: SelectorPopulationSnapshotPayload = Field(default_factory=SelectorPopulationSnapshotPayload)


class SelectorFeatureRollupRowPayload(BaseModel):
    task_name: str
    run_classification: str
    runs: int = 0
    snapshot_runs: int = 0
    candidate_metric_means: dict[str, float] = Field(default_factory=dict)
    selected_metric_means: dict[str, float] = Field(default_factory=dict)
    candidate_share_means: dict[str, float] = Field(default_factory=dict)
    selected_share_means: dict[str, float] = Field(default_factory=dict)
    strategy_signal_means: dict[str, float] = Field(default_factory=dict)


class MutationOutcomePayload(BaseModel):
    created: int | None = None
    merged: int | None = None
    updated: int | None = None
    archived: int | None = None
    degraded: int | None = None
    restored: int | None = None


class RunResultMetadataPayload(BaseModel):
    requested_strategy: str | None = None
    strategy_used: str | None = None
    strategy_fallback_reason: str | None = None
    strategy_selection_mode: str | None = None
    strategy_selection_reason: str | None = None
    strategy_selection_scores: dict[str, float] = Field(default_factory=dict)
    sampler_priority_score: float | None = None
    sampler_priority_explanation: str | None = None
    selector_feature_snapshot: SelectorFeatureSnapshotPayload = Field(default_factory=SelectorFeatureSnapshotPayload)
    candidate_count: int | None = None
    sampled_memory_ids: list[str] = Field(default_factory=list)
    compatibility_group: str | None = None
    claimed_work_item_count: int | None = None
    provider_calls_used: int | None = None
    tool_calls_executed: int | None = None
    mutations: int | None = None
    mutation_outcome: MutationOutcomePayload = Field(default_factory=MutationOutcomePayload)
    work_item_batch_limit: int | None = None
    max_batches_per_run: int | None = None
    requested_grouping_strategy: str | None = None
    grouping_strategy_used: str | None = None
    grouping_fallback_reason: str | None = None
    group_count: int | None = None
    campaign_key: str | None = None
    campaign_origin_family: str | None = None
    campaign_family_keys: list[str] = Field(default_factory=list)
    campaign_continuation_supported: bool | None = None
    compatible_batch_calls: int | None = None
    provider_call_count: int | None = None
    work_items_per_provider_call: float | None = None
    mutations_per_provider_call: float | None = None
    tool_calls_per_provider_call: float | None = None
    quality_evidence_runs: int = 0
    useful_work_count: int = 0
    retrieval_regression_count: int = 0
    zero_result_change: int = 0
    quality_acceptance_met: bool | None = None
    quality_neutral_count: int = 0
    quality_rejected_count: int = 0
    provider_failure_classification: str | None = None
    curation_outcome: str | None = None


class IngestEntryDispositionPayload(BaseModel):
    entry_id: int
    disposition: str
    finalization_status: str | None = None
    memory_id: str | None = None
    memory_title: str | None = None
    reason: str | None = None


class IngestAuditPayload(BaseModel):
    claimed_count: int = 0
    handled_count: int = 0
    released_count: int = 0
    deleted_count: int = 0
    processed_count: int = 0
    meaningful_actions: int = 0
    created_count: int = 0
    touched_count: int = 0
    appended_count: int = 0
    matched_count: int = 0
    tool_calls_executed: int | None = None
    mutations: int | None = None
    provider_reported_tool_calls: int | None = None
    provider_reported_mutations: int | None = None
    created_memory_ids: list[str] = Field(default_factory=list)
    touched_memory_ids: list[str] = Field(default_factory=list)
    appended_memory_ids: list[str] = Field(default_factory=list)
    matched_memory_ids: list[str] = Field(default_factory=list)
    provider_reported_touched_memory_ids: list[str] = Field(default_factory=list)
    provider_reported_matched_memory_ids: list[str] = Field(default_factory=list)
    entry_dispositions: list[IngestEntryDispositionPayload] = Field(default_factory=list)
    provider_reported_entry_outcomes: list[IngestEntryDispositionPayload] = Field(default_factory=list)


class AgentRunPayload(BaseModel):
    task_name: str
    running_count: int = 0
    total_runs: int
    completed_runs: int
    failed_runs: int
    cancelled_runs: int = 0
    retry_runs: int
    avg_duration_seconds: float
    total_lines_compressed: int
    last_status: str | None = None
    last_completed_at: float | None = None
    seconds_since_last_completion: float | None = None
    last_error: str | None = None
    last_result_summary: str | None = None
    last_result_metadata: RunResultMetadataPayload = Field(default_factory=RunResultMetadataPayload)
    last_ingest_audit: IngestAuditPayload = Field(default_factory=IngestAuditPayload)
    next_available_at: float | None = None
    seconds_until_next_run: float | None = None


class AgentRunHistoryPayload(BaseModel):
    task_id: str | None = None
    task_name: str
    status: str
    started_at: float
    completed_at: float
    duration_seconds: float
    error_text: str | None = None
    result_summary: str | None = None
    result_metadata: RunResultMetadataPayload = Field(default_factory=RunResultMetadataPayload)
    ingest_audit: IngestAuditPayload = Field(default_factory=IngestAuditPayload)
    result: dict[str, Any] | None = None
    run_classification: str | None = None
    classification_reason: str | None = None


class AgentRunHistoryListPayload(BaseModel):
    runs: list[AgentRunHistoryPayload] = Field(default_factory=list)


class SamplingSummaryRowPayload(BaseModel):
    name: str
    runs: int = 0
    fallbacks: int = 0
    tasks: list[str] = Field(default_factory=list)


class SelectionStrategyUtilityPayload(BaseModel):
    task_name: str
    strategy_used: str
    runs: int = 0
    fallback_count: int = 0
    mutation_runs: int = 0
    total_mutations: int = 0
    total_tool_calls: int = 0
    average_candidate_count: float | None = None
    mutation_rate: float = 0.0
    mutations_per_run: float = 0.0
    mutations_per_tool_call: float | None = None
    no_op_runs: int = 0
    no_op_rate: float = 0.0
    quality_evidence_runs: int = 0
    useful_work_count: int = 0
    retrieval_regression_count: int = 0
    zero_result_change: int = 0
    productive_mutations: int = 0
    quality_pass_runs: int = 0
    quality_failure_runs: int = 0
    provider_failure_runs: int = 0
    sampler_priority_score: float | None = None
    sampler_priority_explanation: str | None = None


class SelectorBehaviorSummaryPayload(BaseModel):
    task_name: str
    strategy_selection_mode: str
    strategy_used: str
    reason_family: str
    runs: int = 0
    fallback_count: int = 0
    mutation_runs: int = 0
    total_mutations: int = 0
    total_tool_calls: int = 0
    average_candidate_count: float | None = None
    mutation_rate: float = 0.0
    mutations_per_run: float = 0.0
    mutations_per_tool_call: float | None = None
    no_op_runs: int = 0
    no_op_rate: float = 0.0


class TaskSamplingSummaryPayload(BaseModel):
    selection: list[SamplingSummaryRowPayload] = Field(default_factory=list)
    grouping: list[SamplingSummaryRowPayload] = Field(default_factory=list)
    selection_utility: list[SelectionStrategyUtilityPayload] = Field(default_factory=list)
    selector_behavior: list[SelectorBehaviorSummaryPayload] = Field(default_factory=list)


class SelectorStatsSummaryPayload(BaseModel):
    total_runs: int = 0
    selector_signal_runs: int = 0
    fresh_selector_runs: int = 0
    seeded_claimed_runs: int = 0
    unknown_runs: int = 0
    fallback_runs: int = 0
    mutation_runs: int = 0
    no_op_runs: int = 0
    total_mutations: int = 0
    total_tool_calls: int = 0
    average_candidate_count: float | None = None


class SelectorClassificationBreakdownPayload(BaseModel):
    key: str
    label: str
    runs: int = 0


class SelectorOutcomeRowPayload(BaseModel):
    task_name: str
    strategy_used: str
    strategy_selection_mode: str
    reason_family: str
    run_classification: str
    runs: int = 0
    fallback_count: int = 0
    mutation_runs: int = 0
    total_mutations: int = 0
    total_tool_calls: int = 0
    average_candidate_count: float | None = None
    no_op_runs: int = 0
    no_op_rate: float = 0.0


class SelectorRecentDiagnosticPayload(BaseModel):
    task_id: str | None = None
    task_name: str
    status: str
    completed_at: float
    duration_seconds: float
    run_classification: str
    classification_reason: str
    requested_strategy: str | None = None
    strategy_used: str | None = None
    strategy_selection_mode: str | None = None
    strategy_selection_reason: str | None = None
    strategy_fallback_reason: str | None = None
    candidate_count: int | None = None
    claimed_work_item_count: int | None = None
    mutations: int | None = None
    tool_calls_executed: int | None = None
    strategy_selection_scores: dict[str, float] = Field(default_factory=dict)
    selector_feature_snapshot: SelectorFeatureSnapshotPayload = Field(default_factory=SelectorFeatureSnapshotPayload)
    result_summary: str | None = None


class SelectorStatsPayload(BaseModel):
    generated_at: float
    window_hours: int
    run_limit: int
    summary: SelectorStatsSummaryPayload = Field(default_factory=SelectorStatsSummaryPayload)
    classification_breakdown: list[SelectorClassificationBreakdownPayload] = Field(default_factory=list)
    outcome_rows: list[SelectorOutcomeRowPayload] = Field(default_factory=list)
    feature_rollup_rows: list[SelectorFeatureRollupRowPayload] = Field(default_factory=list)
    recent_runs: list[SelectorRecentDiagnosticPayload] = Field(default_factory=list)


class RuntimeLogPayload(BaseModel):
    id: int
    created_at: float
    level: str
    logger_name: str
    source: str
    message: str
    data: dict[str, object] = Field(default_factory=dict)


class RuntimeLogListPayload(BaseModel):
    logs: list[RuntimeLogPayload] = Field(default_factory=list)


class RuntimeLogSummaryPayload(BaseModel):
    total: int = 0
    by_level: dict[str, int] = Field(default_factory=dict)
    by_source: dict[str, int] = Field(default_factory=dict)


class RuntimeLogPrunePayload(BaseModel):
    deleted: int
    max_runtime_logs: int
    max_log_age_days: int


class NerdStatPayload(BaseModel):
    key: str
    label: str
    value: float
    unit: str | None = None


class NerdCountBucketPayload(BaseModel):
    key: str
    label: str
    count: int = 0


class NerdCompositionPayload(BaseModel):
    by_workspace: list[NerdCountBucketPayload] = Field(default_factory=list)
    by_tag: list[NerdCountBucketPayload] = Field(default_factory=list)
    by_content_tag: list[NerdCountBucketPayload] = Field(default_factory=list)
    by_provenance_tag: list[NerdCountBucketPayload] = Field(default_factory=list)
    by_type: list[NerdCountBucketPayload] = Field(default_factory=list)
    by_status: list[NerdCountBucketPayload] = Field(default_factory=list)


class NerdDistributionsPayload(BaseModel):
    created_age_buckets: list[NerdCountBucketPayload] = Field(default_factory=list)
    updated_age_buckets: list[NerdCountBucketPayload] = Field(default_factory=list)
    content_size_buckets: list[NerdCountBucketPayload] = Field(default_factory=list)


class MemoryTimelineBucketPayload(BaseModel):
    bucket_start: float
    created_count: int = 0
    updated_count: int = 0
    total_content_bytes: int = 0


class MaintenanceEventPayload(BaseModel):
    task_id: str
    task_name: str
    status: str
    completed_at: float
    bucket_start: float
    duration_seconds: float = 0.0
    result_summary: str | None = None
    strategy_used: str | None = None
    impact_summary: str | None = None
    candidate_count: int | None = None
    group_count: int | None = None
    created_count: int | None = None
    merged_count: int | None = None
    updated_count: int | None = None
    archived_count: int | None = None
    lines_compressed: int | None = None
    meaningful_actions: int | None = None


class NerdMaintenancePayload(BaseModel):
    events: list[MaintenanceEventPayload] = Field(default_factory=list)


class NerdTimeCountBucketPayload(BaseModel):
    bucket_start: float
    count: int = 0


class NerdTimeShareBucketPayload(BaseModel):
    bucket_start: float
    count: int = 0
    share: float = 0.0


class NerdCountSeriesPayload(BaseModel):
    key: str
    label: str
    buckets: list[NerdTimeCountBucketPayload] = Field(default_factory=list)


class NerdShareSeriesPayload(BaseModel):
    key: str
    label: str
    buckets: list[NerdTimeShareBucketPayload] = Field(default_factory=list)


class NerdLifecycleTrendsPayload(BaseModel):
    status_events: list[NerdCountSeriesPayload] = Field(default_factory=list)
    never_surfaced_backlog: list[NerdTimeCountBucketPayload] = Field(default_factory=list)
    cold_tail: list[NerdTimeCountBucketPayload] = Field(default_factory=list)
    quality_signals: list[NerdCountSeriesPayload] = Field(default_factory=list)


class NerdQualityProducerPayload(BaseModel):
    task_id: str = "unknown"
    task_name: str = "unknown"
    tool_name: str = "unknown"
    provider_key: str = "unknown"
    provider_name: str = "unknown"
    model_name: str = "unknown"


class NerdQualityMemoryRowPayload(BaseModel):
    memory_id: str
    title: str
    summary: str | None = None
    memory_type: str
    status: str
    updated_at: str
    tags: list[str] = Field(default_factory=list)
    producer: NerdQualityProducerPayload = Field(default_factory=NerdQualityProducerPayload)


class NerdQualitySignalDrilldownPayload(BaseModel):
    key: str
    label: str
    count: int = 0
    records: list[NerdQualityMemoryRowPayload] = Field(default_factory=list)


class NerdQualityProducerAttributionPayload(BaseModel):
    signal_key: str
    signal_label: str
    count: int = 0
    repeated: bool = False
    producer: NerdQualityProducerPayload = Field(default_factory=NerdQualityProducerPayload)


class NerdQualityRemediationSignalPayload(BaseModel):
    idempotency_key: str
    defect_family: str
    defect_label: str
    count: int = 0
    threshold: int = 0
    policy_version: str
    window_start: float
    window_end: float
    memory_ids: list[str] = Field(default_factory=list)
    producer: NerdQualityProducerPayload = Field(default_factory=NerdQualityProducerPayload)
    work_item_id: str | None = None
    work_item_status: str | None = None


class NerdQualityDrilldownPayload(BaseModel):
    signals: list[NerdQualitySignalDrilldownPayload] = Field(default_factory=list)
    producer_attributions: list[NerdQualityProducerAttributionPayload] = Field(default_factory=list)


class NerdQualityRemediationPayload(BaseModel):
    stats: list[NerdStatPayload] = Field(default_factory=list)
    activity: list[NerdCountSeriesPayload] = Field(default_factory=list)
    signals: list[NerdQualityRemediationSignalPayload] = Field(default_factory=list)


class NerdGrowthDynamicsPayload(BaseModel):
    top_tag_trends: list[NerdCountSeriesPayload] = Field(default_factory=list)
    workspace_contribution_share: list[NerdShareSeriesPayload] = Field(default_factory=list)


class NerdRetrievalSummaryPayload(BaseModel):
    search_invocations: int = 0
    search_hits: int = 0
    zero_result_searches: int = 0
    read_events: int = 0
    unique_search_memories: int = 0
    unique_read_memories: int = 0


class NerdRetrievalMemoryRowPayload(BaseModel):
    memory_id: str
    title: str
    memory_type: str
    status: str
    tags: list[str] = Field(default_factory=list)
    read_count: int = 0
    search_count: int = 0
    total_count: int = 0
    last_read_at: float | None = None
    last_search_at: float | None = None


class NerdRetrievalTagRowPayload(BaseModel):
    key: str
    label: str
    read_count: int = 0
    search_count: int = 0
    total_count: int = 0


class NerdRetrievalTagTimelinePayload(BaseModel):
    key: str
    label: str
    read_buckets: list[NerdTimeCountBucketPayload] = Field(default_factory=list)
    search_buckets: list[NerdTimeCountBucketPayload] = Field(default_factory=list)


class NerdRetrievalCallerKindRowPayload(BaseModel):
    key: str
    label: str
    search_invocations: int = 0
    search_hits: int = 0
    zero_result_searches: int = 0
    read_events: int = 0
    total_events: int = 0


class NerdRetrievalQueryFamilyRowPayload(BaseModel):
    key: str
    label: str
    search_invocations: int = 0
    search_hits: int = 0
    zero_result_searches: int = 0
    unique_search_memories: int = 0
    converted_search_hits: int = 0
    conversion_rate: float = 0.0


class NerdRetrievalFunnelPayload(BaseModel):
    search_hits: int = 0
    converted_search_hits: int = 0
    conversion_rate: float = 0.0


class NerdRetrievalConversionMemoryRowPayload(BaseModel):
    memory_id: str
    title: str
    memory_type: str
    status: str
    tags: list[str] = Field(default_factory=list)
    read_count: int = 0
    search_count: int = 0
    converted_search_count: int = 0
    conversion_rate: float = 0.0
    last_read_at: float | None = None
    last_search_at: float | None = None


class NerdRetrievalEngagementEvidencePayload(BaseModel):
    memory_id: str
    query_family_key: str
    evidence_kind: str
    strength: str
    exposure_count: int = 0
    co_result_read: bool = False
    graph_provenance: dict[str, object] = Field(default_factory=dict)


class NerdRetrievalPayload(BaseModel):
    summary: NerdRetrievalSummaryPayload = Field(default_factory=NerdRetrievalSummaryPayload)
    funnel: NerdRetrievalFunnelPayload = Field(default_factory=NerdRetrievalFunnelPayload)
    by_caller_kind: list[NerdRetrievalCallerKindRowPayload] = Field(default_factory=list)
    top_query_families: list[NerdRetrievalQueryFamilyRowPayload] = Field(default_factory=list)
    top_zero_result_query_families: list[NerdRetrievalQueryFamilyRowPayload] = Field(default_factory=list)
    top_read_memories: list[NerdRetrievalMemoryRowPayload] = Field(default_factory=list)
    top_search_memories: list[NerdRetrievalMemoryRowPayload] = Field(default_factory=list)
    low_conversion_memories: list[NerdRetrievalConversionMemoryRowPayload] = Field(default_factory=list)
    engagement_evidence: list[NerdRetrievalEngagementEvidencePayload] = Field(default_factory=list)
    top_tags: list[NerdRetrievalTagRowPayload] = Field(default_factory=list)
    tag_timelines: list[NerdRetrievalTagTimelinePayload] = Field(default_factory=list)


class QualityCleanupCriterionPayload(BaseModel):
    key: str
    label: str
    rationale: str
    weight: int = 0
    search_count: int | None = None
    read_count: int | None = None
    converted_search_count: int | None = None
    conversion_rate: float | None = None


class QualityCleanupRecommendationPayload(BaseModel):
    key: str
    label: str
    rationale: str
    weight: int = 0


class QualityCleanupCandidatePayload(BaseModel):
    memory_id: str
    title: str
    summary: str | None = None
    memory_type: str
    status: str
    updated_at: str | None = None
    tags: list[str] = Field(default_factory=list)
    priority_score: int = 0
    criteria: list[QualityCleanupCriterionPayload] = Field(default_factory=list)
    recommendations: list[QualityCleanupRecommendationPayload] = Field(default_factory=list)
    search_count: int | None = None
    read_count: int | None = None
    converted_search_count: int | None = None
    conversion_rate: float | None = None
    last_search_at: float | None = None
    last_read_at: float | None = None


class QualityCleanupCandidatesPayload(BaseModel):
    generated_at: float
    window_hours: int
    bucket_minutes: int
    low_conversion_min_search_count: int
    low_conversion_max_conversion_rate: float
    total_candidates: int = 0
    candidates: list[QualityCleanupCandidatePayload] = Field(default_factory=list)


class NerdMaintenanceSummaryRowPayload(BaseModel):
    key: str
    label: str
    task_names: list[str] = Field(default_factory=list)
    total_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0
    retry_runs: int = 0
    created_count: int = 0
    merged_count: int = 0
    updated_count: int = 0
    archived_count: int = 0
    degraded_count: int = 0
    restored_count: int = 0
    meaningful_actions: int = 0
    mutation_count: int = 0
    lines_compressed: int = 0
    delta_total: int = 0
    valid_plan_count: int = 0
    no_op_count: int = 0
    accepted_mutation_count: int = 0
    verification_failure_count: int = 0
    provider_failure_count: int = 0
    retry_count: int = 0
    valid_plan_rate: float = 0.0
    no_op_rate: float = 0.0
    mutation_categories: dict[str, int] = Field(default_factory=dict)


class NerdMaintenanceAgentYieldPayload(NerdMaintenanceSummaryRowPayload):
    family_key: str
    family_label: str
    actions_per_completed_run: float = 0.0
    lines_per_completed_run: float = 0.0
    delta_per_completed_run: float = 0.0


class NerdMaintenanceDeltaBucketPayload(BaseModel):
    bucket_start: float
    created_count: int = 0
    merged_count: int = 0
    updated_count: int = 0
    archived_count: int = 0
    degraded_count: int = 0
    restored_count: int = 0
    meaningful_actions: int = 0
    mutation_count: int = 0
    lines_compressed: int = 0
    valid_plan_count: int = 0
    no_op_count: int = 0
    accepted_mutation_count: int = 0
    verification_failure_count: int = 0
    provider_failure_count: int = 0
    retry_count: int = 0


class NerdMaintenanceDeltaSeriesPayload(BaseModel):
    key: str
    label: str
    task_names: list[str] = Field(default_factory=list)
    buckets: list[NerdMaintenanceDeltaBucketPayload] = Field(default_factory=list)


class NerdMaintenanceSummaryPayload(BaseModel):
    by_family: list[NerdMaintenanceSummaryRowPayload] = Field(default_factory=list)
    by_agent: list[NerdMaintenanceAgentYieldPayload] = Field(default_factory=list)
    family_delta_series: list[NerdMaintenanceDeltaSeriesPayload] = Field(default_factory=list)


class NerdTimelinesPayload(BaseModel):
    memory_activity: list[MemoryTimelineBucketPayload] = Field(default_factory=list)


class AgentThroughputBucketPayload(BaseModel):
    bucket_start: float
    total_runs: int
    completed_runs: int
    failed_runs: int
    retry_runs: int
    avg_duration_seconds: float


class ProviderLatencyBucketPayload(BaseModel):
    bucket_start: float
    provider_key: str
    provider_name: str
    model_name: str
    call_count: int
    failure_count: int
    avg_duration_seconds: float
    p95_duration_seconds: float


class QueueSnapshotPayload(BaseModel):
    runnable_count: int = 0
    scheduled_count: int = 0
    oldest_age_seconds: float = 0.0


class GraphTopologyPayload(BaseModel):
    total_memories: int = 0
    total_links: int = 0
    average_degree: float = 0.0
    orphan_count: int = 0
    orphan_rate: float = 0.0
    graph_supported_count: int = 0
    graph_supported_rate: float = 0.0
    link_type_counts: dict[str, int] = Field(default_factory=dict)


class MemoryLifecyclePayload(BaseModel):
    by_status: dict[str, int] = Field(default_factory=dict)
    by_type: dict[str, int] = Field(default_factory=dict)
    total_content_bytes: int = 0
    median_content_bytes: float = 0.0
    cold_memory_count: int = 0
    cold_memory_rate: float = 0.0
    never_surfaced_count: int = 0
    stale_count: int = 0
    degraded_count: int = 0


class SearchQualityPayload(BaseModel):
    semantic_enabled: bool = False
    degraded: bool = False
    fallback_count: int = 0
    rebuild_count: int = 0
    graph_supported_count: int = 0
    graph_supported_rate: float = 0.0
    never_surfaced_count: int = 0
    last_error: str | None = None


class NerdAlertPayload(BaseModel):
    key: str
    severity: str
    label: str
    message: str
    value: float
    threshold: float | None = None
    unit: str | None = None


class TaskRouteAuditPayload(BaseModel):
    task_name: str
    task_class: str
    execution_kind: str
    low_priority: bool = False
    configured_primary_route: str | None = None
    configured_fallback_routes: list[str] = Field(default_factory=list)
    resolved_provider_key: str | None = None
    resolved_model_name: str | None = None
    resolved_provider_type: str | None = None
    resolved_supports_agentic: bool | None = None
    recent_provider_key: str | None = None
    recent_model_name: str | None = None
    recent_status: str | None = None
    recent_success_count: int = 0
    recent_failure_count: int = 0
    on_primary_route: bool | None = None


class NerdProviderPolicyTaskPayload(BaseModel):
    task_name: str
    route_exhaustion_count: int = 0
    legacy_fallback_denied_count: int = 0
    admission_skip_count: int = 0
    top_skip_provider_key: str | None = None
    top_skip_model_name: str | None = None
    top_skip_reason_code: str | None = None
    active_admission_provider_count: int = 0


class NerdProviderPolicyProviderPayload(BaseModel):
    provider_key: str
    provider_name: str
    model_name: str
    admission_skip_count: int = 0
    distinct_task_count: int = 0
    top_task_name: str | None = None
    top_reason_code: str | None = None
    active_admission_reason: str | None = None
    active_admission_category: str | None = None
    active_retry_delay_seconds: float | None = None


class NerdProviderPolicyPayload(BaseModel):
    stats: list[NerdStatPayload] = Field(default_factory=list)
    by_task: list[NerdProviderPolicyTaskPayload] = Field(default_factory=list)
    by_provider: list[NerdProviderPolicyProviderPayload] = Field(default_factory=list)


class CurationCandidateMetricsPayload(BaseModel):
    total: int = 0
    dispositions: dict[str, int] = Field(default_factory=dict)
    no_op_candidates: int = 0
    cooldown_candidates: int = 0


class CurationSpecialistRouteMetricsPayload(BaseModel):
    total: int = 0
    by_family: dict[str, int] = Field(default_factory=dict)
    by_reason: dict[str, int] = Field(default_factory=dict)
    by_status: dict[str, int] = Field(default_factory=dict)


class CurationProviderDisclosurePayload(BaseModel):
    mutation_events_with_provider: int = 0
    mutation_events_without_provider: int = 0
    by_provider: dict[str, int] = Field(default_factory=dict)


class CurationHistoryMetricsPayload(BaseModel):
    event_count: int = 0
    applied_event_count: int = 0
    restorable_event_count: int = 0
    restore_request_count: int = 0
    restore_requests_by_status: dict[str, int] = Field(default_factory=dict)
    restore_available: bool = False


class CurationQualityMetricsPayload(BaseModel):
    sampled_action_count: int = 0
    evaluated_action_count: int = 0
    quality_observed_action_count: int = 0
    no_query_action_count: int = 0
    structural_only_action_count: int = 0
    retrieval_regression_count: int = 0
    zero_result_change: int = 0
    payload_size_change: int = 0
    useful_work_count: int = 0
    useful_work_observed_action_count: int = 0
    useful_work_rate: float = 0.0
    content_evaluated_action_count: int = 0
    content_quality_improved_count: int = 0
    content_quality_neutral_count: int = 0
    content_quality_regression_count: int = 0
    content_quality_delta: float = 0.0


class CurationMetricsPayload(BaseModel):
    window_hours: int = 24
    run_states: dict[str, int] = Field(default_factory=dict)
    run_outcomes: dict[str, int] = Field(default_factory=dict)
    operation_counts: dict[str, int] = Field(default_factory=dict)
    receipt_statuses: dict[str, int] = Field(default_factory=dict)
    rejection_reasons: dict[str, int] = Field(default_factory=dict)
    no_op_runs: int = 0
    candidate: CurationCandidateMetricsPayload = Field(default_factory=CurationCandidateMetricsPayload)
    specialist_routes: CurationSpecialistRouteMetricsPayload = Field(default_factory=CurationSpecialistRouteMetricsPayload)
    provider_disclosure: CurationProviderDisclosurePayload = Field(default_factory=CurationProviderDisclosurePayload)
    history: CurationHistoryMetricsPayload = Field(default_factory=CurationHistoryMetricsPayload)
    retrieval_quality: CurationQualityMetricsPayload = Field(default_factory=CurationQualityMetricsPayload)
    verified_yield: float = 0.0
    verified_receipt_count: int = 0
    terminal_receipt_count: int = 0
    valid_plan_count: int = 0
    valid_plan_rate: float = 0.0
    no_op_rate: float = 0.0
    accepted_mutation_count: int = 0
    verification_failure_count: int = 0
    provider_failure_count: int = 0
    retry_count: int = 0
    mutation_categories: dict[str, int] = Field(default_factory=dict)


class MutationHistoryListPayload(BaseModel):
    events: list[dict[str, Any]] = Field(default_factory=list)
    limit: int
    offset: int
    has_more: bool = False
    next_offset: int | None = None


class MutationHistoryDetailPayload(BaseModel):
    event: dict[str, Any]
    receipt: dict[str, Any] | None = None
    curation_run: dict[str, Any] | None = None
    records: list[dict[str, Any]] = Field(default_factory=list)
    links: list[dict[str, Any]] = Field(default_factory=list)
    truncated: bool = False


class MutationHistoryDiffPayload(BaseModel):
    event_id: str
    records: list[dict[str, Any]] = Field(default_factory=list)
    links: list[dict[str, Any]] = Field(default_factory=list)
    truncated: bool = False


class ProtectionListPayload(BaseModel):
    memory_id: str
    protections: list[dict[str, Any]] = Field(default_factory=list)


class ProtectionMutationPayload(BaseModel):
    status: Literal["applied", "removed"]
    memory_id: str
    mode: str
    protection: dict[str, Any] | None = None


class RestoreEligibilityPayload(BaseModel):
    event_id: str
    eligible: bool
    operation: str | None = None
    inverse_operation: str | None = None
    risk: str | None = None
    requires_confirmation: bool = False
    current_record_tokens: dict[str, str] = Field(default_factory=dict)
    current_link_tokens: dict[str, str] = Field(default_factory=dict)
    protections: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    conflict_code: str | None = None
    conflict_reason: str | None = None


class RestoreRequestPayload(BaseModel):
    status: str
    target_event_id: str
    request_id: str | None = None
    event_id: str | None = None
    conflict_reason: str | None = None
    conflict_details: dict[str, Any] = Field(default_factory=dict)


class NerdMetricsPayload(BaseModel):
    generated_at: float
    window_hours: int
    bucket_minutes: int
    stats: list[NerdStatPayload] = Field(default_factory=list)
    composition: NerdCompositionPayload = Field(default_factory=NerdCompositionPayload)
    distributions: NerdDistributionsPayload = Field(default_factory=NerdDistributionsPayload)
    timelines: NerdTimelinesPayload = Field(default_factory=NerdTimelinesPayload)
    maintenance: NerdMaintenancePayload = Field(default_factory=NerdMaintenancePayload)
    lifecycle_trends: NerdLifecycleTrendsPayload = Field(default_factory=NerdLifecycleTrendsPayload)
    growth_dynamics: NerdGrowthDynamicsPayload = Field(default_factory=NerdGrowthDynamicsPayload)
    quality_drilldown: NerdQualityDrilldownPayload = Field(default_factory=NerdQualityDrilldownPayload)
    quality_remediation: NerdQualityRemediationPayload = Field(default_factory=NerdQualityRemediationPayload)
    retrieval: NerdRetrievalPayload = Field(default_factory=NerdRetrievalPayload)
    maintenance_summary: NerdMaintenanceSummaryPayload = Field(default_factory=NerdMaintenanceSummaryPayload)
    queue_snapshot: QueueSnapshotPayload = Field(default_factory=QueueSnapshotPayload)
    graph_topology: GraphTopologyPayload = Field(default_factory=GraphTopologyPayload)
    memory_lifecycle: MemoryLifecyclePayload = Field(default_factory=MemoryLifecyclePayload)
    search_quality: SearchQualityPayload = Field(default_factory=SearchQualityPayload)
    route_audit: list[TaskRouteAuditPayload] = Field(default_factory=list)
    provider_policy: NerdProviderPolicyPayload = Field(default_factory=NerdProviderPolicyPayload)
    curation: CurationMetricsPayload = Field(default_factory=CurationMetricsPayload)
    alerts: list[NerdAlertPayload] = Field(default_factory=list)
    agent_throughput: list[AgentThroughputBucketPayload] = Field(default_factory=list)
    provider_latency: list[ProviderLatencyBucketPayload] = Field(default_factory=list)


class ProviderUsagePayload(BaseModel):
    task_name: str | None = None
    provider_key: str
    provider_name: str
    model_name: str
    calls_last_hour: int
    calls_last_day: int
    failures_last_hour: int
    failures_last_day: int
    skips_last_hour: int = 0
    skips_last_day: int = 0
    avg_duration_last_hour: float
    avg_duration_last_day: float
    top_failure_reason_last_day: str | None = None
    top_skip_reason_last_day: str | None = None
    active_admission_reason: str | None = None
    active_admission_category: str | None = None
    active_retry_delay_seconds: float | None = None
    input_tokens_last_hour: int = 0
    input_tokens_last_day: int = 0
    output_tokens_last_hour: int = 0
    output_tokens_last_day: int = 0
    cached_input_tokens_last_hour: int = 0
    cached_input_tokens_last_day: int = 0
    cache_write_tokens_last_hour: int = 0
    cache_write_tokens_last_day: int = 0
    reasoning_tokens_last_hour: int = 0
    reasoning_tokens_last_day: int = 0
    total_tokens_last_hour: int = 0
    total_tokens_last_day: int = 0
    token_usage_source: str | None = None


class AIConversationPayload(BaseModel):
    id: int
    request_id: str
    attempt: int
    workspace_id: str | None = None
    task_name: str | None = None
    task_id: str | None = None
    provider_key: str
    provider_name: str
    model_name: str
    subprocess_pid: int | None = None
    prompt_text: str
    response_text: str
    parsed: dict | None = None
    status: str
    error_text: str | None = None
    reason_category: str | None = None
    reason_code: str | None = None
    retry_delay_seconds: float | None = None
    started_at: float
    completed_at: float
    duration_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    token_usage_source: str | None = None


class AIConversationListPayload(BaseModel):
    conversations: list[AIConversationPayload] = Field(default_factory=list)


class OverviewPayload(BaseModel):
    memories: OverviewCounts
    embeddings: EmbeddingStatusPayload = Field(default_factory=EmbeddingStatusPayload)
    search: SearchHealthPayload = Field(default_factory=SearchHealthPayload)
    cache: CacheHealthPayload = Field(default_factory=CacheHealthPayload)
    execution_attempts: ExecutionAttemptHealthPayload = Field(default_factory=ExecutionAttemptHealthPayload)
    memory_metrics: MemoryMetricsPayload
    token_usage: TokenUsageSummaryPayload = Field(default_factory=TokenUsageSummaryPayload)
    queue_diagnostics: list[QueueDiagnosticPayload] = Field(default_factory=list)
    agent_runs: list[AgentRunPayload] = Field(default_factory=list)
    provider_usage: list[ProviderUsagePayload] = Field(default_factory=list)
    recent_agent_runs: list[AgentRunHistoryPayload] = Field(default_factory=list)
    recent_logs: list[RuntimeLogPayload] = Field(default_factory=list)
    recent_memories: list[CompactMemoryRecord] = Field(default_factory=list)
    top_read_memories: list[CompactMemoryRecord] = Field(default_factory=list)
    top_read_memories_active: list[CompactMemoryRecord] = Field(default_factory=list)
    tasks: TaskStatusSummary
    failed_tasks: list[dict] = Field(default_factory=list)
    journal: JournalSummary
    storage: StorageSummary


class MemoryDetailPayload(BaseModel):
    record: dict
    relationships: dict[str, list[dict]] = Field(default_factory=dict)
    superseded: list[dict] = Field(default_factory=list)


class TaskListPayload(BaseModel):
    tasks: list[dict] = Field(default_factory=list)


class TaskDetailPayload(BaseModel):
    task: dict[str, Any]
    runs: list[AgentRunHistoryPayload] = Field(default_factory=list)


class MemoryListPayload(BaseModel):
    records: list[CompactMemoryRecord] = Field(default_factory=list)
