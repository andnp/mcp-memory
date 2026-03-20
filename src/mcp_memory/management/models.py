from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EmbeddingStatusPayload(BaseModel):
    model_name: str | None = None
    backend: str | None = None
    model_cached: bool = False


class SearchHealthPayload(BaseModel):
    semantic_enabled: bool = False
    available: bool = False
    degraded: bool = False
    fallback_count: int = 0
    rebuild_count: int = 0
    last_error: str | None = None
    last_failure_at: str | None = None
    last_recovery_at: str | None = None
    last_integrity_check_at: str | None = None
    integrity_check_error: str | None = None


class HealthPayload(BaseModel):
    status: str
    workspace_id: str | None = None
    workspace_root: str | None = None
    memory_path: str | None = None
    db_path: str | None = None
    runtime_active: bool
    client_count: int
    task_queue_enabled: bool
    embeddings: EmbeddingStatusPayload = Field(default_factory=EmbeddingStatusPayload)
    search: SearchHealthPayload = Field(default_factory=SearchHealthPayload)


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


class RunResultMetadataPayload(BaseModel):
    requested_strategy: str | None = None
    strategy_used: str | None = None
    strategy_fallback_reason: str | None = None
    candidate_count: int | None = None
    sampled_memory_ids: list[str] = Field(default_factory=list)
    requested_grouping_strategy: str | None = None
    grouping_strategy_used: str | None = None
    grouping_fallback_reason: str | None = None
    group_count: int | None = None


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


class AgentRunHistoryListPayload(BaseModel):
    runs: list[AgentRunHistoryPayload] = Field(default_factory=list)


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


class NerdGrowthDynamicsPayload(BaseModel):
    top_tag_trends: list[NerdCountSeriesPayload] = Field(default_factory=list)
    workspace_contribution_share: list[NerdShareSeriesPayload] = Field(default_factory=list)


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
    lines_compressed: int = 0
    delta_total: int = 0


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
    lines_compressed: int = 0


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
    maintenance_summary: NerdMaintenanceSummaryPayload = Field(default_factory=NerdMaintenanceSummaryPayload)
    queue_snapshot: QueueSnapshotPayload = Field(default_factory=QueueSnapshotPayload)
    graph_topology: GraphTopologyPayload = Field(default_factory=GraphTopologyPayload)
    memory_lifecycle: MemoryLifecyclePayload = Field(default_factory=MemoryLifecyclePayload)
    search_quality: SearchQualityPayload = Field(default_factory=SearchQualityPayload)
    route_audit: list[TaskRouteAuditPayload] = Field(default_factory=list)
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
    avg_duration_last_hour: float
    avg_duration_last_day: float


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
    started_at: float
    completed_at: float
    duration_seconds: float


class AIConversationListPayload(BaseModel):
    conversations: list[AIConversationPayload] = Field(default_factory=list)


class OverviewPayload(BaseModel):
    memories: OverviewCounts
    embeddings: EmbeddingStatusPayload = Field(default_factory=EmbeddingStatusPayload)
    search: SearchHealthPayload = Field(default_factory=SearchHealthPayload)
    memory_metrics: MemoryMetricsPayload
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
