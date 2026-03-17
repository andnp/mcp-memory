from __future__ import annotations

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
    next_available_at: float | None = None
    seconds_until_next_run: float | None = None


class AgentRunHistoryPayload(BaseModel):
    task_name: str
    status: str
    started_at: float
    completed_at: float
    duration_seconds: float
    error_text: str | None = None
    result_summary: str | None = None
    result_metadata: RunResultMetadataPayload = Field(default_factory=RunResultMetadataPayload)


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


class MemoryListPayload(BaseModel):
    records: list[CompactMemoryRecord] = Field(default_factory=list)
