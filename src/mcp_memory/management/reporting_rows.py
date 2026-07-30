from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import time
from typing import Any, TypeAlias, cast

from pydantic import BaseModel, Field, JsonValue

from mcp_memory.core.task_handlers import MAINTENANCE_TASK_NAMES
from mcp_memory.core.task_results import TaskRunResult, build_task_run_result_summary, decode_task_run_result_payload
from mcp_memory.management.models import (
    IngestAuditPayload,
    IngestEntryDispositionPayload,
    MutationOutcomePayload,
    QueueDiagnosticPayload,
    RunResultMetadataPayload,
    SelectorFeatureSnapshotPayload,
    SelectorMetricSnapshotPayload,
    SelectorPopulationSnapshotPayload,
)
from mcp_memory.management.query_runner import ManagementQueryRunner


JsonObject: TypeAlias = dict[str, JsonValue]

_MUTATION_OUTCOME_KEYS: tuple[str, ...] = (
    "created",
    "merged",
    "updated",
    "archived",
    "degraded",
    "restored",
)
_UNSUPPORTED_JSON_VALUE = object()


class TaskResultView(BaseModel):
    raw_payload: JsonObject = Field(default_factory=dict)
    summary: str | None = None
    metadata: RunResultMetadataPayload = Field(default_factory=RunResultMetadataPayload)
    ingest_audit: IngestAuditPayload = Field(default_factory=IngestAuditPayload)
    meaningful_actions: int | None = None
    mutation_count: int | None = None
    tool_calls_executed: int | None = None
    lines_compressed: int | None = None
    stale: int | None = None

    @property
    def mutation_outcome(self) -> MutationOutcomePayload:
        return self.metadata.mutation_outcome

    @property
    def created(self) -> int | None:
        return self.mutation_outcome.created

    @property
    def merged(self) -> int | None:
        return self.mutation_outcome.merged

    @property
    def updated(self) -> int | None:
        return self.mutation_outcome.updated

    @property
    def archived(self) -> int | None:
        return self.mutation_outcome.archived

    @property
    def degraded(self) -> int | None:
        return self.mutation_outcome.degraded

    @property
    def restored(self) -> int | None:
        return self.mutation_outcome.restored

    def metadata_copy(self) -> RunResultMetadataPayload:
        return self.metadata.model_copy(deep=True)

    def full_ingest_audit(self) -> IngestAuditPayload:
        return self.ingest_audit.model_copy(deep=True)

    def compact_ingest_audit(self) -> IngestAuditPayload:
        updates: dict[str, object] = {}
        if self.ingest_audit.entry_dispositions or self.ingest_audit.provider_reported_entry_outcomes:
            updates = {
                "entry_dispositions": [],
                "provider_reported_entry_outcomes": [],
            }
        return self.ingest_audit.model_copy(deep=True, update=updates)

    def get(self, key: str, default: JsonValue | None = None) -> JsonValue | None:
        return self.raw_payload.get(key, default)

    def __getitem__(self, key: str) -> JsonValue:
        return self.raw_payload[key]

    def __contains__(self, key: object) -> bool:
        return key in self.raw_payload

    def __len__(self) -> int:
        return len(self.raw_payload)

    def __bool__(self) -> bool:
        return bool(self.raw_payload)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            return self.raw_payload == other
        return BaseModel.__eq__(self, other)


TaskResultSource: TypeAlias = TaskResultView | TaskRunResult | Mapping[str, object] | str | None


def coerce_task_result_view(raw_result: object) -> TaskResultView:
    if isinstance(raw_result, TaskResultView):
        return raw_result
    return build_task_result_view(raw_result)


def build_task_result_view(raw_result: object) -> TaskResultView:
    raw_payload = decode_task_result_payload(raw_result)
    metadata = _build_run_result_metadata(raw_payload)
    summary = raw_result.summary if isinstance(raw_result, TaskRunResult) else build_task_run_result_summary(raw_payload)
    meaningful_actions = (
        raw_result.meaningful_actions if isinstance(raw_result, TaskRunResult) else _coerce_int(raw_payload.get("meaningful_actions"))
    )
    mutation_count = _coerce_int(raw_payload.get("mutations"))
    tool_calls_executed = _coerce_int(raw_payload.get("tool_calls_executed"))
    lines_compressed = (
        raw_result.lines_compressed if isinstance(raw_result, TaskRunResult) else _coerce_int(raw_payload.get("lines_compressed"))
    )
    stale = raw_result.stale if isinstance(raw_result, TaskRunResult) else _coerce_int(raw_payload.get("stale"))
    return TaskResultView(
        raw_payload=raw_payload,
        summary=summary,
        metadata=metadata,
        ingest_audit=_build_ingest_audit(raw_payload, include_entries=True),
        meaningful_actions=meaningful_actions,
        mutation_count=mutation_count,
        tool_calls_executed=tool_calls_executed,
        lines_compressed=lines_compressed,
        stale=stale,
    )


def decode_task_result_payload(raw_result: object) -> JsonObject:
    return _coerce_json_object(decode_task_run_result_payload(raw_result))


def _build_run_result_metadata(result: JsonObject) -> RunResultMetadataPayload:
    sampled_memory_ids = result.get("sampled_memory_ids")
    claimed_work_item_count = _coerce_int(result.get("claimed_work_item_count"))
    provider_calls_used = _coerce_int(result.get("provider_calls_used"))
    tool_calls_executed = _coerce_int(result.get("tool_calls_executed"))
    mutation_outcome = _extract_mutation_outcome(result)
    mutations = _coerce_int(result.get("mutations"))
    if mutations is None:
        mutations = _sum_mutation_outcome(mutation_outcome)
    return RunResultMetadataPayload(
        requested_strategy=_coerce_str(result.get("requested_strategy")),
        strategy_used=_coerce_str(result.get("strategy_used")),
        strategy_fallback_reason=_coerce_str(result.get("strategy_fallback_reason")),
        strategy_selection_mode=_coerce_str(result.get("strategy_selection_mode")),
        strategy_selection_reason=_coerce_str(result.get("strategy_selection_reason")),
        strategy_selection_scores=_coerce_score_mapping(result.get("strategy_selection_scores")),
        selector_feature_snapshot=_coerce_selector_feature_snapshot(result.get("selector_feature_snapshot")),
        candidate_count=_coerce_int(result.get("candidate_count")),
        sampled_memory_ids=[item for item in sampled_memory_ids if isinstance(item, str)] if isinstance(sampled_memory_ids, list) else [],
        compatibility_group=_coerce_str(result.get("compatibility_group")),
        claimed_work_item_count=claimed_work_item_count,
        provider_calls_used=provider_calls_used,
        tool_calls_executed=tool_calls_executed,
        mutations=mutations,
        mutation_outcome=mutation_outcome,
        work_item_batch_limit=_coerce_int(result.get("work_item_batch_limit")),
        max_batches_per_run=_coerce_int(result.get("max_batches_per_run")),
        requested_grouping_strategy=_coerce_str(result.get("requested_grouping_strategy")),
        grouping_strategy_used=_coerce_str(result.get("grouping_strategy_used")),
        grouping_fallback_reason=_coerce_str(result.get("grouping_fallback_reason")),
        group_count=_coerce_int(result.get("group_count")),
        campaign_key=_coerce_str(result.get("campaign_key")),
        campaign_origin_family=_coerce_str(result.get("campaign_origin_family")),
        campaign_family_keys=_coerce_str_list(result.get("campaign_family_keys")),
        campaign_continuation_supported=_coerce_bool(result.get("campaign_continuation_supported")),
        compatible_batch_calls=_coerce_int(result.get("compatible_batch_calls")),
        premium_execution_count=provider_calls_used,
        work_items_per_premium_execution=_ratio_or_none(claimed_work_item_count, provider_calls_used),
        mutations_per_premium_execution=_ratio_or_none(mutations, provider_calls_used),
        tool_calls_per_premium_execution=_ratio_or_none(tool_calls_executed, provider_calls_used),
    )


def _build_ingest_audit(
    result: JsonObject,
    *,
    include_entries: bool,
) -> IngestAuditPayload:
    if not _contains_ingest_audit(result):
        return IngestAuditPayload()

    created_memory_ids = _coerce_str_list(result.get("created_memory_ids"))
    touched_memory_ids = _coerce_str_list(result.get("touched_memory_ids"))
    appended_memory_ids = _coerce_str_list(result.get("appended_memory_ids"))
    matched_memory_ids = _coerce_str_list(result.get("matched_memory_ids"))
    provider_reported_touched_memory_ids = _coerce_str_list(result.get("provider_reported_touched_memory_ids"))
    provider_reported_matched_memory_ids = _coerce_str_list(result.get("provider_reported_matched_memory_ids"))
    entry_dispositions = _coerce_ingest_entry_dispositions(result.get("entry_dispositions")) if include_entries else []
    provider_reported_entry_outcomes = (
        _coerce_ingest_entry_dispositions(result.get("provider_reported_entry_outcomes")) if include_entries else []
    )
    processed_count = _result_list_count(result, "processed_entry_ids")
    handled_count = processed_count or _result_list_count(result, "recoverable_entry_ids")
    return IngestAuditPayload(
        claimed_count=_result_list_count(result, "claimed_entry_ids"),
        handled_count=handled_count,
        released_count=_result_list_count(result, "released_entry_ids"),
        deleted_count=_result_list_count(result, "deleted_entry_ids"),
        processed_count=processed_count,
        meaningful_actions=_coerce_int(result.get("meaningful_actions")) or 0,
        created_count=len(created_memory_ids),
        touched_count=len(touched_memory_ids),
        appended_count=len(appended_memory_ids),
        matched_count=len(matched_memory_ids),
        tool_calls_executed=_coerce_int(result.get("tool_calls_executed")),
        mutations=_coerce_int(result.get("mutations")),
        provider_reported_tool_calls=_coerce_int(result.get("provider_reported_tool_calls")),
        provider_reported_mutations=_coerce_int(result.get("provider_reported_mutations")),
        created_memory_ids=created_memory_ids,
        touched_memory_ids=touched_memory_ids,
        appended_memory_ids=appended_memory_ids,
        matched_memory_ids=matched_memory_ids,
        provider_reported_touched_memory_ids=provider_reported_touched_memory_ids,
        provider_reported_matched_memory_ids=provider_reported_matched_memory_ids,
        entry_dispositions=entry_dispositions,
        provider_reported_entry_outcomes=provider_reported_entry_outcomes,
    )


def _coerce_str(value: JsonValue | object) -> str | None:
    return value if isinstance(value, str) else None


def _extract_mutation_outcome(result: JsonObject) -> MutationOutcomePayload:
    return MutationOutcomePayload(**{key: _coerce_int(result.get(key)) for key in _MUTATION_OUTCOME_KEYS})


def _sum_mutation_outcome(mutation_outcome: MutationOutcomePayload) -> int | None:
    values = [value for value in mutation_outcome.model_dump().values() if value is not None]
    if not values:
        return None
    return sum(values)


def _ratio_or_none(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _coerce_str_list(value: JsonValue | object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _coerce_score_mapping(value: JsonValue | object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    scores: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            scores[key] = float(item)
    return scores


def _coerce_selector_feature_snapshot(value: JsonValue | object) -> SelectorFeatureSnapshotPayload:
    if not isinstance(value, Mapping):
        return SelectorFeatureSnapshotPayload()
    return SelectorFeatureSnapshotPayload(
        strategy_signals=_coerce_score_mapping(value.get("strategy_signals")),
        candidate_population=_coerce_selector_population_snapshot(value.get("candidate_population")),
        selected_population=_coerce_selector_population_snapshot(value.get("selected_population")),
    )


def _coerce_selector_population_snapshot(value: JsonValue | object) -> SelectorPopulationSnapshotPayload:
    if not isinstance(value, Mapping):
        return SelectorPopulationSnapshotPayload()
    metrics: dict[str, SelectorMetricSnapshotPayload] = {}
    raw_metrics = value.get("metrics")
    if isinstance(raw_metrics, Mapping):
        for key, item in raw_metrics.items():
            if not isinstance(key, str):
                continue
            metrics[key] = _coerce_selector_metric_snapshot(item)
    return SelectorPopulationSnapshotPayload(
        count=_coerce_int(value.get("count")) or 0,
        metrics=metrics,
        shares=_coerce_score_mapping(value.get("shares")),
    )


def _coerce_selector_metric_snapshot(value: JsonValue | object) -> SelectorMetricSnapshotPayload:
    if not isinstance(value, Mapping):
        return SelectorMetricSnapshotPayload()
    return SelectorMetricSnapshotPayload(
        count=_coerce_int(value.get("count")) or 0,
        min=_coerce_float_or_none(value.get("min")),
        p50=_coerce_float_or_none(value.get("p50")),
        p90=_coerce_float_or_none(value.get("p90")),
        max=_coerce_float_or_none(value.get("max")),
        mean=_coerce_float_or_none(value.get("mean")),
    )


def _coerce_float_or_none(value: JsonValue | object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_ingest_entry_dispositions(value: JsonValue | object) -> list[IngestEntryDispositionPayload]:
    if not isinstance(value, list):
        return []
    payloads: list[IngestEntryDispositionPayload] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        entry_id = _coerce_int(item.get("entry_id"))
        disposition = _coerce_str(item.get("disposition"))
        if entry_id is None or disposition is None:
            continue
        payloads.append(
            IngestEntryDispositionPayload(
                entry_id=entry_id,
                disposition=disposition,
                finalization_status=_coerce_str(item.get("finalization_status")),
                memory_id=_coerce_str(item.get("memory_id")),
                memory_title=_coerce_str(item.get("memory_title")),
                reason=_coerce_str(item.get("reason")),
            )
        )
    return payloads


def _contains_ingest_audit(result: JsonObject) -> bool:
    ingest_keys = {
        "claimed_entry_ids",
        "released_entry_ids",
        "deleted_entry_ids",
        "processed_entry_ids",
        "entry_dispositions",
        "touched_memory_ids",
        "appended_memory_ids",
        "matched_memory_ids",
        "provider_reported_tool_calls",
        "provider_reported_mutations",
        "provider_reported_entry_outcomes",
    }
    return any(key in result for key in ingest_keys)


def _result_list_count(result: JsonObject, key: str) -> int:
    value = result.get(key)
    return len(value) if isinstance(value, list) else 0


def _coerce_json_object(value: object) -> JsonObject:
    if not isinstance(value, Mapping):
        return {}
    raw_payload: JsonObject = {}
    for key, item in value.items():
        coerced_item = _coerce_json_value(item)
        if coerced_item is _UNSUPPORTED_JSON_VALUE:
            continue
        raw_payload[str(key)] = cast(JsonValue, coerced_item)
    return raw_payload


def _coerce_json_value(value: object) -> JsonValue | object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return _coerce_json_object(value)
    if isinstance(value, list):
        coerced_list: list[JsonValue] = []
        for item in value:
            coerced_item = _coerce_json_value(item)
            if coerced_item is _UNSUPPORTED_JSON_VALUE:
                continue
            coerced_list.append(cast(JsonValue, coerced_item))
        return coerced_list
    return _UNSUPPORTED_JSON_VALUE

@dataclass(frozen=True)
class MemoryCountRow:
    memory_type: str
    status: str
    count: int


@dataclass(frozen=True)
class TaskCountRow:
    status: str
    count: int


@dataclass(frozen=True)
class RunningTaskAttemptRow:
    task_id: str
    workspace_id: str | None = None
    updated_at: float = 0.0
    started_at: float | None = None
    claimed_at: float | None = None
    execution_epoch: int | None = None
    attempt_status: str | None = None
    attempt_started_at: float | None = None
    last_heartbeat_at: float | None = None
    attempt_subprocess_pid: int | None = None


@dataclass(frozen=True)
class MemoryMetricsRow:
    total_memories: int = 0
    total_memory_lines: int = 0
    total_summary_lines: int = 0


@dataclass(frozen=True)
class PendingJournalMetricsRow:
    thought_buffer_entries: int = 0
    thought_buffer_lines: int = 0


@dataclass(frozen=True)
class TaskRunRow:
    status: str
    completed_at: float
    duration_seconds: float
    result: TaskResultView = field(default_factory=TaskResultView)

    def __post_init__(self) -> None:
        object.__setattr__(self, "result", coerce_task_result_view(self.result))


@dataclass(frozen=True)
class MaintenanceTaskRunRow:
    task_id: str
    task_name: str
    status: str
    completed_at: float
    duration_seconds: float
    result: TaskResultView = field(default_factory=TaskResultView)
    error_text: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "result", coerce_task_result_view(self.result))


@dataclass(frozen=True)
class ProviderUsageRow:
    provider_key: str
    provider_name: str
    model_name: str
    status: str
    duration_seconds: float
    created_at: float
    task_name: str | None = None
    reason_category: str | None = None
    reason_code: str | None = None
    retry_delay_seconds: float | None = None


@dataclass(frozen=True)
class AiConversationRow:
    provider_key: str
    completed_at: float
    response_text: str | None = None
    parsed_json: str | None = None


@dataclass(frozen=True)
class RuntimeLogRow:
    logger_name: str
    level: str
    message: str
    created_at: float
    task_name: str | None = None


@dataclass(frozen=True)
class ProviderPolicyEventRow:
    event_kind: str
    warning_suppressed: bool
    created_at: float
    task_name: str | None = None
    task_id: str | None = None
    warning_kind: str | None = None
    provider_key: str | None = None
    provider_name: str | None = None
    model_name: str | None = None
    route_key: str | None = None
    candidate_routes: list[str] = field(default_factory=list)
    reason_category: str | None = None
    reason_code: str | None = None
    retry_delay_seconds: float | None = None


@dataclass(frozen=True)
class MemoryToolEventRow:
    invocation_id: str
    event_kind: str
    created_at: float
    workspace_id: str | None = None
    caller_kind: str | None = None
    memory_id: str | None = None
    query_text: str | None = None
    result_rank: int | None = None
    result_count: int = 0
    duration_ms: float | None = None


@dataclass(frozen=True)
class ScopedMemoryRow:
    id: str
    title: str
    memory_type: str
    status: str
    summary: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_accessed_at: datetime | None = None
    last_surfaced_at: datetime | None = None
    content_bytes: int = 0
    workspace_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    has_split_lineage: bool = False


@dataclass(frozen=True)
class LinkRow:
    source_id: str
    target_id: str
    link_type: str


@dataclass(frozen=True)
class CopilotPremiumUsageSummary:
    copilot_premium_requests_today: float = 0.0
    copilot_premium_requests_last_day: float = 0.0


@dataclass(frozen=True)
class AgentRunHistoryRow:
    task_id: str
    task_name: str
    status: str
    started_at: float
    completed_at: float
    duration_seconds: float
    result: TaskResultView = field(default_factory=TaskResultView)
    error_text: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "result", coerce_task_result_view(self.result))


def adapt_memory_count_row(row: Mapping[str, object]) -> MemoryCountRow:
    return MemoryCountRow(
        memory_type=_require_str(row, "type"),
        status=_require_str(row, "status"),
        count=_require_int(row, "count"),
    )


def adapt_task_count_row(row: Mapping[str, object]) -> TaskCountRow:
    return TaskCountRow(
        status=_require_str(row, "status"),
        count=_require_int(row, "count"),
    )


def adapt_running_task_attempt_row(row: Mapping[str, object]) -> RunningTaskAttemptRow:
    return RunningTaskAttemptRow(
        task_id=_require_str(row, "task_id"),
        workspace_id=_optional_str(row, "workspace_id"),
        updated_at=_require_float(row, "updated_at"),
        started_at=_optional_float(row, "started_at"),
        claimed_at=_optional_float(row, "claimed_at"),
        execution_epoch=_optional_int(row, "execution_epoch"),
        attempt_status=_optional_str(row, "attempt_status"),
        attempt_started_at=_optional_float(row, "attempt_started_at"),
        last_heartbeat_at=_optional_float(row, "last_heartbeat_at"),
        attempt_subprocess_pid=_optional_int(row, "attempt_subprocess_pid"),
    )


def adapt_memory_metrics_row(row: Mapping[str, object]) -> MemoryMetricsRow:
    return MemoryMetricsRow(
        total_memories=_require_int(row, "total_memories"),
        total_memory_lines=_require_int(row, "total_memory_lines"),
        total_summary_lines=_require_int(row, "total_summary_lines"),
    )


def adapt_pending_journal_metrics_row(row: Mapping[str, object]) -> PendingJournalMetricsRow:
    return PendingJournalMetricsRow(
        thought_buffer_entries=_require_int(row, "thought_buffer_entries"),
        thought_buffer_lines=_require_int(row, "thought_buffer_lines"),
    )


def adapt_task_run_row(row: Mapping[str, object]) -> TaskRunRow:
    return TaskRunRow(
        status=_require_str(row, "status"),
        completed_at=_require_float(row, "completed_at"),
        duration_seconds=_require_float(row, "duration_seconds"),
        result=coerce_task_result_view(row.get("result_json")),
    )


def adapt_maintenance_task_run_row(row: Mapping[str, object]) -> MaintenanceTaskRunRow:
    return MaintenanceTaskRunRow(
        task_id=_require_str(row, "task_id"),
        task_name=_require_str(row, "task_name"),
        status=_require_str(row, "status"),
        completed_at=_require_float(row, "completed_at"),
        duration_seconds=_require_float(row, "duration_seconds"),
        result=coerce_task_result_view(row.get("result_json")),
        error_text=_optional_str(row, "error_text"),
    )


def adapt_provider_usage_row(row: Mapping[str, object]) -> ProviderUsageRow:
    return ProviderUsageRow(
        task_name=_optional_str(row, "task_name"),
        provider_key=_require_str(row, "provider_key"),
        provider_name=_require_str(row, "provider_name"),
        model_name=_require_str(row, "model_name"),
        status=_require_str(row, "status"),
        duration_seconds=_require_float(row, "duration_seconds"),
        created_at=_require_float(row, "created_at"),
        reason_category=_optional_str(row, "reason_category"),
        reason_code=_optional_str(row, "reason_code"),
        retry_delay_seconds=_optional_float(row, "retry_delay_seconds"),
    )


def adapt_ai_conversation_row(row: Mapping[str, object]) -> AiConversationRow:
    return AiConversationRow(
        provider_key=_require_str(row, "provider_key"),
        completed_at=_require_float(row, "completed_at"),
        response_text=_optional_str(row, "response_text"),
        parsed_json=_optional_str(row, "parsed_json"),
    )


def adapt_runtime_log_row(row: Mapping[str, object]) -> RuntimeLogRow:
    return RuntimeLogRow(
        logger_name=_require_str(row, "logger_name"),
        level=_require_str(row, "level"),
        message=_require_str(row, "message"),
        created_at=_require_float(row, "created_at"),
        task_name=_runtime_log_task_name(_coerce_json_mapping(row.get("data_json"))),
    )


def adapt_provider_policy_event_row(row: Mapping[str, object]) -> ProviderPolicyEventRow:
    return ProviderPolicyEventRow(
        task_name=_optional_str(row, "task_name"),
        task_id=_optional_str(row, "task_id"),
        event_kind=_require_str(row, "event_kind"),
        warning_kind=_optional_str(row, "warning_kind"),
        provider_key=_optional_str(row, "provider_key"),
        provider_name=_optional_str(row, "provider_name"),
        model_name=_optional_str(row, "model_name"),
        route_key=_optional_str(row, "route_key"),
        candidate_routes=_coerce_string_list_json(row.get("candidate_routes_json")),
        reason_category=_optional_str(row, "reason_category"),
        reason_code=_optional_str(row, "reason_code"),
        retry_delay_seconds=_optional_float(row, "retry_delay_seconds"),
        warning_suppressed=_coerce_bool(row.get("warning_suppressed")),
        created_at=_require_float(row, "created_at"),
    )


def adapt_memory_tool_event_row(row: Mapping[str, object]) -> MemoryToolEventRow:
    return MemoryToolEventRow(
        invocation_id=_require_str(row, "invocation_id"),
        workspace_id=_optional_str(row, "workspace_id"),
        caller_kind=_optional_str(row, "caller_kind"),
        event_kind=_require_str(row, "event_kind"),
        memory_id=_optional_str(row, "memory_id"),
        query_text=_optional_str(row, "query_text"),
        result_rank=_optional_int(row, "result_rank"),
        result_count=_optional_int(row, "result_count") or 0,
        duration_ms=_optional_float(row, "duration_ms"),
        created_at=_require_float(row, "created_at"),
    )


def adapt_scoped_memory_row(row: Mapping[str, object]) -> ScopedMemoryRow:
    metadata = _coerce_json_mapping(row.get("metadata"))
    return ScopedMemoryRow(
        id=_require_str(row, "id"),
        title=_require_str(row, "title"),
        summary=_optional_str(row, "summary"),
        memory_type=_require_str(row, "type"),
        status=_require_str(row, "status"),
        created_at=_optional_datetime(row.get("created_at")),
        updated_at=_optional_datetime(row.get("updated_at")),
        last_accessed_at=_optional_datetime(row.get("last_accessed_at")),
        last_surfaced_at=_optional_datetime(row.get("last_surfaced_at")),
        content_bytes=_require_int(row, "content_bytes"),
        workspace_ids=_split_csv_values(_optional_str(row, "workspace_ids_csv")),
        tags=_split_csv_values(_optional_str(row, "tags_csv")),
        metadata=metadata,
        has_split_lineage=_has_split_lineage(metadata),
    )


def adapt_link_row(row: Mapping[str, object]) -> LinkRow:
    return LinkRow(
        source_id=_require_str(row, "source_id"),
        target_id=_require_str(row, "target_id"),
        link_type=_require_str(row, "type"),
    )


def adapt_agent_run_history_row(row: Mapping[str, object]) -> AgentRunHistoryRow:
    return AgentRunHistoryRow(
        task_id=_require_str(row, "task_id"),
        task_name=_require_str(row, "task_name"),
        status=_require_str(row, "status"),
        started_at=_require_float(row, "started_at"),
        completed_at=_require_float(row, "completed_at"),
        duration_seconds=_require_float(row, "duration_seconds"),
        error_text=_optional_str(row, "error_text"),
        result=coerce_task_result_view(row.get("result_json")),
    )


def _optional_str(row: Mapping[str, object], key: str) -> str | None:
    value = row.get(key)
    return value if isinstance(value, str) and value else None


def _require_str(row: Mapping[str, object], key: str) -> str:
    value = _optional_str(row, key)
    if value is not None:
        return value
    raw = row.get(key)
    raise TypeError(f"Expected non-empty string for {key!r}, got {type(raw)!r}")


def _optional_int(row: Mapping[str, object], key: str) -> int | None:
    value = row.get(key)
    return _coerce_int(value)


def _require_int(row: Mapping[str, object], key: str) -> int:
    value = _coerce_int(row.get(key))
    if value is not None:
        return value
    raw = row.get(key)
    raise TypeError(f"Expected int-compatible value for {key!r}, got {type(raw)!r}")


def _optional_float(row: Mapping[str, object], key: str) -> float | None:
    value = row.get(key)
    return _coerce_float(value)


def _require_float(row: Mapping[str, object], key: str) -> float:
    value = _coerce_float(row.get(key))
    if value is not None:
        return value
    raw = row.get(key)
    raise TypeError(f"Expected float-compatible value for {key!r}, got {type(raw)!r}")


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return None


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return None


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return False


def _optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _coerce_json_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): item for key, item in parsed.items()}


def _coerce_string_list_json(value: object) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, str) and item]


def _split_csv_values(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in value.split(",") if item]


def _runtime_log_task_name(log_data: dict[str, object]) -> str | None:
    extra = log_data.get("extra")
    if not isinstance(extra, Mapping):
        return None
    task_name = extra.get("task_name")
    return task_name if isinstance(task_name, str) and task_name.strip() else None


def _has_split_lineage(metadata: dict[str, object]) -> bool:
    return any(
        key in metadata
        for key in ("split_from_memory_id", "split_child_count", "split_child_memory_ids", "split_group_id")
    )

def _fetchall_rows(db_manager, query: str, params: Sequence[object] | None = None) -> list[dict[str, object]]:
    return ManagementQueryRunner(db_manager).fetchall(query, params)


def _fetchone_row(db_manager, query: str, params: Sequence[object] | None = None) -> dict[str, object] | None:
    return ManagementQueryRunner(db_manager).fetchone(query, params)


def fetch_memory_count_rows(db_manager, workspace_id: str | None) -> list[MemoryCountRow]:
    if db_manager is None:
        return []
    params: list[object] = []
    if workspace_id is None:
        query = (
            "SELECT memories.type, memories.status, COUNT(*) AS count "
            "FROM memories"
        )
    else:
        query = (
            "SELECT memories.type, memories.status, COUNT(*) AS count "
            "FROM memories JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id "
            "WHERE memory_workspaces.workspace_id = ?"
        )
        params.append(workspace_id)
    query += " GROUP BY memories.type, memories.status"
    return [adapt_memory_count_row(row) for row in _fetchall_rows(db_manager, query, params)]


def fetch_task_count_rows(db_manager, workspace_id: str | None) -> list[TaskCountRow]:
    if db_manager is None:
        return []
    query = "SELECT status, COUNT(*) AS count FROM tasks"
    params: list[object] = []
    if workspace_id is not None:
        query += " WHERE workspace_id = ?"
        params.append(workspace_id)
    query += " GROUP BY status"
    return [adapt_task_count_row(row) for row in _fetchall_rows(db_manager, query, params)]


def fetch_running_task_attempt_rows(db_manager) -> list[RunningTaskAttemptRow]:
    if db_manager is None:
        return []
    query = (
        "SELECT tasks.id AS task_id, tasks.workspace_id, tasks.updated_at, tasks.started_at, tasks.claimed_at, "
        "tasks.execution_epoch, attempt.status AS attempt_status, attempt.started_at AS attempt_started_at, "
        "attempt.last_heartbeat_at, attempt.subprocess_pid AS attempt_subprocess_pid "
        "FROM tasks "
        "LEFT JOIN task_execution_attempts AS attempt "
        "ON attempt.task_id = tasks.id AND attempt.execution_epoch = tasks.execution_epoch "
        "WHERE tasks.status = 'running'"
    )
    query += " ORDER BY tasks.updated_at DESC, tasks.id DESC"
    return [adapt_running_task_attempt_row(row) for row in _fetchall_rows(db_manager, query)]


def fetch_memory_metrics_row(db_manager, workspace_id: str | None) -> MemoryMetricsRow | None:
    if db_manager is None:
        return None
    params: list[object] = []
    if workspace_id is None:
        query = (
            "SELECT "
            "COUNT(*) AS total_memories, "
            "COALESCE(SUM(CASE WHEN memories.content = '' THEN 0 ELSE 1 + LENGTH(memories.content) - LENGTH(REPLACE(memories.content, CHAR(10), '')) END), 0) AS total_memory_lines, "
            "COALESCE(SUM(CASE WHEN memories.summary IS NULL OR memories.summary = '' THEN 0 ELSE 1 + LENGTH(memories.summary) - LENGTH(REPLACE(memories.summary, CHAR(10), '')) END), 0) AS total_summary_lines "
            "FROM memories"
        )
    else:
        query = (
            "SELECT "
            "COUNT(*) AS total_memories, "
            "COALESCE(SUM(CASE WHEN memories.content = '' THEN 0 ELSE 1 + LENGTH(memories.content) - LENGTH(REPLACE(memories.content, CHAR(10), '')) END), 0) AS total_memory_lines, "
            "COALESCE(SUM(CASE WHEN memories.summary IS NULL OR memories.summary = '' THEN 0 ELSE 1 + LENGTH(memories.summary) - LENGTH(REPLACE(memories.summary, CHAR(10), '')) END), 0) AS total_summary_lines "
            "FROM memories JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id "
            "WHERE memory_workspaces.workspace_id = ?"
        )
        params.append(workspace_id)
    row = _fetchone_row(db_manager, query, params)
    return None if row is None else adapt_memory_metrics_row(row)


def fetch_pending_journal_metrics_row(db_manager, workspace_id: str | None) -> PendingJournalMetricsRow | None:
    if db_manager is None:
        return None
    query = (
        "SELECT "
        "COUNT(*) AS thought_buffer_entries, "
        "COALESCE(SUM(CASE WHEN content = '' THEN 0 ELSE 1 + LENGTH(content) - LENGTH(REPLACE(content, CHAR(10), '')) END), 0) AS thought_buffer_lines "
        "FROM system1_journal WHERE status = 'pending'"
    )
    params: list[object] = []
    if workspace_id is not None:
        query += " AND workspace_id = ?"
        params.append(workspace_id)
    row = _fetchone_row(db_manager, query, params)
    return None if row is None else adapt_pending_journal_metrics_row(row)


def list_task_run_rows_since(db_manager, *, cutoff: float, workspace_id: str | None) -> list[TaskRunRow]:
    if db_manager is None:
        return []
    query = "SELECT status, completed_at, duration_seconds, result_json FROM task_runs WHERE completed_at >= ?"
    params: list[object] = [cutoff]
    if workspace_id is not None:
        query += " AND workspace_id = ?"
        params.append(workspace_id)
    return [adapt_task_run_row(row) for row in _fetchall_rows(db_manager, query, params)]


def list_maintenance_task_run_rows_since(
    db_manager,
    *,
    cutoff: float,
    workspace_id: str | None,
) -> list[MaintenanceTaskRunRow]:
    if db_manager is None:
        return []
    placeholders = ",".join("?" for _ in MAINTENANCE_TASK_NAMES)
    query = (
        f"SELECT task_id, task_name, status, completed_at, duration_seconds, result_json, error_text "
        f"FROM task_runs WHERE task_name IN ({placeholders}) AND completed_at >= ?"
    )
    params: list[object] = [*MAINTENANCE_TASK_NAMES, cutoff]
    if workspace_id is not None:
        query += " AND workspace_id = ?"
        params.append(workspace_id)
    query += " ORDER BY completed_at DESC, started_at DESC"
    return [adapt_maintenance_task_run_row(row) for row in _fetchall_rows(db_manager, query, params)]


def list_provider_usage_rows_since(db_manager, *, cutoff: float, workspace_id: str | None) -> list[ProviderUsageRow]:
    if db_manager is None:
        return []
    query = (
        "SELECT task_name, provider_key, provider_name, model_name, status, duration_seconds, created_at, reason_category, reason_code, retry_delay_seconds "
        "FROM provider_usage WHERE created_at >= ?"
    )
    params: list[object] = [cutoff]
    if workspace_id is not None:
        query += " AND workspace_id = ?"
        params.append(workspace_id)
    return [adapt_provider_usage_row(row) for row in _fetchall_rows(db_manager, query, params)]


def count_recent_conversation_statuses(
    provider_usage_repo,
    *,
    after: float,
    workspace_id: str | None,
    limit: int = 10_000,
) -> dict[str, int]:
    if provider_usage_repo is None:
        return {}
    count_fn = getattr(provider_usage_repo, "count_conversation_statuses_since", None)
    if count_fn is not None:
        return {str(status): int(count) for status, count in count_fn(after=after, workspace_id=workspace_id).items()}
    counts: dict[str, int] = {}
    for record in provider_usage_repo.list_conversations(
        workspace_id=workspace_id,
        limit=limit,
    ):
        if record.completed_at < after:
            continue
        counts[record.status] = counts.get(record.status, 0) + 1
    return counts


def count_recent_memory_updates(
    repository,
    *,
    cutoff: datetime,
    workspace_id: str | None,
) -> int:
    if repository is None:
        return 0
    count_fn = getattr(repository, "count_memories_updated_since", None)
    if count_fn is None:
        return 0
    return int(count_fn(cutoff.isoformat(), workspace_id=workspace_id))


def list_ai_conversation_rows_since(
    db_manager,
    *,
    cutoff: float,
    upper_bound: float,
    workspace_id: str | None,
) -> list[AiConversationRow]:
    if db_manager is None:
        return []
    query = (
        "SELECT provider_key, completed_at, response_text, parsed_json "
        "FROM ai_conversations WHERE completed_at >= ? AND completed_at <= ?"
    )
    params: list[object] = [cutoff, upper_bound]
    if workspace_id is not None:
        query += " AND workspace_id = ?"
        params.append(workspace_id)
    return [adapt_ai_conversation_row(row) for row in _fetchall_rows(db_manager, query, params)]


def summarize_copilot_premium_requests(
    db_manager,
    *,
    workspace_id: str | None,
    now: float | None = None,
) -> CopilotPremiumUsageSummary:
    if db_manager is None:
        return CopilotPremiumUsageSummary()

    current_time = time.time() if now is None else now
    last_day_cutoff = current_time - 86_400
    today_start = datetime.fromtimestamp(current_time, UTC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    query_cutoff = min(last_day_cutoff, today_start)

    premium_requests_today = 0.0
    premium_requests_last_day = 0.0
    for row in list_ai_conversation_rows_since(
        db_manager,
        cutoff=query_cutoff,
        upper_bound=current_time,
        workspace_id=workspace_id,
    ):
        provider_key = row.provider_key
        if not provider_key.startswith("copilot"):
            continue

        premium_requests = extract_copilot_premium_requests(
            response_text=row.response_text,
            parsed_json=row.parsed_json,
        )
        if premium_requests <= 0:
            continue

        completed_at = row.completed_at
        if completed_at >= last_day_cutoff:
            premium_requests_last_day += premium_requests
        if completed_at >= today_start:
            premium_requests_today += premium_requests

    return CopilotPremiumUsageSummary(
        copilot_premium_requests_today=premium_requests_today,
        copilot_premium_requests_last_day=premium_requests_last_day,
    )


def extract_copilot_premium_requests(*, response_text: str | None, parsed_json: str | None) -> float:
    premium_requests: list[float] = []
    premium_requests.extend(_premium_requests_from_maybe_json(parsed_json))
    premium_requests.extend(_premium_requests_from_maybe_json(response_text))
    return max(premium_requests, default=0)


def _premium_requests_from_maybe_json(value: str | None) -> list[float]:
    if not isinstance(value, str):
        return []
    text = value.strip()
    if not text:
        return []

    premium_requests: list[float] = []
    parsed_text = _try_json_loads(text)
    if parsed_text is not None:
        premium_requests.extend(_collect_premium_requests(parsed_text))
        return premium_requests

    for line in text.splitlines():
        parsed_line = _try_json_loads(line.strip())
        if parsed_line is not None:
            premium_requests.extend(_collect_premium_requests(parsed_line))
    return premium_requests


def _try_json_loads(text: str) -> Any | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _collect_premium_requests(payload: Any) -> list[float]:
    results: list[float] = []
    if isinstance(payload, dict):
        premium_value = payload.get("premiumRequests")
        if isinstance(premium_value, bool):
            premium_value = None
        if isinstance(premium_value, (int, float)):
            results.append(float(premium_value))
        for value in payload.values():
            results.extend(_collect_premium_requests(value))
    elif isinstance(payload, list):
        for item in payload:
            results.extend(_collect_premium_requests(item))
    return results


def list_runtime_log_rows_since(
    db_manager,
    *,
    cutoff: float,
    workspace_id: str | None,
    logger_name: str | None = None,
    level: str | None = None,
) -> list[RuntimeLogRow]:
    if db_manager is None:
        return []
    query = (
        "SELECT logger_name, level, message, created_at, data_json "
        "FROM runtime_logs WHERE created_at >= ?"
    )
    params: list[object] = [cutoff]
    if workspace_id is not None:
        query += " AND workspace_id = ?"
        params.append(workspace_id)
    if logger_name is not None:
        query += " AND logger_name = ?"
        params.append(logger_name)
    if level is not None:
        query += " AND level = ?"
        params.append(level)
    return [adapt_runtime_log_row(row) for row in _fetchall_rows(db_manager, query, params)]


def list_provider_policy_event_rows_since(
    db_manager,
    *,
    cutoff: float,
    workspace_id: str | None,
) -> list[ProviderPolicyEventRow]:
    if db_manager is None:
        return []
    query = (
        "SELECT task_name, task_id, event_kind, warning_kind, provider_key, provider_name, model_name, route_key, candidate_routes_json, reason_category, reason_code, retry_delay_seconds, warning_suppressed, created_at "
        "FROM provider_policy_events WHERE created_at >= ?"
    )
    params: list[object] = [cutoff]
    if workspace_id is not None:
        query += " AND workspace_id = ?"
        params.append(workspace_id)
    return [adapt_provider_policy_event_row(row) for row in _fetchall_rows(db_manager, query, params)]


def list_memory_tool_event_rows_since(
    db_manager,
    *,
    cutoff: float,
    workspace_id: str | None,
) -> list[MemoryToolEventRow]:
    if db_manager is None:
        return []
    query = (
        "SELECT invocation_id, workspace_id, caller_kind, event_kind, memory_id, query_text, result_rank, result_count, duration_ms, created_at "
        "FROM memory_tool_events WHERE created_at >= ?"
    )
    params: list[object] = [cutoff]
    if workspace_id is not None:
        query += " AND workspace_id = ?"
        params.append(workspace_id)
    query += " ORDER BY created_at DESC, id DESC"
    return [adapt_memory_tool_event_row(row) for row in _fetchall_rows(db_manager, query, params)]


def list_scoped_memory_rows(db_manager, workspace_id: str | None) -> list[ScopedMemoryRow]:
    if db_manager is None:
        return []
    query = (
        "SELECT memories.id, memories.title, memories.summary, memories.type, memories.status, memories.created_at, memories.updated_at, memories.metadata, "
        "memories.last_accessed_at, memories.last_surfaced_at, LENGTH(COALESCE(memories.content, '')) AS content_bytes, "
        "COALESCE(workspace_agg.workspace_ids, '') AS workspace_ids_csv, COALESCE(tag_agg.tags, '') AS tags_csv "
        "FROM memories "
        "LEFT JOIN ("
        "SELECT memory_id, GROUP_CONCAT(DISTINCT workspace_id) AS workspace_ids "
        "FROM memory_workspaces GROUP BY memory_id"
        ") workspace_agg ON workspace_agg.memory_id = memories.id "
        "LEFT JOIN ("
        "SELECT memory_tags.memory_id, GROUP_CONCAT(DISTINCT tags.name) AS tags "
        "FROM memory_tags JOIN tags ON tags.id = memory_tags.tag_id GROUP BY memory_tags.memory_id"
        ") tag_agg ON tag_agg.memory_id = memories.id"
    )
    params: list[object] = []
    if workspace_id is not None:
        query += (
            " WHERE EXISTS (SELECT 1 FROM memory_workspaces WHERE memory_workspaces.memory_id = memories.id "
            "AND memory_workspaces.workspace_id = ?)"
        )
        params.append(workspace_id)
    return [adapt_scoped_memory_row(row) for row in _fetchall_rows(db_manager, query, params)]


def list_scoped_link_rows(db_manager, workspace_id: str | None, memory_ids: set[str]) -> list[LinkRow]:
    if db_manager is None or not memory_ids:
        return []
    if workspace_id is None:
        return [adapt_link_row(row) for row in _fetchall_rows(db_manager, "SELECT source_id, target_id, type FROM links")]

    placeholders = ",".join("?" for _ in memory_ids)
    params = [*memory_ids, *memory_ids]
    query = (
        f"SELECT source_id, target_id, type FROM links WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})"
    )
    return [adapt_link_row(row) for row in _fetchall_rows(db_manager, query, params)]


def build_queue_diagnostics(task_queue, workspace_id: str | None, limit: int = 8, now: float | None = None) -> list[QueueDiagnosticPayload]:
    current_time = now if now is not None else time.time()
    pending_tasks = task_queue.list_tasks(
        status="pending",
        workspace_id=workspace_id,
        limit=200,
    )
    ordered = sorted(
        pending_tasks,
        key=lambda task: (
            0 if task.available_at <= current_time else 1,
            task.priority,
            task.available_at,
            task.created_at,
        ),
    )
    diagnostics: list[QueueDiagnosticPayload] = []
    for task in ordered[:limit]:
        runnable = task.available_at <= current_time
        diagnostics.append(
            QueueDiagnosticPayload(
                task_id=task.id,
                task_name=task.task_name,
                workspace_id=task.workspace_id,
                priority=task.priority,
                pending_state="runnable" if runnable else "scheduled",
                trigger=task.data.get("trigger") if isinstance(task.data.get("trigger"), str) else None,
                created_at=task.created_at,
                available_at=task.available_at,
                age_seconds=max(current_time - task.created_at, 0.0),
                ready_in_seconds=0.0 if runnable else max(task.available_at - current_time, 0.0),
                overdue_seconds=max(current_time - task.available_at, 0.0) if runnable else 0.0,
            )
        )
    return diagnostics
