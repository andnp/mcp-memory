from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from typing import Mapping


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
    result: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MaintenanceTaskRunRow:
    task_id: str
    task_name: str
    status: str
    completed_at: float
    duration_seconds: float
    result: dict[str, object] = field(default_factory=dict)
    error_text: str | None = None


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
    result: dict[str, object] = field(default_factory=dict)
    error_text: str | None = None


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
        result=_coerce_json_mapping(row.get("result_json")),
    )


def adapt_maintenance_task_run_row(row: Mapping[str, object]) -> MaintenanceTaskRunRow:
    return MaintenanceTaskRunRow(
        task_id=_require_str(row, "task_id"),
        task_name=_require_str(row, "task_name"),
        status=_require_str(row, "status"),
        completed_at=_require_float(row, "completed_at"),
        duration_seconds=_require_float(row, "duration_seconds"),
        result=_coerce_json_mapping(row.get("result_json")),
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
        result=_coerce_json_mapping(row.get("result_json")),
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
