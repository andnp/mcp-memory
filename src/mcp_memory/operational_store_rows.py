from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from typing import Self


type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type _ScalarLike = bool | int | float | str


def encode_json_object(data: Mapping[str, object]) -> str:
    return json.dumps(data, sort_keys=True)


def encode_optional_json_object(data: Mapping[str, object] | None) -> str | None:
    if data is None:
        return None
    return encode_json_object(data)


def decode_json_object(raw: object) -> JsonObject:
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            decoded = json.loads(raw)
        except ValueError:
            return {}
        return _normalize_json_object(decoded)
    return _normalize_json_object(raw)


def decode_optional_json_object(raw: object) -> JsonObject | None:
    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    return decode_json_object(raw)


def _decode_ai_conversation_parsed_json(raw: object) -> JsonObject | None:
    """Decode persisted ``ai_conversations.parsed_json`` payloads.

    Contract:
    - SQL ``NULL`` and blank strings remain missing payloads (``None``)
    - malformed JSON and any non-object payload normalize to ``{}``
    - object payloads normalize to plain JSON-compatible dictionaries

    This keeps legacy/bad rows readable with the same semantics across SQLite text
    storage and Postgres JSONB reads.
    """

    return decode_optional_json_object(raw)


def _normalize_json_object(raw: object) -> JsonObject:
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(key): _normalize_json_value(value)
        for key, value in raw.items()
    }


def _normalize_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_normalize_json_value(item) for item in value]
    return str(value)


def _require_int(value: object, field_name: str) -> int:
    if value is None:
        raise TypeError(f"{field_name} cannot be None")
    if not isinstance(value, bool | int | float | str):
        raise TypeError(f"{field_name} must be int-compatible")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be int-compatible") from exc


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return _require_int(value, "value")


def _require_float(value: object, field_name: str) -> float:
    if value is None:
        raise TypeError(f"{field_name} cannot be None")
    if not isinstance(value, bool | int | float | str):
        raise TypeError(f"{field_name} must be float-compatible")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be float-compatible") from exc


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return _require_float(value, "value")


def _require_str(value: object, field_name: str) -> str:
    if value is None:
        raise TypeError(f"{field_name} cannot be None")
    return str(value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


@dataclass(frozen=True)
class ProviderUsageSummary:
    task_name: str | None
    provider_key: str
    provider_name: str
    model_name: str
    calls_last_hour: int
    calls_last_day: int
    failures_last_hour: int
    failures_last_day: int
    skips_last_hour: int
    skips_last_day: int
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


@dataclass(frozen=True)
class ProviderAdmissionStateRecord:
    provider_key: str
    model_name: str
    reason_category: str
    reason_code: str
    error_text: str | None
    retry_delay_seconds: float | None
    active_until: float
    updated_at: float

    @classmethod
    def from_sqlite_row(cls, row: Mapping[str, object]) -> Self:
        return cls(
            provider_key=_require_str(row["provider_key"], "provider_key"),
            model_name=_require_str(row["model_name"], "model_name"),
            reason_category=_require_str(row["reason_category"], "reason_category"),
            reason_code=_require_str(row["reason_code"], "reason_code"),
            error_text=_optional_str(row["error_text"]),
            retry_delay_seconds=_optional_float(row["retry_delay_seconds"]),
            active_until=_require_float(row["active_until"], "active_until"),
            updated_at=_require_float(row["updated_at"], "updated_at"),
        )

    @classmethod
    def from_postgres_row(cls, row: Sequence[object]) -> Self:
        return cls(
            provider_key=_require_str(row[0], "provider_key"),
            model_name=_require_str(row[1], "model_name"),
            reason_category=_require_str(row[2], "reason_category"),
            reason_code=_require_str(row[3], "reason_code"),
            error_text=_optional_str(row[4]),
            retry_delay_seconds=_optional_float(row[5]),
            active_until=_require_float(row[6], "active_until"),
            updated_at=_require_float(row[7], "updated_at"),
        )


@dataclass(frozen=True)
class AIConversationRecord:
    id: int
    request_id: str
    attempt: int
    workspace_id: str | None
    task_name: str | None
    task_id: str | None
    provider_key: str
    provider_name: str
    model_name: str
    subprocess_pid: int | None
    prompt_text: str
    response_text: str
    parsed: JsonObject | None
    status: str
    error_text: str | None
    reason_category: str | None
    reason_code: str | None
    retry_delay_seconds: float | None
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

    @classmethod
    def from_sqlite_row(cls, row: Mapping[str, object]) -> Self:
        return cls(
            id=_require_int(row["id"], "id"),
            request_id=_require_str(row["request_id"], "request_id"),
            attempt=_require_int(row["attempt"], "attempt"),
            workspace_id=_optional_str(row["workspace_id"]),
            task_name=_optional_str(row["task_name"]),
            task_id=_optional_str(row["task_id"]),
            provider_key=_require_str(row["provider_key"], "provider_key"),
            provider_name=_require_str(row["provider_name"], "provider_name"),
            model_name=_require_str(row["model_name"], "model_name"),
            subprocess_pid=_optional_int(row["subprocess_pid"]),
            prompt_text=_require_str(row["prompt_text"], "prompt_text"),
            response_text=_require_str(row["response_text"], "response_text"),
            parsed=_decode_ai_conversation_parsed_json(row["parsed_json"]),
            status=_require_str(row["status"], "status"),
            error_text=_optional_str(row["error_text"]),
            reason_category=_optional_str(row["reason_category"]),
            reason_code=_optional_str(row["reason_code"]),
            retry_delay_seconds=_optional_float(row["retry_delay_seconds"]),
            started_at=_require_float(row["started_at"], "started_at"),
            completed_at=_require_float(row["completed_at"], "completed_at"),
            duration_seconds=_optional_float(row["duration_seconds"]) or 0.0,
            input_tokens=_optional_int(row["input_tokens"]) if "input_tokens" in row.keys() else None,
            output_tokens=_optional_int(row["output_tokens"]) if "output_tokens" in row.keys() else None,
            cached_input_tokens=_optional_int(row["cached_input_tokens"]) if "cached_input_tokens" in row.keys() else None,
            cache_write_tokens=_optional_int(row["cache_write_tokens"]) if "cache_write_tokens" in row.keys() else None,
            reasoning_tokens=_optional_int(row["reasoning_tokens"]) if "reasoning_tokens" in row.keys() else None,
            total_tokens=_optional_int(row["total_tokens"]) if "total_tokens" in row.keys() else None,
            token_usage_source=_optional_str(row["token_usage_source"]) if "token_usage_source" in row.keys() else None,
        )

    @classmethod
    def from_postgres_row(cls, row: Sequence[object]) -> Self:
        return cls(
            id=_require_int(row[0], "id"),
            request_id=_require_str(row[1], "request_id"),
            attempt=_require_int(row[2], "attempt"),
            workspace_id=_optional_str(row[3]),
            task_name=_optional_str(row[4]),
            task_id=_optional_str(row[5]),
            provider_key=_require_str(row[6], "provider_key"),
            provider_name=_require_str(row[7], "provider_name"),
            model_name=_require_str(row[8], "model_name"),
            subprocess_pid=_optional_int(row[9]),
            prompt_text=_require_str(row[10], "prompt_text"),
            response_text=_require_str(row[11], "response_text"),
            parsed=_decode_ai_conversation_parsed_json(row[12]),
            status=_require_str(row[13], "status"),
            error_text=_optional_str(row[14]),
            reason_category=_optional_str(row[15]),
            reason_code=_optional_str(row[16]),
            retry_delay_seconds=_optional_float(row[17]),
            started_at=_require_float(row[18], "started_at"),
            completed_at=_require_float(row[19], "completed_at"),
            duration_seconds=_optional_float(row[20]) or 0.0,
            input_tokens=_optional_int(row[21]) if len(row) > 21 else None,
            output_tokens=_optional_int(row[22]) if len(row) > 22 else None,
            cached_input_tokens=_optional_int(row[23]) if len(row) > 23 else None,
            cache_write_tokens=_optional_int(row[24]) if len(row) > 24 else None,
            reasoning_tokens=_optional_int(row[25]) if len(row) > 25 else None,
            total_tokens=_optional_int(row[26]) if len(row) > 26 else None,
            token_usage_source=_optional_str(row[27]) if len(row) > 27 else None,
        )


@dataclass(frozen=True)
class RuntimeLogRecord:
    id: int
    created_at: float
    level: str
    logger_name: str
    source: str
    message: str
    data: JsonObject = field(default_factory=dict)

    @classmethod
    def from_sqlite_row(cls, row: Mapping[str, object]) -> Self:
        return cls(
            id=_require_int(row["id"], "id"),
            created_at=_require_float(row["created_at"], "created_at"),
            level=_require_str(row["level"], "level"),
            logger_name=_require_str(row["logger_name"], "logger_name"),
            source=_require_str(row["source"], "source"),
            message=_require_str(row["message"], "message"),
            data=decode_json_object(row["data_json"]),
        )

    @classmethod
    def from_postgres_row(cls, row: Sequence[object]) -> Self:
        return cls(
            id=_require_int(row[0], "id"),
            created_at=_require_float(row[6], "created_at"),
            level=_require_str(row[4], "level"),
            logger_name=_require_str(row[3], "logger_name"),
            source=_require_str(row[2], "source"),
            message=_require_str(row[5], "message"),
            data=decode_json_object(row[7]),
        )


@dataclass(frozen=True)
class RuntimeLogSummary:
    total: int
    by_level: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingIntegrityEventRecord:
    id: int
    workspace_id: str | None
    event_kind: str
    model_name: str | None
    source_kind: str | None
    source_id: str | None
    scanned_row_count: int | None
    invalid_row_count: int | None
    mixed_dimension_group_count: int | None
    created_at: float
    details: JsonObject = field(default_factory=dict)

    @classmethod
    def from_sqlite_row(cls, row: Mapping[str, object]) -> Self:
        return cls(
            id=_require_int(row["id"], "id"),
            workspace_id=_optional_str(row["workspace_id"]),
            event_kind=_require_str(row["event_kind"], "event_kind"),
            model_name=_optional_str(row["model_name"]),
            source_kind=_optional_str(row["source_kind"]),
            source_id=_optional_str(row["source_id"]),
            scanned_row_count=_optional_int(row["scanned_row_count"]),
            invalid_row_count=_optional_int(row["invalid_row_count"]),
            mixed_dimension_group_count=_optional_int(row["mixed_dimension_group_count"]),
            created_at=_require_float(row["created_at"], "created_at"),
            details=decode_json_object(row["details_json"]),
        )

    @classmethod
    def from_postgres_row(cls, row: Sequence[object]) -> Self:
        return cls(
            id=_require_int(row[0], "id"),
            workspace_id=_optional_str(row[1]),
            event_kind=_require_str(row[2], "event_kind"),
            model_name=_optional_str(row[3]),
            source_kind=_optional_str(row[4]),
            source_id=_optional_str(row[5]),
            scanned_row_count=_optional_int(row[6]),
            invalid_row_count=_optional_int(row[7]),
            mixed_dimension_group_count=_optional_int(row[8]),
            created_at=_require_float(row[10], "created_at"),
            details=decode_json_object(row[9]),
        )


@dataclass(frozen=True)
class EmbeddingIntegrityEventSummary:
    total: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    last_scan: EmbeddingIntegrityEventRecord | None = None
    last_blocked_fallback_write: EmbeddingIntegrityEventRecord | None = None


@dataclass(frozen=True)
class ProviderUsageSample:
    task_name: str | None
    task_id: str | None
    execution_epoch: int | None
    request_id: str | None
    attempt: int | None
    attempt_identity: str | None
    provider_key: str
    provider_name: str
    model_name: str
    status: str
    duration_seconds: float
    created_at: float
    reason_code: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    token_usage_source: str | None = None

    @classmethod
    def from_sqlite_row(cls, row: Mapping[str, object]) -> Self:
        return cls(
            task_name=_optional_str(row["task_name"]),
            task_id=_optional_str(row["task_id"]),
            execution_epoch=_optional_int(row["execution_epoch"]),
            request_id=_optional_str(row["request_id"]),
            attempt=_optional_int(row["attempt"]),
            attempt_identity=_optional_str(row["attempt_identity"]),
            provider_key=_require_str(row["provider_key"], "provider_key"),
            provider_name=_require_str(row["provider_name"], "provider_name"),
            model_name=_require_str(row["model_name"], "model_name"),
            status=_require_str(row["status"], "status"),
            duration_seconds=_optional_float(row["duration_seconds"]) or 0.0,
            created_at=_optional_float(row["created_at"]) or 0.0,
            reason_code=_optional_str(row["reason_code"]),
            input_tokens=_optional_int(row["input_tokens"]) if "input_tokens" in row.keys() else None,
            output_tokens=_optional_int(row["output_tokens"]) if "output_tokens" in row.keys() else None,
            cached_input_tokens=_optional_int(row["cached_input_tokens"]) if "cached_input_tokens" in row.keys() else None,
            cache_write_tokens=_optional_int(row["cache_write_tokens"]) if "cache_write_tokens" in row.keys() else None,
            reasoning_tokens=_optional_int(row["reasoning_tokens"]) if "reasoning_tokens" in row.keys() else None,
            total_tokens=_optional_int(row["total_tokens"]) if "total_tokens" in row.keys() else None,
            token_usage_source=_optional_str(row["token_usage_source"]) if "token_usage_source" in row.keys() else None,
        )

    @classmethod
    def from_postgres_row(cls, row: Sequence[object]) -> Self:
        return cls(
            task_name=_optional_str(row[0]),
            task_id=_optional_str(row[1]),
            execution_epoch=_optional_int(row[2]),
            request_id=_optional_str(row[3]),
            attempt=_optional_int(row[4]),
            attempt_identity=_optional_str(row[5]),
            provider_key=_require_str(row[6], "provider_key"),
            provider_name=_require_str(row[7], "provider_name"),
            model_name=_require_str(row[8], "model_name"),
            status=_require_str(row[9], "status"),
            duration_seconds=_optional_float(row[10]) or 0.0,
            created_at=_optional_float(row[11]) or 0.0,
            reason_code=_optional_str(row[12]),
            input_tokens=_optional_int(row[13]) if len(row) > 13 else None,
            output_tokens=_optional_int(row[14]) if len(row) > 14 else None,
            cached_input_tokens=_optional_int(row[15]) if len(row) > 15 else None,
            cache_write_tokens=_optional_int(row[16]) if len(row) > 16 else None,
            reasoning_tokens=_optional_int(row[17]) if len(row) > 17 else None,
            total_tokens=_optional_int(row[18]) if len(row) > 18 else None,
            token_usage_source=_optional_str(row[19]) if len(row) > 19 else None,
        )

    @property
    def group_key(self) -> _ProviderUsageGroupKey:
        return _ProviderUsageGroupKey(
            task_name=self.task_name,
            provider_key=self.provider_key,
            provider_name=self.provider_name,
            model_name=self.model_name,
        )


@dataclass(frozen=True)
class _ProviderUsageGroupKey:
    task_name: str | None
    provider_key: str
    provider_name: str
    model_name: str


@dataclass
class _ProviderUsageAccumulator:
    calls_last_hour: int = 0
    calls_last_day: int = 0
    failures_last_hour: int = 0
    failures_last_day: int = 0
    skips_last_hour: int = 0
    skips_last_day: int = 0
    duration_last_hour: list[float] = field(default_factory=list)
    duration_last_day: list[float] = field(default_factory=list)
    failure_reasons: dict[str, int] = field(default_factory=dict)
    skip_reasons: dict[str, int] = field(default_factory=dict)
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
    token_usage_sources: dict[str, int] = field(default_factory=dict)

    def observe(self, sample: ProviderUsageSample, *, last_hour: float, last_day: float) -> None:
        in_last_hour = sample.created_at >= last_hour
        in_last_day = sample.created_at >= last_day
        if in_last_day:
            self._observe_tokens(sample, in_last_hour=in_last_hour)
        if sample.status == "skipped":
            if in_last_day:
                self.skips_last_day += 1
                skip_key = sample.reason_code or "unclassified_skip"
                self.skip_reasons[skip_key] = self.skip_reasons.get(skip_key, 0) + 1
            if in_last_hour:
                self.skips_last_hour += 1
            return

        if in_last_day:
            self.calls_last_day += 1
            self.duration_last_day.append(sample.duration_seconds)
        if in_last_hour:
            self.calls_last_hour += 1
            self.duration_last_hour.append(sample.duration_seconds)
        if sample.status == "success" or not in_last_day:
            return

        self.failures_last_day += 1
        failure_key = sample.reason_code or "unclassified_error"
        self.failure_reasons[failure_key] = self.failure_reasons.get(failure_key, 0) + 1
        if in_last_hour:
            self.failures_last_hour += 1

    def to_summary(
        self,
        key: _ProviderUsageGroupKey,
        *,
        active_state: ProviderAdmissionStateRecord | None,
        current_time: float,
    ) -> ProviderUsageSummary:
        return ProviderUsageSummary(
            task_name=key.task_name,
            provider_key=key.provider_key,
            provider_name=key.provider_name,
            model_name=key.model_name,
            calls_last_hour=self.calls_last_hour,
            calls_last_day=self.calls_last_day,
            failures_last_hour=self.failures_last_hour,
            failures_last_day=self.failures_last_day,
            skips_last_hour=self.skips_last_hour,
            skips_last_day=self.skips_last_day,
            avg_duration_last_hour=_mean(self.duration_last_hour),
            avg_duration_last_day=_mean(self.duration_last_day),
            top_failure_reason_last_day=_top_reason(self.failure_reasons),
            top_skip_reason_last_day=_top_reason(self.skip_reasons),
            active_admission_reason=None if active_state is None else active_state.reason_code,
            active_admission_category=None if active_state is None else active_state.reason_category,
            active_retry_delay_seconds=(
                None
                if active_state is None
                else max(active_state.active_until - current_time, 0.0)
            ),
            input_tokens_last_hour=self.input_tokens_last_hour,
            input_tokens_last_day=self.input_tokens_last_day,
            output_tokens_last_hour=self.output_tokens_last_hour,
            output_tokens_last_day=self.output_tokens_last_day,
            cached_input_tokens_last_hour=self.cached_input_tokens_last_hour,
            cached_input_tokens_last_day=self.cached_input_tokens_last_day,
            cache_write_tokens_last_hour=self.cache_write_tokens_last_hour,
            cache_write_tokens_last_day=self.cache_write_tokens_last_day,
            reasoning_tokens_last_hour=self.reasoning_tokens_last_hour,
            reasoning_tokens_last_day=self.reasoning_tokens_last_day,
            total_tokens_last_hour=self.total_tokens_last_hour,
            total_tokens_last_day=self.total_tokens_last_day,
            token_usage_source=_top_reason(self.token_usage_sources),
        )

    def _observe_tokens(self, sample: ProviderUsageSample, *, in_last_hour: bool) -> None:
        fields = (
            ("input_tokens", "input_tokens_last_day", "input_tokens_last_hour"),
            ("output_tokens", "output_tokens_last_day", "output_tokens_last_hour"),
            ("cached_input_tokens", "cached_input_tokens_last_day", "cached_input_tokens_last_hour"),
            ("cache_write_tokens", "cache_write_tokens_last_day", "cache_write_tokens_last_hour"),
            ("reasoning_tokens", "reasoning_tokens_last_day", "reasoning_tokens_last_hour"),
            ("total_tokens", "total_tokens_last_day", "total_tokens_last_hour"),
        )
        for source_field, day_field, hour_field in fields:
            value = getattr(sample, source_field)
            if value is None:
                continue
            setattr(self, day_field, getattr(self, day_field) + value)
            if in_last_hour:
                setattr(self, hour_field, getattr(self, hour_field) + value)
        if sample.token_usage_source:
            self.token_usage_sources[sample.token_usage_source] = (
                self.token_usage_sources.get(sample.token_usage_source, 0) + 1
            )


def build_provider_usage_summaries(
    *,
    samples: Iterable[ProviderUsageSample],
    active_states: Iterable[ProviderAdmissionStateRecord],
    current_time: float,
) -> list[ProviderUsageSummary]:
    aggregate: dict[_ProviderUsageGroupKey, _ProviderUsageAccumulator] = {}
    last_hour = current_time - 3600
    last_day = current_time - 86400
    for sample in samples:
        aggregate.setdefault(sample.group_key, _ProviderUsageAccumulator()).observe(
            sample,
            last_hour=last_hour,
            last_day=last_day,
        )

    active_state_map = {
        (state.provider_key, state.model_name): state
        for state in active_states
    }
    summaries = [
        bucket.to_summary(
            key,
            active_state=active_state_map.get((key.provider_key, key.model_name)),
            current_time=current_time,
        )
        for key, bucket in aggregate.items()
    ]
    summaries.sort(
        key=lambda item: (
            -item.calls_last_day,
            -item.skips_last_day,
            item.task_name or "",
            item.provider_key,
        )
    )
    return summaries


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _top_reason(counts: Mapping[str, int]) -> str | None:
    if not counts:
        return None
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]
