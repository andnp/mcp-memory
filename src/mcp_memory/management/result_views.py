from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias, cast

from pydantic import BaseModel, Field, JsonValue

from mcp_memory.core.task_results import TaskRunResult, build_task_run_result_summary, decode_task_run_result_payload
from mcp_memory.management.models import (
    IngestAuditPayload,
    IngestEntryDispositionPayload,
    MutationOutcomePayload,
    RunResultMetadataPayload,
    SelectorFeatureSnapshotPayload,
    SelectorMetricSnapshotPayload,
    SelectorPopulationSnapshotPayload,
)

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


def _coerce_int(value: JsonValue | object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _coerce_bool(value: JsonValue | object) -> bool | None:
    return value if isinstance(value, bool) else None


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