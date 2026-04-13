from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias


TaskRunResultSource: TypeAlias = "TaskRunResult | Mapping[str, object] | str | None"

_PREFERRED_SUMMARY_KEYS: tuple[str, ...] = (
    "created",
    "merged",
    "updated",
    "archived",
    "absorbed_observations",
    "degraded",
    "restored",
    "deleted_tasks",
    "deleted_journal_entries",
    "claimed_entry_ids",
    "recoverable_entry_ids",
    "deleted_entry_ids",
    "released_entry_ids",
    "meaningful_actions",
    "processed_entry_ids",
    "created_memory_ids",
    "lines_compressed",
    "requested_strategy",
    "strategy_used",
    "strategy_fallback_reason",
    "candidate_count",
    "sampled_memory_ids",
    "compatibility_group",
    "claimed_work_item_count",
    "provider_calls_used",
    "tool_calls_executed",
    "mutations",
    "work_item_batch_limit",
    "max_batches_per_run",
    "requested_grouping_strategy",
    "grouping_strategy_used",
    "grouping_fallback_reason",
    "group_count",
    "campaign_key",
    "campaign_origin_family",
    "campaign_family_keys",
    "campaign_continuation_supported",
    "compatible_batch_calls",
)


@dataclass(frozen=True)
class TaskRunResult(Mapping[str, Any]):
    payload: dict[str, Any] = field(default_factory=dict)
    summary: str | None = None
    meaningful_actions: int | None = None
    lines_compressed: int | None = None
    stale: int | None = None

    def __post_init__(self) -> None:
        normalized_payload = decode_task_run_result_payload(self.payload)
        object.__setattr__(self, "payload", normalized_payload)
        object.__setattr__(
            self,
            "summary",
            self.summary if self.summary is not None else build_task_run_result_summary(normalized_payload),
        )
        object.__setattr__(self, "meaningful_actions", _coerce_optional_int(normalized_payload.get("meaningful_actions")))
        object.__setattr__(self, "lines_compressed", _coerce_optional_int(normalized_payload.get("lines_compressed")))
        object.__setattr__(self, "stale", _coerce_optional_int(normalized_payload.get("stale")))

    @property
    def raw_payload(self) -> dict[str, Any]:
        return dict(self.payload)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def copy(self) -> dict[str, Any]:
        return self.to_dict()

    def get(self, key: str, default: object | None = None) -> Any:
        return self.payload.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.payload)

    def __len__(self) -> int:
        return len(self.payload)

    def __bool__(self) -> bool:
        return bool(self.payload)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TaskRunResult):
            return self.payload == other.payload
        if isinstance(other, Mapping):
            return self.payload == {str(key): value for key, value in other.items()}
        return False


def coerce_task_run_result(raw_result: TaskRunResultSource | object) -> TaskRunResult:
    if isinstance(raw_result, TaskRunResult):
        return raw_result
    return TaskRunResult(payload=decode_task_run_result_payload(raw_result))


def decode_task_run_result_payload(raw_result: TaskRunResultSource | object) -> dict[str, Any]:
    if isinstance(raw_result, TaskRunResult):
        return raw_result.to_dict()
    if isinstance(raw_result, Mapping):
        return {str(key): value for key, value in raw_result.items()}
    if not isinstance(raw_result, str) or not raw_result.strip():
        return {}
    try:
        decoded = json.loads(raw_result)
    except ValueError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def build_task_run_result_summary(result: Mapping[str, object]) -> str | None:
    if not result:
        return None

    formatted_parts: list[str] = []
    for key in _PREFERRED_SUMMARY_KEYS:
        if key not in result:
            continue
        value = result[key]
        if isinstance(value, list):
            formatted_parts.append(f"{key}={len(value)}")
        else:
            formatted_parts.append(f"{key}={value}")

    if formatted_parts:
        return ", ".join(formatted_parts)

    for key in sorted(result):
        value = result[key]
        if isinstance(value, (str, int, float, bool)):
            formatted_parts.append(f"{key}={value}")
    return ", ".join(formatted_parts) if formatted_parts else None


def _coerce_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None