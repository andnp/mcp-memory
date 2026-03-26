from __future__ import annotations

import json
from typing import Any, cast

from mcp_memory.context import ApplicationContext
from mcp_memory.mcp.internal_ingest_keys import (
    INGEST_ENTRY_DISPOSITIONS_TASK_DATA_KEY,
    INGEST_HANDLED_ENTRY_IDS_TASK_DATA_KEY,
    INGEST_TOUCHED_MEMORY_IDS_TASK_DATA_KEY,
    INGEST_TOOL_INVOCATIONS_TASK_DATA_KEY,
)


def _build_ingest_result(
    *,
    created_ids: list[str],
    claimed_ids: list[int],
    deleted_ids: list[int],
    recoverable_ids: list[int],
    released_ids: list[int],
    meaningful_actions: int,
    semantic_entry_dispositions: list[dict[str, Any]] | None = None,
    recorded_touched_memory_ids: list[str] | None = None,
) -> dict[str, Any]:
    entry_dispositions = _finalize_entry_dispositions(
        claimed_ids=claimed_ids,
        recoverable_ids=recoverable_ids,
        released_ids=released_ids,
        semantic_entry_dispositions=semantic_entry_dispositions or [],
    )
    touched_memory_ids = _merge_string_lists(
        _memory_ids_for_dispositions(entry_dispositions, {"appended", "created", "matched_existing"}),
        recorded_touched_memory_ids or [],
    )
    return {
        "created_memory_ids": created_ids,
        "claimed_entry_ids": claimed_ids,
        "deleted_entry_ids": deleted_ids,
        "recoverable_entry_ids": recoverable_ids,
        "released_entry_ids": released_ids,
        "meaningful_actions": meaningful_actions,
        "processed_entry_ids": recoverable_ids,
        "entry_dispositions": entry_dispositions,
        "touched_memory_ids": touched_memory_ids,
        "appended_memory_ids": _memory_ids_for_dispositions(entry_dispositions, {"appended"}),
        "matched_memory_ids": _memory_ids_for_dispositions(entry_dispositions, {"appended", "matched_existing"}),
    }


def _normalize_ingest_agentic_result(agentic_result: Any) -> dict[str, Any]:
    parsed = agentic_result.parsed if isinstance(getattr(agentic_result, "parsed", None), dict) else {}
    response_payload = parsed
    response_text = parsed.get("response")
    if not _looks_like_ingest_agentic_final_json(response_payload) and isinstance(response_text, str):
        nested = _extract_embedded_json_object(response_text)
        if isinstance(nested, dict):
            response_payload = nested
    raw_tool_stats = parsed.get("stats")
    tool_stats = raw_tool_stats if isinstance(raw_tool_stats, dict) else {}
    raw_tool_payload = tool_stats.get("tools")
    tool_payload = raw_tool_payload if isinstance(raw_tool_payload, dict) else {}
    created_memory_ids = _coerce_string_list(response_payload.get("created_memory_ids"))
    entry_outcomes = _coerce_ingest_entry_outcomes(response_payload.get("entry_outcomes"))
    cluster_outcomes = _coerce_ingest_cluster_outcomes(response_payload.get("cluster_outcomes"))
    response_has_touched_memory_ids = "touched_memory_ids" in response_payload
    response_has_matched_memory_ids = "matched_memory_ids" in response_payload
    touched_memory_ids = _coerce_string_list(response_payload.get("touched_memory_ids"))
    matched_memory_ids = _coerce_string_list(response_payload.get("matched_memory_ids"))
    if not response_has_touched_memory_ids:
        touched_memory_ids = _merge_string_lists(
            created_memory_ids,
            _memory_ids_from_entry_outcomes(entry_outcomes),
        )
    if not response_has_matched_memory_ids:
        matched_memory_ids = _memory_ids_from_entry_outcomes(entry_outcomes, {"appended", "matched_existing"})
    return {
        "summary": _coerce_text_summary(getattr(agentic_result, "summary", None)) or _coerce_text_summary(response_payload.get("summary")),
        "created_memory_ids": created_memory_ids,
        "entry_outcomes": entry_outcomes,
        "cluster_outcomes": cluster_outcomes,
        "touched_memory_ids": touched_memory_ids,
        "matched_memory_ids": matched_memory_ids,
        "meaningful_actions": _coerce_non_negative_int(response_payload.get("meaningful_actions")),
        "tool_calls_executed": _coerce_non_negative_int(tool_payload.get("totalCalls")),
        "mutations": _count_mutating_agentic_tool_calls(tool_payload.get("byName")),
        "tool_names_used": _extract_agentic_tool_names(tool_payload.get("byName")),
    }


def _reset_recorded_ingest_handled_entry_ids(ctx: ApplicationContext, task_id: str) -> None:
    if ctx.task_queue is None:
        return
    try:
        ctx.task_queue.clear_running_task_data_keys(
            task_id,
            field_names=[
                INGEST_HANDLED_ENTRY_IDS_TASK_DATA_KEY,
                INGEST_ENTRY_DISPOSITIONS_TASK_DATA_KEY,
                INGEST_TOUCHED_MEMORY_IDS_TASK_DATA_KEY,
                INGEST_TOOL_INVOCATIONS_TASK_DATA_KEY,
            ],
        )
    except ValueError:
        return


def _recorded_ingest_tool_usage(ctx: ApplicationContext, task_id: str) -> dict[str, Any]:
    if ctx.task_queue is None:
        return {
            "tool_calls_executed": 0,
            "mutations": 0,
            "tool_names_used": [],
        }
    try:
        task = ctx.task_queue.get_task(task_id)
    except ValueError:
        return {
            "tool_calls_executed": 0,
            "mutations": 0,
            "tool_names_used": [],
        }

    raw_invocations = task.data.get(INGEST_TOOL_INVOCATIONS_TASK_DATA_KEY)
    if not isinstance(raw_invocations, list):
        return {
            "tool_calls_executed": 0,
            "mutations": 0,
            "tool_names_used": [],
        }

    normalized_invocations: list[dict[str, Any]] = []
    for item in raw_invocations:
        if not isinstance(item, dict):
            continue
        tool_name = item.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            continue
        normalized_invocations.append(
            {
                "tool_name": tool_name.strip(),
                "mutation": bool(item.get("mutation")),
            }
        )

    return {
        "tool_calls_executed": len(normalized_invocations),
        "mutations": sum(1 for item in normalized_invocations if item["mutation"]),
        "tool_names_used": sorted({item["tool_name"] for item in normalized_invocations}),
    }


def _recorded_ingest_handled_entry_ids(ctx: ApplicationContext, task_id: str) -> list[int]:
    if ctx.task_queue is None:
        return []
    try:
        task = ctx.task_queue.get_task(task_id)
    except ValueError:
        return []
    raw_entry_ids = task.data.get(INGEST_HANDLED_ENTRY_IDS_TASK_DATA_KEY)
    if not isinstance(raw_entry_ids, list):
        return []
    return [
        entry_id
        for entry_id in raw_entry_ids
        if isinstance(entry_id, int) and not isinstance(entry_id, bool) and entry_id > 0
    ]


def _recorded_ingest_entry_dispositions(ctx: ApplicationContext, task_id: str) -> list[dict[str, Any]]:
    if ctx.task_queue is None:
        return []
    try:
        task = ctx.task_queue.get_task(task_id)
    except ValueError:
        return []
    raw_entry_dispositions = task.data.get(INGEST_ENTRY_DISPOSITIONS_TASK_DATA_KEY)
    if not isinstance(raw_entry_dispositions, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw_entry_dispositions:
        if not isinstance(item, dict):
            continue
        entry_id = item.get("entry_id")
        if not isinstance(entry_id, int) or isinstance(entry_id, bool) or entry_id <= 0:
            continue
        disposition = item.get("disposition")
        if not isinstance(disposition, str) or not disposition.strip():
            continue
        normalized.append(
            _build_semantic_entry_disposition(
                entry_id=entry_id,
                disposition=disposition.strip(),
                memory_id=item.get("memory_id"),
                memory_title=item.get("memory_title"),
                reason=item.get("reason"),
            )
        )
    return normalized


def _recorded_ingest_touched_memory_ids(ctx: ApplicationContext, task_id: str) -> list[str]:
    if ctx.task_queue is None:
        return []
    try:
        task = ctx.task_queue.get_task(task_id)
    except ValueError:
        return []
    raw_memory_ids = task.data.get(INGEST_TOUCHED_MEMORY_IDS_TASK_DATA_KEY)
    if not isinstance(raw_memory_ids, list):
        return []
    return sorted(
        {
            memory_id.strip()
            for item in raw_memory_ids
            for memory_id in [item if isinstance(item, str) else item.get("memory_id") if isinstance(item, dict) else None]
            if isinstance(memory_id, str) and memory_id.strip()
        }
    )


def _finalize_claimed_ingest_entries(
    ctx: ApplicationContext,
    *,
    task_id: str,
    handled_entry_ids: list[int],
) -> tuple[list[int], list[int], list[int]]:
    if ctx.journal is None:
        return [], [], []
    claimed_ids = ctx.journal.get_claimed_entry_ids(task_id)
    claimed_entry_id_set = set(claimed_ids)
    handled_claimed_entry_ids = [entry_id for entry_id in handled_entry_ids if entry_id in claimed_entry_id_set]
    recoverable_ids = ctx.journal.move_claimed_entry_ids_to_recoverable(task_id, handled_claimed_entry_ids)
    recoverable_entry_id_set = set(recoverable_ids)
    released_ids = ctx.journal.release_claimed_entry_ids(
        task_id,
        [entry_id for entry_id in claimed_ids if entry_id not in recoverable_entry_id_set],
    )
    return claimed_ids, recoverable_ids, released_ids


def _build_semantic_entry_dispositions(
    entries,
    *,
    disposition: str,
    memory_id: str | None = None,
    memory_title: str | None = None,
    reason: str | None = None,
) -> list[dict[str, Any]]:
    return [
        _build_semantic_entry_disposition(
            entry_id=entry.id,
            disposition=disposition,
            memory_id=memory_id,
            memory_title=memory_title,
            reason=reason,
        )
        for entry in entries
    ]


def _build_semantic_entry_disposition(
    *,
    entry_id: int,
    disposition: str,
    memory_id: object = None,
    memory_title: object = None,
    reason: object = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "entry_id": entry_id,
        "disposition": disposition,
    }
    if isinstance(memory_id, str) and memory_id.strip():
        payload["memory_id"] = memory_id.strip()
    if isinstance(memory_title, str) and memory_title.strip():
        payload["memory_title"] = memory_title.strip()
    if isinstance(reason, str) and reason.strip():
        payload["reason"] = reason.strip()
    return payload


def _coerce_non_negative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    return 0


def _coerce_text_summary(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _coerce_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if stripped:
            normalized.append(stripped)
    return normalized


def _looks_like_ingest_agentic_final_json(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    ingest_contract_keys = {
        "summary",
        "created_memory_ids",
        "entry_outcomes",
        "touched_memory_ids",
        "matched_memory_ids",
        "meaningful_actions",
    }
    return any(key in value for key in ingest_contract_keys)


def _coerce_ingest_entry_outcomes(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        entry_id = item.get("entry_id")
        if not isinstance(entry_id, int) or isinstance(entry_id, bool) or entry_id <= 0:
            continue
        raw_disposition = item.get("disposition", item.get("outcome"))
        if not isinstance(raw_disposition, str) or not raw_disposition.strip():
            continue
        normalized.append(
            _build_semantic_entry_disposition(
                entry_id=entry_id,
                disposition=raw_disposition.strip(),
                memory_id=item.get("memory_id"),
                memory_title=item.get("memory_title"),
                reason=item.get("reason"),
            )
        )
    return normalized


def _coerce_ingest_cluster_outcomes(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        entry_ids = [
            entry_id
            for entry_id in item.get("entry_ids", [])
            if isinstance(entry_id, int) and not isinstance(entry_id, bool) and entry_id > 0
        ]
        memory_ids = _coerce_string_list(item.get("memory_ids"))
        disposition = item.get("disposition")
        if not entry_ids or not isinstance(disposition, str) or not disposition.strip():
            continue
        payload: dict[str, Any] = {
            "entry_ids": sorted(set(entry_ids)),
            "disposition": disposition.strip(),
        }
        if memory_ids:
            payload["memory_ids"] = memory_ids
        reason = item.get("reason")
        if isinstance(reason, str) and reason.strip():
            payload["reason"] = reason.strip()
        normalized.append(payload)
    return normalized


def _memory_ids_from_entry_outcomes(
    entry_outcomes: list[dict[str, Any]],
    included_dispositions: set[str] | None = None,
) -> list[str]:
    dispositions = included_dispositions or {"created", "appended", "matched_existing"}
    return _memory_ids_for_dispositions(entry_outcomes, dispositions)


def _merge_string_lists(*values: list[str]) -> list[str]:
    return sorted({item for value in values for item in value})


def _extract_embedded_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start < 0 or json_end <= json_start:
            return None
        try:
            parsed = json.loads(text[json_start:json_end])
        except json.JSONDecodeError:
            return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _extract_agentic_tool_names(value: object) -> list[str]:
    if not isinstance(value, dict):
        return []
    return sorted(str(name) for name, payload in value.items() if isinstance(name, str) and isinstance(payload, dict))


def _count_mutating_agentic_tool_calls(value: object) -> int:
    if not isinstance(value, dict):
        return 0
    by_name_payload = cast(dict[Any, Any], value)
    read_only_tool_names = {
        "mcp_mcp-memory-internal_task_complete",
        "mcp_mcp-memory-internal_internal_get_next_ingest_batch",
        "mcp_mcp-memory-internal_internal_search_memory_records",
        "mcp_mcp-memory-internal_internal_read_memory_record",
        "mcp_mcp-memory-internal_internal_list_memory_records",
        "mcp_mcp-memory-internal_internal_task_complete",
    }
    total = 0
    for name, payload in by_name_payload.items():
        if not isinstance(name, str) or name in read_only_tool_names or not isinstance(payload, dict):
            continue
        total += _coerce_non_negative_int(payload.get("count"))
    return total


def _finalize_entry_dispositions(
    *,
    claimed_ids: list[int],
    recoverable_ids: list[int],
    released_ids: list[int],
    semantic_entry_dispositions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    recoverable_entry_id_set = set(recoverable_ids)
    released_entry_id_set = set(released_ids)
    semantic_by_entry_id: dict[int, dict[str, Any]] = {}
    for disposition in semantic_entry_dispositions:
        entry_id = disposition.get("entry_id")
        if isinstance(entry_id, int) and not isinstance(entry_id, bool) and entry_id > 0:
            semantic_by_entry_id[entry_id] = disposition

    finalized: list[dict[str, Any]] = []
    for entry_id in claimed_ids:
        finalization_status = "recoverable" if entry_id in recoverable_entry_id_set else "released"
        semantic_disposition = semantic_by_entry_id.get(entry_id)
        if semantic_disposition is None:
            finalized.append(
                {
                    "entry_id": entry_id,
                    "disposition": "released_unhandled" if entry_id in released_entry_id_set else "recoverable_unclassified",
                    "finalization_status": finalization_status,
                }
            )
            continue

        finalized_disposition = {
            **semantic_disposition,
            "finalization_status": finalization_status,
        }
        finalized.append(finalized_disposition)
    return finalized


def _memory_ids_for_dispositions(entry_dispositions: list[dict[str, Any]], included_dispositions: set[str]) -> list[str]:
    return sorted(
        {
            memory_id.strip()
            for disposition in entry_dispositions
            if disposition.get("disposition") in included_dispositions
            for memory_id in [disposition.get("memory_id")]
            if isinstance(memory_id, str) and memory_id.strip()
        }
    )
