from __future__ import annotations

from typing import Any, cast

from mcp_memory.core.ingest_claim_lifecycle import (
    _finalize_claimed_ingest_entries as _finalize_claimed_ingest_entries,
    _recorded_ingest_entry_dispositions as _recorded_ingest_entry_dispositions,
    _recorded_ingest_handled_entry_ids as _recorded_ingest_handled_entry_ids,
    _recorded_ingest_run_metadata as _recorded_ingest_run_metadata,
    _recorded_ingest_tool_usage as _recorded_ingest_tool_usage,
    _recorded_ingest_touched_memory_ids as _recorded_ingest_touched_memory_ids,
    _reset_recorded_ingest_handled_entry_ids as _reset_recorded_ingest_handled_entry_ids,
)
from mcp_memory.core.task_handlers.agentic_result_support import (
    build_tool_usage_summary,
    coerce_non_negative_int,
    coerce_text_summary,
    count_mutating_agentic_tool_calls,
    extract_agentic_tool_names,
    extract_embedded_json_object,
)


_INGEST_READ_ONLY_TOOL_NAMES = {
    "mcp_mcp-memory-internal_task_complete",
    "mcp_mcp-memory-internal_internal_get_next_ingest_batch",
    "mcp_mcp-memory-internal_internal_search_memory_records",
    "mcp_mcp-memory-internal_internal_read_memory_record",
    "mcp_mcp-memory-internal_internal_list_memory_records",
    "mcp_mcp-memory-internal_internal_task_complete",
}


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
    } | build_tool_usage_summary(parsed, read_only_tool_names=_INGEST_READ_ONLY_TOOL_NAMES)


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
    return coerce_non_negative_int(value)


def _coerce_text_summary(value: object) -> str | None:
    return coerce_text_summary(value)


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
        item_dict = cast(dict[str, Any], item)
        entry_id = item_dict.get("entry_id")
        if not isinstance(entry_id, int) or isinstance(entry_id, bool) or entry_id <= 0:
            continue
        raw_disposition = item_dict.get("disposition", item_dict.get("outcome"))
        if not isinstance(raw_disposition, str) or not raw_disposition.strip():
            continue
        normalized.append(
            _build_semantic_entry_disposition(
                entry_id=entry_id,
                disposition=raw_disposition.strip(),
                memory_id=item_dict.get("memory_id"),
                memory_title=item_dict.get("memory_title"),
                reason=item_dict.get("reason"),
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
        item_dict = cast(dict[str, Any], item)
        entry_ids = [
            entry_id
            for entry_id in item_dict.get("entry_ids", [])
            if isinstance(entry_id, int) and not isinstance(entry_id, bool) and entry_id > 0
        ]
        memory_ids = _coerce_string_list(item_dict.get("memory_ids"))
        disposition = item_dict.get("disposition")
        if not entry_ids or not isinstance(disposition, str) or not disposition.strip():
            continue
        payload: dict[str, Any] = {
            "entry_ids": sorted(set(entry_ids)),
            "disposition": disposition.strip(),
        }
        if memory_ids:
            payload["memory_ids"] = memory_ids
        reason = item_dict.get("reason")
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
    return extract_embedded_json_object(text)


def _extract_agentic_tool_names(value: object) -> list[str]:
    return extract_agentic_tool_names(value)


def _count_mutating_agentic_tool_calls(value: object) -> int:
    return count_mutating_agentic_tool_calls(value, read_only_tool_names=_INGEST_READ_ONLY_TOOL_NAMES)


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
