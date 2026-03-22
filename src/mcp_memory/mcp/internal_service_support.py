from __future__ import annotations

import sqlite3
import re
from typing import Any

from mcp_memory.context import ApplicationContext


_WHITESPACE_RE = re.compile(r"\s+")
_OVERSIZED_MEMORY_WARNING_CHARS = 4_000


def _append_content(existing: str, addition: str) -> str:
    normalized_addition = addition.strip()
    if not normalized_addition or normalized_addition in existing:
        return existing
    return f"{existing.rstrip()}\n\n{normalized_addition}".strip()


def _normalize_tags(tags: list[str]) -> list[str]:
    normalized = []
    seen: set[str] = set()
    for tag in tags:
        value = tag.strip().lower().replace("_", "-")
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return sorted(normalized)


def _merge_memory_metadata(existing: dict[str, Any], override: dict[str, object] | None) -> dict[str, object] | None:
    if override is None:
        return None
    merged: dict[str, object] = dict(existing)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, list) and isinstance(value, list):
            merged[key] = _merge_metadata_lists(current, value)
        else:
            merged[key] = value
    return merged


def _merge_metadata_lists(existing: list[Any], override: list[Any]) -> list[Any]:
    int_values: set[int] = set()
    if all(isinstance(item, int) and not isinstance(item, bool) for item in [*existing, *override]):
        for item in [*existing, *override]:
            int_values.add(int(item))
        return sorted(int_values)
    str_values: set[str] = set()
    if all(isinstance(item, str) for item in [*existing, *override]):
        for item in [*existing, *override]:
            stripped = str(item).strip()
            if stripped:
                str_values.add(stripped)
        return sorted(str_values)
    return list(override)


def _enqueue_summary_task(ctx: ApplicationContext, memory_id: str, workspace_ids: list[str]) -> None:
    if ctx.task_queue is None:
        return
    from mcp_memory.core.task_handlers.constants import SUMMARIZE_MEMORY_PRIORITY, SUMMARIZE_MEMORY_TASK_NAME

    workspace_id = workspace_ids[0] if workspace_ids else ctx.workspace_id
    try:
        ctx.task_queue.enqueue(
            task_name=SUMMARIZE_MEMORY_TASK_NAME,
            task_id=f"{SUMMARIZE_MEMORY_TASK_NAME}:{memory_id}",
            workspace_id=workspace_id,
            data={"memory_id": memory_id},
            priority=SUMMARIZE_MEMORY_PRIORITY,
        )
    except sqlite3.IntegrityError:
        return


def _memory_write_quality_error(
    *,
    title: str,
    content: str,
    summary: str | None,
) -> str | None:
    normalized_title = _normalize_quality_text(title)
    normalized_summary = _normalize_quality_text(summary or "")
    normalized_content = _normalize_quality_text(content)

    title_like_completion = normalized_title.startswith("task_complete") or normalized_title.startswith("task complete")
    summary_like_completion = normalized_summary.startswith("task id:") or "completion marker" in normalized_summary
    content_has_completion_shape = (
        "task id:" in normalized_content
        and ("task name:" in normalized_content or normalized_content.startswith("task:"))
        and (
            "completion marker" in normalized_content
            or "completionrecorded" in normalized_content
            or "merged:" in normalized_content
            or "archived:" in normalized_content
            or "absorbed_observations:" in normalized_content
        )
    )

    if title_like_completion or summary_like_completion or content_has_completion_shape:
        return "routine completion/status traces must use task_complete instead of creating memories"
    return None


def _memory_write_quality_warnings(
    *,
    summary: str | None,
    memory_type: str,
    tags: list[str],
    content: str | None = None,
) -> list[str]:
    warnings: list[str] = []
    normalized_summary = _normalize_quality_text(summary or "")
    normalized_memory_type = memory_type.strip().lower()

    if normalized_summary.startswith("covers "):
        warnings.append("generic_summary")
    if normalized_memory_type == "observation" and not tags:
        warnings.append("observation_missing_tags")
    if content is not None and len(content.strip()) >= _OVERSIZED_MEMORY_WARNING_CHARS:
        warnings.append("memory_needs_split")
    return warnings


def _normalize_quality_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value.strip().lower())
