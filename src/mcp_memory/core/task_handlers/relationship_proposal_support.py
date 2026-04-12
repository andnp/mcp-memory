from __future__ import annotations

import json


def build_candidate_prompt_entries(candidates: list, *, limit: int = 20) -> str:
    return json.dumps(
        [
            {
                "id": record.id,
                "type": record.type,
                "status": record.status,
                "title": record.title,
                "summary": truncate_text(record.summary or record.content, 180),
                "tags": record.tags,
            }
            for record in candidates[:limit]
        ],
        sort_keys=True,
        ensure_ascii=False,
    )


def normalize_graph_link_proposals(proposals: object) -> list[tuple[str, str, str, str]]:
    if not isinstance(proposals, list):
        return []
    normalized: list[tuple[str, str, str, str]] = []
    for item in proposals:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id", "")).strip()
        target_id = str(item.get("target_id", "")).strip()
        link_type = str(item.get("link_type", "")).strip() or "DEPENDS_ON"
        context = str(item.get("context", "")).strip() or "Auto-linked by graph linker."
        if source_id and target_id:
            normalized.append((source_id, target_id, link_type, context))
    return normalized


def normalize_conflict_proposals(proposals: object) -> list[tuple[str, str, str]]:
    if not isinstance(proposals, list):
        return []
    normalized: list[tuple[str, str, str]] = []
    for item in proposals:
        if not isinstance(item, dict):
            continue
        left_id = str(item.get("left_id", "")).strip()
        right_id = str(item.get("right_id", "")).strip()
        context = str(item.get("context", "")).strip() or "Potential contradiction detected."
        if left_id and right_id:
            normalized.append((left_id, right_id, context))
    return normalized


def truncate_text(value: str | None, limit: int) -> str:
    text = "" if value is None else " ".join(value.strip().split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"