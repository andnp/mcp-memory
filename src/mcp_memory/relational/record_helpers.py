"""Pure record normalization and hydration helpers shared by repositories."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mcp_memory.core.ports.memory import (
    VALID_MEMORY_STATUSES,
    VALID_MEMORY_TYPES,
    MemoryRecord,
)


@dataclass(frozen=True, slots=True)
class NormalizedMemoryRecord:
    id: str
    title: str
    content: str
    summary: str | None
    type: str
    status: str
    created_at: str
    updated_at: str
    read_count: int
    access_score: float
    last_accessed_at: str | None
    last_surfaced_at: str | None
    metadata: dict[str, object]
    workspace_ids: list[str]
    tags: list[str]
    memory_ref: int | None = None
    archived_at: str | None = None


def normalize_values(values: Sequence[str]) -> list[str]:
    normalized_values: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized_value = value.strip()
        if not normalized_value or normalized_value in seen:
            continue
        seen.add(normalized_value)
        normalized_values.append(normalized_value)
    return normalized_values


def validate_memory_type(memory_type: str) -> str:
    normalized_type = memory_type.strip()
    if normalized_type not in VALID_MEMORY_TYPES:
        raise ValueError(
            f"invalid memory_type: {memory_type!r}. Expected one of {sorted(VALID_MEMORY_TYPES)}"
        )
    return normalized_type


def validate_memory_status(status: str) -> str:
    normalized_status = status.strip()
    if normalized_status not in VALID_MEMORY_STATUSES:
        raise ValueError(
            f"invalid status: {status!r}. Expected one of {sorted(VALID_MEMORY_STATUSES)}"
        )
    return normalized_status


def validate_required_text(field_name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be non-empty")


def normalize_link_type(link_type: str) -> str:
    normalized_link_type = re.sub(r"[\s-]+", "_", link_type.strip()).upper()
    if not normalized_link_type:
        raise ValueError("link_type must be non-empty")
    return normalized_link_type


def serialize_metadata(metadata: Mapping[str, object] | None) -> str:
    return json.dumps(metadata or {}, sort_keys=True)


def hydrate_memory_record(values: NormalizedMemoryRecord) -> MemoryRecord:
    return MemoryRecord(
        id=values.id,
        memory_ref=values.memory_ref,
        title=values.title,
        content=values.content,
        summary=values.summary,
        type=values.type,
        status=values.status,
        created_at=values.created_at,
        updated_at=values.updated_at,
        archived_at=values.archived_at,
        read_count=values.read_count,
        access_score=values.access_score,
        last_accessed_at=values.last_accessed_at,
        last_surfaced_at=values.last_surfaced_at,
        metadata=values.metadata,
        workspace_ids=values.workspace_ids,
        tags=values.tags,
    )
