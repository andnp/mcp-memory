"""Provider-neutral memory records and repository contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from mcp_memory.core.summaries import build_deterministic_summary

VALID_MEMORY_TYPES = frozenset({"journal", "plan", "fact", "observation", "reflection"})
VALID_MEMORY_STATUSES = frozenset({"active", "stale", "degraded", "archived"})
FTS_QUERY_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_:-]+")
MEMORY_REF_PREFIX = "mem-"
_MEMORY_REF_PATTERN = re.compile(r"(?:mem-)?([1-9][0-9]*)\Z", re.IGNORECASE)
_DEFAULT_CANDIDATE_LIMIT = 50
_DEFAULT_OVERSIZED_CANDIDATE_MIN_CHARS = 3_000
_DEFAULT_THIN_CANDIDATE_MAX_CHARS = 800
_DEFAULT_LOW_SUPPORT_MAX = 1
_DEFAULT_QUALITY_OVERSIZED_MIN_CHARS = 4_000
_QUALITY_SIGNAL_ALIASES = {
    "trace_like": "trace_like_memory_count",
    "generic_summary": "generic_summary_count",
    "untagged_observation": "untagged_observation_count",
    "oversized": "oversized_memory_count",
    "raw_ingress": "raw_ingress_count",
}


@dataclass
class MemoryRecord:
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
    metadata: dict[str, object] = field(default_factory=dict)
    workspace_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    memory_ref: int | None = None


def format_memory_ref(memory_ref: int | None) -> str | None:
    if memory_ref is None:
        return None
    if memory_ref < 1:
        raise ValueError("memory_ref must be a positive integer")
    return f"{MEMORY_REF_PREFIX}{memory_ref}"


def parse_memory_ref(value: str) -> int | None:
    match = _MEMORY_REF_PATTERN.fullmatch(value.strip())
    return None if match is None else int(match.group(1))


@dataclass
class MemoryLink:
    source_id: str
    target_id: str
    link_type: str
    context: str


@dataclass
class MemoryReadContext:
    record: MemoryRecord
    relationships: dict[str, list[MemoryLink]]
    superseded: list[MemoryRecord]


@dataclass
class RankedMemoryCandidate:
    record: MemoryRecord
    incoming_links_count: int
    has_incoming_supersedes: bool = False
    incoming_link_type_counts: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class MemoryReadPort(Protocol):
    def get_search_epochs(self) -> dict[str, int]: ...

    def get_read_cache_validation_tokens(self, memory_ids: list[str]) -> dict[str, str]: ...

    def get_searchable_memories(
        self,
        memory_ids: list[str],
        *,
        status: str | None = None,
        include_superseded: bool = False,
    ) -> list[MemoryRecord]: ...

    def search_keyword_memory_ids(
        self,
        query: str,
        *,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        tags: Sequence[str] | None = None,
        include_superseded: bool = False,
        limit: int = 50,
    ) -> list[str]: ...

    def get_ranking_candidates(
        self,
        memory_ids: list[str],
        *,
        status: str | None = None,
        include_superseded: bool = False,
    ) -> list[RankedMemoryCandidate]: ...

    def get_memory(self, memory_id: str) -> MemoryRecord | None: ...

    def resolve_memory_id(self, memory_id: str) -> str | None: ...

    def list_memories(
        self,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[MemoryRecord]: ...

    def list_memory_ids(
        self,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[str]: ...

    def get_links(
        self,
        memory_id: str,
        direction: str = "outgoing",
        link_type: str | None = None,
    ) -> list[MemoryLink]: ...

    def count_incoming_links(self, memory_id: str) -> int: ...

    def touch_last_surfaced(
        self,
        memory_ids: list[str],
        surfaced_at: str,
        *,
        best_effort: bool = False,
    ) -> int: ...

    def record_access(
        self,
        memory_id: str,
        access_score: float,
        accessed_at: str,
        increment_read_count: bool = False,
    ) -> MemoryRecord | None: ...


@runtime_checkable
class MemoryMutationPort(Protocol):
    def create_memory(
        self,
        title: str,
        content: str,
        workspace_ids: list[str],
        tags: list[str] | None = None,
        summary: str | None = None,
        memory_type: str = "journal",
        status: str = "active",
        metadata: dict[str, object] | None = None,
        memory_id: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> MemoryRecord | None: ...

    def update_memory(
        self,
        memory_id: str,
        title: str | None = None,
        content: str | None = None,
        summary: str | None | object = ...,
        memory_type: str | None = None,
        status: str | None = None,
        metadata: dict[str, object] | None = None,
        workspace_ids: list[str] | None = None,
        tags: list[str] | None = None,
        access_score: float | None = None,
        last_accessed_at: str | None = None,
        last_surfaced_at: str | None = None,
    ) -> MemoryRecord | None: ...

    def delete_memory(self, memory_id: str) -> MemoryRecord | None: ...

    def append_workspace_ids(self, memory_id: str, workspace_ids: list[str]) -> MemoryRecord | None: ...

    def record_access(
        self,
        memory_id: str,
        access_score: float,
        accessed_at: str,
        increment_read_count: bool = False,
    ) -> MemoryRecord | None: ...

    def touch_last_surfaced(
        self,
        memory_ids: list[str],
        surfaced_at: str,
        *,
        best_effort: bool = False,
    ) -> int: ...


@runtime_checkable
class MemoryLinkPort(Protocol):
    def add_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str,
        context: str = "",
    ) -> MemoryLink: ...

    def remove_link(self, source_id: str, target_id: str, link_type: str) -> bool: ...

    def get_links(
        self,
        memory_id: str,
        direction: str = "outgoing",
        link_type: str | None = None,
    ) -> list[MemoryLink]: ...

    def has_incoming_link(self, memory_id: str, link_type: str) -> bool: ...

    def count_incoming_links(self, memory_id: str) -> int: ...


@runtime_checkable
class MemoryMaintenanceReadPort(Protocol):
    def peek_memory(self, memory_id: str) -> MemoryReadContext | None: ...

    def search_memories_for_maintenance(
        self,
        query: str,
        *,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        limit: int = 50,
    ) -> list[MemoryReadContext]: ...


class MemoryRepositoryPort(
    MemoryReadPort,
    MemoryMutationPort,
    MemoryLinkPort,
    MemoryMaintenanceReadPort,
    Protocol,
):
    pass


def build_read_cache_validation_token(
    *,
    record: MemoryRecord,
    outgoing_links: list[MemoryLink],
    incoming_links: list[MemoryLink],
    superseded_records: list[MemoryRecord],
) -> str:
    payload = {
        "memory_id": record.id,
        "record": {
            "status": record.status,
            "updated_at": record.updated_at,
        },
        "relationships": {
            "incoming": _serialized_link_tuples(incoming_links),
            "outgoing": _serialized_link_tuples(outgoing_links),
        },
        "superseded": [
            {
                "memory_id": superseded_record.id,
                "status": superseded_record.status,
                "updated_at": superseded_record.updated_at,
            }
            for superseded_record in sorted(
                superseded_records,
                key=lambda item: (item.id, item.updated_at, item.status),
            )
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"v1:{hashlib.sha256(encoded).hexdigest()}"


def build_memory_summary(*, title: str, content: str, memory_type: str | None = None) -> str:
    return build_deterministic_summary(title=title, content=content, memory_type=memory_type)


def _serialized_link_tuples(links: list[MemoryLink]) -> list[tuple[str, str, str, str]]:
    return sorted(
        (link.source_id, link.target_id, link.link_type, link.context)
        for link in links
    )


__all__ = [
    "FTS_QUERY_TOKEN_PATTERN",
    "MemoryLink",
    "MemoryMaintenanceReadPort",
    "MemoryMutationPort",
    "MemoryReadContext",
    "MemoryReadPort",
    "MemoryRecord",
    "MemoryRepositoryPort",
    "RankedMemoryCandidate",
    "VALID_MEMORY_STATUSES",
    "VALID_MEMORY_TYPES",
    "build_memory_summary",
    "build_read_cache_validation_token",
]
