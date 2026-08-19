from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypeAlias
from uuid import UUID

import pytest
from searchkernel.utils.similarity import cosine_similarity_lists

from mcp_memory.config import Config, SearchKernelConfig
from mcp_memory.core.ports.memory import (
    MemoryLinkPort,
    MemoryMaintenanceReadPort,
    MemoryMutationPort,
    MemoryReadPort,
)
from mcp_memory.embeddings import EmbeddingRecord
from mcp_memory.relational.search import RelationalMemorySearchService
from mcp_memory.storage.postgres_repository import PostgresRelationalMemoryRepository
from tests.small.maintenance_candidate_query_contract import (
    assert_maintenance_candidate_query_contract,
)
from tests.small.maintenance_read_repository_contract import assert_maintenance_read_preserves_telemetry

pytestmark = pytest.mark.small


SqlParams: TypeAlias = tuple[object, ...]


@dataclass
class FakePostgresState:
    memories: dict[str, dict[str, object]] = field(default_factory=dict)
    memory_workspaces: dict[str, set[str]] = field(default_factory=dict)
    tags: dict[int, str] = field(default_factory=dict)
    tag_ids_by_name: dict[str, int] = field(default_factory=dict)
    memory_tags: dict[str, set[int]] = field(default_factory=dict)
    links: dict[tuple[str, str, str], str] = field(default_factory=dict)
    memory_search_documents: set[str] = field(default_factory=set)
    query_log: list[str] = field(default_factory=list)
    next_tag_id: int = 1
    next_memory_ref: int = 1


class FakeCursor:
    def __init__(self, state: FakePostgresState) -> None:
        self._state = state
        self._result: list[tuple[object, ...]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        normalized = " ".join(query.split())
        arguments: SqlParams = tuple(() if params is None else params)
        self._state.query_log.append(normalized)

        if normalized.startswith("INSERT INTO memories ("):
            self._insert_memory(arguments)
        elif normalized.startswith(
            "SELECT memories.id, memories.title, memories.content, memories.summary, memories.type, memories.status, memories.created_at, memories.updated_at, memories.read_count, memories.access_score, memories.last_accessed_at, memories.last_surfaced_at, memories.metadata, memories.memory_ref,"
        ) and "incoming_counts" in normalized:
            self._select_maintenance_candidates(normalized, arguments)
        elif normalized.startswith(
            "SELECT memories.id, memories.title, memories.content, memories.summary, memories.type, memories.status, memories.created_at, memories.updated_at, memories.read_count, memories.access_score, memories.last_accessed_at, memories.last_surfaced_at, memories.metadata, memories.memory_ref,"
        ):
            self._select_searchable_memories(normalized, arguments)
        elif normalized.startswith(
            "SELECT id, title, content, summary, type, status, created_at, updated_at, read_count, access_score, last_accessed_at, last_surfaced_at, metadata, memory_ref, archived_at FROM memories WHERE id = ANY("
        ):
            self._select_memories_by_ids(arguments)
        elif normalized.startswith("SELECT id, title, content, summary, type, status, created_at, updated_at,") and "FROM memories WHERE id = %s" in normalized:
            self._select_memory(arguments)
        elif normalized == "SELECT id FROM memories WHERE memory_ref = %s":
            memory_ref = self._as_int(arguments[0])
            memory = next(
                (memory for memory in self._state.memories.values() if memory["memory_ref"] == memory_ref),
                None,
            )
            self._result = [] if memory is None else [(memory["id"],)]
        elif normalized.startswith("UPDATE memories SET"):
            self._update_memory(normalized, arguments)
        elif normalized == "DELETE FROM memory_workspaces WHERE memory_id = %s":
            self._state.memory_workspaces[str(arguments[0])] = set()
            self._result = []
        elif normalized == "INSERT INTO memory_workspaces (memory_id, workspace_id) VALUES (%s, %s)":
            memory_id, workspace_id = str(arguments[0]), str(arguments[1])
            self._state.memory_workspaces.setdefault(memory_id, set()).add(workspace_id)
            self._result = []
        elif normalized == "DELETE FROM memory_tags WHERE memory_id = %s":
            self._state.memory_tags[str(arguments[0])] = set()
            self._result = []
        elif normalized == "INSERT INTO tags (name) VALUES (%s) ON CONFLICT (name) DO NOTHING":
            tag_name = str(arguments[0])
            if tag_name not in self._state.tag_ids_by_name:
                tag_id = self._state.next_tag_id
                self._state.next_tag_id += 1
                self._state.tag_ids_by_name[tag_name] = tag_id
                self._state.tags[tag_id] = tag_name
            self._result = []
        elif normalized == "SELECT id FROM tags WHERE name = %s":
            tag_name = str(arguments[0])
            tag_id = self._state.tag_ids_by_name.get(tag_name)
            self._result = [] if tag_id is None else [(tag_id,)]
        elif normalized == "INSERT INTO memory_tags (memory_id, tag_id) VALUES (%s, %s) ON CONFLICT DO NOTHING":
            memory_id, tag_id = str(arguments[0]), self._as_int(arguments[1])
            self._state.memory_tags.setdefault(memory_id, set()).add(tag_id)
            self._result = []
        elif normalized == "SELECT workspace_id FROM memory_workspaces WHERE memory_id = %s ORDER BY workspace_id ASC":
            workspaces = sorted(self._state.memory_workspaces.get(str(arguments[0]), set()))
            self._result = [(workspace_id,) for workspace_id in workspaces]
        elif normalized == "SELECT memory_id, workspace_id FROM memory_workspaces WHERE memory_id = ANY(%s::text[]) ORDER BY memory_id ASC, workspace_id ASC":
            memory_ids = self._as_str_sequence(arguments[0])
            rows: list[tuple[object, ...]] = []
            for memory_id in sorted(memory_ids):
                for workspace_id in sorted(self._state.memory_workspaces.get(memory_id, set())):
                    rows.append((memory_id, workspace_id))
            self._result = rows
        elif normalized.startswith("SELECT tags.name FROM tags JOIN memory_tags ON memory_tags.tag_id = tags.id"):
            memory_id = str(arguments[0])
            tag_names = sorted(
                self._state.tags[tag_id]
                for tag_id in self._state.memory_tags.get(memory_id, set())
                if tag_id in self._state.tags
            )
            self._result = [(tag_name,) for tag_name in tag_names]
        elif normalized == "SELECT memory_tags.memory_id, tags.name FROM memory_tags JOIN tags ON tags.id = memory_tags.tag_id WHERE memory_tags.memory_id = ANY(%s::text[]) ORDER BY memory_tags.memory_id ASC, tags.name ASC":
            memory_ids = self._as_str_sequence(arguments[0])
            rows = []
            for memory_id in sorted(memory_ids):
                tag_names = sorted(
                    self._state.tags[tag_id]
                    for tag_id in self._state.memory_tags.get(memory_id, set())
                    if tag_id in self._state.tags
                )
                rows.extend((memory_id, tag_name) for tag_name in tag_names)
            self._result = rows
        elif normalized.startswith("SELECT memories.id FROM memories"):
            self._select_memory_ids(normalized, arguments)
        elif normalized.startswith("SELECT id, title, content, summary, type, status, created_at, updated_at,"):
            self._select_memories(normalized, arguments)
        elif normalized.startswith("SELECT DISTINCT memories.id,"):
            self._search_keyword_memory_ids(normalized, arguments)
        elif normalized.startswith("WITH input_ids AS (") or normalized.startswith(
            "WITH input_ids AS ( SELECT memory_id, ordinality FROM unnest(%s::text[]) WITH ORDINALITY AS requested(memory_id, ordinality) ), workspace_agg AS ( SELECT memory_workspaces.memory_id, array_agg(DISTINCT memory_workspaces.workspace_id ORDER BY memory_workspaces.workspace_id) AS workspace_ids FROM memory_workspaces JOIN input_ids ON input_ids.memory_id = memory_workspaces.memory_id GROUP BY memory_workspaces.memory_id ), tag_agg AS ( SELECT memory_tags.memory_id, array_agg(DISTINCT tags.name ORDER BY tags.name) AS tags FROM memory_tags JOIN tags ON tags.id = memory_tags.tag_id JOIN input_ids ON input_ids.memory_id = memory_tags.memory_id GROUP BY memory_tags.memory_id ), link_counts AS ( SELECT links.target_id AS memory_id, COUNT(*) AS incoming_links_count, MAX(CASE WHEN links.type = 'SUPERSEDES' THEN 1 ELSE 0 END) AS has_incoming_supersedes, SUM(CASE WHEN links.type = 'DEPENDS_ON' THEN 1 ELSE 0 END) AS incoming_depends_on_count, SUM(CASE WHEN links.type = 'AMENDS' THEN 1 ELSE 0 END) AS incoming_amends_count, SUM(CASE WHEN links.type = 'CONTRADICTS' THEN 1 ELSE 0 END) AS incoming_contradicts_count, SUM(CASE WHEN links.type = 'SUPERSEDES' THEN 1 ELSE 0 END) AS incoming_supersedes_count FROM links JOIN input_ids ON input_ids.memory_id = links.target_id GROUP BY links.target_id ) SELECT memories.id, memories.title, memories.content, memories.summary, memories.type, memories.status, memories.created_at, memories.updated_at, memories.read_count, memories.access_score, memories.last_accessed_at, memories.last_surfaced_at, memories.metadata, COALESCE(workspace_agg.workspace_ids, ARRAY[]::text[]) AS workspace_ids, COALESCE(tag_agg.tags, ARRAY[]::text[]) AS tags, COALESCE(link_counts.incoming_links_count, 0) AS incoming_links_count, COALESCE(link_counts.has_incoming_supersedes, 0) AS has_incoming_supersedes, COALESCE(link_counts.incoming_depends_on_count, 0) AS incoming_depends_on_count, COALESCE(link_counts.incoming_amends_count, 0) AS incoming_amends_count, COALESCE(link_counts.incoming_contradicts_count, 0) AS incoming_contradicts_count, COALESCE(link_counts.incoming_supersedes_count, 0) AS incoming_supersedes_count FROM input_ids JOIN memories ON memories.id = input_ids.memory_id LEFT JOIN workspace_agg ON workspace_agg.memory_id = memories.id LEFT JOIN tag_agg ON tag_agg.memory_id = memories.id LEFT JOIN link_counts ON link_counts.memory_id = memories.id"
        ):
            self._select_ranking_candidates(normalized, arguments)
        elif normalized.startswith("INSERT INTO memory_search_documents ("):
            self._state.memory_search_documents.add(str(arguments[0]))
            self._result = []
        elif normalized == "DELETE FROM memory_search_documents WHERE memory_id = %s":
            self._state.memory_search_documents.discard(str(arguments[0]))
            self._result = []
        elif normalized.startswith("INSERT INTO links (source_id, target_id, type, context)"):
            source_id, target_id, link_type, context = map(str, arguments)
            self._state.links[(source_id, target_id, link_type)] = context
            self._result = []
        elif normalized.startswith("SELECT source_id, target_id, type, context FROM links WHERE"):
            self._select_links(normalized, arguments)
        elif normalized == "DELETE FROM links WHERE source_id = %s AND target_id = %s AND type = %s":
            self._state.links.pop((str(arguments[0]), str(arguments[1]), str(arguments[2])), None)
            self._result = []
        elif normalized == "SELECT 1 FROM links WHERE target_id = %s AND type = %s LIMIT 1":
            target_id, link_type = str(arguments[0]), str(arguments[1])
            exists = any(key[1] == target_id and key[2] == link_type for key in self._state.links)
            self._result = [(1,)] if exists else []
        elif normalized == "SELECT COUNT(*) FROM links WHERE target_id = %s":
            target_id = str(arguments[0])
            count = sum(1 for source_id, linked_target_id, _link_type in self._state.links if linked_target_id == target_id)
            self._result = [(count,)]
        elif normalized == "DELETE FROM links WHERE source_id = %s OR target_id = %s":
            doomed = [key for key in self._state.links if key[0] == str(arguments[0]) or key[1] == str(arguments[1])]
            for key in doomed:
                self._state.links.pop(key, None)
            self._result = []
        elif normalized == "DELETE FROM memories WHERE id = %s":
            memory_id = str(arguments[0])
            self._state.memories.pop(memory_id, None)
            self._state.memory_workspaces.pop(memory_id, None)
            self._state.memory_tags.pop(memory_id, None)
            self._result = []
        else:
            raise AssertionError(f"Unhandled query: {normalized}")

    def fetchone(self) -> tuple[object, ...] | None:
        return None if not self._result else self._result[0]

    def fetchall(self) -> list[tuple[object, ...]]:
        return list(self._result)

    def executemany(self, query: str, rows: Sequence[tuple[object, ...]]) -> None:
        for row in rows:
            self.execute(query, tuple(row))

    def _insert_memory(self, arguments: SqlParams) -> None:
        memory_id = str(arguments[0])
        self._state.memories[memory_id] = {
            "id": memory_id,
            "title": str(arguments[1]),
            "content": str(arguments[2]),
            "summary": None if arguments[3] is None else str(arguments[3]),
            "type": str(arguments[4]),
            "status": str(arguments[5]),
            "created_at": str(arguments[6]),
            "updated_at": str(arguments[7]),
            "read_count": self._as_int(arguments[8]),
            "access_score": self._as_float(arguments[9]),
            "last_accessed_at": arguments[10],
            "last_surfaced_at": arguments[11],
            "metadata": str(arguments[12]),
            "archived_at": arguments[13],
            "memory_ref": self._state.next_memory_ref,
        }
        self._state.next_memory_ref += 1
        self._result = []

    def _select_memory(self, arguments: SqlParams) -> None:
        memory = self._state.memories.get(str(arguments[0]))
        if memory is None:
            self._result = []
            return
        self._result = [self._memory_row(memory)]

    def _select_memories_by_ids(self, arguments: SqlParams) -> None:
        memory_ids = self._as_str_sequence(arguments[0])
        status = str(arguments[1]) if len(arguments) > 1 else None
        rows = []
        for memory_id in memory_ids:
            memory = self._state.memories.get(memory_id)
            if memory is None:
                continue
            if status is not None and str(memory["status"]) != status:
                continue
            rows.append(self._memory_row(memory))
        self._result = rows

    def _select_searchable_memories(self, normalized: str, arguments: SqlParams) -> None:
        memory_ids = self._as_str_sequence(arguments[0])
        status = str(arguments[1]) if "memories.status = %s" in normalized else None
        include_superseded = "NOT EXISTS (SELECT 1 FROM links supersedes" not in normalized
        rows: list[tuple[object, ...]] = []
        for memory_id in memory_ids:
            memory = self._state.memories.get(memory_id)
            if memory is None:
                continue
            if status is not None and str(memory["status"]) != status:
                continue
            if not include_superseded and any(
                target_id == memory_id and link_type == "SUPERSEDES"
                for _source_id, target_id, link_type in self._state.links
            ):
                continue
            workspace_ids = sorted(self._state.memory_workspaces.get(memory_id, set()))
            tags = sorted(
                self._state.tags[tag_id]
                for tag_id in self._state.memory_tags.get(memory_id, set())
                if tag_id in self._state.tags
            )
            rows.append((*self._memory_row(memory), workspace_ids, tags))
        self._result = rows

    def _select_memories(self, normalized: str, arguments: SqlParams) -> None:
        params = list(arguments)
        limit = self._as_int(params.pop())
        workspace_id: str | None = None
        memory_type: str | None = None
        status: str | None = None

        if "memory_workspaces.workspace_id = %s" in normalized:
            workspace_id = str(params.pop(0))
        if "memories.type = %s" in normalized:
            memory_type = str(params.pop(0))
        if "memories.status = %s" in normalized:
            status = str(params.pop(0))

        rows = []
        for memory in self._sorted_memories():
            memory_id = str(memory["id"])
            if workspace_id is not None and workspace_id not in self._state.memory_workspaces.get(memory_id, set()):
                continue
            if memory_type is not None and str(memory["type"]) != memory_type:
                continue
            if status is not None and str(memory["status"]) != status:
                continue
            rows.append(self._memory_row(memory))
            if len(rows) >= limit:
                break
        self._result = rows

    def _select_memory_ids(self, normalized: str, arguments: SqlParams) -> None:
        params = list(arguments)
        limit = self._as_int(params.pop())
        workspace_id: str | None = None
        memory_type: str | None = None
        status: str | None = None

        if "memory_workspaces.workspace_id = %s" in normalized:
            workspace_id = str(params.pop(0))
        if "memories.type = %s" in normalized:
            memory_type = str(params.pop(0))
        if "memories.status = %s" in normalized:
            status = str(params.pop(0))

        rows: list[tuple[object, ...]] = []
        for memory in self._sorted_memories():
            memory_id = str(memory["id"])
            if workspace_id is not None and workspace_id not in self._state.memory_workspaces.get(memory_id, set()):
                continue
            if memory_type is not None and str(memory["type"]) != memory_type:
                continue
            if status is not None and str(memory["status"]) != status:
                continue
            rows.append((memory_id,))
            if len(rows) >= limit:
                break
        self._result = rows

    def _update_memory(self, normalized: str, arguments: SqlParams) -> None:
        if normalized == "UPDATE memories SET last_surfaced_at = %s WHERE id = ANY(%s::text[])":
            surfaced_at = arguments[0]
            memory_ids = self._as_str_sequence(arguments[1])
            for memory_id in memory_ids:
                self._state.memories[memory_id]["last_surfaced_at"] = surfaced_at
            self._result = []
            return
        assignments, _where_clause = normalized.split(" WHERE id = %s", 1)
        memory_id = str(arguments[-1])
        memory = self._state.memories[memory_id]
        column_tokens = [segment.strip() for segment in assignments.removeprefix("UPDATE memories SET ").split(",")]
        values = list(arguments[:-1])
        for column_token, value in zip(column_tokens, values, strict=False):
            column_name = column_token.split(" = %s", 1)[0].split(" = %s::jsonb", 1)[0].strip()
            if column_name == "metadata":
                memory[column_name] = str(value)
            else:
                memory[column_name] = value
        self._result = []

    def _search_keyword_memory_ids(self, normalized: str, arguments: SqlParams) -> None:
        raw_tokens = arguments[0]
        if not isinstance(raw_tokens, list | tuple):
            raise TypeError("expected token sequence")
        tokens = [str(token).lower() for token in raw_tokens]
        limit = self._as_int(arguments[-1])
        workspace_id: str | None = None
        memory_type: str | None = None
        status: str | None = None
        include_superseded = "NOT EXISTS (SELECT 1 FROM links supersedes" not in normalized
        argument_index = 2
        if "JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id" in normalized:
            workspace_id = str(arguments[argument_index])
            argument_index += 1
        if "memories.type = %s" in normalized:
            memory_type = str(arguments[argument_index])
            argument_index += 1
        if "memories.status = %s" in normalized:
            status = str(arguments[argument_index])

        ranked_rows: list[tuple[str, float, str]] = []
        for memory in self._state.memories.values():
            memory_id = str(memory["id"])
            if self._state.memory_search_documents and memory_id not in self._state.memory_search_documents:
                continue
            if workspace_id is not None and workspace_id not in self._state.memory_workspaces.get(memory_id, set()):
                continue
            if memory_type is not None and str(memory["type"]) != memory_type:
                continue
            if status is not None and str(memory["status"]) != status:
                continue
            if not include_superseded and any(
                target_id == memory_id and link_type == "SUPERSEDES"
                for _source_id, target_id, link_type in self._state.links
            ):
                continue
            score = self._keyword_score(memory_id, memory, tokens)
            if score <= 0.0:
                continue
            ranked_rows.append((memory_id, score, str(memory["updated_at"])))

        ranked_rows.sort(key=lambda item: (item[1], item[2], item[0]), reverse=True)
        self._result = [(memory_id, score) for memory_id, score, _updated_at in ranked_rows[:limit]]

    def _select_links(self, normalized: str, arguments: SqlParams) -> None:
        direction_key = "target_id" if "WHERE target_id" in normalized else "source_id"
        if " = ANY(%s::text[])" in normalized:
            memory_ids = set(self._as_str_sequence(arguments[0]))
        else:
            memory_ids = {str(arguments[0])}
        link_type = str(arguments[1]) if len(arguments) > 1 else None
        rows: list[tuple[object, ...]] = []
        for source_id, target_id, stored_link_type in sorted(self._state.links):
            matched_memory_id = source_id if direction_key == "source_id" else target_id
            if matched_memory_id not in memory_ids:
                continue
            if link_type is not None and stored_link_type != link_type:
                continue
            rows.append((source_id, target_id, stored_link_type, self._state.links[(source_id, target_id, stored_link_type)]))
        self._result = rows

    def _select_ranking_candidates(self, normalized: str, arguments: SqlParams) -> None:
        memory_ids = self._as_str_sequence(arguments[0])
        argument_index = 1
        status = None
        if "WHERE memories.status = %s" in normalized or "AND memories.status = %s" in normalized:
            status = str(arguments[argument_index])
            argument_index += 1
        include_superseded = "COALESCE(link_counts.has_incoming_supersedes, 0) = 0" not in normalized

        rows: list[tuple[object, ...]] = []
        for memory_id in memory_ids:
            memory = self._state.memories.get(memory_id)
            if memory is None:
                continue
            if status is not None and str(memory["status"]) != status:
                continue
            counts = {
                "DEPENDS_ON": 0,
                "AMENDS": 0,
                "CONTRADICTS": 0,
                "SUPERSEDES": 0,
            }
            total = 0
            for _source_id, target_id, link_type in self._state.links:
                if target_id != memory_id:
                    continue
                total += 1
                counts[link_type] = counts.get(link_type, 0) + 1
            has_incoming_supersedes = counts["SUPERSEDES"] > 0
            if has_incoming_supersedes and not include_superseded:
                continue
            workspace_ids = sorted(self._state.memory_workspaces.get(memory_id, set()))
            tag_names = sorted(
                self._state.tags[tag_id]
                for tag_id in self._state.memory_tags.get(memory_id, set())
                if tag_id in self._state.tags
            )
            rows.append(
                (*self._memory_row(memory), workspace_ids, tag_names,
                    total,
                    int(has_incoming_supersedes),
                    counts["DEPENDS_ON"],
                    counts["AMENDS"],
                    counts["CONTRADICTS"],
                    counts["SUPERSEDES"],
                )
            )
        self._result = rows

    def _select_maintenance_candidates(self, normalized: str, arguments: SqlParams) -> None:
        params = list(arguments)
        limit = self._as_int(params.pop())
        workspace_id: str | None = None
        status: str | None = None
        if "scoped_workspace.workspace_id = %s" in normalized:
            workspace_id = str(params.pop(0))
        if "memories.status = %s" in normalized:
            status = str(params.pop(0))

        seed = ""
        oversized_min_chars = 0
        thin_max_chars = 0
        low_support_max = 0
        quality_min_chars: int | None = None
        if "md5(CONCAT" in normalized:
            strategy = "seeded-random"
            seed = str(params.pop(0))
        elif "LENGTH(COALESCE(memories.content, '')) > %s" in normalized:
            strategy = "oversized/thin"
            oversized_min_chars = self._as_int(params.pop(0))
            thin_max_chars = self._as_int(params.pop(0))
        elif "COALESCE(incoming_counts.incoming_links_count, 0) <= %s" in normalized:
            strategy = "orphan/low-support"
            low_support_max = self._as_int(params.pop(0))
        elif (
            "LOWER(TRIM" in normalized
            or "memories.type = 'observation' AND NOT EXISTS" in normalized
            or "LENGTH(COALESCE(memories.content, '')) >= %s" in normalized
        ):
            strategy = "quality-signal"
            if "LOWER(TRIM(memories.title)) LIKE %s" in normalized:
                for _ in range(3):
                    params.pop(0)
            if "LOWER(TRIM(COALESCE(memories.summary" in normalized:
                for _ in range(2):
                    params.pop(0)
            quality_min_chars = self._as_int(params.pop(0)) if ">= %s" in normalized else None
        elif "memories.last_surfaced_at IS NULL" in normalized:
            strategy = "never-surfaced"
        else:
            strategy = "cold-storage"

        rows: list[tuple[Any, dict[str, object]]] = []
        for memory in self._state.memories.values():
            memory_id = str(memory["id"])
            if workspace_id is not None and workspace_id not in self._state.memory_workspaces.get(memory_id, set()):
                continue
            if status is not None and str(memory["status"]) != status:
                continue
            if (
                "NOT EXISTS (SELECT 1 FROM links supersedes" in normalized
                and any(target_id == memory_id and link_type == "SUPERSEDES" for _, target_id, link_type in self._state.links)
            ):
                continue
            content_length = len(str(memory["content"]))
            metadata = json.loads(str(memory["metadata"]))
            if strategy == "never-surfaced" and memory["last_surfaced_at"] is not None:
                continue
            if strategy == "oversized/thin" and not (
                content_length > oversized_min_chars
                or (content_length <= thin_max_chars and "split_from_memory_id" in metadata)
            ):
                continue
            incoming_count = sum(1 for _, target_id, _ in self._state.links if target_id == memory_id)
            if strategy == "orphan/low-support" and incoming_count > low_support_max:
                continue
            if strategy == "quality-signal":
                title = str(memory["title"]).strip().lower()
                summary = str(memory["summary"] or "").strip().lower()
                signal_matches = (
                    title.startswith("task_complete")
                    or title.startswith("task complete")
                    or title.startswith("task_complete_record")
                    or summary.startswith("covers ")
                    or (
                        str(memory["type"]) == "observation"
                        and not self._state.memory_tags.get(memory_id)
                    )
                    or (quality_min_chars is not None and content_length >= quality_min_chars)
                )
                all_quality_signals = all(
                    marker in normalized
                    for marker in (
                        "LOWER(TRIM(memories.title)) LIKE %s",
                        "LOWER(TRIM(COALESCE(memories.summary",
                        "memories.type = 'observation'",
                        ">= %s",
                    )
                )
                if (
                    not all_quality_signals
                    and "LOWER(TRIM(memories.title)) LIKE %s" in normalized
                    and "LOWER(TRIM(COALESCE(memories.summary" not in normalized
                ):
                    signal_matches = title.startswith("task_complete")
                elif (
                    not all_quality_signals
                    and "LOWER(TRIM(COALESCE(memories.summary" in normalized
                    and "LOWER(TRIM(memories.title)) LIKE" not in normalized
                ):
                    signal_matches = summary.startswith("covers ")
                elif (
                    not all_quality_signals
                    and "memories.type = 'observation'" in normalized
                    and ">= %s" not in normalized
                ):
                    signal_matches = (
                        str(memory["type"]) == "observation"
                        and not self._state.memory_tags.get(memory_id)
                    )
                elif not all_quality_signals and ">= %s" in normalized:
                    signal_matches = quality_min_chars is not None and content_length >= quality_min_chars
                if not signal_matches:
                    continue

            if strategy == "seeded-random":
                sort_key = (hashlib.md5(f"{seed}:{memory_id}".encode()).hexdigest(), memory_id)
            elif strategy == "cold-storage":
                last_accessed = memory["last_accessed_at"]
                sort_key = (
                    0 if last_accessed is None else 1,
                    "" if last_accessed is None else str(last_accessed),
                    str(memory["updated_at"]),
                    str(memory["created_at"]),
                    memory_id,
                )
            elif strategy == "oversized/thin":
                sort_key = (-content_length, self._as_int(memory["read_count"]), str(memory["updated_at"]), memory_id)
            elif strategy == "orphan/low-support":
                last_surfaced = memory["last_surfaced_at"]
                sort_key = (
                    incoming_count,
                    self._as_int(memory["read_count"]),
                    self._as_float(memory["access_score"]),
                    0 if last_surfaced is None else 1,
                    "" if last_surfaced is None else str(last_surfaced),
                    memory_id,
                )
            else:
                sort_key = (str(memory["updated_at"]), str(memory["created_at"]), memory_id)
            rows.append((sort_key, memory))

        rows.sort(key=lambda row: row[0])
        self._result = [
            (*self._memory_row(memory),
             sorted(self._state.memory_workspaces.get(str(memory["id"]), set())),
             sorted(
                 self._state.tags[tag_id]
                 for tag_id in self._state.memory_tags.get(str(memory["id"]), set())
                 if tag_id in self._state.tags
             ))
            for _, memory in rows[:limit]
        ]

    def _memory_row(self, memory: dict[str, object]) -> tuple[object, ...]:
        return (
            memory["id"],
            memory["title"],
            memory["content"],
            memory["summary"],
            memory["type"],
            memory["status"],
            memory["created_at"],
            memory["updated_at"],
            memory["read_count"],
            memory["access_score"],
            memory["last_accessed_at"],
            memory["last_surfaced_at"],
            memory["metadata"],
            memory["memory_ref"],
            memory["archived_at"],
        )

    def _sorted_memories(self) -> Iterable[dict[str, object]]:
        return sorted(
            self._state.memories.values(),
            key=lambda memory: (str(memory["updated_at"]), str(memory["created_at"]), str(memory["id"])),
            reverse=True,
        )

    def _keyword_score(self, memory_id: str, memory: dict[str, object], tokens: list[str]) -> float:
        tag_names = [
            self._state.tags[tag_id]
            for tag_id in self._state.memory_tags.get(memory_id, set())
            if tag_id in self._state.tags
        ]
        searchable_fields = [
            (str(memory["title"]).lower(), 3.0),
            (str(memory.get("summary") or "").lower(), 2.0),
            (" ".join(sorted(tag_names)).lower(), 1.5),
            (str(memory["content"]).lower(), 1.0),
        ]
        total = 0.0
        for token in tokens:
            for haystack, weight in searchable_fields:
                if token and token in haystack:
                    total += weight
        return total

    def _as_int(self, value: object) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return int(value)
        raise TypeError("expected integer-compatible value")

    def _as_float(self, value: object) -> float:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            return float(value)
        raise TypeError("expected float-compatible value")

    def _as_str_sequence(self, value: object) -> list[str]:
        if isinstance(value, list | tuple):
            return [str(item) for item in value]
        raise TypeError("expected string sequence")


class FakeConnection:
    def __init__(self, state: FakePostgresState) -> None:
        self._state = state
        self.commit_count = 0
        self.rollback_count = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self._state)

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1


class FakeLease:
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection

    def __enter__(self) -> FakeConnection:
        return self._connection

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self._connection.rollback()
        return False

    def close(self) -> None:
        return None


class FakeSessionManager:
    def __init__(self, state: FakePostgresState) -> None:
        self._state = state
        self.connections: list[FakeConnection] = []

    def __enter__(self) -> FakeSessionManager:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def open_connection(self) -> FakeLease:
        connection = FakeConnection(self._state)
        self.connections.append(connection)
        return FakeLease(connection)

    def close(self) -> None:
        return None


@pytest.fixture
def postgres_repository() -> Iterator[tuple[PostgresRelationalMemoryRepository, FakeSessionManager]]:
    session_manager = FakeSessionManager(FakePostgresState())
    repository = PostgresRelationalMemoryRepository(session_manager)
    try:
        yield repository, session_manager
    finally:
        repository.close()


def test_postgres_repository_implements_memory_ports(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, _ = postgres_repository

    assert isinstance(repository, MemoryReadPort)
    assert isinstance(repository, MemoryMutationPort)
    assert isinstance(repository, MemoryLinkPort)
    assert isinstance(repository, MemoryMaintenanceReadPort)


def test_postgres_repository_create_read_update_and_list_memory(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, session_manager = postgres_repository

    created = repository.create_memory(
        title="Epic 01 bootstrap",
        content="Add the first relational schema slice.",
        summary="Tracks the first relational bootstrap step.",
        memory_type="plan",
        workspace_ids=["workspace-a", "workspace-a", "workspace-b"],
        tags=["sqlite", "testing", "sqlite"],
        metadata={"priority": "high"},
    )
    secondary = repository.create_memory(
        title="Standalone fact",
        content="Second record for list filtering.",
        memory_type="fact",
        workspace_ids=["workspace-c"],
        tags=["facts"],
    )
    archived = repository.create_memory(
        title="Archived observation",
        content="An archived record keeps its archive timestamp.",
        workspace_ids=["workspace-archive"],
        tags=["retention"],
        status="archived",
    )

    assert created is not None and secondary is not None and archived is not None

    UUID(created.id)
    assert created.type == "plan"
    assert created.workspace_ids == ["workspace-a", "workspace-b"]
    assert created.tags == ["sqlite", "testing"]
    assert created.metadata == {"priority": "high"}
    assert secondary.summary == "Second record for list filtering."
    assert archived.status == "archived"
    assert archived.archived_at is not None

    fetched = repository.get_memory(created.id)
    assert fetched is not None
    assert fetched.summary == "Tracks the first relational bootstrap step."

    updated = repository.update_memory(
        created.id,
        title="Epic 01 relational bootstrap",
        content="Add schema and a repository slice.",
        summary="Updated after wiring the repository.",
        status="stale",
        metadata={"priority": "medium", "phase": 1},
        workspace_ids=["workspace-b"],
        tags=["repository", "sqlite"],
        access_score=2.5,
        last_accessed_at="2026-03-14T12:00:00+00:00",
        last_surfaced_at="2026-03-14T13:00:00+00:00",
    )

    assert updated is not None
    assert updated.title == "Epic 01 relational bootstrap"
    assert updated.content == "Add schema and a repository slice."
    assert updated.summary == "Updated after wiring the repository."
    assert updated.status == "stale"
    assert updated.workspace_ids == ["workspace-b"]
    assert updated.tags == ["repository", "sqlite"]
    assert updated.metadata == {"phase": 1, "priority": "medium"}
    assert updated.access_score == 2.5
    assert updated.last_accessed_at == "2026-03-14T12:00:00+00:00"
    assert updated.last_surfaced_at == "2026-03-14T13:00:00+00:00"
    assert datetime.fromisoformat(updated.updated_at) >= datetime.fromisoformat(created.updated_at)

    workspace_filtered = repository.list_memories(workspace_id="workspace-b")
    fact_filtered = repository.list_memories(memory_type="fact")
    stale_filtered = repository.list_memories(status="stale")

    assert [record.id for record in workspace_filtered] == [created.id]
    assert [record.id for record in fact_filtered] == [secondary.id]
    assert [record.id for record in stale_filtered] == [created.id]
    assert any(connection.commit_count > 0 for connection in session_manager.connections)


def test_postgres_repository_persists_and_resolves_memory_references(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, _session_manager = postgres_repository

    first = repository.create_memory(
        title="First memory",
        content="First content.",
        workspace_ids=["workspace-a"],
    )
    second = repository.create_memory(
        title="Second memory",
        content="Second content.",
        workspace_ids=["workspace-a"],
    )

    assert first is not None and second is not None
    assert (first.memory_ref, second.memory_ref) == (1, 2)
    first_by_ref = repository.get_memory("mem-1")
    second_by_ref = repository.get_memory("2")
    assert first_by_ref is not None and first_by_ref.id == first.id
    assert second_by_ref is not None and second_by_ref.id == second.id

    updated = repository.update_memory("mem-2", title="Updated second memory")
    assert updated is not None
    assert updated.id == second.id
    assert updated.title == "Updated second memory"

    deleted = repository.delete_memory("mem-1")
    assert deleted is not None
    assert deleted.id == first.id
    assert repository.get_memory("mem-1") is None


def test_postgres_repository_rejects_invalid_domain_values(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, _session_manager = postgres_repository

    with pytest.raises(ValueError, match="workspace_ids must contain at least one non-empty value"):
        repository.create_memory(
            title="Bad memory",
            content="No workspace IDs should fail.",
            workspace_ids=["", "   "],
        )

    with pytest.raises(ValueError, match="invalid memory_type"):
        repository.create_memory(
            title="Bad memory",
            content="Unsupported type should fail.",
            workspace_ids=["workspace-a"],
            memory_type="todo",
        )

    created = repository.create_memory(
        title="Valid memory",
        content="This one is okay.",
        workspace_ids=["workspace-a"],
    )

    assert created is not None

    with pytest.raises(ValueError, match="invalid status"):
        repository.update_memory(created.id, status="unknown")


def test_postgres_repository_normalizes_link_types_and_collapses_semantic_duplicates(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, _session_manager = postgres_repository

    source = repository.create_memory(
        title="Source fact",
        content="Depends on the canonical auth architecture.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
    )
    target = repository.create_memory(
        title="Target fact",
        content="Canonical auth architecture.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
    )

    assert source is not None and target is not None

    first = repository.add_link(source.id, target.id, "depends_on", "first context")
    second = repository.add_link(source.id, target.id, "Depends-On", "updated context")
    outgoing = repository.get_links(source.id, direction="outgoing")
    filtered = repository.get_links(source.id, direction="outgoing", link_type="depends on")

    assert first.link_type == "DEPENDS_ON"
    assert second.link_type == "DEPENDS_ON"
    assert len(outgoing) == 1
    assert outgoing[0].link_type == "DEPENDS_ON"
    assert outgoing[0].context == "updated context"
    assert filtered[0].link_type == "DEPENDS_ON"
    assert repository.has_incoming_link(target.id, "depends-on") is True
    assert repository.remove_link(source.id, target.id, "depends on") is True
    assert repository.get_links(source.id, direction="outgoing") == []


def test_postgres_repository_delete_memory_removes_links(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, _session_manager = postgres_repository

    source = repository.create_memory(
        title="Source fact",
        content="Depends on the canonical auth architecture.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
    )
    target = repository.create_memory(
        title="Target fact",
        content="Canonical auth architecture.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
    )
    assert source is not None and target is not None
    repository.add_link(source.id, target.id, "depends_on", "context")

    deleted = repository.delete_memory(target.id)

    assert deleted is not None
    assert deleted.id == target.id
    assert repository.get_memory(target.id) is None
    assert repository.count_incoming_links(target.id) == 0


def test_postgres_repository_loads_json_metadata_objects(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, session_manager = postgres_repository
    created = repository.create_memory(
        title="Metadata fact",
        content="Checks JSON round-trip.",
        workspace_ids=["workspace-a"],
        metadata={"priority": "high", "phase": 2},
    )
    assert created is not None

    state = session_manager.connections[0]._state
    raw_metadata = state.memories[created.id]["metadata"]
    assert json.loads(str(raw_metadata)) == {"phase": 2, "priority": "high"}
    fetched = repository.get_memory(created.id)
    assert fetched is not None
    assert fetched.metadata == {"phase": 2, "priority": "high"}


def test_postgres_repository_keyword_candidates_use_postgres_fts_shape_and_hide_superseded(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, _session_manager = postgres_repository

    old_plan = repository.create_memory(
        title="Legacy auth rollout",
        content="Old auth rollout plan.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    current_plan = repository.create_memory(
        title="Current auth rollout",
        content="Current auth rollout plan.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    cross_workspace = repository.create_memory(
        title="Auth notes",
        content="Shared auth notes.",
        memory_type="fact",
        workspace_ids=["workspace-beta"],
        tags=["auth"],
    )
    assert old_plan is not None and current_plan is not None and cross_workspace is not None

    repository.add_link(current_plan.id, old_plan.id, "SUPERSEDES")

    ids = repository.search_keyword_memory_ids(
        "auth rollout",
        workspace_id="workspace-alpha",
        limit=10,
    )

    assert ids == [current_plan.id]


def _postgres_telemetry_snapshot(
    session_manager: FakeSessionManager,
    memory_ids: Sequence[str],
) -> dict[str, tuple[object, object, object, object]]:
    normalized_ids = list(dict.fromkeys(memory_ids))
    return {
        memory_id: (
            memory["read_count"],
            memory["access_score"],
            memory["last_accessed_at"],
            memory["last_surfaced_at"],
        )
        for memory_id in normalized_ids
        if (memory := session_manager._state.memories.get(memory_id)) is not None
    }


def test_postgres_repository_maintenance_peek_returns_context_without_changing_telemetry(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, session_manager = postgres_repository
    old_fact = repository.create_memory(
        title="Postgres fact",
        content="Use Postgres for shared storage.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    current_fact = repository.create_memory(
        title="Postgres fact refined",
        content="Use Postgres for shared storage with maintenance reads.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    supporter = repository.create_memory(
        title="Postgres support",
        content="Supports the refined storage fact.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    assert old_fact is not None and current_fact is not None and supporter is not None

    repository.add_link(current_fact.id, old_fact.id, "SUPERSEDES", "Refined after testing")
    repository.add_link(supporter.id, current_fact.id, "DEPENDS_ON", "Supports the refinement")
    session_manager._state.memories[current_fact.id].update(
        read_count=1,
        access_score=4.0,
        last_accessed_at="2026-07-14T12:00:00+00:00",
        last_surfaced_at="2026-07-14T12:01:00+00:00",
    )

    result = assert_maintenance_read_preserves_telemetry(
        repository,
        [current_fact.id, old_fact.id, supporter.id],
        lambda: repository.peek_memory(current_fact.id),
        telemetry_snapshot=lambda ids: _postgres_telemetry_snapshot(session_manager, ids),
    )

    ordinary_record = repository.get_memory(current_fact.id)
    assert result is not None and ordinary_record is not None
    assert result.record == ordinary_record
    assert result.relationships["outgoing"] == repository.get_links(current_fact.id, "outgoing")
    assert result.relationships["incoming"] == repository.get_links(current_fact.id, "incoming")
    assert [record.id for record in result.superseded] == [old_fact.id]


def test_postgres_repository_maintenance_search_returns_filtered_context_without_changing_telemetry(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, session_manager = postgres_repository
    old_fact = repository.create_memory(
        title="Legacy Postgres maintenance investigation",
        content="Legacy investigation details.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    current_fact = repository.create_memory(
        title="Current Postgres maintenance investigation",
        content="Authoritative investigation details for shared storage.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    other_type = repository.create_memory(
        title="Postgres maintenance investigation plan",
        content="A plan should be excluded by the type filter.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
    )
    assert old_fact is not None and current_fact is not None and other_type is not None
    repository.add_link(current_fact.id, old_fact.id, "SUPERSEDES", "Current investigation supersedes legacy")
    session_manager._state.memories[current_fact.id].update(
        read_count=1,
        access_score=2.5,
        last_accessed_at="2026-07-14T12:00:00+00:00",
        last_surfaced_at="2026-07-14T12:01:00+00:00",
    )

    default_results, included_results = assert_maintenance_read_preserves_telemetry(
        repository,
        [current_fact.id, old_fact.id, other_type.id],
        lambda: (
            repository.search_memories_for_maintenance(
                "Postgres maintenance investigation",
                workspace_id="workspace-alpha",
                memory_type="fact",
                limit=10,
            ),
            repository.search_memories_for_maintenance(
                "Postgres maintenance investigation",
                workspace_id="workspace-alpha",
                memory_type="fact",
                include_superseded=True,
                limit=10,
            ),
        ),
        telemetry_snapshot=lambda ids: _postgres_telemetry_snapshot(session_manager, ids),
    )

    assert [context.record.id for context in default_results] == [current_fact.id]
    assert [context.record.id for context in included_results] == [current_fact.id, old_fact.id]
    assert [record.id for record in default_results[0].superseded] == [old_fact.id]
    assert included_results[1].superseded == []
    assert default_results[0].record == repository.get_memory(current_fact.id)
    assert default_results[0].relationships == {
        "outgoing": repository.get_links(current_fact.id, "outgoing"),
        "incoming": repository.get_links(current_fact.id, "incoming"),
    }


def test_postgres_repository_maintenance_search_batches_context_hydration(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, session_manager = postgres_repository
    current = repository.create_memory(
        title="Current batched maintenance fact",
        content="Current maintenance details.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["current", "maintenance"],
        created_at="2026-07-14T12:03:00+00:00",
        updated_at="2026-07-14T12:03:00+00:00",
    )
    peer = repository.create_memory(
        title="Peer batched maintenance fact",
        content="Peer maintenance details.",
        memory_type="fact",
        workspace_ids=["workspace-alpha", "workspace-beta"],
        tags=["maintenance", "peer"],
        created_at="2026-07-14T12:02:00+00:00",
        updated_at="2026-07-14T12:02:00+00:00",
    )
    cross_workspace = repository.create_memory(
        title="Cross workspace batched maintenance fact",
        content="Should not match the workspace filter.",
        memory_type="fact",
        workspace_ids=["workspace-beta"],
    )
    old = repository.create_memory(
        title="Legacy batched maintenance fact",
        content="Legacy maintenance details.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["legacy"],
        created_at="2026-07-14T12:01:00+00:00",
        updated_at="2026-07-14T12:01:00+00:00",
    )
    supporter = repository.create_memory(
        title="Support note",
        content="Supports the current fact.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    assert (
        current is not None
        and peer is not None
        and cross_workspace is not None
        and old is not None
        and supporter is not None
    )

    repository.add_link(current.id, old.id, "SUPERSEDES", "Current replaces legacy")
    repository.add_link(supporter.id, current.id, "DEPENDS_ON", "Support for current")
    state = session_manager.connections[0]._state
    query_start = len(state.query_log)
    connection_start = len(session_manager.connections)

    contexts = repository.search_memories_for_maintenance(
        "batched maintenance",
        workspace_id="workspace-alpha",
        memory_type="fact",
        include_superseded=True,
        limit=10,
    )

    queries = state.query_log[query_start:]
    assert [context.record.id for context in contexts] == [current.id, peer.id, old.id]
    assert cross_workspace.id not in [context.record.id for context in contexts]
    assert contexts[0].record.workspace_ids == ["workspace-alpha"]
    assert contexts[0].record.tags == ["current", "maintenance"]
    assert [(link.source_id, link.target_id, link.link_type, link.context) for link in contexts[0].relationships["outgoing"]] == [
        (current.id, old.id, "SUPERSEDES", "Current replaces legacy"),
    ]
    assert [(link.source_id, link.target_id, link.link_type, link.context) for link in contexts[0].relationships["incoming"]] == [
        (supporter.id, current.id, "DEPENDS_ON", "Support for current"),
    ]
    assert [record.id for record in contexts[0].superseded] == [old.id]
    assert contexts[0].superseded[0].workspace_ids == ["workspace-alpha"]
    assert contexts[0].superseded[0].tags == ["legacy"]
    assert contexts[1].record.workspace_ids == ["workspace-alpha", "workspace-beta"]
    assert contexts[1].record.tags == ["maintenance", "peer"]
    assert contexts[2].relationships == {"outgoing": [], "incoming": [contexts[0].relationships["outgoing"][0]]}
    assert len(session_manager.connections) - connection_start == 2
    assert len(queries) <= 9
    assert "SELECT workspace_id FROM memory_workspaces WHERE memory_id = %s ORDER BY workspace_id ASC" not in queries
    assert not any(query.startswith("SELECT tags.name FROM tags JOIN memory_tags") for query in queries)


def test_postgres_repository_batches_ranking_candidate_hydration_and_preserves_filters(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, session_manager = postgres_repository

    preferred = repository.create_memory(
        title="Preferred auth rollout",
        content="Preferred active auth rollout plan.",
        memory_type="plan",
        status="active",
        workspace_ids=["workspace-alpha"],
        tags=["auth", "rollout"],
    )
    peer = repository.create_memory(
        title="Peer auth rollout",
        content="Second visible active auth rollout plan.",
        memory_type="plan",
        status="active",
        workspace_ids=["workspace-beta"],
        tags=["auth"],
    )
    superseded = repository.create_memory(
        title="Superseded auth rollout",
        content="Older auth rollout plan.",
        memory_type="plan",
        status="active",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    stale = repository.create_memory(
        title="Stale auth rollout",
        content="Stale auth rollout plan.",
        memory_type="plan",
        status="stale",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    supporter = repository.create_memory(
        title="Support fact",
        content="Supports the preferred rollout.",
        memory_type="fact",
        status="active",
        workspace_ids=["workspace-alpha"],
    )
    superseder = repository.create_memory(
        title="Replacement rollout",
        content="Replaces the old rollout.",
        memory_type="plan",
        status="active",
        workspace_ids=["workspace-alpha"],
    )
    assert preferred is not None and peer is not None and superseded is not None and stale is not None
    assert supporter is not None and superseder is not None

    repository.add_link(supporter.id, preferred.id, "DEPENDS_ON")
    repository.add_link(superseder.id, superseded.id, "SUPERSEDES")

    state = session_manager.connections[0]._state
    query_start = len(state.query_log)
    timing_ms: dict[str, float] = {}

    candidates = repository.get_ranking_candidates(
        [preferred.id, superseded.id, stale.id, peer.id],
        status="active",
        include_superseded=False,
        timing_ms=timing_ms,
    )

    queries = state.query_log[query_start:]
    assert len(queries) == 1
    assert queries[0].startswith("WITH input_ids AS ( SELECT memory_id, ordinality FROM unnest(%s::text[]) WITH ORDINALITY AS requested(memory_id, ordinality) )")
    assert "SELECT workspace_id FROM memory_workspaces WHERE memory_id = %s ORDER BY workspace_id ASC" not in queries
    assert not any(query.startswith("SELECT source_id, target_id, type, context FROM links WHERE") for query in queries)

    assert [candidate.record.id for candidate in candidates] == [preferred.id, peer.id]
    assert candidates[0].incoming_links_count == 1
    assert candidates[0].incoming_link_type_counts == {
        "DEPENDS_ON": 1,
        "AMENDS": 0,
        "CONTRADICTS": 0,
        "SUPERSEDES": 0,
    }
    assert candidates[1].incoming_links_count == 0
    assert timing_ms["candidate_hydration_query_execution"] >= 0.0
    assert timing_ms["candidate_hydration_row_fetch"] >= 0.0
    assert timing_ms["candidate_hydration_candidate_build"] >= 0.0


def test_postgres_repository_list_memory_ids_matches_list_memories_ordering_and_filters(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, _session_manager = postgres_repository

    oldest = repository.create_memory(
        title="Old alpha fact",
        content="Old alpha fact content.",
        memory_type="fact",
        status="active",
        workspace_ids=["workspace-alpha"],
        updated_at="2026-03-01T00:00:00+00:00",
        created_at="2026-03-01T00:00:00+00:00",
    )
    newest = repository.create_memory(
        title="New alpha fact",
        content="New alpha fact content.",
        memory_type="fact",
        status="active",
        workspace_ids=["workspace-alpha"],
        updated_at="2026-03-03T00:00:00+00:00",
        created_at="2026-03-03T00:00:00+00:00",
    )
    other_type = repository.create_memory(
        title="Alpha plan",
        content="Alpha plan content.",
        memory_type="plan",
        status="active",
        workspace_ids=["workspace-alpha"],
        updated_at="2026-03-02T00:00:00+00:00",
        created_at="2026-03-02T00:00:00+00:00",
    )
    other_workspace = repository.create_memory(
        title="Beta fact",
        content="Beta fact content.",
        memory_type="fact",
        status="active",
        workspace_ids=["workspace-beta"],
        updated_at="2026-03-04T00:00:00+00:00",
        created_at="2026-03-04T00:00:00+00:00",
    )
    stale = repository.create_memory(
        title="Stale alpha fact",
        content="Stale alpha fact content.",
        memory_type="fact",
        status="stale",
        workspace_ids=["workspace-alpha"],
        updated_at="2026-03-05T00:00:00+00:00",
        created_at="2026-03-05T00:00:00+00:00",
    )
    assert oldest is not None and newest is not None and other_type is not None
    assert other_workspace is not None and stale is not None

    expected = repository.list_memories(
        workspace_id="workspace-alpha",
        memory_type="fact",
        status="active",
        limit=10,
    )
    observed_ids = repository.list_memory_ids(
        workspace_id="workspace-alpha",
        memory_type="fact",
        status="active",
        limit=10,
    )

    assert observed_ids == [record.id for record in expected] == [newest.id, oldest.id]


def test_postgres_repository_list_memories_batches_workspace_and_tag_hydration(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, session_manager = postgres_repository

    primary = repository.create_memory(
        title="Primary fact",
        content="Primary fact content.",
        memory_type="fact",
        status="active",
        workspace_ids=["workspace-beta", "workspace-alpha"],
        tags=["beta", "alpha"],
    )
    secondary = repository.create_memory(
        title="Secondary fact",
        content="Secondary fact content.",
        memory_type="fact",
        status="active",
        workspace_ids=["workspace-gamma"],
        tags=["gamma"],
    )
    assert primary is not None and secondary is not None

    state = session_manager.connections[0]._state
    query_start = len(state.query_log)

    listed = repository.list_memories(status="active", limit=10)

    queries = state.query_log[query_start:]
    assert len(queries) == 3
    assert queries[0].startswith(
        "SELECT id, title, content, summary, type, status, created_at, updated_at, read_count, access_score, last_accessed_at, last_surfaced_at, metadata, memory_ref, archived_at FROM memories"
    )
    assert queries[1] == "SELECT memory_id, workspace_id FROM memory_workspaces WHERE memory_id = ANY(%s::text[]) ORDER BY memory_id ASC, workspace_id ASC"
    assert queries[2] == "SELECT memory_tags.memory_id, tags.name FROM memory_tags JOIN tags ON tags.id = memory_tags.tag_id WHERE memory_tags.memory_id = ANY(%s::text[]) ORDER BY memory_tags.memory_id ASC, tags.name ASC"
    assert not any(query == "SELECT workspace_id FROM memory_workspaces WHERE memory_id = %s ORDER BY workspace_id ASC" for query in queries)
    assert not any(
        query.startswith("SELECT tags.name FROM tags JOIN memory_tags ON memory_tags.tag_id = tags.id")
        for query in queries
    )

    assert [record.id for record in listed] == [secondary.id, primary.id]
    hydrated_primary = next(record for record in listed if record.id == primary.id)
    assert hydrated_primary.workspace_ids == ["workspace-alpha", "workspace-beta"]
    assert hydrated_primary.tags == ["alpha", "beta"]


def test_postgres_repository_ranking_candidates_aggregate_without_join_fanout_inflation(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, _session_manager = postgres_repository

    primary = repository.create_memory(
        title="Primary authority",
        content="Canonical authority with multiple workspaces, tags, and link types.",
        memory_type="fact",
        status="active",
        workspace_ids=["workspace-beta", "workspace-alpha"],
        tags=["alpha", "beta"],
    )
    secondary = repository.create_memory(
        title="Secondary note",
        content="Secondary active note.",
        memory_type="fact",
        status="active",
        workspace_ids=["workspace-zeta"],
        tags=["gamma"],
    )
    depends = repository.create_memory(
        title="Depends supporter",
        content="Depends on primary.",
        memory_type="plan",
        status="active",
        workspace_ids=["workspace-alpha"],
    )
    amends = repository.create_memory(
        title="Amends supporter",
        content="Amends primary.",
        memory_type="fact",
        status="active",
        workspace_ids=["workspace-alpha"],
    )
    contradicts = repository.create_memory(
        title="Contradiction",
        content="Contradicts primary.",
        memory_type="fact",
        status="active",
        workspace_ids=["workspace-alpha"],
    )

    assert primary is not None and secondary is not None
    assert depends is not None and amends is not None and contradicts is not None

    repository.add_link(depends.id, primary.id, "DEPENDS_ON")
    repository.add_link(amends.id, primary.id, "AMENDS")
    repository.add_link(contradicts.id, primary.id, "CONTRADICTS")

    candidates = repository.get_ranking_candidates([secondary.id, primary.id], status="active")

    assert [candidate.record.id for candidate in candidates] == [secondary.id, primary.id]
    assert candidates[0].record.workspace_ids == ["workspace-zeta"]
    assert candidates[0].record.tags == ["gamma"]

    primary_candidate = candidates[1]
    assert primary_candidate.record.workspace_ids == ["workspace-alpha", "workspace-beta"]
    assert primary_candidate.record.tags == ["alpha", "beta"]
    assert primary_candidate.incoming_links_count == 3
    assert primary_candidate.incoming_link_type_counts == {
        "DEPENDS_ON": 1,
        "AMENDS": 1,
        "CONTRADICTS": 1,
        "SUPERSEDES": 0,
    }
    assert primary_candidate.has_incoming_supersedes is False


def test_postgres_repository_get_searchable_memories_matches_sqlite_semantics(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, _session_manager = postgres_repository

    primary = repository.create_memory(
        title="Primary Postgres backend note",
        content="Primary Postgres backend guidance for bounded semantic candidates.",
        summary="Primary backend summary.",
        memory_type="fact",
        status="active",
        workspace_ids=["workspace-zeta", "workspace-alpha"],
        tags=["backend", "postgres"],
        metadata={"priority": "high", "source": "test"},
    )
    secondary = repository.create_memory(
        title="Secondary Postgres backend note",
        content="Secondary active note.",
        summary="Secondary backend summary.",
        memory_type="fact",
        status="active",
        workspace_ids=["workspace-beta"],
        tags=["search"],
    )
    stale = repository.create_memory(
        title="Stale Postgres backend note",
        content="Stale note.",
        summary="Stale backend summary.",
        memory_type="fact",
        status="stale",
        workspace_ids=["workspace-alpha"],
        tags=["postgres"],
    )
    superseded = repository.create_memory(
        title="Legacy Postgres backend note",
        content="Legacy note hidden by default.",
        summary="Legacy backend summary.",
        memory_type="fact",
        status="active",
        workspace_ids=["workspace-gamma"],
        tags=["legacy"],
        metadata={"priority": "low"},
    )
    superseder = repository.create_memory(
        title="Replacement Postgres backend note",
        content="Replacement note.",
        summary="Replacement backend summary.",
        memory_type="fact",
        status="active",
        workspace_ids=["workspace-gamma"],
    )

    assert primary is not None and secondary is not None and stale is not None
    assert superseded is not None and superseder is not None

    repository.add_link(superseder.id, superseded.id, "SUPERSEDES")

    default_visible = repository.get_searchable_memories(
        [secondary.id, superseded.id, primary.id, stale.id],
        status="active",
    )
    include_superseded = repository.get_searchable_memories(
        [secondary.id, superseded.id, primary.id, stale.id],
        status="active",
        include_superseded=True,
    )

    assert [record.id for record in default_visible] == [secondary.id, primary.id]
    assert [record.id for record in include_superseded] == [secondary.id, superseded.id, primary.id]

    hydrated_primary = next(record for record in default_visible if record.id == primary.id)
    assert hydrated_primary.metadata == {"priority": "high", "source": "test"}
    assert set(hydrated_primary.workspace_ids) == {"workspace-alpha", "workspace-zeta"}
    assert set(hydrated_primary.tags) == {"backend", "postgres"}


def test_postgres_search_service_prioritizes_workspace_and_hides_superseded(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, _session_manager = postgres_repository
    service = RelationalMemorySearchService(repository, Config())

    old_plan = repository.create_memory(
        title="Legacy auth plan",
        content="Old auth plan for workspace alpha.",
        summary="Old plan summary.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
        created_at="2026-02-01T10:00:00+00:00",
        updated_at="2026-02-01T10:00:00+00:00",
    )
    current_plan = repository.create_memory(
        title="Current auth plan",
        content="Current auth plan for workspace alpha.",
        summary="Current plan summary.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
        created_at="2026-03-10T10:00:00+00:00",
        updated_at="2026-03-10T10:00:00+00:00",
    )
    cross_workspace = repository.create_memory(
        title="Cross workspace auth fact",
        content="Shared auth fact for another workspace.",
        summary="Shared fact summary.",
        memory_type="fact",
        workspace_ids=["workspace-beta"],
        tags=["auth"],
        created_at="2026-03-09T10:00:00+00:00",
        updated_at="2026-03-09T10:00:00+00:00",
    )
    assert old_plan is not None and current_plan is not None and cross_workspace is not None

    repository.add_link(current_plan.id, old_plan.id, "SUPERSEDES", "Replaced during redesign")

    results = service.search_memories("auth plan", workspace_id="workspace-alpha", limit=5)

    assert [result.memory_id for result in results] == [current_plan.id]
    assert results[0].summary == "Current plan summary."
    assert results[0].workspace_ids == ["workspace-alpha"]


def test_postgres_search_service_updates_last_surfaced_timestamps(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, _session_manager = postgres_repository
    service = RelationalMemorySearchService(
        repository, Config(searchkernel=SearchKernelConfig(rerank_policy="disabled", rerank_budget=0))
    )

    active = repository.create_memory(
        title="Search pipeline active",
        content="Active search pipeline note.",
        summary="Active summary.",
        memory_type="plan",
        status="active",
        workspace_ids=["workspace-alpha"],
        tags=["search"],
    )
    stale = repository.create_memory(
        title="Search pipeline stale",
        content="Stale search pipeline note.",
        summary="Stale summary.",
        memory_type="plan",
        status="stale",
        workspace_ids=["workspace-alpha"],
        tags=["search"],
    )
    assert active is not None and stale is not None

    results = service.search_memories("search pipeline", workspace_id="workspace-alpha", limit=5)

    assert [result.memory_id for result in results] == [active.id, stale.id]
    repository.flush()
    refreshed_active = repository.get_memory(active.id)
    refreshed_stale = repository.get_memory(stale.id)
    assert refreshed_active is not None and refreshed_active.last_surfaced_at is not None
    assert refreshed_stale is not None and refreshed_stale.last_surfaced_at is not None


def test_postgres_repository_best_effort_last_surfaced_flushes_on_close(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, _session_manager = postgres_repository
    record = repository.create_memory(
        title="Buffered surfaced write",
        content="Checks close flush behavior.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
    )
    assert record is not None

    repository.touch_last_surfaced([record.id], "2026-03-29T00:00:00+00:00", best_effort=True)
    repository.close()

    refreshed = repository.get_memory(record.id)

    assert refreshed is not None
    assert refreshed.last_surfaced_at == "2026-03-29T00:00:00+00:00"


def test_postgres_repository_read_cache_validation_tokens_are_stable_for_unchanged_data(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, session_manager = postgres_repository

    current = repository.create_memory(
        title="Current fact",
        content="Current canonical fact.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
    )
    superseded = repository.create_memory(
        title="Legacy fact",
        content="Legacy fact content.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
    )
    incoming = repository.create_memory(
        title="Supporting fact",
        content="Supports the current fact.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
    )

    assert current is not None and superseded is not None and incoming is not None

    repository.add_link(current.id, superseded.id, "SUPERSEDES", "replacement")
    repository.add_link(incoming.id, current.id, "DEPENDS_ON", "supporting evidence")

    state = session_manager.connections[0]._state
    query_start = len(state.query_log)
    first = repository.get_read_cache_validation_tokens([current.id, "missing-memory"])
    first_queries = state.query_log[query_start:]
    second = repository.get_read_cache_validation_tokens([current.id, "missing-memory"])

    assert list(first) == [current.id]
    assert second == first
    assert len(first_queries) == 4
    assert first_queries[0].startswith(
        "SELECT id, title, content, summary, type, status, created_at, updated_at, read_count, access_score, last_accessed_at, last_surfaced_at, metadata, memory_ref, archived_at FROM memories WHERE id = ANY(%s::text[])"
    )
    assert first_queries[1] == "SELECT source_id, target_id, type, context FROM links WHERE source_id = ANY(%s::text[]) ORDER BY source_id ASC, target_id ASC"
    assert first_queries[2] == "SELECT source_id, target_id, type, context FROM links WHERE target_id = ANY(%s::text[]) ORDER BY source_id ASC, target_id ASC"
    assert first_queries[3].startswith(
        "SELECT id, title, content, summary, type, status, created_at, updated_at, read_count, access_score, last_accessed_at, last_surfaced_at, metadata, memory_ref, archived_at FROM memories WHERE id = ANY(%s::text[])"
    )
    assert not any(query == "SELECT workspace_id FROM memory_workspaces WHERE memory_id = %s ORDER BY workspace_id ASC" for query in first_queries)
    assert not any(
        query.startswith("SELECT tags.name FROM tags JOIN memory_tags ON memory_tags.tag_id = tags.id")
        for query in first_queries
    )


def test_postgres_repository_read_cache_validation_tokens_invalidate_for_link_and_superseded_changes(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, _session_manager = postgres_repository

    current = repository.create_memory(
        title="Current fact",
        content="Current canonical fact.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
    )
    superseded = repository.create_memory(
        title="Legacy fact",
        content="Legacy fact content.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
    )
    incoming = repository.create_memory(
        title="Supporting fact",
        content="Supports the current fact.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
    )

    assert current is not None and superseded is not None and incoming is not None

    repository.add_link(current.id, superseded.id, "SUPERSEDES", "replacement")
    repository.add_link(incoming.id, current.id, "DEPENDS_ON", "supporting evidence")

    initial_token = repository.get_read_cache_validation_tokens([current.id])[current.id]

    repository.add_link(incoming.id, current.id, "DEPENDS_ON", "updated supporting evidence")
    link_updated_token = repository.get_read_cache_validation_tokens([current.id])[current.id]

    repository.update_memory(
        superseded.id,
        content="Legacy fact content, revised.",
    )
    superseded_updated_token = repository.get_read_cache_validation_tokens([current.id])[current.id]

    assert link_updated_token != initial_token
    assert superseded_updated_token != link_updated_token


class _PostgresSearchFakeEmbedder:
    model_name = "fake-postgres-mini"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if any(token in lowered for token in ["ripgrep", "scan", "scanning", "repository"]):
                vectors.append([0.0, 1.0])
            elif any(token in lowered for token in ["permission", "permissions", "identity", "authentication", "token"]):
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.2, 0.2])
        return vectors


class _RecordingCandidateAwareVectorStore:
    supports_candidate_filtering = True

    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str], EmbeddingRecord] = {}
        self.last_candidate_ids: list[str] | None = None

    def upsert(
        self,
        *,
        source_kind: str,
        source_id: str,
        workspace_id: str | None,
        model_name: str,
        embedding: list[float],
        source_updated_at: str | None = None,
    ) -> None:
        _ = source_updated_at
        self._records[(source_kind, source_id, model_name)] = EmbeddingRecord(
            source_kind=source_kind,
            source_id=source_id,
            workspace_id=workspace_id,
            model_name=model_name,
            embedding=list(embedding),
            updated_at=1.0,
        )

    def get(
        self,
        *,
        source_kind: str,
        source_id: str,
        model_name: str,
    ) -> EmbeddingRecord | None:
        return self._records.get((source_kind, source_id, model_name))

    def get_updated_at_map(
        self,
        *,
        source_kind: str,
        model_name: str,
        source_ids: list[str],
    ) -> dict[str, float]:
        return {
            source_id: self._records[(source_kind, source_id, model_name)].updated_at
            for source_id in source_ids
            if (source_kind, source_id, model_name) in self._records
        }

    def search(
        self,
        *,
        source_kind: str,
        model_name: str,
        query_embedding: list[float],
        candidate_ids: list[str] | None = None,
        diagnostics: dict[str, object] | None = None,
        workspace_id: str | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        del diagnostics
        del workspace_id
        self.last_candidate_ids = None if candidate_ids is None else list(candidate_ids)
        candidate_id_filter = None if candidate_ids is None else set(candidate_ids)
        scored = [
            (
                record.source_id,
                cosine_similarity_lists(query_embedding, record.embedding),
            )
            for (stored_source_kind, _source_id, stored_model_name), record in self._records.items()
            if stored_source_kind == source_kind
            and stored_model_name == model_name
            and (candidate_id_filter is None or record.source_id in candidate_id_filter)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]


def test_postgres_search_service_bounds_semantic_scoring_when_lexical_hits_are_strong(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, _session_manager = postgres_repository
    vector_store = _RecordingCandidateAwareVectorStore()
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_PostgresSearchFakeEmbedder(),
        vector_store=vector_store,
    )

    exact = repository.create_memory(
        title="Ripgrep ban",
        content="Avoid broad ripgrep scans in large repositories.",
        summary="Ripgrep ban summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["grep"],
    )
    distractor = repository.create_memory(
        title="Repository scanning guidance",
        content="Use targeted repository scanning alternatives for large codebases.",
        summary="Repository scanning summary.",
        memory_type="fact",
        workspace_ids=["workspace-beta"],
        tags=["search"],
    )
    assert exact is not None and distractor is not None

    results = service.search_memories("ripgrep ban", workspace_id="workspace-alpha", limit=5)

    assert results
    assert results[0].memory_id == exact.id
    assert distractor.id not in {result.memory_id for result in results}
    assert vector_store.last_candidate_ids == [exact.id]


def test_postgres_search_service_delegates_weak_lexical_queries_to_kernel(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, _session_manager = postgres_repository
    vector_store = _RecordingCandidateAwareVectorStore()
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_PostgresSearchFakeEmbedder(),
        vector_store=vector_store,
    )

    weak_lexical = repository.create_memory(
        title="Zebra checklist",
        content="Zebra checklist for unrelated operational work.",
        summary="Zebra summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["operations"],
    )
    semantic_match = repository.create_memory(
        title="Identity policy",
        content="Authentication token rotation and credential policy.",
        summary="Identity controls.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    assert weak_lexical is not None and semantic_match is not None

    results = service.search_memories("permissions zebra", workspace_id="workspace-alpha", limit=5)

    assert results
    assert semantic_match.id in {result.memory_id for result in results}
    assert vector_store.last_candidate_ids is None


def test_postgres_search_service_reports_kernel_owned_diagnostics(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, _session_manager = postgres_repository
    vector_store = _RecordingCandidateAwareVectorStore()
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_PostgresSearchFakeEmbedder(),
        vector_store=vector_store,
    )
    lexical_records = []
    for index in range(3):
        record = repository.create_memory(
            title=f"Permissions security note {index}",
            content="Permissions security guidance for access control reviews.",
            summary="Permissions security summary.",
            memory_type="fact",
            workspace_ids=["workspace-alpha"],
            tags=["security"],
        )
        assert record is not None
        lexical_records.append(record)

    semantic_only = repository.create_memory(
        title="Identity policy",
        content="Authentication token rotation and credential policy.",
        summary="Identity controls.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    assert semantic_only is not None
    for record in lexical_records:
        vector_store.upsert(
            source_kind="memory",
            source_id=record.id,
            workspace_id=None,
            model_name=_PostgresSearchFakeEmbedder.model_name,
            embedding=[1.0, 0.0],
        )
    vector_store.upsert(
        source_kind="memory",
        source_id=semantic_only.id,
        workspace_id=None,
        model_name=_PostgresSearchFakeEmbedder.model_name,
        embedding=[1.0, 0.0],
    )

    results, diagnostics = service.search_memories_with_diagnostics(
        "permissions security access control latency",
        workspace_id="workspace-alpha",
        limit=4,
        debug=True,
    )

    assert results
    assert diagnostics.timing_ms["total"] >= 0.0
    assert "semantic_speculative_fallback" not in diagnostics.timing_ms


def test_postgres_maintenance_candidate_query_contract(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, session_manager = postgres_repository

    assert_maintenance_candidate_query_contract(repository)

    candidate_queries = [
        query
        for query in session_manager._state.query_log
        if "incoming_counts" in query
    ]
    assert candidate_queries
    assert all(" LIMIT %s" in query for query in candidate_queries)
