from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypeAlias
from uuid import UUID

import pytest

from mcp_memory.config import Config
from mcp_memory.relational.search import RelationalMemorySearchService
from mcp_memory.storage.postgres_repository import PostgresRelationalMemoryRepository


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
    next_tag_id: int = 1


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

        if normalized.startswith("INSERT INTO memories ("):
            self._insert_memory(arguments)
        elif normalized.startswith("SELECT id, title, content, summary, type, status, created_at, updated_at,"):
            self._select_memory(arguments)
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
        elif normalized.startswith("SELECT tags.name FROM tags JOIN memory_tags ON memory_tags.tag_id = tags.id"):
            memory_id = str(arguments[0])
            tag_names = sorted(
                self._state.tags[tag_id]
                for tag_id in self._state.memory_tags.get(memory_id, set())
                if tag_id in self._state.tags
            )
            self._result = [(tag_name,) for tag_name in tag_names]
        elif normalized.startswith("SELECT DISTINCT id, title, content, summary, type, status, created_at, updated_at,"):
            self._select_memories(normalized, arguments)
        elif normalized.startswith("SELECT DISTINCT memories.id,"):
            self._search_keyword_memory_ids(normalized, arguments)
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
        }
        self._result = []

    def _select_memory(self, arguments: SqlParams) -> None:
        memory = self._state.memories.get(str(arguments[0]))
        if memory is None:
            self._result = []
            return
        self._result = [self._memory_row(memory)]

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

    def _update_memory(self, normalized: str, arguments: SqlParams) -> None:
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
        direction_key = "target_id" if "WHERE target_id = %s" in normalized else "source_id"
        memory_id = str(arguments[0])
        link_type = str(arguments[1]) if len(arguments) > 1 else None
        rows = []
        for source_id, target_id, stored_link_type in sorted(self._state.links):
            if direction_key == "source_id" and source_id != memory_id:
                continue
            if direction_key == "target_id" and target_id != memory_id:
                continue
            if link_type is not None and stored_link_type != link_type:
                continue
            rows.append((source_id, target_id, stored_link_type, self._state.links[(source_id, target_id, stored_link_type)]))
        self._result = rows

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
def postgres_repository() -> tuple[PostgresRelationalMemoryRepository, FakeSessionManager]:
    session_manager = FakeSessionManager(FakePostgresState())
    return PostgresRelationalMemoryRepository(session_manager), session_manager


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

    assert created is not None
    assert secondary is not None

    UUID(created.id)
    assert created.type == "plan"
    assert created.workspace_ids == ["workspace-a", "workspace-b"]
    assert created.tags == ["sqlite", "testing"]
    assert created.metadata == {"priority": "high"}

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

    assert [result.memory_id for result in results] == [current_plan.id, cross_workspace.id]
    assert results[0].summary == "Current plan summary."
    assert results[0].workspace_ids == ["workspace-alpha"]


def test_postgres_search_service_updates_last_surfaced_timestamps(
    postgres_repository: tuple[PostgresRelationalMemoryRepository, FakeSessionManager],
) -> None:
    repository, _session_manager = postgres_repository
    service = RelationalMemorySearchService(repository, Config())

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
    refreshed_active = repository.get_memory(active.id)
    refreshed_stale = repository.get_memory(stale.id)
    assert refreshed_active is not None and refreshed_active.last_surfaced_at is not None
    assert refreshed_stale is not None and refreshed_stale.last_surfaced_at is not None