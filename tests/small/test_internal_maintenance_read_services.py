from __future__ import annotations

import pytest

from mcp_memory.config import Config
from mcp_memory.context import ApplicationContext
from mcp_memory.internal_tool_call_tracking import internal_tool_is_mutating
from mcp_memory.mcp.internal_read_services import (
    internal_bounded_adjacency_service,
    internal_list_relationships_service,
    internal_maintenance_search_service,
    internal_peek_record_service,
)
from mcp_memory.mcp.internal_tools import get_internal_maintenance_tools
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.relational.search import RelationalMemorySearchService
from tests.small.maintenance_read_repository_contract import assert_maintenance_read_preserves_telemetry


pytestmark = pytest.mark.small


class _ExplodingReadCache:
    def __getattr__(self, name: str):
        raise AssertionError(f"maintenance read unexpectedly touched shared cache: {name}")


def _context(db_manager) -> tuple[ApplicationContext, RelationalMemoryRepository]:
    repository = RelationalMemoryRepository(db_manager)
    context = ApplicationContext(
        db_manager=db_manager,
        workspace_id="workspace-alpha",
        repository=repository,
        relational_search=RelationalMemorySearchService(repository, Config()),
        read_cache=_ExplodingReadCache(),
    )
    return context, repository


def test_internal_maintenance_reads_are_bounded_authoritative_and_instrumented(db_manager) -> None:
    context, repository = _context(db_manager)
    root = repository.create_memory(
        title="Maintenance root",
        content="Authoritative maintenance search phrase.",
        workspace_ids=["workspace-alpha"],
    )
    neighbor = repository.create_memory(
        title="Maintenance neighbor",
        content="Neighbor content.",
        workspace_ids=["workspace-alpha"],
    )
    assert root is not None and neighbor is not None
    repository.add_link(root.id, neighbor.id, "DEPENDS_ON", "maintenance edge")
    repository.record_access(
        root.id,
        access_score=3.5,
        accessed_at="2026-07-14T12:00:00+00:00",
        increment_read_count=True,
    )
    repository.touch_last_surfaced([root.id], "2026-07-14T12:01:00+00:00")

    peek = assert_maintenance_read_preserves_telemetry(
        repository,
        [root.id, neighbor.id],
        lambda: internal_peek_record_service(context, {"memory_id": root.id}),
    )
    search = internal_maintenance_search_service(
        context,
        {"query": "authoritative maintenance", "limit": 1},
    )
    relationships = internal_list_relationships_service(
        context,
        {"memory_id": root.id, "direction": "outgoing", "limit": 1},
    )
    adjacency = internal_bounded_adjacency_service(
        context,
        {"memory_id": root.id, "direction": "outgoing", "limit": 1},
    )

    assert peek["status"] == "ok"
    assert peek["record"]["id"] == root.id
    assert peek["budget"] == {
        "requested_limit": 1,
        "effective_limit": 1,
        "returned_count": 1,
        "remaining_budget": 0,
        "truncated": False,
    }
    assert search["status"] == "ok"
    assert [item["record"]["id"] for item in search["results"]] == [root.id]
    assert search["budget"]["remaining_budget"] == 0
    assert relationships["relationships"][0]["link"]["target_id"] == neighbor.id
    assert adjacency["neighbors"][0]["record"]["id"] == neighbor.id
    assert adjacency["budget"]["remaining_budget"] == 0

    context.retrieval_telemetry.flush()
    rows = db_manager.get_connection().execute(
        "SELECT event_kind, caller_kind, memory_id FROM memory_tool_events WHERE caller_kind = 'internal'"
    ).fetchall()
    assert {str(row[0]) for row in rows} == {"read", "search"}
    assert all(str(row[1]) == "internal" for row in rows)

    after = repository.get_memory(root.id)
    assert after is not None
    assert (after.read_count, after.access_score, after.last_accessed_at, after.last_surfaced_at) == (
        1,
        3.5,
        "2026-07-14T12:00:00+00:00",
        "2026-07-14T12:01:00+00:00",
    )


def test_new_internal_maintenance_tools_are_read_only_to_tracking() -> None:
    names = {tool.name for tool in get_internal_maintenance_tools()}
    expected = {
        "internal_peek_record",
        "internal_maintenance_search",
        "internal_list_relationships",
        "internal_bounded_adjacency",
    }
    assert expected <= names
    assert all(not internal_tool_is_mutating(name) for name in expected)
