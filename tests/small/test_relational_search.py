from datetime import datetime, timedelta, timezone

import pytest

from mcp_memory.config import Config
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.relational.search import RelationalMemorySearchService


pytestmark = pytest.mark.small


def test_search_memories_prioritizes_workspace_and_hides_superseded(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
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
    assert old_plan is not None

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
    assert current_plan is not None

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
    assert cross_workspace is not None

    repository.add_link(current_plan.id, old_plan.id, "SUPERSEDES", "Replaced during redesign")

    results = service.search_memories("auth plan", workspace_id="workspace-alpha", limit=5)

    assert [result.memory_id for result in results] == [current_plan.id, cross_workspace.id]
    assert results[0].summary == "Current plan summary."
    assert results[0].workspace_ids == ["workspace-alpha"]


def test_read_memory_returns_superseded_breadcrumbs_and_updates_access_score(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())

    old_fact = repository.create_memory(
        title="SQLite fact",
        content="Use SQLite locally.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["database"],
        created_at="2026-03-01T08:00:00+00:00",
        updated_at="2026-03-01T08:00:00+00:00",
    )
    current_fact = repository.create_memory(
        title="SQLite fact refined",
        content="Use SQLite locally with WAL mode.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["database"],
        created_at="2026-03-05T08:00:00+00:00",
        updated_at="2026-03-05T08:00:00+00:00",
    )
    assert old_fact is not None and current_fact is not None

    repository.add_link(current_fact.id, old_fact.id, "SUPERSEDES", "Refined after testing")
    repository.record_access(
        current_fact.id,
        access_score=4.0,
        accessed_at=(datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
    )

    result = service.read_memory(current_fact.id)

    assert result is not None
    assert result.record.id == current_fact.id
    assert result.record.access_score > 2.9
    assert [memory.id for memory in result.superseded] == [old_fact.id]
    assert result.relationships["outgoing"][0].link_type == "SUPERSEDES"


def test_search_memories_penalizes_stale_records_and_updates_last_surfaced(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
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
