from datetime import datetime, timedelta, timezone

import pytest

from mcp_memory.config import Config
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.relational.search import RelationalMemorySearchService


pytestmark = pytest.mark.medium


def test_ranking_ladder_orders_working_and_fresh_above_stale_generic_match(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())
    now = datetime.now(timezone.utc)

    evergreen = repository.create_memory(
        title="Platform design note",
        content="Generic platform note for the search overhaul.",
        summary="Evergreen fact.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["generic", "search"],
        created_at=(now - timedelta(days=240)).isoformat(),
        updated_at=(now - timedelta(days=240)).isoformat(),
    )
    fresh = repository.create_memory(
        title="Platform journal update",
        content="Generic platform note capturing the newest search work.",
        summary="Fresh journal.",
        memory_type="journal",
        workspace_ids=["workspace-alpha"],
        tags=["generic", "search"],
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )
    obsolete = repository.create_memory(
        title="Platform stale plan",
        content="Generic platform note for an obsolete search plan.",
        summary="Obsolete plan.",
        memory_type="plan",
        status="stale",
        workspace_ids=["workspace-alpha"],
        tags=["generic", "search"],
        created_at=(now - timedelta(days=30)).isoformat(),
        updated_at=(now - timedelta(days=30)).isoformat(),
    )
    working = repository.create_memory(
        title="Platform working thread",
        content="Generic platform note for the currently active search task.",
        summary="Working memory.",
        memory_type="observation",
        workspace_ids=["workspace-alpha"],
        tags=["generic", "search"],
        created_at=(now - timedelta(days=2)).isoformat(),
        updated_at=(now - timedelta(days=2)).isoformat(),
    )
    assert evergreen is not None and fresh is not None and obsolete is not None and working is not None

    repository.record_access(
        working.id,
        access_score=64.0,
        accessed_at=now.isoformat(),
    )
    repository.add_link(fresh.id, evergreen.id, "DEPENDS_ON")
    repository.add_link(working.id, evergreen.id, "DEPENDS_ON")

    results = service.search_memories("generic platform note", workspace_id="workspace-alpha", limit=4)

    assert [result.memory_id for result in results] == [working.id, fresh.id, evergreen.id, obsolete.id]
    assert results[0].score >= results[1].score >= results[2].score >= results[3].score