from __future__ import annotations

import pytest

from mcp_memory.config import Config
from mcp_memory.context import ApplicationContext
from mcp_memory.mcp.services import (
    read_memory_record_service,
    read_memory_records_service,
    search_memory_records_service,
)
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.relational.search import RelationalMemorySearchService


pytestmark = pytest.mark.small


def _context(db_manager) -> tuple[ApplicationContext, RelationalMemoryRepository]:
    repository = RelationalMemoryRepository(db_manager)
    return (
        ApplicationContext(
            db_manager=db_manager,
            workspace_id="workspace-alpha",
            repository=repository,
            relational_search=RelationalMemorySearchService(repository, Config()),
        ),
        repository,
    )


def test_agent_search_read_batch_workflow_uses_stable_references(db_manager) -> None:
    context, repository = _context(db_manager)
    first = repository.create_memory(
        title="Stable reference workflow alpha",
        content="Agents can read this stable reference workflow memory.",
        workspace_ids=["workspace-alpha"],
    )
    second = repository.create_memory(
        title="Stable reference workflow beta",
        content="Batch reads can retrieve this stable reference workflow memory.",
        workspace_ids=["workspace-alpha"],
    )
    assert first is not None and second is not None

    search = search_memory_records_service(
        context,
        {"query": "stable reference workflow", "limit": 5},
    )
    assert search["status"] == "ok"
    references = [result["memory_ref"] for result in search["results"]]
    assert references
    assert all(reference.startswith("mem-") for reference in references)
    assert all("memory_id" not in result for result in search["results"])

    batch = read_memory_records_service(
        context,
        {"memory_refs": references + ["mem-999999"]},
    )
    assert batch["status"] == "ok"
    assert {record["memory_ref"] for record in batch["records"]} == set(references)
    assert batch["missing"] == ["mem-999999"]

    singular = read_memory_record_service(context, {"memory_id": references[0]})
    assert singular["status"] == "ok"
    assert singular["record"]["memory_ref"] == references[0]

    legacy = read_memory_record_service(context, {"memory_id": first.id})
    assert legacy["status"] == "ok"
    assert legacy["record"]["memory_ref"] == f"mem-{first.memory_ref}"

    context.retrieval_telemetry.flush()
    rows = db_manager.get_connection().execute(
        "SELECT memory_id FROM memory_tool_events WHERE event_kind = 'read'"
    ).fetchall()
    expected_read_ids = {
        repository.resolve_memory_id(reference) for reference in references
    } | {first.id}
    assert {str(row[0]) for row in rows} == expected_read_ids


def test_batch_read_deduplicates_references_and_enforces_bound(db_manager) -> None:
    context, repository = _context(db_manager)
    record = repository.create_memory(
        title="Bounded batch reference",
        content="Bounded batch reference content.",
        workspace_ids=["workspace-alpha"],
    )
    assert record is not None
    reference = f"mem-{record.memory_ref}"

    batch = read_memory_records_service(
        context,
        {"memory_refs": [reference, reference]},
    )
    assert batch["status"] == "ok"
    assert len(batch["records"]) == 1
    assert batch["budget"]["requested"] == 1

    with pytest.raises(ValueError, match="at most 20"):
        read_memory_records_service(
            context,
            {"memory_refs": [f"mem-{index}" for index in range(1, 22)]},
        )
