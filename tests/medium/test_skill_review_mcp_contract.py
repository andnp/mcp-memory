from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from mcp.types import TextContent

from mcp_memory.application.skill_review_contract import (
    SkillReviewCommitRequest,
    SkillReviewDisposition,
    SkillReviewEvidence,
)
from mcp_memory.context import ApplicationContext
from mcp_memory.mcp.handlers import call_memory_tool
from mcp_memory.relational.repository import SQLiteRelationalMemoryRepository
from mcp_memory.utils.db import DatabaseManager


pytestmark = pytest.mark.medium


def _request() -> SkillReviewCommitRequest:
    """Build one valid serialized commit request for the MCP contract."""
    return SkillReviewCommitRequest(
        review_run_id=str(uuid4()),
        workspace_id="skill-manager-workspace",
        evidence=SkillReviewEvidence(
            skills=("delegate-code-review",),
            source_hashes={"delegate-code-review": "a" * 64},
            deployment_status="passed",
            deployment_receipt_hash="b" * 64,
        ),
        dispositions=(SkillReviewDisposition("mem-123", "verified", "Checked."),),
    )


def _context(tmp_path: Path) -> ApplicationContext:
    """Build the smallest real application context accepted by MCP dispatch."""
    manager = DatabaseManager(tmp_path / "memory.db")
    repository = SQLiteRelationalMemoryRepository(manager)
    repository.create_memory(
        memory_id="mem-123",
        title="Skill observation",
        content="The skill behavior is already implemented.",
        summary="Skill observation",
        workspace_ids=["skill-manager-workspace"],
        tags=["skill-observation"],
        memory_type="observation",
        metadata={"skill": "delegate-code-review", "review_status": "open"},
    )
    return ApplicationContext(
        db_manager=manager,
        repository=repository,
        workspace_id="skill-manager-workspace",
    )


@pytest.mark.asyncio
async def test_commit_skill_review_serializes_a_committed_response(tmp_path: Path) -> None:
    """Serialize the batch response through the public MCP tool dispatcher."""
    context = _context(tmp_path)
    request = _request()

    contents = await call_memory_tool(context, "commit_skill_review", request.as_dict())

    assert len(contents) == 1
    assert isinstance(contents[0], TextContent)
    payload = json.loads(contents[0].text)
    assert payload == {
        "dispositions": [{"memory_id": "mem-123", "resolution": "verified", "status": "stale"}],
        "idempotent": False,
        "protocol_version": 1,
        "review_run_id": request.review_run_id,
        "status": "committed",
    }


@pytest.mark.asyncio
async def test_commit_skill_review_serializes_invalid_input_without_mutation(tmp_path: Path) -> None:
    """Return a protocol error for invalid input before any record is changed."""
    context = _context(tmp_path)
    request = _request().as_dict()
    request["protocol_version"] = 2

    contents = await call_memory_tool(context, "commit_skill_review", request)

    assert json.loads(contents[0].text)["error"] == "invalid_arguments"
    record = context.repository.get_memory("mem-123")
    assert record is not None
    assert record.status == "active"


def _create_ledger_record(
    context: ApplicationContext,
    *,
    memory_id: str,
    workspace_id: str,
    memory_type: str = "observation",
    status: str = "active",
    tags: list[str] | None = None,
    updated_at: str,
) -> None:
    """Insert one explicit record for ledger predicate and ordering tests."""
    context.repository.create_memory(
        memory_id=memory_id,
        title=memory_id,
        content=f"content for {memory_id}",
        summary=f"summary for {memory_id}",
        workspace_ids=[workspace_id],
        tags=tags or ["skill-observation"],
        memory_type=memory_type,
        status=status,
        metadata={"skill": "task-observer"},
        created_at=updated_at,
        updated_at=updated_at,
    )


@pytest.mark.asyncio
async def test_skill_review_ledger_filters_and_orders_authoritative_records(tmp_path: Path) -> None:
    """Return only active observations carrying the required tag and workspace."""
    context = _context(tmp_path)
    _create_ledger_record(
        context,
        memory_id="valid-late",
        workspace_id="skill-manager-workspace",
        updated_at="2026-08-08T12:00:02+00:00",
    )
    _create_ledger_record(
        context,
        memory_id="valid-early",
        workspace_id="skill-manager-workspace",
        updated_at="2026-08-08T12:00:01+00:00",
    )
    _create_ledger_record(
        context,
        memory_id="wrong-status",
        workspace_id="skill-manager-workspace",
        status="stale",
        updated_at="2026-08-08T12:00:00+00:00",
    )
    _create_ledger_record(
        context,
        memory_id="wrong-type",
        workspace_id="skill-manager-workspace",
        memory_type="fact",
        updated_at="2026-08-08T12:00:00+00:00",
    )
    _create_ledger_record(
        context,
        memory_id="wrong-tag",
        workspace_id="skill-manager-workspace",
        tags=["other"],
        updated_at="2026-08-08T12:00:00+00:00",
    )
    _create_ledger_record(
        context,
        memory_id="wrong-workspace",
        workspace_id="other-workspace",
        updated_at="2026-08-08T12:00:00+00:00",
    )

    contents = await call_memory_tool(
        context,
        "get_skill_review_ledger",
        {"protocol_version": 1, "workspace_id": "skill-manager-workspace", "page_size": 10},
    )
    payload = json.loads(contents[0].text)

    assert [record["memory_id"] for record in payload["records"]] == [
        "valid-early",
        "valid-late",
        "mem-123",
    ]
    assert all("content" not in record for record in payload["records"])
    assert payload["total_count"] == 3
    assert payload["next_page_token"] is None


@pytest.mark.asyncio
async def test_skill_review_ledger_paginates_every_record_with_one_snapshot(tmp_path: Path) -> None:
    """Traverse all pages while retaining one deterministic ledger identity."""
    context = _context(tmp_path)
    for index in range(3):
        _create_ledger_record(
            context,
            memory_id=f"page-{index}",
            workspace_id="skill-manager-workspace",
            updated_at=f"2026-08-08T12:00:0{index}+00:00",
        )

    records: list[str] = []
    page_token = None
    snapshot_id = None
    while True:
        arguments: dict[str, object] = {
            "protocol_version": 1,
            "workspace_id": "skill-manager-workspace",
            "page_size": 1,
        }
        if page_token is not None:
            arguments["page_token"] = page_token
        payload = json.loads(
            (await call_memory_tool(context, "get_skill_review_ledger", arguments))[0].text,
        )
        records.extend(record["memory_id"] for record in payload["records"])
        snapshot_id = snapshot_id or payload["snapshot_id"]
        assert payload["snapshot_id"] == snapshot_id
        page_token = payload["next_page_token"]
        if page_token is None:
            break

    assert records == ["page-0", "page-1", "page-2", "mem-123"]


@pytest.mark.asyncio
async def test_skill_review_ledger_rejects_a_changed_snapshot_cursor(tmp_path: Path) -> None:
    """Fail closed when a record changes after the first ledger page."""
    context = _context(tmp_path)
    _create_ledger_record(
        context,
        memory_id="page-1",
        workspace_id="skill-manager-workspace",
        updated_at="2026-08-08T12:00:01+00:00",
    )
    first = json.loads(
        (
            await call_memory_tool(
                context,
                "get_skill_review_ledger",
                {"protocol_version": 1, "workspace_id": "skill-manager-workspace", "page_size": 1},
            )
        )[0].text,
    )
    record = context.repository.get_memory("mem-123")
    assert record is not None
    context.repository.update_memory(record.id, summary="changed")

    second = json.loads(
        (
            await call_memory_tool(
                context,
                "get_skill_review_ledger",
                {
                    "protocol_version": 1,
                    "workspace_id": "skill-manager-workspace",
                    "page_size": 1,
                    "page_token": first["next_page_token"],
                },
            )
        )[0].text,
    )

    assert second["error"] == "ledger_snapshot_changed"
