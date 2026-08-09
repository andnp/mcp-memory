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
