from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from mcp_memory.core.curation_direct_mcp import _direct_curator_prompt
from mcp_memory.core.curation_validation import CurationMutationBudget
from mcp_memory.core.task_handlers.constants import CURATOR_TASK_NAME
from mcp_memory.core.task_handlers.curator_support import (
    curator_seed_payload_item,
    retrieval_friction_flags,
)
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.relational.repository import MemoryRecord

pytestmark = pytest.mark.small


def _record(
    memory_id: str,
    *,
    title: str,
    content: str,
    summary: str = "Focused durable summary.",
    memory_type: str = "fact",
) -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        title=title,
        content=content,
        summary=summary,
        type=memory_type,
        status="active",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        read_count=0,
        access_score=0.0,
        last_accessed_at=None,
        last_surfaced_at=None,
        workspace_ids=["workspace-benchmark"],
        tags=["curation"],
        metadata={},
    )


def _task() -> TaskRecord:
    return TaskRecord(
        id="curator-durability-benchmark",
        task_name=CURATOR_TASK_NAME,
        data={},
        workspace_id="workspace-benchmark",
        status="running",
        priority=100,
        retries_count=0,
        max_retries=3,
        created_at=0.0,
        updated_at=0.0,
        available_at=0.0,
        claimed_at=0.0,
        started_at=0.0,
        completed_at=None,
        last_error=None,
    )


@pytest.mark.parametrize(
    ("title", "content", "expected_flag"),
    (
        (
            "2026-01-15 work log",
            "Worked on the migration and recorded the progress update.",
            "dated_work_log",
        ),
        (
            "Migration status",
            "Completed the migration; ran tests with a temporary workaround.",
            "task_completion_residue",
        ),
    ),
)
def test_transient_work_and_status_residue_are_advisory_risks(
    title: str, content: str, expected_flag: str
) -> None:
    """Classify transient work history without making retention an automatic action."""
    record = _record(title.lower().replace(" ", "-"), title=title, content=content, memory_type="journal")

    flags = set(retrieval_friction_flags(record))
    payload = curator_seed_payload_item(record)

    assert expected_flag in flags
    assert expected_flag in payload["retrieval_friction_flags"]
    assert payload["retrieval_friction_flags"] == retrieval_friction_flags(record)


@pytest.mark.parametrize(
    "content",
    (
        "The deadline is 2026-01-15 for the release candidate.",
        "The 2026-01-15 incident postmortem records the root cause.",
        "Historical decision: retain the compatibility boundary after the 2026-01-15 release.",
    ),
)
def test_durable_dates_are_not_misclassified_as_work_logs(content: str) -> None:
    """Preserve dates whose surrounding language carries durable operational meaning."""
    record = _record("durable-date", title="Durable dated record", content=content)

    flags = set(retrieval_friction_flags(record))

    assert "dated_work_log" not in flags
    assert not {"task_completion_residue", "transient_execution_detail"}.intersection(flags)


def test_mixed_content_exposes_both_cleanup_risk_and_durable_claim_guardrail() -> None:
    """Surface mixed-content risk while retaining the durable decision for curator review."""
    record = _record(
        "mixed-record",
        title="Migration decision and status",
        content=(
            "Decision: keep the boundary. Completed the migration after debugging; "
            "ran tests with a temporary workaround."
        ),
    )

    flags = set(retrieval_friction_flags(record))

    assert {"task_completion_residue", "transient_execution_detail", "mixed_durability_content"} <= flags


def test_already_durable_record_has_no_durability_risk_flags() -> None:
    """Treat a focused durable record as a no-op candidate for retention purposes."""
    record = _record(
        "durable-no-op",
        title="Release deadline decision",
        content="The release deadline is 2026-01-15; retain this planning constraint.",
    )

    flags = set(retrieval_friction_flags(record))
    payload = curator_seed_payload_item(record)

    assert not {"dated_work_log", "task_completion_residue", "transient_execution_detail", "mixed_durability_content"}.intersection(flags)
    assert payload["retrieval_friction_flags"] == retrieval_friction_flags(record)


def test_boundary_date_without_work_log_language_is_not_transient() -> None:
    """Avoid flagging a date alone when no transient work-log signal is present."""
    record = _record(
        "date-only",
        title="Migration note",
        content="Migration target: 2026-01-15.",
    )

    assert "dated_work_log" not in retrieval_friction_flags(record)


def test_direct_prompt_requires_guarded_curator_judgment() -> None:
    """Keep direct-agent instructions aligned with benchmark risk classifications."""
    prompt = _direct_curator_prompt(
        task=cast(TaskRecord, SimpleNamespace(id="curator-durability-benchmark")),
        seed_records=[],
        campaign_hypothesis=None,
        mutation_budget=CurationMutationBudget(max_accepted_mutations=2),
    )

    assert "distinguish durable content" in prompt
    assert "transient content" in prompt
    assert "mixed content" in prompt
    assert "Flag work logs, status updates, task-complete summaries, and execution residue" in prompt
    assert "Never archive or delete solely due to age, date, or access" in prompt
    assert "preserve every durable claim and its meaningful qualifiers" in prompt
    assert "Omit records that need no change" in prompt
