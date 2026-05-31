from __future__ import annotations

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.maintenance_schedule import SWEEPER_TASK_NAME
from mcp_memory.core.task_handlers.maintenance_housekeeping import handle_sweeper_task
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.relational.repository import RelationalMemoryRepository


pytestmark = pytest.mark.small


def _task(workspace_id: str = "workspace-a") -> TaskRecord:
    return TaskRecord(
        id="sweeper-lineage-task",
        task_name=SWEEPER_TASK_NAME,
        data={"workspace_id": workspace_id},
        workspace_id=workspace_id,
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


def test_sweeper_reports_lineage_hotspots_and_prunes_sqlite_dead_metadata(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    child_ids = [f"child-{index:02d}" for index in range(12)]
    original = repository.create_memory(
        title="Large split original",
        content="Original content that was split but retained active.",
        workspace_ids=["workspace-a"],
        metadata={
            "split_child_memory_ids": child_ids,
            "split_sibling_memory_ids": [f"sibling-{index:02d}" for index in range(12)],
            "split_group_id": "split-group-a",
            "custom_keep": "still useful",
        },
    )
    assert original is not None
    for index in range(20):
        target = repository.create_memory(
            title=f"Related memory {index:02d}",
            content="Related content.",
            workspace_ids=["workspace-a"],
        )
        assert target is not None
        repository.add_link(original.id, target.id, "AMENDS", "Auto-linked from shared tags (lineage)")

    result = handle_sweeper_task(
        ApplicationContext(db_manager=db_manager, repository=repository, workspace_id="workspace-a"),
        _task(),
    )

    assert result["gc_metadata_records"] == 1
    assert result["lineage_hotspots"]["active_split_original_records"] == 1
    assert result["lineage_hotspots"]["oversized_lineage_metadata_records"] == 1
    assert result["lineage_hotspots"]["high_relationship_density_records"] == 1
    assert result["lineage_hotspots"]["examples"] == [
        {
            "memory_id": original.id,
            "reasons": [
                "active_split_original",
                "oversized_lineage_metadata",
                "high_relationship_density",
            ],
            "metadata_bytes": result["lineage_hotspots"]["examples"][0]["metadata_bytes"],
            "relationship_count": 20,
        }
    ]

    refreshed = repository.get_memory(original.id)
    assert refreshed is not None
    assert refreshed.metadata == {"custom_keep": "still useful"}


def test_sweeper_lineage_hotspots_are_empty_for_clean_memory(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    record = repository.create_memory(
        title="Focused memory",
        content="One compact durable takeaway.",
        workspace_ids=["workspace-a"],
        metadata={"custom_keep": "small"},
    )
    assert record is not None

    result = handle_sweeper_task(
        ApplicationContext(db_manager=db_manager, repository=repository, workspace_id="workspace-a"),
        _task(),
    )

    assert result["gc_metadata_records"] == 0
    assert result["lineage_hotspots"] == {
        "active_split_original_records": 0,
        "oversized_lineage_metadata_records": 0,
        "high_relationship_density_records": 0,
        "examples": [],
    }
