from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from mcp_memory.application.memory_use_cases import (
    RecordSkillObservationUseCase,
    ResolveSkillObservationUseCase,
)
from mcp_memory.application.ports import MemoryMutationDependencies
from mcp_memory.core.ports.memory import MemoryRecord


pytestmark = pytest.mark.small


def _observation(
    memory_id: str = "observation-1",
    *,
    memory_type: str = "observation",
    status: str = "active",
    title: str = "Observation title",
    content: str = "Observation content.",
    summary: str = "Observation summary.",
    workspace_ids: list[str] | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        title=title,
        content=content,
        summary=summary,
        type=memory_type,
        status=status,
        created_at="2026-08-08T12:00:00+00:00",
        updated_at="2026-08-08T12:00:00+00:00",
        read_count=0,
        access_score=0.0,
        last_accessed_at=None,
        last_surfaced_at=None,
        metadata=metadata or {},
        workspace_ids=workspace_ids or ["workspace-context"],
        tags=tags or ["skill-observation"],
        memory_ref=42,
    )


class _MemoryRepository:
    def __init__(self, records: list[MemoryRecord] | None = None, *, update: bool = True) -> None:
        self.records = {record.id: record for record in records or []}
        self.update_enabled = update

    def create_memory(
        self,
        *,
        title: str,
        content: str,
        summary: str,
        workspace_ids: list[str],
        tags: list[str],
        memory_type: str,
        status: str,
        metadata: dict[str, object],
    ) -> MemoryRecord:
        record = _observation(
            title=title,
            content=content,
            summary=summary,
            workspace_ids=workspace_ids,
            tags=tags,
            metadata=metadata,
        )
        self.records[record.id] = record
        return record

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        return self.records.get(memory_id)

    def update_memory(
        self,
        memory_id: str,
        *,
        status: str,
        metadata: dict[str, object],
    ) -> MemoryRecord | None:
        if not self.update_enabled:
            return None
        record = self.records.get(memory_id)
        if record is None:
            return None
        record.status = status
        record.metadata = metadata
        return record


def _context(
    repository: object = None, workspace_id: str | None = None
) -> MemoryMutationDependencies:
    return cast(
        MemoryMutationDependencies,
        SimpleNamespace(
            config=None,
            journal=None,
            repository=repository,
            task_queue=None,
            workspace_id=workspace_id,
        ),
    )


def _record_arguments(**overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "title": "A useful observation",
        "content": "The skill produced a useful result.",
        "summary": "Useful skill observation",
        "skill": "write-good-tests",
        "observation_kind": "improvement",
        "privacy_classification": "private",
    }
    arguments.update(overrides)
    return arguments


def test_record_skill_observation_uses_explicit_workspace_and_persists_tags_metadata() -> None:
    repository = _MemoryRepository()
    context = _context(repository, "workspace-context")

    result = RecordSkillObservationUseCase(context).execute(
        _record_arguments(workspace_id="  workspace-explicit  ")
    )

    assert result == {
        "status": "ok",
        "record": {
            "memory_id": next(iter(repository.records)),
            "memory_ref": 42,
            "title": "A useful observation",
            "summary": "Useful skill observation",
            "status": "active",
            "memory_type": "observation",
            "tags": [
                "skill-observation",
                "skill:write-good-tests",
                "kind:improvement",
                "privacy:private",
            ],
        },
    }
    record = next(iter(repository.records.values()))
    assert record.workspace_ids == ["workspace-explicit"]
    assert record.tags == [
        "skill-observation",
        "skill:write-good-tests",
        "kind:improvement",
        "privacy:private",
    ]
    assert record.metadata == {
        "skill": "write-good-tests",
        "observation_kind": "improvement",
        "privacy_classification": "private",
        "review_status": "open",
    }


def test_record_skill_observation_reports_missing_repository() -> None:
    context = _context(workspace_id="workspace-context")

    result = RecordSkillObservationUseCase(context).execute(_record_arguments())

    assert result == {"status": "error", "error": "repository_not_initialized"}


def test_record_skill_observation_reports_missing_workspace() -> None:
    context = _context(_MemoryRepository())

    result = RecordSkillObservationUseCase(context).execute(_record_arguments())

    assert result == {"status": "error", "error": "workspace_not_initialized"}


@pytest.mark.parametrize(
    ("resolution", "expected_status"),
    [("actioned", "archived"), ("deferred", "stale")],
)
def test_resolve_skill_observation_maps_resolution_to_status(
    resolution: str, expected_status: str
) -> None:
    record = _observation(metadata={"review_status": "open", "source": "test"})
    repository = _MemoryRepository([record])
    context = _context(repository, "workspace-context")

    result = ResolveSkillObservationUseCase(context).execute(
        {"memory_id": record.id, "resolution": resolution, "note": "Reviewed."}
    )

    assert result == {
        "status": "ok",
        "record": {
            "memory_id": record.id,
            "memory_ref": 42,
            "title": "Observation title",
            "summary": "Observation summary.",
            "status": expected_status,
            "memory_type": "observation",
            "tags": ["skill-observation"],
        },
    }
    assert record.status == expected_status
    assert record.metadata == {
        "review_status": resolution,
        "source": "test",
        "resolution_note": "Reviewed.",
    }


def test_resolve_skill_observation_reports_missing_repository() -> None:
    context = _context(workspace_id="workspace-context")

    result = ResolveSkillObservationUseCase(context).execute(
        {"memory_id": "observation-1", "resolution": "actioned", "note": "Reviewed."}
    )

    assert result == {"status": "error", "error": "repository_not_initialized"}


def test_resolve_skill_observation_rejects_ordinary_memory() -> None:
    record = _observation(memory_type="fact", tags=[])
    context = _context(_MemoryRepository([record]), "workspace-context")

    result = ResolveSkillObservationUseCase(context).execute(
        {"memory_id": record.id, "resolution": "actioned", "note": "Reviewed."}
    )

    assert result == {"status": "error", "error": "not_skill_observation"}


def test_resolve_skill_observation_returns_memory_not_found_when_update_fails() -> None:
    record = _observation()
    repository = _MemoryRepository([record], update=False)
    context = _context(repository, "workspace-context")

    result = ResolveSkillObservationUseCase(context).execute(
        {"memory_id": record.id, "resolution": "deferred", "note": "Later."}
    )

    assert result == {"status": "error", "error": "memory_not_found"}
