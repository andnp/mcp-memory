from __future__ import annotations

from pydantic import BaseModel, Field


class HealthPayload(BaseModel):
    status: str
    project_name: str | None = None
    memory_path: str | None = None
    db_path: str | None = None
    runtime_active: bool
    client_count: int
    task_queue_enabled: bool


class OverviewCounts(BaseModel):
    total: int
    by_type: dict[str, int] = Field(default_factory=dict)
    by_status: dict[str, int] = Field(default_factory=dict)


class CompactMemoryRecord(BaseModel):
    id: str
    title: str
    summary: str | None = None
    type: str
    status: str
    updated_at: str
    workspace_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class TaskStatusSummary(BaseModel):
    by_status: dict[str, int] = Field(default_factory=dict)
    failed_count: int


class StorageSummary(BaseModel):
    sqlite_bytes: int
    sqlite_path: str | None = None


class JournalSummary(BaseModel):
    pending_count: int


class OverviewPayload(BaseModel):
    memories: OverviewCounts
    recent_memories: list[CompactMemoryRecord] = Field(default_factory=list)
    tasks: TaskStatusSummary
    failed_tasks: list[dict] = Field(default_factory=list)
    journal: JournalSummary
    storage: StorageSummary


class TaskListPayload(BaseModel):
    tasks: list[dict] = Field(default_factory=list)


class MemoryDetailPayload(BaseModel):
    record: dict
    relationships: dict[str, list[dict]] = Field(default_factory=dict)
    superseded: list[dict] = Field(default_factory=list)
