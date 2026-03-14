from __future__ import annotations

from mcp_memory.core.tasks import TaskRecord
from mcp_memory.management.models import CompactMemoryRecord
from mcp_memory.relational.repository import MemoryLink, RelationalMemoryRecord
from mcp_memory.relational.search import RelationalSearchResult


def memory_record_payload(record: RelationalMemoryRecord) -> dict:
    return {
        "id": record.id,
        "title": record.title,
        "content": record.content,
        "summary": record.summary,
        "type": record.type,
        "status": record.status,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "access_score": record.access_score,
        "last_accessed_at": record.last_accessed_at,
        "last_surfaced_at": record.last_surfaced_at,
        "metadata": record.metadata,
        "workspace_ids": list(record.workspace_ids),
        "tags": list(record.tags),
    }


def compact_memory_record_payload(record: RelationalMemoryRecord) -> CompactMemoryRecord:
    return CompactMemoryRecord(
        id=record.id,
        title=record.title,
        summary=record.summary,
        type=record.type,
        status=record.status,
        updated_at=record.updated_at,
        workspace_ids=list(record.workspace_ids),
        tags=list(record.tags),
    )


def link_payload(link: MemoryLink) -> dict:
    return {
        "source_id": link.source_id,
        "target_id": link.target_id,
        "link_type": link.link_type,
        "context": link.context,
    }


def search_result_payload(result: RelationalSearchResult) -> dict:
    return {
        "memory_id": result.memory_id,
        "title": result.title,
        "summary": result.summary,
        "memory_type": result.memory_type,
        "status": result.status,
        "tags": list(result.tags),
        "workspace_ids": list(result.workspace_ids),
        "score": result.score,
    }


def task_payload(task: TaskRecord) -> dict:
    return {
        "id": task.id,
        "task_name": task.task_name,
        "workspace_id": task.workspace_id,
        "status": task.status,
        "priority": task.priority,
        "retries_count": task.retries_count,
        "max_retries": task.max_retries,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "available_at": task.available_at,
        "claimed_at": task.claimed_at,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
        "last_error": task.last_error,
    }