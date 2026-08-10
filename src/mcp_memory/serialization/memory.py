from __future__ import annotations

from mcp_memory.core.ports.tasks import TaskRecord
from mcp_memory.management.models import CompactMemoryRecord
from mcp_memory.core.ports.memory import (
    MemoryLink,
    MemoryRecord,
    format_memory_ref,
)

from mcp_memory.relational.search import RelationalSearchResult


def memory_record_payload(record: MemoryRecord) -> dict:
    return {
        "id": record.id,
        "memory_ref": format_memory_ref(getattr(record, "memory_ref", None)),
        "title": record.title,
        "content": record.content,
        "summary": record.summary,
        "type": record.type,
        "status": record.status,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "read_count": record.read_count,
        "access_score": record.access_score,
        "last_accessed_at": record.last_accessed_at,
        "last_surfaced_at": record.last_surfaced_at,
        "metadata": record.metadata,
        "workspace_ids": list(record.workspace_ids),
        "tags": list(record.tags),
    }


def internal_mutation_record_payload(record: MemoryRecord) -> dict:
    """Compact receipt for internal mutation follow-up."""
    return {
        "id": record.id,
        "memory_ref": format_memory_ref(getattr(record, "memory_ref", None)),
        "title": record.title,
        "summary": record.summary,
        "type": record.type,
        "status": record.status,
        "updated_at": record.updated_at,
        "workspace_ids": list(record.workspace_ids),
        "tags": list(record.tags),
    }


_AUTO_LINK_PREFIX = "Auto-linked from shared tags"


def agent_memory_record_payload(record: MemoryRecord) -> dict:
    """Minimal record payload for agent consumption.

    Contains only the fields a model needs to understand the memory:
    memory_ref for referencing, title for identity, content for the body.
    Summary is dropped because it's auto-generated from content.
    Everything else (type, status, timestamps, tags, internal fields)
    burns context without adding agent-useful information.
    """
    return {
        "memory_ref": format_memory_ref(getattr(record, "memory_ref", None)) or record.id,
        "title": record.title,
        "content": record.content,
    }


def agent_memory_record_payload_with_metadata(record: MemoryRecord) -> dict:
    """Agent-facing record payload with explicit audit metadata included.

    This is intentionally opt-in for MCP tools: metadata can be useful for
    maintenance/debugging, but routine reads should not spend tokens on lineage,
    counters, workspace routing, or cache bookkeeping.
    """
    return agent_memory_record_payload(record) | {
        "metadata": record.metadata,
        "workspace_ids": list(record.workspace_ids),
    }


def agent_link_payload(link: MemoryLink) -> dict:
    """Link payload for agent consumption.

    Compresses verbose auto-link context strings (which merely repeat the
    tag list already present on the record) to a short sentinel.
    """
    context = link.context
    if context and context.startswith(_AUTO_LINK_PREFIX):
        context = "auto-linked"
    return {
        "source_id": link.source_id,
        "target_id": link.target_id,
        "link_type": link.link_type,
        "context": context,
    }


def compact_memory_record_payload(
    record: MemoryRecord,
) -> CompactMemoryRecord:
    return CompactMemoryRecord(
        id=record.id,
        title=record.title,
        summary=record.summary,
        type=record.type,
        status=record.status,
        updated_at=record.updated_at,
        read_count=record.read_count,
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
        "memory_ref": format_memory_ref(getattr(result, "memory_ref", None)),
        "title": result.title,
        "summary": result.summary,
        "memory_type": result.memory_type,
        "status": result.status,
        "created_at": getattr(result, "created_at", None),
        "updated_at": getattr(result, "updated_at", None),
        "tags": list(result.tags),
        "workspace_ids": list(result.workspace_ids),
        "score": result.score,
    }


def search_result_payload_compact(result: RelationalSearchResult) -> dict:
    """Minimal search result for agent consumption.

    Returns the fields needed to decide whether to read a record, including
    objective status and temporal metadata.
    """
    return {
        "memory_ref": format_memory_ref(getattr(result, "memory_ref", None)) or result.memory_id,
        "title": result.title,
        "summary": result.summary,
        "status": getattr(result, "status", None),
        "created_at": getattr(result, "created_at", None),
        "updated_at": getattr(result, "updated_at", None),
    }


def search_result_payload_with_debug_fields(result: RelationalSearchResult) -> dict:
    return search_result_payload(result) | {
        "workspace_ids": list(result.workspace_ids),
        "score": result.score,
    }


def task_payload(task: TaskRecord) -> dict:
    return {
        "id": task.id,
        "task_name": task.task_name,
        "trigger": task.data.get("trigger") if isinstance(task.data, dict) else None,
        "strategy": task.data.get("strategy")
        if isinstance(task.data.get("strategy"), str)
        else None,
        "grouping_strategy": task.data.get("grouping_strategy")
        if isinstance(task.data.get("grouping_strategy"), str)
        else None,
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
        "subprocess_pid": task.subprocess_pid,
        "active_request_id": task.active_request_id,
        "cancellation_requested_at": task.cancellation_requested_at,
        "cancelled_at": task.cancelled_at,
        "cancellation_reason": task.cancellation_reason,
        "cancelled_by": task.cancelled_by,
    }
