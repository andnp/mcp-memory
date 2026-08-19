from __future__ import annotations

from mcp_memory.context import ApplicationContext
from mcp_memory.core.ports.work_items import compatibility_group_families
from mcp_memory.toolkit.arg_validation import optional_positive_int, optional_string, require_string, string_list


def normalize_internal_workspace_scope(workspace_id: str | None) -> str | None:
    if workspace_id is None:
        return None
    normalized = workspace_id.strip()
    if not normalized:
        return None
    lowered = normalized.lower()
    if lowered == "global":
        return None
    if lowered in {"all", "any"}:
        return "*"
    return normalized


def task_workspace_scope(ctx: ApplicationContext, task_id: str | None) -> str | None:
    if not task_id:
        return None
    task_queue = getattr(ctx, "task_queue", None)
    if task_queue is None:
        return None
    try:
        task = task_queue.get_task(task_id)
    except ValueError:
        return None
    task_workspace = task.data.get("workspace_id")
    if isinstance(task_workspace, str):
        return normalize_internal_workspace_scope(task_workspace)
    if isinstance(task.workspace_id, str):
        return normalize_internal_workspace_scope(task.workspace_id)
    return None


def resolve_internal_workspace_scope(
    ctx: ApplicationContext,
    arguments: dict,
    *,
    task_id_key: str = "task_id",
) -> str | None:
    explicit_workspace_id = normalize_internal_workspace_scope(optional_string(arguments, "workspace_id"))
    if "workspace_id" in arguments:
        return explicit_workspace_id
    return task_workspace_scope(ctx, optional_string(arguments, task_id_key))


def internal_get_work_batch_service(ctx: ApplicationContext, arguments: dict) -> dict:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None:
        return {"status": "error", "error": "work_items_not_initialized"}

    task_id = require_string(arguments, "task_id")
    family_key = require_string(arguments, "family_key")
    execution_lane = require_string(arguments, "execution_lane")
    workspace_id = resolve_internal_workspace_scope(ctx, arguments)
    limit = optional_positive_int(arguments, "limit", 20)
    lease_ttl_seconds = float(optional_positive_int(arguments, "lease_ttl_seconds", 1800))
    records = work_items.claim_batch(
        family_key=family_key,
        execution_lane=execution_lane,
        lease_owner=task_id,
        limit=limit,
        workspace_id=workspace_id,
        lease_ttl_seconds=lease_ttl_seconds,
    )
    return {
        "status": "ok",
        "task_id": task_id,
        "family_key": family_key,
        "execution_lane": execution_lane,
        "claimed_count": len(records),
        "limit": limit,
        "records": [
            {
                "id": record.id,
                "family_key": record.family_key,
                "execution_lane": record.execution_lane,
                "workspace_id": record.workspace_id,
                "payload": record.payload,
                "status": record.status,
                "priority": record.priority,
                "attempt_count": record.attempt_count,
                "lease_owner": record.lease_owner,
                "lease_expires_at": record.lease_expires_at,
            }
            for record in records
        ],
    }


def internal_get_compatible_work_batch_service(ctx: ApplicationContext, arguments: dict) -> dict:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None:
        return {"status": "error", "error": "work_items_not_initialized"}

    task_id = require_string(arguments, "task_id")
    compatibility_group = require_string(arguments, "compatibility_group")
    execution_lane = require_string(arguments, "execution_lane")
    workspace_id = resolve_internal_workspace_scope(ctx, arguments)
    limit = optional_positive_int(arguments, "limit", 20)
    lease_ttl_seconds = float(optional_positive_int(arguments, "lease_ttl_seconds", 1800))
    family_keys = compatibility_group_families(
        compatibility_group,
        allowed_families=string_list(arguments, "allowed_families"),
    )
    records = work_items.claim_compatible_batch(
        family_keys=family_keys,
        execution_lane=execution_lane,
        lease_owner=task_id,
        limit=limit,
        workspace_id=workspace_id,
        lease_ttl_seconds=lease_ttl_seconds,
    )
    return {
        "status": "ok",
        "task_id": task_id,
        "compatibility_group": compatibility_group,
        "family_keys": list(family_keys),
        "execution_lane": execution_lane,
        "claimed_count": len(records),
        "limit": limit,
        "records": [
            {
                "id": record.id,
                "family_key": record.family_key,
                "execution_lane": record.execution_lane,
                "workspace_id": record.workspace_id,
                "payload": record.payload,
                "status": record.status,
                "priority": record.priority,
                "attempt_count": record.attempt_count,
                "lease_owner": record.lease_owner,
                "lease_expires_at": record.lease_expires_at,
            }
            for record in records
        ],
    }


def internal_heartbeat_work_item_service(ctx: ApplicationContext, arguments: dict) -> dict:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None:
        return {"status": "error", "error": "work_items_not_initialized"}

    task_id = require_string(arguments, "task_id")
    work_item_id = require_string(arguments, "work_item_id")
    lease_ttl_seconds = float(optional_positive_int(arguments, "lease_ttl_seconds", 1800))
    record = work_items.heartbeat_item(
        work_item_id,
        lease_owner=task_id,
        lease_ttl_seconds=lease_ttl_seconds,
    )
    return {
        "status": "ok",
        "task_id": task_id,
        "record": {
            "id": record.id,
            "family_key": record.family_key,
            "execution_lane": record.execution_lane,
            "workspace_id": record.workspace_id,
            "payload": record.payload,
            "status": record.status,
            "priority": record.priority,
            "attempt_count": record.attempt_count,
            "lease_owner": record.lease_owner,
            "lease_expires_at": record.lease_expires_at,
        },
    }


def internal_complete_work_item_service(ctx: ApplicationContext, arguments: dict) -> dict:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None:
        return {"status": "error", "error": "work_items_not_initialized"}

    work_item_id = require_string(arguments, "work_item_id")
    record = work_items.complete_item(work_item_id)
    return {
        "status": "ok",
        "record": {
            "id": record.id,
            "status": record.status,
            "completed_at": record.completed_at,
            "last_error": record.last_error,
        },
    }


def internal_defer_work_item_service(ctx: ApplicationContext, arguments: dict) -> dict:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None:
        return {"status": "error", "error": "work_items_not_initialized"}

    work_item_id = require_string(arguments, "work_item_id")
    error = require_string(arguments, "error")
    retry_delay_seconds = float(optional_positive_int(arguments, "retry_delay_seconds", 0))
    record = work_items.defer_item(
        work_item_id,
        error=error,
        retry_delay_seconds=retry_delay_seconds,
    )
    return {
        "status": "ok",
        "record": {
            "id": record.id,
            "status": record.status,
            "available_at": record.available_at,
            "last_error": record.last_error,
        },
    }


def internal_release_work_item_service(ctx: ApplicationContext, arguments: dict) -> dict:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None:
        return {"status": "error", "error": "work_items_not_initialized"}

    work_item_id = require_string(arguments, "work_item_id")
    record = work_items.release_item(work_item_id)
    return {
        "status": "ok",
        "record": {
            "id": record.id,
            "status": record.status,
            "lease_owner": record.lease_owner,
            "lease_expires_at": record.lease_expires_at,
        },
    }
