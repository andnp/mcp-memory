from __future__ import annotations

from datetime import UTC, datetime

from mcp_memory.context import ApplicationContext
from mcp_memory.core.ingress_evidence import IngressActionReceipt, IngressReceiptStatus
from mcp_memory.core.ingress_identity import action_identity, canonical_payload_digest
from mcp_memory.core.ports.ingress import IngressActionReceiptIdentityConflictError
from mcp_memory.core.ingest_claim_lifecycle import (
    _normalize_ingest_entry_ids,
    _record_ingest_tool_invocation,
    _record_successful_ingest_entry_dispositions,
    _record_successful_ingest_entry_ids,
    _record_touched_memory_ids,
)
from mcp_memory.mcp.internal_ingest_keys import (
    INGEST_ENTRY_DISPOSITIONS_TASK_DATA_KEY,
    INGEST_HANDLED_ENTRY_IDS_TASK_DATA_KEY,
    INGEST_TOOL_INVOCATIONS_TASK_DATA_KEY,
)
from mcp_memory.core.ingest_provenance import (
    build_ingest_appended_metadata,
    build_ingest_created_metadata,
)
from mcp_memory.core.task_handlers.ingest import build_next_ingest_batch_payload
from mcp_memory.mcp.internal_service_support import (
    _append_content,
    _memory_write_quality_error,
    _memory_write_quality_warnings,
    _merge_memory_metadata,
    _normalize_tags,
)
from mcp_memory.mcp.validation import (
    optional_object,
    optional_string,
    require_string,
    string_list,
    validate_ingest_mutation_payload,
)
from mcp_memory.serialization import internal_mutation_record_payload
from mcp_memory.storage.ingress_mutation_transaction import IngressMutationResult


__all__ = [
    "INGEST_ENTRY_DISPOSITIONS_TASK_DATA_KEY",
    "INGEST_HANDLED_ENTRY_IDS_TASK_DATA_KEY",
    "INGEST_TOOL_INVOCATIONS_TASK_DATA_KEY",
    "internal_append_to_existing_memory_for_ingest_service",
    "internal_create_memory_record_for_ingest_service",
    "internal_get_next_ingest_batch_service",
]

def internal_get_next_ingest_batch_service(ctx: ApplicationContext, arguments: dict) -> dict:
    return build_next_ingest_batch_payload(ctx, arguments)


def internal_append_to_existing_memory_for_ingest_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    try:
        memory_id = require_string(arguments, "memory_id")
        content = require_string(arguments, "content")
        task_id = require_string(arguments, "task_id")
        batch_id = optional_string(arguments, "batch_id")
        entry_ids = _normalize_ingest_entry_ids(arguments, "entry_ids")
        validate_ingest_mutation_payload(entry_ids=entry_ids, content=content)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": "invalid_ingest_payload", "detail": str(exc)}
    if _ingress_requires_atomic_boundary(ctx, batch_id=batch_id):
        return {"status": "error", "error": "atomic_boundary_required"}
    if _ingress_is_enforced(ctx):
        assert batch_id is not None
        return _execute_atomic_append(
            ctx,
            batch_id=batch_id,
            memory_id=memory_id,
            content=content,
            task_id=task_id,
            entry_ids=entry_ids,
            arguments=arguments,
        )
    record = ctx.repository.get_memory(memory_id)
    if record is None:
        return {"status": "error", "error": "memory_not_found"}
    if record.status != "active":
        return {"status": "error", "error": "memory_not_active"}

    summary = optional_string(arguments, "summary") if "summary" in arguments else record.summary
    quality_error = _memory_write_quality_error(
        title=record.title,
        content=content,
        summary=optional_string(arguments, "summary"),
    )
    if quality_error is not None:
        return {"status": "error", "error": "low_value_memory_rejected", "detail": quality_error}

    tags = string_list(arguments, "tags") if "tags" in arguments else []
    workspace_ids = string_list(arguments, "workspace_ids") if "workspace_ids" in arguments else []
    metadata_override = optional_object(arguments, "metadata") or {}
    merged_content = _append_content(record.content, content)
    merged_tags = _normalize_tags([*record.tags, *tags])
    warnings = _memory_write_quality_warnings(
        summary=summary,
        memory_type=record.type,
        tags=merged_tags,
        content=merged_content,
    )
    merged_metadata = _merge_memory_metadata(
        record.metadata,
        metadata_override,
    )
    merged_metadata = build_ingest_appended_metadata(
        task_id=task_id,
        entry_ids=entry_ids,
        metadata=merged_metadata,
    )
    updated = ctx.repository.update_memory(
        memory_id,
        content=merged_content,
        summary=optional_string(arguments, "summary") if "summary" in arguments else None,
        tags=merged_tags,
        workspace_ids=sorted({*record.workspace_ids, *workspace_ids}),
        metadata=merged_metadata,
    )
    if updated is None:
        return {"status": "error", "error": "memory_not_found"}
    _record_successful_ingest_entry_ids(ctx, task_id=task_id, entry_ids=entry_ids)
    _record_successful_ingest_entry_dispositions(
        ctx,
        task_id=task_id,
        entry_dispositions=[
            {
                "entry_id": entry_id,
                "disposition": "appended",
                "memory_id": updated.id,
                "memory_title": updated.title,
            }
            for entry_id in entry_ids
        ],
    )
    _record_ingest_tool_invocation(
        ctx,
        task_id=task_id,
        tool_name="internal_ingest_append_memory",
        mutation=True,
    )
    _record_touched_memory_ids(ctx, task_id=task_id, memory_ids=[updated.id])
    ingress_metadata = _persist_ingress_action_receipt(
        ctx,
        batch_id=batch_id,
        operation="append",
        entry_ids=entry_ids,
        target_ids=[updated.id],
        payload={
            "content": content,
            "metadata": metadata_override,
            "summary": summary,
            "tags": merged_tags,
            "workspace_ids": sorted({*record.workspace_ids, *workspace_ids}),
        },
    )
    payload: dict[str, object] = {"status": "ok", "record": internal_mutation_record_payload(updated), "handled_entry_ids": entry_ids}
    payload.update(ingress_metadata)
    if warnings:
        payload["warnings"] = warnings
    return payload


def internal_create_memory_record_for_ingest_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    try:
        task_id = require_string(arguments, "task_id")
        batch_id = optional_string(arguments, "batch_id")
        entry_ids = _normalize_ingest_entry_ids(arguments, "entry_ids")
        title = require_string(arguments, "title")
        content = require_string(arguments, "content")
        validate_ingest_mutation_payload(entry_ids=entry_ids, content=content, title=title)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": "invalid_ingest_payload", "detail": str(exc)}
    if _ingress_requires_atomic_boundary(ctx, batch_id=batch_id):
        return {"status": "error", "error": "atomic_boundary_required"}
    if _ingress_is_enforced(ctx):
        assert batch_id is not None
        return _execute_atomic_create(
            ctx,
            batch_id=batch_id,
            task_id=task_id,
            entry_ids=entry_ids,
            title=title,
            content=content,
            arguments=arguments,
        )
    workspace_ids = string_list(arguments, "workspace_ids") or ([ctx.workspace_id] if ctx.workspace_id is not None else ["workspace-unknown"])
    metadata_override = optional_object(arguments, "metadata") or {}
    summary = optional_string(arguments, "summary")
    memory_type = optional_string(arguments, "memory_type") or "observation"
    tags = _normalize_tags([*string_list(arguments, "tags")])
    quality_error = _memory_write_quality_error(title=title, content=content, summary=summary)
    if quality_error is not None:
        return {"status": "error", "error": "low_value_memory_rejected", "detail": quality_error}
    warnings = _memory_write_quality_warnings(summary=summary, memory_type=memory_type, tags=tags, content=content)
    record = ctx.repository.create_memory(
        title=title,
        content=content,
        summary=summary,
        memory_type=memory_type,
        status=optional_string(arguments, "status") or "active",
        workspace_ids=workspace_ids,
        tags=tags,
        metadata=build_ingest_created_metadata(
            task_id=task_id,
            entry_ids=entry_ids,
            metadata=metadata_override,
        ),
    )
    assert record is not None
    _record_successful_ingest_entry_ids(ctx, task_id=task_id, entry_ids=entry_ids)
    _record_successful_ingest_entry_dispositions(
        ctx,
        task_id=task_id,
        entry_dispositions=[
            {
                "entry_id": entry_id,
                "disposition": "created",
                "memory_id": record.id,
                "memory_title": record.title,
            }
            for entry_id in entry_ids
        ],
    )
    _record_ingest_tool_invocation(
        ctx,
        task_id=task_id,
        tool_name="internal_ingest_create_memory",
        mutation=True,
    )
    _record_touched_memory_ids(ctx, task_id=task_id, memory_ids=[record.id])
    ingress_metadata = _persist_ingress_action_receipt(
        ctx,
        batch_id=batch_id,
        operation="create",
        entry_ids=entry_ids,
        target_ids=[],
        payload={
            "content": content,
            "metadata": metadata_override,
            "memory_type": memory_type,
            "status": optional_string(arguments, "status") or "active",
            "summary": summary,
            "tags": tags,
            "title": title,
            "workspace_ids": workspace_ids,
        },
    )
    payload: dict[str, object] = {"status": "ok", "record": internal_mutation_record_payload(record), "handled_entry_ids": entry_ids}
    payload.update(ingress_metadata)
    if warnings:
        payload["warnings"] = warnings
    return payload


def _persist_ingress_action_receipt(
    ctx: ApplicationContext,
    *,
    batch_id: str | None,
    operation: str,
    entry_ids: list[int],
    target_ids: list[str],
    payload: dict[str, object],
) -> dict[str, str]:
    mode = getattr(getattr(ctx, "config", None), "ingress_evidence_mode", "off")
    repository = getattr(ctx, "ingress_action_receipts", None)
    if batch_id is None or repository is None or mode not in {"shadow", "detect"}:
        return {}
    identity = action_identity(operation, [str(entry_id) for entry_id in entry_ids], target_ids)
    digest = canonical_payload_digest(payload)
    receipt = IngressActionReceipt(
        action_id=identity.action_id,
        batch_id=batch_id,
        operation=operation,
        entry_ids=identity.entry_ids,
        target_ids=identity.target_ids,
        canonical_payload_digest=digest,
        status=IngressReceiptStatus.APPLIED_UNVERIFIED,
        mutation_evidence_id=None,
        created_at=datetime.now(UTC).isoformat(),
    )
    try:
        repository.save(receipt)
    except Exception:
        return {}
    return {"ingress_action_id": identity.action_id, "ingress_payload_digest": digest}


class _AtomicIngressServiceError(RuntimeError):
    def __init__(self, error: str, detail: str | None = None) -> None:
        self.error = error
        self.detail = detail
        super().__init__(detail or error)


def _execute_atomic_append(
    ctx: ApplicationContext,
    *,
    batch_id: str,
    memory_id: str,
    content: str,
    task_id: str,
    entry_ids: list[int],
    arguments: dict,
) -> dict:
    transaction = ctx.ingress_mutation_transaction
    requested_summary = optional_string(arguments, "summary") if "summary" in arguments else None
    requested_tags = string_list(arguments, "tags") if "tags" in arguments else []
    requested_workspace_ids = string_list(arguments, "workspace_ids") if "workspace_ids" in arguments else []
    metadata_override = optional_object(arguments, "metadata") or {}
    request_payload = {
        "content": content,
        "metadata": metadata_override,
        "summary": requested_summary,
        "tags": _normalize_tags(requested_tags),
        "workspace_ids": sorted(set(requested_workspace_ids)),
    }

    def apply(domain) -> IngressMutationResult:
        record = domain.get_memory(memory_id)
        if record is None:
            raise _AtomicIngressServiceError("memory_not_found")
        if record.status != "active":
            raise _AtomicIngressServiceError("memory_not_active")
        quality_error = _memory_write_quality_error(
            title=record.title,
            content=content,
            summary=requested_summary,
        )
        if quality_error is not None:
            raise _AtomicIngressServiceError("low_value_memory_rejected", quality_error)

        merged_tags = _normalize_tags([*record.tags, *requested_tags])
        merged_metadata = _merge_memory_metadata(record.metadata, metadata_override)
        merged_metadata = build_ingest_appended_metadata(
            task_id=task_id,
            entry_ids=entry_ids,
            metadata=merged_metadata,
        )
        updated = domain.append_memory(
            memory_id,
            content,
            summary=record.summary if "summary" not in arguments else requested_summary,
            tags=merged_tags,
            workspace_ids=sorted({*record.workspace_ids, *requested_workspace_ids}),
            metadata=merged_metadata,
        )
        return IngressMutationResult("append", [updated.id])

    try:
        receipt = transaction.execute(
            batch_id=batch_id,
            operation="append",
            entry_ids=[str(entry_id) for entry_id in entry_ids],
            target_ids=[memory_id],
            payload=request_payload,
            apply=apply,
            journal_task_id=task_id,
        )
    except _AtomicIngressServiceError as exc:
        payload: dict[str, object] = {"status": "error", "error": exc.error}
        if exc.detail is not None:
            payload["detail"] = exc.detail
        return payload
    except IngressActionReceiptIdentityConflictError as exc:
        return {"status": "error", "error": "ingress_action_identity_collision", "detail": str(exc)}

    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}
    updated = ctx.repository.get_memory(memory_id)
    if updated is None:
        return {"status": "error", "error": "memory_not_found"}
    _record_successful_ingest_entry_ids(ctx, task_id=task_id, entry_ids=entry_ids)
    _record_successful_ingest_entry_dispositions(
        ctx,
        task_id=task_id,
        entry_dispositions=[
            {
                "entry_id": entry_id,
                "disposition": "appended",
                "memory_id": updated.id,
                "memory_title": updated.title,
            }
            for entry_id in entry_ids
        ],
    )
    _record_ingest_tool_invocation(
        ctx,
        task_id=task_id,
        tool_name="internal_ingest_append_memory",
        mutation=True,
    )
    _record_touched_memory_ids(ctx, task_id=task_id, memory_ids=[updated.id])
    payload = {
        "status": "ok",
        "record": internal_mutation_record_payload(updated),
        "handled_entry_ids": entry_ids,
        **_atomic_ingress_metadata(receipt),
    }
    warnings = _memory_write_quality_warnings(
        summary=updated.summary,
        memory_type=updated.type,
        tags=updated.tags,
        content=updated.content,
    )
    if warnings:
        payload["warnings"] = warnings
    return payload


def _execute_atomic_create(
    ctx: ApplicationContext,
    *,
    batch_id: str,
    task_id: str,
    entry_ids: list[int],
    title: str,
    content: str,
    arguments: dict,
) -> dict:
    transaction = ctx.ingress_mutation_transaction
    workspace_ids = string_list(arguments, "workspace_ids") or (
        [ctx.workspace_id] if ctx.workspace_id is not None else ["workspace-unknown"]
    )
    metadata_override = optional_object(arguments, "metadata") or {}
    summary = optional_string(arguments, "summary")
    memory_type = optional_string(arguments, "memory_type") or "observation"
    status = optional_string(arguments, "status") or "active"
    tags = _normalize_tags([*string_list(arguments, "tags")])
    quality_error = _memory_write_quality_error(title=title, content=content, summary=summary)
    if quality_error is not None:
        return {"status": "error", "error": "low_value_memory_rejected", "detail": quality_error}
    warnings = _memory_write_quality_warnings(summary=summary, memory_type=memory_type, tags=tags, content=content)
    request_payload = {
        "content": content,
        "metadata": metadata_override,
        "memory_type": memory_type,
        "status": status,
        "summary": summary,
        "tags": tags,
        "title": title,
        "workspace_ids": workspace_ids,
    }
    created_memory_id: str | None = None

    def apply(domain) -> IngressMutationResult:
        nonlocal created_memory_id
        record = domain.create_memory(
            title=title,
            content=content,
            summary=summary,
            memory_type=memory_type,
            status=status,
            workspace_ids=workspace_ids,
            tags=tags,
            metadata=build_ingest_created_metadata(
                task_id=task_id,
                entry_ids=entry_ids,
                metadata=metadata_override,
            ),
        )
        created_memory_id = record.id
        return IngressMutationResult("create", [record.id])

    try:
        receipt = transaction.execute(
            batch_id=batch_id,
            operation="create",
            entry_ids=[str(entry_id) for entry_id in entry_ids],
            target_ids=[],
            payload=request_payload,
            apply=apply,
            journal_task_id=task_id,
        )
    except IngressActionReceiptIdentityConflictError as exc:
        return {"status": "error", "error": "ingress_action_identity_collision", "detail": str(exc)}

    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}
    memory_id = created_memory_id or next(iter(receipt.after_revision_tokens), None)
    if memory_id is None:
        return {"status": "error", "error": "memory_not_found"}
    record = ctx.repository.get_memory(memory_id)
    if record is None:
        return {"status": "error", "error": "memory_not_found"}
    _record_successful_ingest_entry_ids(ctx, task_id=task_id, entry_ids=entry_ids)
    _record_successful_ingest_entry_dispositions(
        ctx,
        task_id=task_id,
        entry_dispositions=[
            {
                "entry_id": entry_id,
                "disposition": "created",
                "memory_id": record.id,
                "memory_title": record.title,
            }
            for entry_id in entry_ids
        ],
    )
    _record_ingest_tool_invocation(
        ctx,
        task_id=task_id,
        tool_name="internal_ingest_create_memory",
        mutation=True,
    )
    _record_touched_memory_ids(ctx, task_id=task_id, memory_ids=[record.id])
    payload: dict[str, object] = {
        "status": "ok",
        "record": internal_mutation_record_payload(record),
        "handled_entry_ids": entry_ids,
        **_atomic_ingress_metadata(receipt),
    }
    if warnings:
        payload["warnings"] = warnings
    return payload


def _atomic_ingress_metadata(receipt) -> dict[str, str]:
    return {
        "ingress_action_id": receipt.action_id,
        "ingress_payload_digest": receipt.canonical_payload_digest,
    }


def _ingress_is_enforced(ctx: ApplicationContext) -> bool:
    return getattr(getattr(ctx, "config", None), "ingress_evidence_mode", "off") == "enforce"


def _ingress_requires_atomic_boundary(ctx: ApplicationContext, *, batch_id: str | None) -> bool:
    if not _ingress_is_enforced(ctx):
        return False
    transaction = getattr(ctx, "ingress_mutation_transaction", None)
    return batch_id is None or not callable(getattr(transaction, "execute", None))
