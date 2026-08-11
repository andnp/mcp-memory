"""Compatibility adapters for the three public MCP memory use cases.

The application use cases live outside the protocol layer.  These functions
remain stable because the CLI, transport, and existing integrations import
them directly.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from mcp_memory.application.memory_use_cases import (
    CommitSkillReviewUseCase,
    ReadSkillReviewLedgerUseCase,
    ReadMemoryRecordUseCase,
    RecordSkillObservationUseCase,
    RecordThoughtUseCase,
    ResolveSkillObservationUseCase,
    SearchMemoryRecordsUseCase,
)
from mcp_memory.application.ports import MemoryReadContext, MemoryReadDependencies
from mcp_memory.context import ApplicationContext
from mcp_memory.mcp.adapters import (
    parse_batch_read_arguments,
    parse_commit_skill_review_arguments,
    parse_read_arguments,
    parse_record_thought_arguments,
    parse_resolve_skill_observation_arguments,
    parse_search_arguments,
    parse_skill_review_ledger_arguments,
    parse_skill_observation_arguments,
)
from mcp_memory.mcp.cache_policy import (
    _load_validated_cached_projection_entries,
)
from mcp_memory.mcp.telemetry import (
    McpRetrievalTelemetryAdapter,
    _record_read_invocation,
    _record_search_invocation,
    _retrieval_telemetry_repository,
)
from mcp_memory.relational.operations import SearchMemoryRecordsOperation
from mcp_memory.storage.shared_mode_cache import resolve_shared_mode_cache_state


class _ContextBoundRetrievalTelemetry:
    def __init__(self, ctx: MemoryReadContext) -> None:
        self._ctx = ctx
        self._adapter = McpRetrievalTelemetryAdapter()

    def record_search(
        self,
        *,
        caller_kind: str,
        query: str,
        surfaced_memory_ids: list[str],
        graph_provenance: Mapping[str, object] | None = None,
        duration_ms: float,
    ) -> None:
        self._adapter.record_search(
            self._ctx,
            caller_kind=caller_kind,
            query=query,
            surfaced_memory_ids=surfaced_memory_ids,
            graph_provenance=graph_provenance,
            duration_ms=duration_ms,
        )

    def record_read(
        self,
        *,
        caller_kind: str,
        memory_id: str,
        duration_ms: float,
    ) -> None:
        self._adapter.record_read(
            self._ctx,
            caller_kind=caller_kind,
            memory_id=memory_id,
            duration_ms=duration_ms,
        )


def _read_capability(ctx: MemoryReadContext, name: str, default=None):
    memory = getattr(ctx, "memory", None)
    value = getattr(memory, name, None) if memory is not None else None
    if value is not None:
        return value
    return getattr(ctx, name, default)


def _memory_read_dependencies(ctx: MemoryReadContext) -> MemoryReadDependencies:
    return MemoryReadDependencies(
        config=_read_capability(ctx, "config"),
        workspace_id=_read_capability(ctx, "workspace_id"),
        repository=_read_capability(ctx, "repository"),
        surface_tracker=_read_capability(ctx, "repository"),
        relational_search=_read_capability(ctx, "relational_search"),
        memory_retrieval=_read_capability(ctx, "memory_retrieval"),
        read_cache=_read_capability(ctx, "read_cache"),
        vector_store=_read_capability(ctx, "vector_store"),
        embedder=_read_capability(ctx, "embedder"),
        embedding_maintenance=_read_capability(ctx, "embedding_maintenance"),
    )


def record_thought_service(ctx: ApplicationContext, arguments: dict) -> dict:
    content = parse_record_thought_arguments(arguments)
    cache_state = resolve_shared_mode_cache_state(
        ctx.config,
        storage_backend=ctx.storage_backend,
        read_cache=getattr(ctx, "read_cache", None),
    )
    return RecordThoughtUseCase(
        ctx,
        writeback_cache=cache_state.writeback_cache,
        max_outbox_entries=cache_state.max_outbox_entries,
    ).execute(content)


def record_skill_observation_service(ctx: ApplicationContext, arguments: dict) -> dict:
    return RecordSkillObservationUseCase(ctx).execute(parse_skill_observation_arguments(arguments))


def resolve_skill_observation_service(ctx: ApplicationContext, arguments: dict) -> dict:
    return ResolveSkillObservationUseCase(ctx).execute(parse_resolve_skill_observation_arguments(arguments))


def commit_skill_review_service(ctx: ApplicationContext, arguments: dict) -> dict:
    return CommitSkillReviewUseCase(ctx).execute(parse_commit_skill_review_arguments(arguments))


def skill_review_ledger_service(ctx: MemoryReadContext, arguments: dict) -> dict:
    return ReadSkillReviewLedgerUseCase(_memory_read_dependencies(ctx)).execute(
        parse_skill_review_ledger_arguments(arguments)
    )


def search_memory_records_service(
    ctx: MemoryReadContext,
    arguments: dict,
    *,
    caller_kind: str = "external",
) -> dict:
    parsed = parse_search_arguments(arguments)
    return SearchMemoryRecordsUseCase(
        _memory_read_dependencies(ctx), _ContextBoundRetrievalTelemetry(ctx)
    ).execute(parsed, caller_kind=caller_kind)


async def search_memory_records_async_service(
    ctx: MemoryReadContext,
    arguments: dict,
    *,
    caller_kind: str = "external",
) -> dict:
    parsed = parse_search_arguments(arguments)
    if ctx.repository is not None:
        return await SearchMemoryRecordsUseCase(
            _memory_read_dependencies(ctx),
            _ContextBoundRetrievalTelemetry(ctx),
        ).execute_async(parsed, caller_kind=caller_kind)
    return await asyncio.to_thread(
        search_memory_records_service,
        ctx,
        arguments,
        caller_kind=caller_kind,
    )


def read_memory_record_service(
    ctx: MemoryReadContext,
    arguments: dict,
    *,
    caller_kind: str = "external",
) -> dict:
    parsed = parse_read_arguments(arguments, caller_kind=caller_kind)
    return ReadMemoryRecordUseCase(
        _memory_read_dependencies(ctx), _ContextBoundRetrievalTelemetry(ctx)
    ).execute(parsed, caller_kind=caller_kind)


def read_memory_records_service(
    ctx: MemoryReadContext,
    arguments: dict,
    *,
    caller_kind: str = "external",
) -> dict:
    parsed = parse_batch_read_arguments(arguments, caller_kind=caller_kind)
    records: list[dict] = []
    missing: list[str] = []
    for memory_id in parsed["memory_ids"]:
        response = read_memory_record_service(
            ctx,
            {
                "memory_id": memory_id,
                "include_relationships": parsed["include_relationships"],
                "include_superseded": parsed["include_superseded"],
                "include_metadata": parsed["include_metadata"],
                "summary_only": parsed["summary_only"],
                "content_offset": parsed["content_offset"],
                "content_limit": parsed["content_limit"],
            },
            caller_kind=caller_kind,
        )
        if response.get("status") == "ok":
            record = response.get("record")
            if isinstance(record, dict):
                records.append(record)
            continue
        if response.get("error") == "memory_not_found":
            missing.append(memory_id)
            continue
        return response
    payload: dict[str, object] = {"status": "ok", "records": records}
    if missing:
        payload["missing"] = missing
    return payload


__all__ = [
    "record_thought_service",
    "record_skill_observation_service",
    "resolve_skill_observation_service",
    "commit_skill_review_service",
    "skill_review_ledger_service",
    "search_memory_records_service",
    "search_memory_records_async_service",
    "read_memory_record_service",
    "read_memory_records_service",
    "_record_read_invocation",
    "_record_search_invocation",
    "_retrieval_telemetry_repository",
    "_load_validated_cached_projection_entries",
    "SearchMemoryRecordsOperation",
]
