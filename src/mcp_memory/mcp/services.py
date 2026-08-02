"""Compatibility adapters for the three public MCP memory use cases.

The application use cases live outside the protocol layer.  These functions
remain stable because the CLI, transport, and existing integrations import
them directly.
"""

from __future__ import annotations

import asyncio

from mcp_memory.application.memory_use_cases import (
    ReadMemoryRecordUseCase,
    RecordThoughtUseCase,
    SearchMemoryRecordsUseCase,
)
from mcp_memory.application.ports import MemoryReadContext, MemoryReadDependencies
from mcp_memory.context import ApplicationContext
from mcp_memory.mcp.adapters import (
    parse_batch_read_arguments,
    parse_read_arguments,
    parse_record_thought_arguments,
    parse_search_arguments,
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
        duration_ms: float,
    ) -> None:
        self._adapter.record_search(
            self._ctx,
            caller_kind=caller_kind,
            query=query,
            surfaced_memory_ids=surfaced_memory_ids,
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


def _memory_read_dependencies(ctx: MemoryReadContext) -> MemoryReadDependencies:
    return MemoryReadDependencies(
        config=ctx.config,
        workspace_id=ctx.workspace_id,
        repository=ctx.repository,
        surface_tracker=ctx.repository,
        relational_search=ctx.relational_search,
        memory_retrieval=getattr(ctx, "memory_retrieval", None),
        read_cache=getattr(ctx, "read_cache", None),
        vector_store=ctx.vector_store,
        embedder=ctx.embedder,
        embedding_maintenance=getattr(ctx, "embedding_maintenance", None),
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
    return {
        "status": "ok",
        "records": records,
        "missing": missing,
        "budget": {
            "requested": len(parsed["memory_ids"]),
            "returned": len(records),
            "missing": len(missing),
        },
    }


__all__ = [
    "record_thought_service",
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
