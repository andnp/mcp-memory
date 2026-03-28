from __future__ import annotations

import logging
from time import perf_counter, time
from uuid import uuid4

from mcp_memory.context import ApplicationContext
from mcp_memory.core.journal_operations import (
    RecordThoughtOperation,
)
from mcp_memory.mcp.validation import (
    optional_bool,
    optional_positive_int,
    optional_string,
    require_string,
)
from mcp_memory.relational.operations import (
    ReadMemoryRecordOperation,
    SearchMemoryRecordsOperation,
)
from mcp_memory.retrieval_telemetry_store import RetrievalTelemetryRepository
from mcp_memory.serialization import (
    link_payload,
    memory_record_payload,
    search_result_payload,
)


SEARCH_READ_GUIDANCE = (
    "Use these summary-first results to identify the most promising memories, then call read_memory_record for full context on the specific memory_id values you want to inspect."
)
_SLOW_MEMORY_TOOL_WARNING_MS = 2_000.0


logger = logging.getLogger(__name__)


def record_thought_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.journal is None:
        return {"status": "error", "error": "journal_not_initialized"}

    suppression_config = None if ctx.config is None else ctx.config.ingest_suppression
    operation = RecordThoughtOperation(ctx.journal, ctx.task_queue, ctx.workspace_id, suppression_config)
    return operation.execute(require_string(arguments, "content"))


def _retrieval_telemetry_repository(ctx: ApplicationContext) -> RetrievalTelemetryRepository:
    return RetrievalTelemetryRepository(ctx.db_manager, workspace_id=ctx.workspace_id)


def _log_slow_memory_tool_operation(
    ctx: ApplicationContext,
    *,
    tool_name: str,
    duration_ms: float,
    data: dict[str, object],
) -> None:
    if duration_ms < _SLOW_MEMORY_TOOL_WARNING_MS:
        return
    runtime_logs = getattr(ctx, "runtime_logs", None)
    if runtime_logs is not None:
        runtime_logs.write_log(
            source="memory-tool",
            logger_name=__name__,
            level="WARNING",
            message=f"Slow {tool_name} operation",
            created_at=time(),
            data={"tool_name": tool_name, "duration_ms": round(duration_ms, 3)} | data,
        )
        return
    logger.warning("Slow %s operation: %.3fms %s", tool_name, duration_ms, data)


def search_memory_records_service(
    ctx: ApplicationContext,
    arguments: dict,
    *,
    caller_kind: str = "external",
) -> dict:
    if ctx.relational_search is None:
        return {"status": "error", "error": "relational_search_not_initialized"}

    query = require_string(arguments, "query")
    operation = SearchMemoryRecordsOperation(ctx.relational_search)
    started_at = perf_counter()
    results = operation.execute(
        query=query,
        workspace_id=ctx.workspace_id,
        limit=optional_positive_int(arguments, "limit", 5),
        adaptive_limit="limit" not in arguments,
        memory_type=optional_string(arguments, "memory_type"),
        status=optional_string(arguments, "status"),
        include_superseded=optional_bool(arguments, "include_superseded", False),
        debug=optional_bool(arguments, "debug", False),
    )
    duration_ms = (perf_counter() - started_at) * 1000.0
    _retrieval_telemetry_repository(ctx).record_search(
        invocation_id=str(uuid4()),
        caller_kind=caller_kind,
        query=query,
        surfaced_memory_ids=[result.memory_id for result in results],
        duration_ms=duration_ms,
    )
    _log_slow_memory_tool_operation(
        ctx,
        tool_name="search_memory_records",
        duration_ms=duration_ms,
        data={
            "caller_kind": caller_kind,
            "query": query,
            "result_count": len(results),
            "storage_backend": ctx.storage_backend or "sqlite",
        },
    )
    debug_enabled = optional_bool(arguments, "debug", False)
    payload = {
        "status": "ok",
        "recommended_follow_up_tool": "read_memory_record",
        "results": [
            search_result_payload(result)
            | ({"ranking_debug": result.ranking_debug} if debug_enabled and result.ranking_debug is not None else {})
            for result in results
        ],
        "guidance": SEARCH_READ_GUIDANCE,
    }
    if debug_enabled:
        payload["timing_ms"] = {"total": round(duration_ms, 3)}
    return payload


def read_memory_record_service(
    ctx: ApplicationContext,
    arguments: dict,
    *,
    caller_kind: str = "external",
) -> dict:
    if ctx.relational_search is None:
        return {"status": "error", "error": "relational_search_not_initialized"}

    operation = ReadMemoryRecordOperation(ctx.relational_search)
    memory_id = require_string(arguments, "memory_id")
    started_at = perf_counter()
    result = operation.execute(memory_id)
    duration_ms = (perf_counter() - started_at) * 1000.0
    if result is None:
        return {"status": "error", "error": "memory_not_found"}
    _retrieval_telemetry_repository(ctx).record_read(
        invocation_id=str(uuid4()),
        caller_kind=caller_kind,
        memory_id=memory_id,
        duration_ms=duration_ms,
    )
    _log_slow_memory_tool_operation(
        ctx,
        tool_name="read_memory_record",
        duration_ms=duration_ms,
        data={
            "caller_kind": caller_kind,
            "memory_id": memory_id,
            "storage_backend": ctx.storage_backend or "sqlite",
        },
    )

    return {
        "status": "ok",
        "record": memory_record_payload(result.record),
        "relationships": {
            direction: [link_payload(link) for link in links]
            for direction, links in result.relationships.items()
        },
        "superseded": [memory_record_payload(record) for record in result.superseded],
    }
