from __future__ import annotations

from logging import getLogger
from time import time
from uuid import uuid4

from mcp_memory.application.ports import MemoryReadContext
from mcp_memory.retrieval_telemetry_store import RetrievalTelemetryRepository


_SLOW_MEMORY_TOOL_WARNING_MS = 2_000.0
logger = getLogger(__name__)

def _retrieval_telemetry_repository(
    ctx: MemoryReadContext,
) -> RetrievalTelemetryRepository:
    repository = ctx.retrieval_telemetry
    if repository is None:
        repository = RetrievalTelemetryRepository(
            ctx.db_manager,
            workspace_id=ctx.workspace_id,
            storage_backend=ctx.storage_backend,
        )
        ctx.retrieval_telemetry = repository
    return repository


def _log_slow_memory_tool_operation(
    ctx: MemoryReadContext,
    *,
    tool_name: str,
    duration_ms: float,
    data: dict[str, object],
) -> None:
    if duration_ms < _SLOW_MEMORY_TOOL_WARNING_MS:
        return
    runtime_logs = getattr(ctx, "runtime_logs", None)
    if runtime_logs is not None:
        try:
            runtime_logs.write_log(
                source="memory-tool",
                logger_name=__name__,
                level="WARNING",
                message=f"Slow {tool_name} operation",
                created_at=time(),
                data={"tool_name": tool_name, "duration_ms": round(duration_ms, 3)}
                | data,
            )
            return
        except (
            Exception
        ):  # pragma: no cover - defensive fallback for locked telemetry/log stores
            logger.warning(
                "Failed to persist slow %s log; falling back to process logger",
                tool_name,
                exc_info=True,
            )
    logger.warning("Slow %s operation: %.3fms %s", tool_name, duration_ms, data)


def _record_search_invocation(
    ctx: MemoryReadContext,
    *,
    caller_kind: str,
    query: str,
    surfaced_memory_ids: list[str],
    duration_ms: float,
) -> None:
    _retrieval_telemetry_repository(ctx).record_search(
        invocation_id=str(uuid4()),
        caller_kind=caller_kind,
        query=query,
        surfaced_memory_ids=surfaced_memory_ids,
        duration_ms=duration_ms,
    )
    _log_slow_memory_tool_operation(
        ctx,
        tool_name="search_memory_records",
        duration_ms=duration_ms,
        data={
            "caller_kind": caller_kind,
            "query": query,
            "result_count": len(surfaced_memory_ids),
            "storage_backend": ctx.storage_backend or "sqlite",
        },
    )


def _record_read_invocation(
    ctx: MemoryReadContext,
    *,
    caller_kind: str,
    memory_id: str,
    duration_ms: float,
) -> None:
    repository = _retrieval_telemetry_repository(ctx)
    repository.record_read(
        invocation_id=str(uuid4()),
        caller_kind=caller_kind,
        memory_id=memory_id,
        duration_ms=duration_ms,
    )
    try:
        repository.flush()
    except Exception:
        logger.debug(
            "Read telemetry flush failed; continuing without blocking the read response",
            exc_info=True,
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


class McpRetrievalTelemetryAdapter:
    def record_search(
        self,
        ctx: MemoryReadContext,
        *,
        caller_kind: str,
        query: str,
        surfaced_memory_ids: list[str],
        duration_ms: float,
    ) -> None:
        _record_search_invocation(
            ctx,
            caller_kind=caller_kind,
            query=query,
            surfaced_memory_ids=surfaced_memory_ids,
            duration_ms=duration_ms,
        )

    def record_read(
        self,
        ctx: MemoryReadContext,
        *,
        caller_kind: str,
        memory_id: str,
        duration_ms: float,
    ) -> None:
        _record_read_invocation(
            ctx,
            caller_kind=caller_kind,
            memory_id=memory_id,
            duration_ms=duration_ms,
        )
