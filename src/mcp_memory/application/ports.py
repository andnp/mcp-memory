from __future__ import annotations

from typing import Protocol

from mcp_memory.context import ApplicationContext


class RetrievalTelemetryPort(Protocol):
    def record_search(
        self,
        ctx: ApplicationContext,
        *,
        caller_kind: str,
        query: str,
        surfaced_memory_ids: list[str],
        duration_ms: float,
    ) -> None: ...

    def record_read(
        self,
        ctx: ApplicationContext,
        *,
        caller_kind: str,
        memory_id: str,
        duration_ms: float,
    ) -> None: ...
