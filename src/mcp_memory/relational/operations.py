from __future__ import annotations

from time import perf_counter
from typing import Any

from mcp_memory.core.ports.memory import parse_memory_ref
from mcp_memory.integrations.memory_retrieval import (
    MemoryRetrievalPort,
    MemorySearchRequest,
)
from mcp_memory.relational.search import (
    RelationalSearchResult,
    SearchExecutionDiagnostics,
)


class SearchMemoryRecordsOperation:
    def __init__(self, retrieval: MemoryRetrievalPort) -> None:
        self._retrieval = retrieval

    def execute(
        self,
        *,
        query: str,
        workspace_id: str | None,
        limit: int,
        adaptive_limit: bool,
        memory_type: str | None,
        status: str | None,
        include_superseded: bool,
        ranking_workspace_id: str | None = None,
        debug: bool = False,
    ) -> list[RelationalSearchResult]:
        _ = debug
        request = MemorySearchRequest(
            query=query,
            workspace_id=workspace_id,
            limit=limit,
            adaptive_limit=adaptive_limit,
            memory_type=memory_type,
            status=status,
            include_superseded=include_superseded,
            ranking_workspace_id=ranking_workspace_id,
        )
        outcome = self._retrieval.search_sync(request)
        return [_to_relational_result(result) for result in outcome.results]

    def execute_with_diagnostics(
        self,
        *,
        query: str,
        workspace_id: str | None,
        limit: int,
        adaptive_limit: bool,
        memory_type: str | None,
        status: str | None,
        include_superseded: bool,
        ranking_workspace_id: str | None = None,
        debug: bool = False,
    ) -> tuple[list[RelationalSearchResult], SearchExecutionDiagnostics]:
        started_at = perf_counter()
        results = self.execute(
            query=query,
            workspace_id=workspace_id,
            limit=limit,
            adaptive_limit=adaptive_limit,
            memory_type=memory_type,
            status=status,
            include_superseded=include_superseded,
            ranking_workspace_id=ranking_workspace_id,
            debug=debug,
        )
        return results, SearchExecutionDiagnostics(
            timing_ms={"total": round((perf_counter() - started_at) * 1000.0, 3)}
        )


def _to_relational_result(result: Any) -> RelationalSearchResult:
    record = result.record
    memory_ref_value = record.metadata.get("memory_ref")
    memory_ref = (
        memory_ref_value
        if isinstance(memory_ref_value, int) and memory_ref_value > 0
        else (
            parse_memory_ref(memory_ref_value)
            if isinstance(memory_ref_value, str)
            else None
        )
    )
    return RelationalSearchResult(
        memory_id=record.source_id,
        memory_ref=memory_ref,
        title=record.title,
        summary=str(record.metadata.get("summary", "")),
        memory_type=str(record.metadata.get("memory_type", "")),
        status=str(record.metadata.get("memory_status", record.status.value)),
        tags=[str(tag) for tag in record.metadata.get("tags", []) if isinstance(tag, str)],
        workspace_ids=[
            str(workspace_id)
            for workspace_id in record.metadata.get("workspace_ids", [])
            if isinstance(workspace_id, str)
        ],
        score=round(result.score, 6),
        ranking_debug={
            "provenance": result.provenance.to_dict(),
            "canonical_id": record.storage_key,
        },
    )


class ReadMemoryRecordOperation:
    def __init__(self, search_service) -> None:
        self._search_service = search_service

    def execute(self, memory_id: str):
        return self._search_service.read_memory(memory_id)
