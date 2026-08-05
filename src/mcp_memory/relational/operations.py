from __future__ import annotations

from time import perf_counter

from mcp_memory.integrations.memory_retrieval import (
    MemoryRetrievalPort,
    MemorySearchRequest,
)
from mcp_memory.relational.search import (
    RelationalSearchResult,
    SearchExecutionDiagnostics,
    _to_relational_search_result,
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
        return [_to_relational_search_result(result) for result in outcome.results]

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
        results = [_to_relational_search_result(result) for result in outcome.results]
        trace = outcome.trace.to_dict() if debug and outcome.trace is not None else None
        return results, SearchExecutionDiagnostics(
            timing_ms={"total": round((perf_counter() - started_at) * 1000.0, 3)},
            kernel_diagnostics=list(outcome.diagnostics),
            cache_diagnostics=list(outcome.cache_diagnostics),
            failure_count=len(outcome.failures),
            missing_record_count=len(outcome.missing_record_ids),
            degraded=bool(outcome.failures or outcome.degraded),
            trace=trace,
        )


class ReadMemoryRecordOperation:
    def __init__(self, search_service) -> None:
        self._search_service = search_service

    def execute(self, memory_id: str):
        return self._search_service.read_memory(memory_id)
