from __future__ import annotations

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
        tags: tuple[str, ...] = (),
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
            tags=tags,
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
        tags: tuple[str, ...] = (),
        ranking_workspace_id: str | None = None,
        debug: bool = False,
    ) -> tuple[list[RelationalSearchResult], SearchExecutionDiagnostics]:
        request = MemorySearchRequest(
            query=query,
            workspace_id=workspace_id,
            limit=limit,
            adaptive_limit=adaptive_limit,
            memory_type=memory_type,
            status=status,
            tags=tags,
            include_superseded=include_superseded,
            ranking_workspace_id=ranking_workspace_id,
        )
        outcome, diagnostics = self._retrieval.search_sync_with_diagnostics(
            request,
            debug=debug,
        )
        results = [_to_relational_search_result(result) for result in outcome.results]
        for result in results:
            if result.ranking_debug is not None:
                result.ranking_debug["final_duplicate"] = (
                    result.memory_id in (diagnostics.final_duplicate_ids or [])
                )
        return results, diagnostics


class ReadMemoryRecordOperation:
    def __init__(self, search_service) -> None:
        self._search_service = search_service

    def execute(self, memory_id: str):
        return self._search_service.read_memory(memory_id)
