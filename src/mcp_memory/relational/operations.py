from __future__ import annotations

from mcp_memory.relational.search import RelationalMemorySearchService


class SearchMemoryRecordsOperation:
    def __init__(self, search_service: RelationalMemorySearchService) -> None:
        self._search_service = search_service

    def execute(
        self,
        *,
        query: str,
        workspace_id: str | None,
        limit: int,
        memory_type: str | None,
        status: str | None,
        include_superseded: bool,
        debug: bool = False,
    ):
        return self._search_service.search_memories(
            query=query,
            workspace_id=workspace_id,
            limit=limit,
            memory_type=memory_type,
            status=status,
            include_superseded=include_superseded,
            debug=debug,
        )


class ReadMemoryRecordOperation:
    def __init__(self, search_service: RelationalMemorySearchService) -> None:
        self._search_service = search_service

    def execute(self, memory_id: str):
        return self._search_service.read_memory(memory_id)
