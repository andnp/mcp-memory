from __future__ import annotations

from pathlib import Path

from mcp_memory.relational.importer import import_markdown_memory
from mcp_memory.relational.queries import RelationalMemoryQueries
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.relational.search import RelationalMemorySearchService


class CreateMemoryRecordOperation:
    def __init__(
        self,
        repository: RelationalMemoryRepository,
        default_workspace_id: str | None,
    ) -> None:
        self._repository = repository
        self._default_workspace_id = default_workspace_id

    def execute(
        self,
        *,
        title: str,
        content: str,
        workspace_ids: list[str],
        tags: list[str],
        summary: str | None,
        memory_type: str,
        status: str,
        metadata: dict[str, object],
    ):
        resolved_workspace_ids = list(workspace_ids)
        if not resolved_workspace_ids and self._default_workspace_id is not None:
            resolved_workspace_ids = [self._default_workspace_id]
        if not resolved_workspace_ids:
            raise ValueError("workspace_ids are required")

        return self._repository.create_memory(
            title=title,
            content=content,
            workspace_ids=resolved_workspace_ids,
            tags=tags,
            summary=summary,
            memory_type=memory_type,
            status=status,
            metadata=metadata,
        )


class GetMemoryRecordOperation:
    def __init__(self, queries: RelationalMemoryQueries) -> None:
        self._queries = queries

    def execute(self, memory_id: str):
        return self._queries.get_memory(memory_id)


class ListMemoryRecordsOperation:
    def __init__(self, queries: RelationalMemoryQueries) -> None:
        self._queries = queries

    def execute(
        self,
        *,
        workspace_id: str | None,
        memory_type: str | None,
        status: str | None,
        limit: int,
    ):
        return self._queries.list_memories(
            workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            limit=limit,
        )


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
    ):
        return self._search_service.search_memories(
            query=query,
            workspace_id=workspace_id,
            limit=limit,
            memory_type=memory_type,
            status=status,
            include_superseded=include_superseded,
        )


class ReadMemoryRecordOperation:
    def __init__(self, search_service: RelationalMemorySearchService) -> None:
        self._search_service = search_service

    def execute(self, memory_id: str):
        return self._search_service.read_memory(memory_id)


class ImportMarkdownMemoryFileOperation:
    def __init__(
        self,
        repository: RelationalMemoryRepository,
        default_workspace_id: str | None,
    ) -> None:
        self._repository = repository
        self._default_workspace_id = default_workspace_id

    def execute(self, file_path: str, workspace_ids: list[str]):
        resolved_workspace_ids = list(workspace_ids)
        if not resolved_workspace_ids and self._default_workspace_id is not None:
            resolved_workspace_ids = [self._default_workspace_id]
        if not resolved_workspace_ids:
            raise ValueError("workspace_ids are required when workspace_id is unavailable")

        import_path = Path(file_path)
        if not import_path.exists():
            raise FileNotFoundError(f"markdown memory file not found: {file_path}")

        return import_markdown_memory(self._repository, import_path, resolved_workspace_ids)