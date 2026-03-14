from __future__ import annotations

from mcp_memory.relational.repository import MemoryLink, RelationalMemoryRecord, RelationalMemoryRepository


class RelationalMemoryQueries:
    def __init__(self, repository: RelationalMemoryRepository) -> None:
        self._repository = repository

    def get_memory(self, memory_id: str) -> RelationalMemoryRecord | None:
        return self._repository.get_memory(memory_id)

    def list_memories(
        self,
        *,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[RelationalMemoryRecord]:
        return self._repository.list_memories(
            workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            limit=limit,
        )

    def get_links(
        self,
        memory_id: str,
        *,
        direction: str = "outgoing",
        link_type: str | None = None,
    ) -> list[MemoryLink]:
        return self._repository.get_links(
            memory_id,
            direction=direction,
            link_type=link_type,
        )

    def get_superseded_records(self, memory_id: str) -> list[RelationalMemoryRecord]:
        superseded: list[RelationalMemoryRecord] = []
        for link in self.get_links(memory_id, direction="outgoing"):
            if link.link_type != "SUPERSEDES":
                continue
            target = self.get_memory(link.target_id)
            if target is not None:
                superseded.append(target)
        return superseded