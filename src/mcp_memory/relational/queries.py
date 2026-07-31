from __future__ import annotations

from mcp_memory.core.ports.memory import MemoryLink, MemoryReadPort, MemoryRecord


class RelationalMemoryQueries:
    def __init__(self, repository: MemoryReadPort) -> None:
        self._repository = repository

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        return self._repository.get_memory(memory_id)

    def list_memories(
        self,
        *,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[MemoryRecord]:
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

    def get_superseded_records(self, memory_id: str) -> list[MemoryRecord]:
        superseded: list[MemoryRecord] = []
        for link in self.get_links(memory_id, direction="outgoing"):
            if link.link_type != "SUPERSEDES":
                continue
            target = self.get_memory(link.target_id)
            if target is not None:
                superseded.append(target)
        return superseded