from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.relational.queries import RelationalMemoryQueries
from mcp_memory.runtime_facades import JournalFacade, RuntimeInfoFacade, TaskQueueFacade


@dataclass(frozen=True)
class MemoryPipeline:
    runtime_info: RuntimeInfoFacade
    journal: JournalFacade
    task_queue: TaskQueueFacade
    memory_queries: RelationalMemoryQueries | None
    repository: Any
    search: Any

    @classmethod
    def from_context(
        cls,
        ctx: ApplicationContext,
        controller: Any | None = None,
    ) -> MemoryPipeline:
        return cls(
            runtime_info=RuntimeInfoFacade.from_context(ctx, controller),
            journal=JournalFacade.from_context(ctx),
            task_queue=TaskQueueFacade.from_context(ctx),
            memory_queries=(
                RelationalMemoryQueries(ctx.repository)
                if ctx.repository is not None
                else None
            ),
            repository=ctx.repository,
            search=ctx.relational_search,
        )