from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from mcp_memory.application.ports import MemorySearchPort
from mcp_memory.context import MemoryPipelineContext, MemoryReadCapabilities, MutationCapabilities
from mcp_memory.integrations.memory_retrieval import MemoryRetrievalPort
from mcp_memory.relational.queries import RelationalMemoryQueries
from mcp_memory.runtime_facades import JournalFacade, RuntimeInfoFacade, TaskQueueFacade


@dataclass(frozen=True)
class MemoryPipeline:
    runtime_info: RuntimeInfoFacade
    journal: JournalFacade
    task_queue: TaskQueueFacade
    memory_queries: RelationalMemoryQueries | None
    repository: object | None
    search: MemorySearchPort | None
    retrieval: MemoryRetrievalPort | None

    @classmethod
    def from_context(
        cls,
        ctx: MemoryReadCapabilities | MemoryPipelineContext,
        controller: Any | None = None,
        *,
        mutation: MutationCapabilities | None = None,
    ) -> MemoryPipeline:
        capabilities = (
            ctx
            if isinstance(ctx, MemoryReadCapabilities)
            else MemoryReadCapabilities.from_context(ctx)
        )
        compatibility_mutation = (
            mutation
            if mutation is not None
            else (
                None
                if isinstance(ctx, MemoryReadCapabilities)
                else MutationCapabilities.from_context(ctx)
            )
        )
        task_queue = (
            compatibility_mutation.task_queue
            if compatibility_mutation is not None
            else getattr(ctx, "task_queue", None)
        )
        journal = (
            compatibility_mutation.journal
            if compatibility_mutation is not None
            else None
        )
        return cls(
            runtime_info=RuntimeInfoFacade.from_capabilities(
                capabilities,
                task_queue=task_queue,
                controller=controller,
            ),
            journal=JournalFacade(journal=cast(Any, journal)),
            task_queue=TaskQueueFacade(task_queue=cast(Any, task_queue)),
            memory_queries=(
                RelationalMemoryQueries(cast(Any, capabilities.repository))
                if capabilities.repository is not None
                else None
            ),
            repository=capabilities.repository,
            search=capabilities.relational_search,
            retrieval=cast(
                MemoryRetrievalPort | None,
                getattr(ctx, "memory_retrieval", None),
            ),
        )
