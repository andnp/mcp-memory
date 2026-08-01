from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp_memory.application.memory_embedding_maintenance import (
    MemoryEmbeddingMaintenance,
)
from mcp_memory.core.ports.tasks import TaskRecord

if TYPE_CHECKING:
    from mcp_memory.context import ApplicationContext


DEFAULT_EMBEDDING_REPAIR_BATCH_SIZE = 32
DEFAULT_EMBEDDING_REPAIR_MAX_BATCHES_PER_RUN = 8
DEFAULT_EMBEDDING_REPAIR_PRUNE_LIMIT = 256


async def handle_embedding_repair_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    maintenance = getattr(ctx, "embedding_maintenance", None)
    if not isinstance(maintenance, MemoryEmbeddingMaintenance):
        maintenance = MemoryEmbeddingMaintenance.from_context(ctx)
    return await maintenance.repair_task(task)
