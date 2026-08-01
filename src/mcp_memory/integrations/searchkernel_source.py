"""Memory SearchableSource adapter for searchkernel federation.

This adapter wraps the mcp-memory search service (RelationalMemorySearchService)
as a federated SearchableSource, allowing memory to be searched through
searchkernel's unified search_anything interface.

The adapter honors memory's lifecycle (status: active/stale/archived) and
supersession graph, ensuring archived and superseded records never surface.
"""

from __future__ import annotations

from typing import Any, Iterable

from searchkernel.domain import ScoredRef, canonical_storage_key
from searchkernel.kernel import SearchKernel
from searchkernel.search.record_pipeline import RecordSearchResult
from searchkernel.ports.content_source import SearchableSource

from mcp_memory.integrations.memory_retrieval import MemoryRetrievalPort
from mcp_memory.core.ports.memory import format_memory_ref


def build_memory_search_kernel(
    retrieval: MemoryRetrievalPort,
    *,
    extra_sources: Iterable[SearchableSource] = (),
    per_source_timeout_s: float = 5.0,
    side_effect_free: bool = False,
) -> SearchKernel:
    """Build a federated kernel with memory plus optional source adapters."""
    sources = [
        MemorySearchableSource(retrieval, side_effect_free=side_effect_free),
        *extra_sources,
    ]
    return SearchKernel.build(
        sources=sources,
        per_source_timeout_s=per_source_timeout_s,
    )


class MemorySearchableSource:
    """SearchableSource wrapping mcp-memory's RelationalMemorySearchService.

    Implements the SearchableSource protocol to allow federated searching:
    the kernel never stores memory content; memory ranks itself via its
    existing 10-stage gauntlet, and the kernel fuses results via RRF.

    The adapter ensures:
    - Archived records are skipped (status filtering)
    - Stale/degraded records are downranked (handled by the search service)
    - Superseded records never rank above their canonical targets
      (via include_superseded=False enforcement)
    - Result text is available for retrieve-then-rerank-once
    """

    source_kind = "memory"

    def __init__(
        self,
        retrieval: MemoryRetrievalPort,
        *,
        side_effect_free: bool = False,
    ):
        """Initialize the adapter with a RelationalMemorySearchService.

        Args:
            retrieval: The canonical memory retrieval facade.
        """
        self._retrieval = retrieval
        self._side_effect_free = side_effect_free

    async def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> Iterable[ScoredRef]:
        """Run canonical memory retrieval and return federated references."""
        filters = filters or {}
        workspace_id = filters.get("workspace_id") if isinstance(filters.get("workspace_id"), str) else None
        memory_type = filters.get("memory_type") if isinstance(filters.get("memory_type"), str) else None
        status = filters.get("status") if isinstance(filters.get("status"), str) else None
        include_superseded = bool(filters.get("include_superseded", False))
        outcome = await self._retrieval.search(
            query,
            limit=k,
            workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            include_superseded=include_superseded,
        )
        return [self._to_scored_ref(result) for result in outcome.results]

    @staticmethod
    def _to_scored_ref(result: RecordSearchResult) -> ScoredRef:
        """Convert a facade search result to a ScoredRef for federation.

        Args:
            result: A ranked search result from the memory search service.

        Returns:
            A ScoredRef with source_id mapped from memory_id, score preserved,
            and metadata containing the candidate text (title + summary) for
            retrieve-then-rerank-once, plus additional context.
        """
        record = result.record
        summary = str(record.metadata.get("summary", ""))
        workspace_ids = [
            str(workspace_id)
            for workspace_id in record.metadata.get("workspace_ids", [])
            if isinstance(workspace_id, str)
        ]
        memory_ref = record.metadata.get("memory_ref")
        source_id = record.source_id
        title = record.title
        memory_type = record.metadata.get("memory_type", "")
        status = record.metadata.get("memory_status", record.status.value)
        tags = list(record.metadata.get("tags", []))
        workspace_id = workspace_ids[0] if workspace_ids else record.workspace_id
        candidate_text = title
        if summary:
            candidate_text = f"{title}\n\n{summary}"
        return ScoredRef(
            source_id=source_id,
            score=result.score,
            source_kind="memory",
            workspace_id=workspace_id,
            metadata={
                "text": candidate_text,
                "title": title,
                "summary": summary,
                "memory_type": memory_type,
                "status": status,
                "tags": tags,
                "workspace_ids": workspace_ids,
                "memory_ref": memory_ref,
                "memory_reference": (
                    format_memory_ref(memory_ref)
                    if isinstance(memory_ref, int)
                    else memory_ref
                ),
                "canonical_id": canonical_storage_key(
                    workspace_id,
                    "memory",
                    source_id,
                ),
            },
        )
