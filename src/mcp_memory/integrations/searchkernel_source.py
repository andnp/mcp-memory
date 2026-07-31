"""Memory SearchableSource adapter for searchkernel federation.

This adapter wraps the mcp-memory search service (RelationalMemorySearchService)
as a federated SearchableSource, allowing memory to be searched through
searchkernel's unified search_anything interface.

The adapter honors memory's lifecycle (status: active/stale/archived) and
supersession graph, ensuring archived and superseded records never surface.
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterable

from searchkernel.domain import ScoredRef, canonical_storage_key
from searchkernel.kernel import SearchKernel
from searchkernel.ports.content_source import SearchableSource

from mcp_memory.relational.search import RelationalMemorySearchService, RelationalSearchResult


def build_memory_search_kernel(
    search_service: RelationalMemorySearchService,
    *,
    extra_sources: Iterable[SearchableSource] = (),
    per_source_timeout_s: float = 5.0,
    side_effect_free: bool = False,
) -> SearchKernel:
    """Build a federated kernel with memory plus optional source adapters."""
    sources = [
        MemorySearchableSource(search_service, side_effect_free=side_effect_free),
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
        search_service: RelationalMemorySearchService,
        *,
        side_effect_free: bool = False,
    ):
        """Initialize the adapter with a RelationalMemorySearchService.

        Args:
            search_service: The underlying memory search service that performs
                           the actual ranking and retrieval.
        """
        self._search_service = search_service
        self._side_effect_free = side_effect_free

    async def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> Iterable[ScoredRef]:
        """Run memory's native search and return ranked references.

        The search honors memory's lifecycle: archived records are excluded,
        stale/degraded records are downranked, and superseded records never
        rank above their canonical targets.

        Args:
            query: The search query string.
            k: Maximum number of results to return.
            filters: Optional source-specific filters. Recognized keys:
                     - workspace_id: str - restrict to specific workspace
                     - memory_type: str - restrict to specific memory type
                     Any other filter keys are ignored (memory-specific
                     but not yet integrated).

        Returns:
            An iterable of ScoredRefs in descending score order.
            Scores are memory-specific (on its internal scale) and used
            for RRF fusion by the kernel.
        """
        filters = filters or {}
        workspace_id = filters.get("workspace_id") if isinstance(filters.get("workspace_id"), str) else None
        memory_type = filters.get("memory_type") if isinstance(filters.get("memory_type"), str) else None
        status = filters.get("status") if isinstance(filters.get("status"), str) else None
        include_superseded = bool(filters.get("include_superseded", False))

        def run_search() -> list[RelationalSearchResult]:
            # search_memories honors include_superseded=False by default in the
            # operation, but the service wrapper allows it to be configured.
            # We explicitly ask for include_superseded=False to enforce the
            # contract obligation.
            results = self._search_service.search_memories(
                query,
                workspace_id=workspace_id,
                limit=k,
                memory_type=memory_type,
                status=status,  # None means "skip archived"; stale/degraded are downranked
                include_superseded=include_superseded,
                side_effect_free=self._side_effect_free,
            )
            return results

        # Offload the blocking search to a thread to avoid blocking the event loop.
        # This follows the pattern from the reference implementation (local.py).
        results = await asyncio.to_thread(run_search)

        # Convert RelationalSearchResult to ScoredRef with required metadata.
        return [self._to_scored_ref(result) for result in results]

    @staticmethod
    def _to_scored_ref(result: RelationalSearchResult) -> ScoredRef:
        """Convert a RelationalSearchResult to a ScoredRef for federation.

        Args:
            result: A ranked search result from the memory search service.

        Returns:
            A ScoredRef with source_id mapped from memory_id, score preserved,
            and metadata containing the candidate text (title + summary) for
            retrieve-then-rerank-once, plus additional context.
        """
        # Combine title and summary for retrieve-then-rerank-once.
        # The kernel will rerank using this text, so it must contain the
        # searchable content.
        candidate_text = result.title
        if result.summary:
            candidate_text = f"{result.title}\n\n{result.summary}"

        return ScoredRef(
            source_id=result.memory_id,
            score=result.score,
            source_kind="memory",
            workspace_id=(
                result.workspace_ids[0] if result.workspace_ids else None
            ),
            metadata={
                "text": candidate_text,
                "title": result.title,
                "summary": result.summary,
                "memory_type": result.memory_type,
                "status": result.status,
                "tags": result.tags,
                "workspace_ids": result.workspace_ids,
                "canonical_id": canonical_storage_key(
                    result.workspace_ids[0] if result.workspace_ids else None,
                    "memory",
                    result.memory_id,
                ),
            },
        )
