"""Storage-agnostic graph candidate expansion for memory search."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from mcp_memory.core.ports.memory import MemoryLink
from searchkernel.search.bounded_graph import TypedGraphEdge, expand_bounded_typed_graph

GRAPH_EXPANSION_MAX_SEEDS = 3
GRAPH_EXPANSION_MAX_NEIGHBORS_PER_SEED = 10
GRAPH_EXPANSION_DISCOUNTS = {
    "DEPENDS_ON": 0.7,
    "AMENDS": 0.6,
    "CONTRADICTS": 0.35,
}


@dataclass(frozen=True, slots=True)
class GraphExpansionInfo:
    rrf_score: float
    seed_id: str
    link_type: str


class GraphCandidateExpander:
    """Expand ranked seed IDs through supported outgoing memory links."""

    def __init__(self, link_reader: Callable[[str], list[MemoryLink]]) -> None:
        self._link_reader = link_reader

    def expand(self, rrf_scores: Mapping[str, float]) -> dict[str, GraphExpansionInfo]:
        def outgoing_edges(seed_id: str) -> Iterable[TypedGraphEdge[str, str]]:
            return (
                TypedGraphEdge(target_id=link.target_id, edge_type=link.link_type)
                for link in self._link_reader(seed_id)
            )

        expanded = expand_bounded_typed_graph(
            rrf_scores,
            outgoing_edges,
            GRAPH_EXPANSION_DISCOUNTS,
            max_seed_count=GRAPH_EXPANSION_MAX_SEEDS,
            max_neighbors_per_seed=GRAPH_EXPANSION_MAX_NEIGHBORS_PER_SEED,
        )
        return {
            target_id: GraphExpansionInfo(
                rrf_score=result.contribution,
                seed_id=result.provenance.seed_id,
                link_type=result.provenance.edge_type,
            )
            for target_id, result in expanded.items()
        }
