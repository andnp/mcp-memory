"""Storage-agnostic graph candidate expansion for memory search."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from mcp_memory.core.ports.memory import MemoryLink

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
        expanded: dict[str, GraphExpansionInfo] = {}
        seed_items = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:GRAPH_EXPANSION_MAX_SEEDS]
        for seed_id, seed_score in seed_items:
            expanded_neighbors = 0
            for link in self._link_reader(seed_id):
                discount = GRAPH_EXPANSION_DISCOUNTS.get(link.link_type)
                if discount is None:
                    continue
                expanded_neighbors += 1
                if expanded_neighbors > GRAPH_EXPANSION_MAX_NEIGHBORS_PER_SEED:
                    break
                expanded_score = seed_score * discount
                current = expanded.get(link.target_id)
                if current is None or expanded_score > current.rrf_score:
                    expanded[link.target_id] = GraphExpansionInfo(
                        rrf_score=expanded_score,
                        seed_id=seed_id,
                        link_type=link.link_type,
                    )
        return expanded
