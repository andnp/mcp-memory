from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar

from mcp_memory.search.score_pipeline import ScorePipelineConfig


T = TypeVar("T")


@dataclass
class SearchExecutionContext:
    query: str
    vector_results: list[dict] = field(default_factory=list)
    keyword_results: list[dict] = field(default_factory=list)
    graph_results: list[dict] = field(default_factory=list)


class BaseSearchOrchestrator(Generic[T]):
    def __init__(self, vector, keyword, graph, config, documents_path: Path | None = None):
        self._vector = vector
        self._keyword = keyword
        self._graph = graph
        self._config = config
        self._documents_path = documents_path

    async def _execute_parallel_search(self, query: str, top_k: int) -> SearchExecutionContext:
        vector_task = asyncio.to_thread(self._vector.search, query, top_k)
        keyword_task = asyncio.to_thread(
            self._keyword.search,
            query,
            top_k,
            None,
            self._documents_path,
        )
        vector_results, keyword_results = await asyncio.gather(vector_task, keyword_task)
        return SearchExecutionContext(
            query=query,
            vector_results=list(vector_results),
            keyword_results=list(keyword_results),
        )

    def _apply_tag_expansion(self, ctx: SearchExecutionContext, top_k: int) -> None:
        return None

    def _build_strategy_results(self, ctx: SearchExecutionContext) -> dict[str, list[dict]]:
        return {
            "vector": ctx.vector_results,
            "keyword": ctx.keyword_results,
            "graph": ctx.graph_results,
        }

    def _get_base_weights(self) -> dict[str, float]:
        return {
            "vector": getattr(self._config.search, "semantic_weight", 1.0),
            "keyword": getattr(self._config.search, "keyword_weight", 1.0),
            "graph": 1.0,
        }

    def _build_score_pipeline_config(self, weights: dict[str, float]) -> ScorePipelineConfig:
        return ScorePipelineConfig(strategy_weights=weights)

    def _apply_score_pipeline(
        self,
        strategy_results: dict[str, list[dict]],
        weights: dict[str, float],
    ) -> list[tuple[str, float]]:
        aggregated: dict[str, float] = defaultdict(float)
        for strategy_name, results in strategy_results.items():
            weight = weights.get(strategy_name, 1.0)
            for rank, result in enumerate(results, start=1):
                chunk_id = str(result.get("chunk_id") or result.get("doc_id"))
                base_score = float(result.get("score", 0.0))
                rrf_score = weight / (rank + 60)
                aggregated[chunk_id] += max(base_score, rrf_score)
        return sorted(aggregated.items(), key=lambda item: item[1], reverse=True)