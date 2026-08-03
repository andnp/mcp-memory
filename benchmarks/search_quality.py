"""Run a deterministic local search-quality benchmark.

This benchmark intentionally uses a fixed topic encoder so relevance and
latency can be compared without model downloads or external services.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from searchkernel.eval import BenchmarkConfig, run_benchmark
from searchkernel.eval.golden import GoldenEntry, GoldenSet

from mcp_memory.config import Config
from mcp_memory.embeddings import SQLiteVectorStore
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.relational.search import RelationalMemorySearchService
from mcp_memory.utils.db import DatabaseManager


class TopicEmbedder:
    model_name = "benchmark-topics"
    dim = 6
    _topics = {
        "authentication": 0,
        "auth": 0,
        "token": 0,
        "credential": 0,
        "security": 0,
        "database": 1,
        "sqlite": 1,
        "postgres": 1,
        "storage": 1,
        "persistence": 1,
        "search": 2,
        "retrieval": 2,
        "ranking": 2,
        "query": 2,
        "fts": 2,
        "daemon": 3,
        "worker": 3,
        "runtime": 3,
        "background": 3,
        "lifecycle": 3,
        "embedding": 4,
        "semantic": 4,
        "vector": 4,
        "model": 4,
        "graph": 5,
        "links": 5,
        "relationships": 5,
        "authority": 5,
    }

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in text.lower().replace("-", " ").split():
                topic = self._topics.get(token.strip(".,?!:;"))
                if topic is not None:
                    vector[topic] += 1.0
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector] if norm else vector)
        return vectors


CASES = (
    ("rotate credentials after security review", "Authentication token rotation"),
    ("local persistence database", "SQLite WAL persistence"),
    ("semantic retrieval vectors", "Postgres vector ranking"),
    ("exact query intent ranking", "Search ranking ladder"),
    ("background shutdown worker", "Daemon worker lifecycle"),
    ("relationship authority support", "Graph authority links"),
    ("repair semantic embeddings", "Embedding repair queue"),
    ("auth policy", "Authentication token rotation"),
)


def main() -> None:
    with TemporaryDirectory() as directory:
        db_path = Path(directory) / "search-quality.db"
        manager = DatabaseManager(db_path)
        try:
            repository = RelationalMemoryRepository(manager)
            for title, content, tags in (
                ("Authentication token rotation", "Rotate credentials after the security review.", "auth security"),
                ("SQLite WAL persistence", "Use SQLite storage with WAL for durable local persistence.", "database storage"),
                ("Postgres vector ranking", "Use Postgres server-side vector ranking for semantic retrieval.", "database search semantic vector"),
                ("Search ranking ladder", "Keyword and semantic search ranking should preserve exact query intent.", "search ranking retrieval"),
                ("Daemon worker lifecycle", "Background worker lifecycle and daemon shutdown behavior.", "daemon runtime worker"),
                ("Graph authority links", "Relationship links provide authority support for graph-aware ranking.", "graph relationships authority"),
                ("Embedding repair queue", "Repair missing semantic vector embeddings in the background.", "embedding semantic vector"),
                ("Generic architecture note", "Architecture decisions and implementation details.", "architecture"),
            ):
                repository.create_memory(
                    title,
                    content,
                    ["workspace"],
                    summary=content,
                    tags=tags.split(),
                    memory_type="fact",
                )

            service = RelationalMemorySearchService(
                repository,
                Config(),
                embedder=TopicEmbedder(),
                vector_store=SQLiteVectorStore(manager),
                db_manager=manager,
            )
            title_to_id = {
                record.title: record.id
                for record in repository.list_memories(limit=100)
            }
            golden = GoldenSet(
                [
                    GoldenEntry(query=query, relevant_ids=[title_to_id[title]])
                    for query, title in CASES
                ]
            )

            started = time.perf_counter()
            report = run_benchmark(
                golden,
                lambda query: [
                    result.memory_id
                    for result in service.search_memories(
                        query,
                        workspace_id="workspace",
                        limit=5,
                        side_effect_free=True,
                    )
                ],
                k=5,
                config=BenchmarkConfig(
                    backend="sqlite",
                    model_fingerprint=TopicEmbedder.model_name,
                    metadata={"case_count": len(CASES)},
                ),
            )
            payload = report.to_dict()
            payload["elapsed_seconds"] = round(time.perf_counter() - started, 6)
            print(json.dumps(payload, indent=2, sort_keys=True))
        finally:
            manager.close()


if __name__ == "__main__":
    main()
