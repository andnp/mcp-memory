"""Run the labeled search-quality corpus in a local SQLite process."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

from mcp_memory.config import Config
from mcp_memory.embeddings import SQLiteVectorStore
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.relational.search import RelationalMemorySearchService
from mcp_memory.utils.db import DatabaseManager

from .corpus import SearchQualityCase, SearchQualityCorpus, load_corpus
from .metrics import SearchObservation, SearchQualityMetrics, evaluate_corpus


class TopicEmbedder:
    """Fixed topic encoder that requires no model download or service."""

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
        """Encode text into normalized topic vectors."""
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


SearchCaseRunner = Callable[[SearchQualityCase], SearchObservation]


def run_search_quality(
    corpus: SearchQualityCorpus,
    search_case: SearchCaseRunner,
) -> SearchQualityMetrics:
    """Evaluate a corpus through a caller-provided in-process search function."""
    observations = {
        case.evaluation_label: search_case(case) for case in corpus.entries
    }
    return evaluate_corpus(corpus, observations)


def run_in_process(
    corpus: SearchQualityCorpus | None = None,
) -> SearchQualityMetrics:
    """Run the default corpus against temporary relational and vector stores."""
    selected_corpus = corpus or load_corpus()
    with TemporaryDirectory() as directory:
        manager = DatabaseManager(Path(directory) / "search-quality.db")
        try:
            repository = RelationalMemoryRepository(manager)
            label_to_id = _seed_records(repository)
            service = RelationalMemorySearchService(
                repository,
                Config(),
                embedder=TopicEmbedder(),
                vector_store=SQLiteVectorStore(manager),
                db_manager=manager,
            )
            id_to_label = {memory_id: label for label, memory_id in label_to_id.items()}

            def search_case(case: SearchQualityCase) -> SearchObservation:
                started = time.perf_counter()
                results = service.search_memories(
                    case.query,
                    workspace_id=case.workspace,
                    limit=case.acceptable_top_k,
                    side_effect_free=True,
                )
                return SearchObservation(
                    result_labels=tuple(
                        id_to_label.get(result.memory_id, "unknown")
                        for result in results
                    ),
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )

            return run_search_quality(selected_corpus, search_case)
        finally:
            manager.close()


def _seed_records(repository: RelationalMemoryRepository) -> dict[str, str]:
    records = (
        (
            "authentication-token-rotation",
            "Authentication token rotation",
            "Rotate credentials after the security review.",
            "auth security",
        ),
        (
            "sqlite-wal-persistence",
            "SQLite WAL persistence",
            "Use SQLite storage with WAL for durable local persistence.",
            "database storage",
        ),
        (
            "postgres-vector-ranking",
            "Postgres vector ranking",
            "Use Postgres server-side vector ranking for semantic retrieval.",
            "database search semantic vector",
        ),
        (
            "search-ranking-ladder",
            "Search ranking ladder",
            "Keyword and semantic search ranking should preserve exact query intent.",
            "search ranking retrieval",
        ),
        (
            "daemon-worker-lifecycle",
            "Daemon worker lifecycle",
            "Background worker lifecycle and daemon shutdown behavior.",
            "daemon runtime worker",
        ),
        (
            "graph-authority-links",
            "Graph authority links",
            "Relationship links provide authority support for graph-aware ranking.",
            "graph relationships authority",
        ),
        (
            "embedding-repair-queue",
            "Embedding repair queue",
            "Repair missing semantic vector embeddings in the background.",
            "embedding semantic vector",
        ),
        (
            "generic-architecture-note",
            "Generic architecture note",
            "Architecture decisions and implementation details.",
            "architecture",
        ),
    )
    label_to_id: dict[str, str] = {}
    for label, title, content, tags in records:
        created = repository.create_memory(
            title,
            content,
            ["workspace"],
            summary=content,
            tags=tags.split(),
            memory_type="fact",
        )
        if created is None:
            raise RuntimeError(f"failed to create benchmark record: {label}")
        label_to_id[label] = created.id
    return label_to_id
