"""Small, deterministic search-quality evaluation helpers."""

from .corpus import (
    DEFAULT_CORPUS_PATH,
    QueryClass,
    SearchQualityCase,
    SearchQualityCorpus,
    load_corpus,
)
from .metrics import (
    CaseMetrics,
    QueryClassMetrics,
    SearchObservation,
    SearchQualityMetrics,
    evaluate_corpus,
)
from .runner import (
    TopicEmbedder,
    run_in_process,
    run_search_quality,
    seed_search_quality_records,
)

__all__ = [
    "DEFAULT_CORPUS_PATH",
    "QueryClass",
    "SearchQualityCase",
    "SearchQualityCorpus",
    "CaseMetrics",
    "QueryClassMetrics",
    "SearchObservation",
    "SearchQualityMetrics",
    "evaluate_corpus",
    "load_corpus",
    "TopicEmbedder",
    "run_in_process",
    "run_search_quality",
    "seed_search_quality_records",
]
