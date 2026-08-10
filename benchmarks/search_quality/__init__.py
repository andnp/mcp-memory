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
]
