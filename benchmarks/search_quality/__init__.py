"""Small, deterministic search-quality evaluation helpers."""

from .corpus import (
    DEFAULT_CORPUS_PATH,
    QueryClass,
    SearchQualityCase,
    SearchQualityCorpus,
    load_corpus,
)

__all__ = [
    "DEFAULT_CORPUS_PATH",
    "QueryClass",
    "SearchQualityCase",
    "SearchQualityCorpus",
    "load_corpus",
]
