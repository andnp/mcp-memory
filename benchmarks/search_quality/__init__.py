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
    LocalPolicyBuilders,
    SearchPolicyComparison,
    SearchPolicyIdentity,
    SearchPolicyObservation,
    SearchPolicyReport,
    TopicEmbedder,
    build_local_policy_builders,
    run_in_process,
    run_policy_comparison,
    run_search_quality,
    seed_search_quality_records,
)

__all__ = [
    "DEFAULT_CORPUS_PATH",
    "CaseMetrics",
    "LocalPolicyBuilders",
    "QueryClass",
    "QueryClassMetrics",
    "SearchObservation",
    "SearchPolicyComparison",
    "SearchPolicyIdentity",
    "SearchPolicyObservation",
    "SearchPolicyReport",
    "SearchQualityCase",
    "SearchQualityCorpus",
    "SearchQualityMetrics",
    "TopicEmbedder",
    "build_local_policy_builders",
    "evaluate_corpus",
    "load_corpus",
    "run_in_process",
    "run_policy_comparison",
    "run_search_quality",
    "seed_search_quality_records",
]
