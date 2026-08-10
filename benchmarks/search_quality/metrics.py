"""Deterministic metrics for the labeled search-quality corpus."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Mapping

from .corpus import QueryClass, SearchQualityCase, SearchQualityCorpus


@dataclass(frozen=True, slots=True)
class SearchObservation:
    """One search response expressed in stable evaluation labels."""

    result_labels: tuple[str, ...]
    latency_ms: float | None = None
    semantic_abstained: bool | None = None
    diagnostics: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class CaseMetrics:
    """Metric values for one corpus case."""

    evaluation_label: str
    query_class: QueryClass
    hit_at_1: bool
    hit_at_5: bool
    reciprocal_rank: float
    latency_ms: float | None
    semantic_abstained: bool | None


@dataclass(frozen=True, slots=True)
class QueryClassMetrics:
    """Aggregate metrics for one query class."""

    query_count: int
    hit_at_1: float
    hit_at_5: float
    mean_reciprocal_rank: float
    semantic_abstention_rate: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None


@dataclass(frozen=True, slots=True)
class SearchQualityMetrics:
    """Overall and query-class metrics for one corpus evaluation."""

    corpus_version: str
    query_count: int
    hit_at_1: float
    hit_at_5: float
    mean_reciprocal_rank: float
    semantic_abstention_rate: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    latency_max_ms: float | None
    by_query_class: Mapping[str, QueryClassMetrics]
    cases: tuple[CaseMetrics, ...]

    def to_mapping(self) -> dict[str, object]:
        """Serialize metrics with stable ordering for benchmark diffs."""
        return {
            "corpus_version": self.corpus_version,
            "query_count": self.query_count,
            "hit_at_1": self.hit_at_1,
            "hit_at_5": self.hit_at_5,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "semantic_abstention_rate": self.semantic_abstention_rate,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_max_ms": self.latency_max_ms,
            "by_query_class": {
                query_class: {
                    "query_count": summary.query_count,
                    "hit_at_1": summary.hit_at_1,
                    "hit_at_5": summary.hit_at_5,
                    "mean_reciprocal_rank": summary.mean_reciprocal_rank,
                    "semantic_abstention_rate": summary.semantic_abstention_rate,
                    "latency_p50_ms": summary.latency_p50_ms,
                    "latency_p95_ms": summary.latency_p95_ms,
                }
                for query_class, summary in sorted(self.by_query_class.items())
            },
            "cases": [
                {
                    "evaluation_label": case.evaluation_label,
                    "query_class": case.query_class.value,
                    "hit_at_1": case.hit_at_1,
                    "hit_at_5": case.hit_at_5,
                    "reciprocal_rank": case.reciprocal_rank,
                    "latency_ms": case.latency_ms,
                    "semantic_abstained": case.semantic_abstained,
                }
                for case in self.cases
            ],
        }


def evaluate_corpus(
    corpus: SearchQualityCorpus,
    observations: Mapping[str, SearchObservation],
) -> SearchQualityMetrics:
    """Calculate stable retrieval, abstention, and latency metrics."""
    missing = [
        entry.evaluation_label
        for entry in corpus.entries
        if entry.evaluation_label not in observations
    ]
    if missing:
        raise ValueError(f"missing observations: {', '.join(missing)}")

    cases = tuple(
        _case_metrics(entry, observations[entry.evaluation_label])
        for entry in corpus.entries
    )
    return SearchQualityMetrics(
        corpus_version=corpus.version,
        query_count=len(cases),
        hit_at_1=_mean(case.hit_at_1 for case in cases),
        hit_at_5=_mean(case.hit_at_5 for case in cases),
        mean_reciprocal_rank=_mean(case.reciprocal_rank for case in cases),
        semantic_abstention_rate=_abstention_rate(cases),
        latency_p50_ms=_percentile(_latencies(cases), 0.50),
        latency_p95_ms=_percentile(_latencies(cases), 0.95),
        latency_max_ms=max(_latencies(cases), default=None),
        by_query_class=_by_query_class(cases),
        cases=cases,
    )


def _case_metrics(entry: SearchQualityCase, observation: SearchObservation) -> CaseMetrics:
    query_class = entry.query_class
    expected_labels = set(entry.expected_labels)
    result_labels = observation.result_labels
    rank = next(
        (index for index, label in enumerate(result_labels, start=1) if label in expected_labels),
        None,
    )
    return CaseMetrics(
        evaluation_label=entry.evaluation_label,
        query_class=query_class,
        hit_at_1=rank == 1,
        hit_at_5=rank is not None and rank <= 5,
        reciprocal_rank=0.0 if rank is None else 1.0 / rank,
        latency_ms=observation.latency_ms,
        semantic_abstained=observation.semantic_abstained,
    )


def _by_query_class(cases: tuple[CaseMetrics, ...]) -> dict[str, QueryClassMetrics]:
    grouped: dict[QueryClass, list[CaseMetrics]] = {}
    for case in cases:
        grouped.setdefault(case.query_class, []).append(case)
    return {
        query_class.value: QueryClassMetrics(
            query_count=len(grouped[query_class]),
            hit_at_1=_mean(case.hit_at_1 for case in grouped[query_class]),
            hit_at_5=_mean(case.hit_at_5 for case in grouped[query_class]),
            mean_reciprocal_rank=_mean(
                case.reciprocal_rank for case in grouped[query_class]
            ),
            semantic_abstention_rate=_abstention_rate(grouped[query_class]),
            latency_p50_ms=_percentile(_latencies(grouped[query_class]), 0.50),
            latency_p95_ms=_percentile(_latencies(grouped[query_class]), 0.95),
        )
        for query_class in sorted(grouped, key=lambda value: value.value)
    }


def _abstention_rate(cases: Iterable[CaseMetrics]) -> float | None:
    values = [case.semantic_abstained for case in cases if case.semantic_abstained is not None]
    return None if not values else sum(values) / len(values)


def _latencies(cases: Iterable[CaseMetrics]) -> list[float]:
    return [case.latency_ms for case in cases if case.latency_ms is not None]


def _mean(values: Iterable[float | bool]) -> float:
    numbers = [float(value) for value in values]
    return sum(numbers) / len(numbers) if numbers else 0.0


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]
