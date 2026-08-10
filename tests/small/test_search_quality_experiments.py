from __future__ import annotations

import pytest

from benchmarks.search_quality import (
    QueryClass,
    SearchObservation,
    SearchPolicyObservation,
    SearchQualityCase,
    SearchQualityCorpus,
    run_policy_comparison,
)


pytestmark = pytest.mark.small


def _corpus() -> SearchQualityCorpus:
    return SearchQualityCorpus(
        version="experiment-test",
        entries=(
            SearchQualityCase(
                case_id="exact-case",
                evaluation_label="exact",
                query="exact query",
                query_class=QueryClass.EXACT,
                intent="find the exact record",
                workspace="workspace",
                expected_labels=("target",),
            ),
            SearchQualityCase(
                case_id="broad-case",
                evaluation_label="broad",
                query="broad query",
                query_class=QueryClass.BROAD,
                intent="find a related record",
                workspace="workspace",
                expected_labels=("target",),
            ),
            SearchQualityCase(
                case_id="degraded-case",
                evaluation_label="degraded",
                query="degraded query",
                query_class=QueryClass.DEGRADED,
                intent="measure degraded retrieval",
                workspace="workspace",
                expected_labels=("target",),
            ),
        ),
    )


def _builder(
    name: str,
    built: list[str],
    calls: list[tuple[str, str]],
    *,
    degraded_label: str | None = None,
):
    def build():
        built.append(name)

        def search(case: SearchQualityCase) -> SearchPolicyObservation:
            calls.append((name, case.evaluation_label))
            labels = ("other", "target") if case.evaluation_label == "broad" else ("target",)
            return SearchPolicyObservation(
                SearchObservation(
                    labels,
                    latency_ms=float(len(calls)),
                    semantic_abstained=case.evaluation_label == "broad",
                ),
                degraded=case.evaluation_label == degraded_label,
            )

        return search

    return build


def test_policy_comparison_runs_isolated_builders_in_stable_order() -> None:
    """Evaluate every supplied policy over the same ordered corpus.

    Each builder creates a separate runner, while metrics retain query-class,
    latency, degradation, and abstention information for that policy.
    """
    built: list[str] = []
    calls: list[tuple[str, str]] = []
    comparison = run_policy_comparison(
        _corpus(),
        baseline_builder=_builder("baseline", built, calls),
        calibrated_fusion_builder=_builder("calibrated", built, calls),
        query_expansion_builder=_builder(
            "expanded", built, calls, degraded_label="degraded"
        ),
        reranking_builder=lambda _reranker: _builder("reranked", built, calls)(),
        reranker=object(),
    )

    assert [report.policy for report in comparison.policies] == [
        "baseline",
        "calibrated_fusion",
        "query_expansion",
        "reranking",
    ]
    assert built == ["baseline", "calibrated", "expanded", "reranked"]
    assert calls[:3] == [("baseline", "exact"), ("baseline", "broad"), ("baseline", "degraded")]

    baseline = comparison.by_policy["baseline"]
    assert baseline.metrics is not None
    assert baseline.metrics.hit_at_1 == pytest.approx(2 / 3)
    assert baseline.metrics.hit_at_5 == 1.0
    assert baseline.metrics.mean_reciprocal_rank == pytest.approx(5 / 6)
    assert baseline.metrics.semantic_abstention_rate == pytest.approx(1 / 3)
    assert baseline.metrics.by_query_class["broad"].latency_p95_ms == 2.0
    assert baseline.degraded_case_count == 0
    assert baseline.degradation_rate == 0.0

    expanded = comparison.by_policy["query_expansion"]
    assert expanded.degraded_case_count == 1
    assert expanded.degradation_rate == pytest.approx(1 / 3)
    serialized_policies = comparison.to_mapping()["policies"]
    assert isinstance(serialized_policies, list)
    assert isinstance(serialized_policies[-1], dict)
    assert serialized_policies[-1]["status"] == "completed"


def test_policy_comparison_skips_reranking_without_a_reranker() -> None:
    """Do not construct a reranking policy when its dependency is absent."""
    reranking_builds: list[object] = []

    def reranking_builder(reranker: object):
        reranking_builds.append(reranker)
        return _builder("reranked", [], [])()

    comparison = run_policy_comparison(
        _corpus(),
        baseline_builder=_builder("baseline", [], []),
        reranking_builder=reranking_builder,
    )

    reranking = comparison.by_policy["reranking"]
    assert reranking.status == "skipped"
    assert reranking.skip_reason == "reranker_not_supplied"
    assert reranking.metrics is None
    assert reranking.degradation_rate is None
    assert reranking_builds == []


def test_policy_comparison_injects_supplied_reranker() -> None:
    """Pass the supplied reranker only to the reranking builder."""
    reranker = object()
    received: list[object] = []

    def reranking_builder(supplied: object):
        received.append(supplied)
        return _builder("reranked", [], [])()

    comparison = run_policy_comparison(
        _corpus(),
        baseline_builder=_builder("baseline", [], []),
        reranking_builder=reranking_builder,
        reranker=reranker,
    )

    assert received == [reranker]
    assert comparison.by_policy["reranking"].status == "completed"


def test_policy_comparison_reports_missing_optional_builders() -> None:
    """Represent unavailable experiments explicitly instead of hiding them."""
    comparison = run_policy_comparison(
        _corpus(),
        baseline_builder=_builder("baseline", [], []),
    )

    assert comparison.by_policy["calibrated_fusion"].skip_reason == "builder_not_supplied"
    assert comparison.by_policy["query_expansion"].skip_reason == "builder_not_supplied"
    assert comparison.by_policy["reranking"].skip_reason == "builder_not_supplied"
