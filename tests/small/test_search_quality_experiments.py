from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import pytest

from benchmarks.search_quality import (
    QueryClass,
    SearchObservation,
    SearchPolicyObservation,
    SearchQualityCase,
    SearchQualityCorpus,
    build_local_policy_builders,
    evaluate_corpus,
    load_corpus,
    run_policy_comparison,
    seed_search_quality_records,
)
from benchmarks.search_quality.runner import (
    SearchPolicyAcceptanceThresholds,
    SearchPolicyReport,
    evaluate_policy_acceptance,
)
from mcp_memory.config import Config, SearchKernelConfig
from mcp_memory.embeddings import SQLiteVectorStore
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.utils.db import DatabaseManager

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


def _complete_diagnostics() -> dict[str, object]:
    return {
        "timing_ms": {"search": 1.0},
        "candidate_counts": {"keyword": 1, "vector": 1},
        "degraded": False,
        "lane_decisions": {
            "enabled": ["keyword", "vector"],
            "budgets": {"keyword": 15, "vector": 15},
            "skipped": ["graph:awaiting_seed_confidence"],
        },
        "overlap": {
            "raw_lane_overlap_count": 1,
            "multi_lane_result_count": 1,
            "final_duplicate_count": 0,
        },
        "duplicate_candidate_ids": ["target"],
        "multi_lane_candidate_ids": ["target"],
        "final_duplicate_ids": [],
        "semantic": {
            "candidate_count": 1,
            "semantic_only_candidate_count": 1,
            "abstention_count": 0,
            "abstention_rate": 0.0,
        },
        "semantic_abstention_rate": 0.0,
    }


def _acceptance_corpus() -> SearchQualityCorpus:
    return SearchQualityCorpus(
        version="acceptance-test",
        entries=tuple(
            SearchQualityCase(
                case_id=query_class,
                evaluation_label=query_class,
                query=query_class,
                query_class=QueryClass(query_class),
                intent=query_class,
                workspace="workspace",
                expected_labels=("target",),
            )
            for query_class in ("exact", "broad", "historical")
        ),
    )


def _acceptance_report(
    policy: Literal["baseline", "calibrated_fusion", "query_expansion", "reranking"],
    *,
    exact_rank: int = 1,
    broad_rank: int = 1,
    historical_rank: int = 1,
    latency_ms: float = 10.0,
    degraded_case_count: int = 0,
    diagnostics: bool = False,
    duplicate_case_count: int = 0,
    semantic_abstained: bool | None = None,
) -> SearchPolicyReport:
    def labels(rank: int) -> tuple[str, ...]:
        return tuple("target" if index == rank else "other" for index in range(1, 6))

    corpus = _acceptance_corpus()
    metrics = evaluate_corpus(
        corpus,
        {
            "exact": SearchObservation(
                labels(exact_rank), latency_ms, diagnostics={"stage": "exact"}
                if diagnostics
                else None,
                semantic_abstained=semantic_abstained,
            ),
            "broad": SearchObservation(
                labels(broad_rank), latency_ms, diagnostics={"stage": "broad"}
                if diagnostics
                else None,
                semantic_abstained=semantic_abstained,
            ),
            "historical": SearchObservation(
                labels(historical_rank), latency_ms, diagnostics={"stage": "historical"}
                if diagnostics
                else None,
                semantic_abstained=semantic_abstained,
            ),
        },
    )
    return SearchPolicyReport(
        policy=policy,
        status="completed",
        metrics=metrics,
        degraded_case_count=degraded_case_count,
        degradation_rate=degraded_case_count / 3,
        diagnostics_complete=diagnostics,
        duplicate_case_count=duplicate_case_count,
    )


def test_policy_acceptance_passes_with_stable_serializable_decision() -> None:
    """Accept a candidate that preserves exact quality and meets all gates."""
    decision = evaluate_policy_acceptance(
        _acceptance_report("query_expansion", latency_ms=12.0),
        _acceptance_report("baseline", latency_ms=10.0),
        thresholds=SearchPolicyAcceptanceThresholds(max_p95_latency_ms=12.0),
    )

    assert decision.accepted
    assert decision.reasons == ()
    assert decision.to_mapping() == {
        "accepted": True,
        "candidate_policy": "query_expansion",
        "baseline_policy": "baseline",
        "reasons": [],
        "thresholds": {
            "min_broad_hit_at_5_delta": 0.0,
            "min_historical_hit_at_5_delta": 0.0,
            "max_degradation_rate_delta": 0.0,
            "max_p95_latency_ms": 12.0,
        },
    }


@pytest.mark.parametrize(
    ("candidate", "thresholds", "reason"),
    [
        (
            _acceptance_report("query_expansion", exact_rank=2),
            SearchPolicyAcceptanceThresholds(),
            "exact_hit_at_1_not_preserved",
        ),
        (
            _acceptance_report("query_expansion", broad_rank=6),
            SearchPolicyAcceptanceThresholds(),
            "broad_hit_at_5_below_threshold",
        ),
        (
            _acceptance_report("query_expansion", historical_rank=6),
            SearchPolicyAcceptanceThresholds(),
            "historical_hit_at_5_below_threshold",
        ),
        (
            _acceptance_report("query_expansion", degraded_case_count=1),
            SearchPolicyAcceptanceThresholds(),
            "degradation_rate_above_threshold",
        ),
        (
            _acceptance_report("query_expansion", latency_ms=20.0),
            SearchPolicyAcceptanceThresholds(max_p95_latency_ms=15.0),
            "candidate_p95_latency_above_threshold",
        ),
    ],
)
def test_policy_acceptance_rejects_each_failed_gate(
    candidate: SearchPolicyReport,
    thresholds: SearchPolicyAcceptanceThresholds,
    reason: str,
) -> None:
    """Reject candidates that violate any named acceptance gate."""
    decision = evaluate_policy_acceptance(
        candidate,
        _acceptance_report("baseline"),
        thresholds=thresholds,
    )

    assert not decision.accepted
    assert reason in decision.reasons


def test_policy_acceptance_fails_closed_for_missing_metrics() -> None:
    """Return an explicit failure when a completed report lacks metrics."""
    candidate = SearchPolicyReport(
        policy="query_expansion",
        status="completed",
        metrics=None,
        degraded_case_count=0,
        degradation_rate=0.0,
    )

    decision = evaluate_policy_acceptance(candidate, _acceptance_report("baseline"))

    assert not decision.accepted
    assert decision.reasons == ("candidate_metrics_missing",)


def test_policy_acceptance_fails_closed_for_skipped_reranking() -> None:
    """Do not accept an optional reranking policy that was skipped."""
    candidate = SearchPolicyReport(
        policy="reranking",
        status="skipped",
        metrics=None,
        degraded_case_count=0,
        degradation_rate=None,
        skip_reason="reranker_not_supplied",
    )

    decision = evaluate_policy_acceptance(candidate, _acceptance_report("baseline"))

    assert not decision.accepted
    assert decision.reasons == ("candidate_not_completed", "candidate_metrics_missing")


def test_policy_acceptance_requires_diagnostics_when_opted_in() -> None:
    """Reject completed reports without observations diagnostics when required."""
    decision = evaluate_policy_acceptance(
        _acceptance_report("query_expansion", diagnostics=False),
        _acceptance_report("baseline", diagnostics=True),
        thresholds=SearchPolicyAcceptanceThresholds(require_diagnostics=True),
    )

    assert not decision.accepted
    assert decision.reasons == ("candidate_diagnostics_missing",)
    assert decision.to_mapping()["diagnostics_complete"] is False
    assert decision.to_mapping()["thresholds"] == {
        "min_broad_hit_at_5_delta": 0.0,
        "min_historical_hit_at_5_delta": 0.0,
        "max_degradation_rate_delta": 0.0,
        "max_p95_latency_ms": None,
        "require_diagnostics": True,
    }


def test_policy_acceptance_passes_with_required_diagnostics() -> None:
    """Accept completed reports when every observation has diagnostics."""
    decision = evaluate_policy_acceptance(
        _acceptance_report("query_expansion", diagnostics=True),
        _acceptance_report("baseline", diagnostics=True),
        thresholds=SearchPolicyAcceptanceThresholds(require_diagnostics=True),
    )

    assert decision.accepted
    assert decision.diagnostics_complete is True


@pytest.mark.parametrize(
    ("candidate_count", "baseline_count", "reason"),
    [
        (1, 0, "candidate_duplicate_case_count_above_threshold"),
        (0, 1, "baseline_duplicate_case_count_above_threshold"),
    ],
)
def test_policy_acceptance_rejects_duplicate_case_threshold(
    candidate_count: int,
    baseline_count: int,
    reason: str,
) -> None:
    """Reject completed reports exceeding the configured duplicate threshold."""
    decision = evaluate_policy_acceptance(
        _acceptance_report(
            "query_expansion", duplicate_case_count=candidate_count
        ),
        _acceptance_report("baseline", duplicate_case_count=baseline_count),
        thresholds=SearchPolicyAcceptanceThresholds(max_duplicate_case_count=0),
    )

    assert not decision.accepted
    assert decision.reasons == (reason,)
    assert decision.to_mapping()["thresholds"] == {
        "min_broad_hit_at_5_delta": 0.0,
        "min_historical_hit_at_5_delta": 0.0,
        "max_degradation_rate_delta": 0.0,
        "max_p95_latency_ms": None,
        "max_duplicate_case_count": 0,
    }


def test_policy_acceptance_preserves_defaults_without_duplicate_threshold() -> None:
    """Keep the existing decision mapping when duplicate gating is disabled."""
    decision = evaluate_policy_acceptance(
        _acceptance_report("query_expansion", duplicate_case_count=1),
        _acceptance_report("baseline"),
    )

    assert decision.accepted
    thresholds = decision.to_mapping()["thresholds"]
    assert isinstance(thresholds, dict)
    assert "max_duplicate_case_count" not in thresholds


def test_policy_acceptance_rejects_semantic_abstention_delta() -> None:
    """Reject a candidate whose abstention rate exceeds the configured delta."""
    decision = evaluate_policy_acceptance(
        _acceptance_report("query_expansion", semantic_abstained=True),
        _acceptance_report("baseline", semantic_abstained=False),
        thresholds=SearchPolicyAcceptanceThresholds(
            max_semantic_abstention_rate_delta=0.5
        ),
    )

    assert not decision.accepted
    assert decision.reasons == (
        "candidate_semantic_abstention_rate_above_threshold",
    )
    assert decision.to_mapping()["thresholds"] == {
        "min_broad_hit_at_5_delta": 0.0,
        "min_historical_hit_at_5_delta": 0.0,
        "max_degradation_rate_delta": 0.0,
        "max_p95_latency_ms": None,
        "max_semantic_abstention_rate_delta": 0.5,
    }


@pytest.mark.parametrize(
    ("candidate_abstained", "baseline_abstained", "reason"),
    [
        (True, None, "baseline_semantic_abstention_rate_missing"),
        (None, False, "candidate_semantic_abstention_rate_missing"),
    ],
)
def test_policy_acceptance_fails_closed_for_missing_semantic_rates(
    candidate_abstained: bool | None,
    baseline_abstained: bool | None,
    reason: str,
) -> None:
    """Reject configured abstention gating when either rate is unavailable."""
    decision = evaluate_policy_acceptance(
        _acceptance_report(
            "query_expansion", semantic_abstained=candidate_abstained
        ),
        _acceptance_report("baseline", semantic_abstained=baseline_abstained),
        thresholds=SearchPolicyAcceptanceThresholds(
            max_semantic_abstention_rate_delta=0.5
        ),
    )

    assert not decision.accepted
    assert reason in decision.reasons


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


def test_policy_comparison_carries_observed_diagnostic_contract() -> None:
    """Retain planner evidence instead of treating callback execution as proof."""

    def build():
        def search(_case: SearchQualityCase) -> SearchPolicyObservation:
            return SearchPolicyObservation(
                SearchObservation(
                    ("target",),
                    diagnostics=_complete_diagnostics(),
                    semantic_abstained=False,
                )
            )

        return search

    comparison = run_policy_comparison(_corpus(), baseline_builder=build)
    report = comparison.by_policy["baseline"]
    observation = report.observations["exact"]

    assert report.policy_fingerprint
    assert report.corpus_fingerprint == comparison.corpus_fingerprint
    assert observation.policy_fingerprint == report.policy_fingerprint
    assert observation.corpus_fingerprint == report.corpus_fingerprint
    assert observation.lane_decisions == _complete_diagnostics()["lane_decisions"]
    assert observation.stage_timings_ms == {"search": 1.0}
    assert observation.diagnostics_complete
    assert observation.degraded is False
    assert observation.duplicate_semantics == {
        "duplicate_candidate_ids": ["target"],
        "multi_lane_candidate_ids": ["target"],
        "final_duplicate_ids": [],
        "overlap": _complete_diagnostics()["overlap"],
    }
    assert observation.semantic_abstention_available is True
    assert observation.semantic_abstention_rate == 0.0


def test_policy_comparison_fails_diagnostics_gate_for_incomplete_observations() -> None:
    """Reject a completed-looking experiment when required evidence is absent."""
    comparison = run_policy_comparison(
        _corpus(),
        baseline_builder=_builder("baseline", [], []),
    )
    report = comparison.by_policy["baseline"]

    decision = evaluate_policy_acceptance(
        report,
        report,
        thresholds=SearchPolicyAcceptanceThresholds(require_diagnostics=True),
    )

    assert report.diagnostics_complete is False
    assert decision.accepted is False
    assert "baseline_diagnostics_missing" in decision.reasons
    assert "candidate_diagnostics_missing" in decision.reasons


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


def test_local_builders_default_to_baseline_and_stable_metadata() -> None:
    """Build only baseline by default with reproducible policy metadata."""
    with TemporaryDirectory() as directory:
        manager = DatabaseManager(Path(directory) / "search-quality.db")
        try:
            repository = RelationalMemoryRepository(manager)
            label_to_id = seed_search_quality_records(repository)
            builders = build_local_policy_builders(
                repository,
                label_to_id,
                db_manager=manager,
                vector_store=SQLiteVectorStore(manager),
            )

            assert builders.calibrated_fusion_builder is None
            assert builders.query_expansion_builder is None
            assert builders.reranking_builder is None
            assert list(builders.identities) == ["baseline"]
            assert builders.metadata() == {
                "baseline": {
                    "policy": "baseline",
                    "identity": "mcp-memory-search-quality-v1:baseline",
                    "fingerprint": "baseline",
                }
            }
            assert builders.baseline_builder() is not builders.baseline_builder()
        finally:
            manager.close()


def test_local_builders_apply_requested_config_overrides() -> None:
    """Construct requested experimental policies without changing the base config."""
    with TemporaryDirectory() as directory:
        manager = DatabaseManager(Path(directory) / "search-quality.db")
        try:
            repository = RelationalMemoryRepository(manager)
            label_to_id = seed_search_quality_records(repository)
            config = Config(
                searchkernel=SearchKernelConfig(query_expansion_policy="synonym")
            )
            builders = build_local_policy_builders(
                repository,
                label_to_id,
                db_manager=manager,
                vector_store=SQLiteVectorStore(manager),
                config=config,
                requested_policies=("calibrated_fusion", "query_expansion"),
            )

            assert builders.calibrated_fusion_builder is not None
            assert builders.query_expansion_builder is not None
            assert set(builders.identities) == {
                "baseline",
                "calibrated_fusion",
                "query_expansion",
            }
            assert builders.identities["baseline"].fingerprint == "baseline"
            assert (
                builders.identities["calibrated_fusion"].fingerprint
                == "calibrated-fusion"
            )
            assert (
                builders.identities["query_expansion"].fingerprint
                == "query-expansion:synonym"
            )
            assert config.searchkernel.active_feature_fingerprint() == (
                "rerank:cross_encoder:10:BAAI/bge-reranker-v2-m3"
            )

            comparison = run_policy_comparison(
                load_corpus(),
                **builders.comparison_kwargs(),
            )
            assert all(
                report.status == "completed"
                for report in comparison.policies[:3]
            )
            assert all(
                report.metrics is not None
                for report in comparison.policies[:3]
            )
            assert [
                report.policy_fingerprint for report in comparison.policies[:3]
            ] == [
                builders.identities[policy].fingerprint
                for policy in ("baseline", "calibrated_fusion", "query_expansion")
            ]
            assert all(
                report.corpus_fingerprint == comparison.corpus_fingerprint
                for report in comparison.policies
            )
            assert all(
                report.diagnostics_complete for report in comparison.policies[:3]
            )
        finally:
            manager.close()


def test_local_builder_reranking_is_skipped_without_dependency() -> None:
    """Leave reranking unbuilt when the local adapter has no reranker."""
    with TemporaryDirectory() as directory:
        manager = DatabaseManager(Path(directory) / "search-quality.db")
        try:
            repository = RelationalMemoryRepository(manager)
            label_to_id = seed_search_quality_records(repository)
            builders = build_local_policy_builders(
                repository,
                label_to_id,
                db_manager=manager,
                vector_store=SQLiteVectorStore(manager),
                requested_policies=("reranking",),
            )
            comparison = run_policy_comparison(
                SearchQualityCorpus(version="empty", entries=()),
                baseline_builder=builders.baseline_builder,
                calibrated_fusion_builder=builders.calibrated_fusion_builder,
                query_expansion_builder=builders.query_expansion_builder,
                reranking_builder=builders.reranking_builder,
            )

            reranking = comparison.by_policy["reranking"]
            assert builders.reranking_builder is None
            assert reranking.status == "skipped"
            assert reranking.skip_reason == "builder_not_supplied"
        finally:
            manager.close()
