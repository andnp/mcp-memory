from __future__ import annotations

import pytest

from benchmarks.search_quality import (
    QueryClass,
    SearchObservation,
    SearchQualityCase,
    SearchQualityCorpus,
    evaluate_corpus,
    load_corpus,
    run_in_process,
    run_search_quality,
)


pytestmark = pytest.mark.small


def _case(
    label: str,
    query_class: QueryClass = QueryClass.EXACT,
    *,
    expected_labels: tuple[str, ...] = ("target",),
) -> SearchQualityCase:
    return SearchQualityCase(
        case_id=f"case-{label}",
        evaluation_label=label,
        query=f"query for {label}",
        query_class=query_class,
        intent="retrieve a labeled record",
        workspace="workspace",
        expected_labels=expected_labels,
    )


def test_search_quality_corpus_fixture_has_stable_labels() -> None:
    """Load the versioned fixture without private memory payloads.

    Stable labels are the evaluation oracle for local benchmark records.
    """
    corpus = load_corpus()

    assert corpus.version == "search-quality-v1"
    assert len(corpus.entries) == 9
    assert len({entry.evaluation_label for entry in corpus.entries}) == 9
    assert all(entry.workspace == "workspace" for entry in corpus.entries)


def test_search_quality_corpus_rejects_duplicate_labels() -> None:
    """Reject ambiguous labels before they can distort aggregate metrics."""
    first = _case("duplicate")
    second = _case("duplicate")

    with pytest.raises(ValueError, match="case_id values must be unique"):
        SearchQualityCorpus(version="test", entries=(first, second))


def test_search_quality_metrics_calculate_retrieval_and_latency() -> None:
    """Calculate ranking metrics from labeled results deterministically.

    Optional abstention values contribute only when the runner supplies them.
    """
    corpus = SearchQualityCorpus(
        version="test",
        entries=(
            _case("second", QueryClass.EXACT),
            _case("first", QueryClass.BROAD),
            _case("miss", QueryClass.HISTORICAL),
        ),
    )
    metrics = evaluate_corpus(
        corpus,
        {
            "second": SearchObservation(("other", "target"), 10.0, False),
            "first": SearchObservation(("target",), 20.0, True),
            "miss": SearchObservation(("other",), None),
        },
    )

    assert metrics.hit_at_1 == pytest.approx(1 / 3)
    assert metrics.hit_at_5 == pytest.approx(2 / 3)
    assert metrics.mean_reciprocal_rank == pytest.approx(0.5)
    assert metrics.semantic_abstention_rate == pytest.approx(0.5)
    assert metrics.latency_p50_ms == 10.0
    assert metrics.latency_p95_ms == 20.0
    assert metrics.latency_max_ms == 20.0
    assert metrics.by_query_class["exact"].mean_reciprocal_rank == 0.5
    assert metrics.by_query_class["historical"].hit_at_5 == 0.0


def test_search_quality_metrics_require_every_corpus_observation() -> None:
    """Fail clearly when a runner omits a corpus case."""
    corpus = SearchQualityCorpus(version="test", entries=(_case("missing"),))

    with pytest.raises(ValueError, match="missing observations: missing"):
        evaluate_corpus(corpus, {})


def test_search_observation_diagnostics_are_optional_and_preserved() -> None:
    """Keep diagnostic payloads available without changing scored metrics."""
    corpus = SearchQualityCorpus(version="test", entries=(_case("diagnostic"),))
    diagnostics = {"candidate_counts": {"keyword": 2}, "degraded": False}
    observation = SearchObservation(("target",), diagnostics=diagnostics)

    metrics = evaluate_corpus(corpus, {"diagnostic": observation})

    assert observation.diagnostics == diagnostics
    assert metrics.cases[0].hit_at_1
    serialized_cases = metrics.to_mapping()["cases"]
    assert isinstance(serialized_cases, list)
    assert serialized_cases[0] == {
        "evaluation_label": "diagnostic",
        "query_class": "exact",
        "hit_at_1": True,
        "hit_at_5": True,
        "reciprocal_rank": 1.0,
        "latency_ms": None,
        "semantic_abstained": None,
    }


def test_search_quality_runner_evaluates_cases_in_corpus_order() -> None:
    """Keep runner invocation order aligned with the versioned corpus."""
    corpus = SearchQualityCorpus(
        version="test",
        entries=(_case("first"), _case("second", QueryClass.BROAD)),
    )
    seen: list[str] = []

    def search_case(case: SearchQualityCase) -> SearchObservation:
        seen.append(case.evaluation_label)
        return SearchObservation(("target",))

    metrics = run_search_quality(corpus, search_case)

    assert seen == ["first", "second"]
    assert metrics.query_count == 2
    assert metrics.hit_at_1 == 1.0


def test_search_quality_in_process_runner_uses_local_search_path() -> None:
    """Exercise the real local benchmark path without external services."""
    metrics = run_in_process(load_corpus())

    assert metrics.corpus_version == "search-quality-v1"
    assert metrics.query_count == 9
    assert metrics.hit_at_1 == 1.0
    assert metrics.hit_at_5 == 1.0
    assert all(case.latency_ms is not None for case in metrics.cases)
    assert metrics.semantic_abstention_rate is None
