from datetime import datetime, timezone

import pytest
from searchkernel.search.adaptive_limit import resolve_adaptive_result_limit

from mcp_memory.config import Config
from mcp_memory.core.ports.memory import MemoryRecord, RankedMemoryCandidate
from mcp_memory.core.search_ranking import (
    EXACT_IDENTIFIER_MATCH_MULTIPLIER,
    RankingEngine,
    RankingSignals,
    ScoringWeights,
    _has_exact_identifier_match,
)

pytestmark = pytest.mark.small


def _record(
    memory_id: str,
    workspace_ids: list[str],
    *,
    title: str | None = None,
    summary: str = "summary",
    tags: list[str] | None = None,
) -> MemoryRecord:
    now = datetime.now(timezone.utc).isoformat()
    return MemoryRecord(
        id=memory_id,
        title=title or memory_id,
        content="search content",
        summary=summary,
        type="fact",
        status="active",
        created_at=now,
        updated_at=now,
        read_count=0,
        access_score=0.0,
        last_accessed_at=None,
        last_surfaced_at=None,
        workspace_ids=workspace_ids,
        tags=tags or [],
    )


def test_fuse_reciprocal_rank_preserves_kernel_formula_and_configured_k() -> None:
    engine = RankingEngine(Config(), weights=ScoringWeights(rrf_k=10.0))

    assert engine.fuse_reciprocal_rank(["doc1"], []) == {"doc1": 1 / 11}
    assert engine.fuse_reciprocal_rank(["doc1", "doc2"], ["doc2", "doc1"]) == {
        "doc1": 1 / 11 + 1 / 12,
        "doc2": 1 / 12 + 1 / 11,
    }


def test_adaptive_result_bands_use_calibrated_scores() -> None:
    engine = RankingEngine(Config())
    scores = [
        engine.calibrate_score(rrf_score)
        for rrf_score in (0.05, 0.049, 0.048, 0.02)
    ]

    result_limit = resolve_adaptive_result_limit(
        scores,
        requested_limit=1,
        adaptive_enabled=True,
        maximum_limit=4,
        score_ratio_floor=0.7,
        minimum_score=0.0,
        maximum_score_gap=0.08,
    )

    assert result_limit == 3


def test_pure_ranking_uses_explicit_candidate_authority_without_storage() -> None:
    engine = RankingEngine(Config())
    supported = _record("supported", ["workspace-alpha"])
    unsupported = _record("unsupported", ["workspace-alpha"])
    candidates = [
        RankedMemoryCandidate(
            record=supported,
            incoming_links_count=2,
            incoming_link_type_counts={"DEPENDS_ON": 2},
        ),
        RankedMemoryCandidate(record=unsupported, incoming_links_count=0),
    ]

    ranked = engine.rank_records(
        candidates,
        {"supported": 0.04, "unsupported": 0.04},
        workspace_id="workspace-alpha",
        ranking_signals={
            "supported": RankingSignals(matched_by_keyword=True, keyword_token_coverage=1.0),
            "unsupported": RankingSignals(matched_by_keyword=True, keyword_token_coverage=1.0),
        },
        keyword_candidates_present=True,
    )

    assert ranked[0][0].id == "supported"
    assert ranked[0][1] > ranked[1][1]


def test_equal_scores_have_input_order_independent_id_tiebreaking() -> None:
    engine = RankingEngine(Config())
    records = [_record("record-b", []), _record("record-a", [])]
    rrf_scores = {record.id: 0.04 for record in records}

    ranked = engine.rank_records(records, rrf_scores)
    reversed_ranked = engine.rank_records(list(reversed(records)), rrf_scores)

    assert [record.id for record, _ in ranked] == ["record-a", "record-b"]
    assert [record.id for record, _ in reversed_ranked] == ["record-a", "record-b"]


@pytest.mark.parametrize(
    ("query", "title", "summary"),
    [
        ("SearchFilters", "SearchFilters compatibility", "query filters"),
        ("mcp-memory/config.toml", "Embedding configuration", "mcp-memory/config.toml selects Ollama"),
        ("qwen3-embedding:0.6b", "Embedding backend", "Uses qwen3-embedding:0.6b for local vectors"),
    ],
)
def test_exact_technical_identifier_matches_title_or_summary(
    query: str,
    title: str,
    summary: str,
) -> None:
    record = _record("record", [], title=title, summary=summary)

    assert _has_exact_identifier_match(query, record)


def test_exact_identifier_signal_ignores_tags_and_partial_matches() -> None:
    tag_only = _record(
        "tag-only",
        [],
        title="Embedding backend",
        summary="Uses a local embedding provider",
        tags=["qwen3-embedding:0.6b"],
    )
    partial = _record(
        "partial",
        [],
        title="SearchFilter compatibility",
        summary="query filters",
    )

    assert not _has_exact_identifier_match("qwen3-embedding:0.6b", tag_only)
    assert not _has_exact_identifier_match("SearchFilters", partial)


def test_exact_identifier_boost_orders_match_and_explains_signal() -> None:
    engine = RankingEngine(Config())
    exact = _record("exact", [], title="SearchFilters compatibility")
    generic = _record("generic", [], title="Query filtering guidance")
    exact_signals = RankingSignals(
        matched_by_keyword=True,
        keyword_token_coverage=1.0,
        exact_identifier_match=True,
    )
    generic_signals = RankingSignals(
        matched_by_keyword=True,
        keyword_token_coverage=1.0,
    )

    ranked = engine.rank_records(
        [generic, exact],
        {"generic": 0.04, "exact": 0.04},
        ranking_signals={"generic": generic_signals, "exact": exact_signals},
        keyword_candidates_present=True,
    )
    explanation = engine.explain_candidate(
        exact,
        0.04,
        signals=exact_signals,
        keyword_candidates_present=True,
    )

    assert ranked[0][0].id == "exact"
    assert explanation["exact_identifier_match"] is True
    assert explanation["exact_identifier_multiplier"] == EXACT_IDENTIFIER_MATCH_MULTIPLIER
    assert explanation["ranking_signal_multiplier"] == EXACT_IDENTIFIER_MATCH_MULTIPLIER
