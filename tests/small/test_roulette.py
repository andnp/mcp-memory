from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from mcp_memory.core.sampling import (
    ANOMALY_STRATEGY,
    BOUNDED_NOISE_STRATEGY,
    COLD_STORAGE_STRATEGY,
    COOLDOWN_ESCAPE_STRATEGY,
    NEVER_SURFACED_STRATEGY,
    ORPHAN_LOW_SUPPORT_STRATEGY,
    QUALITY_SIGNAL_STRATEGY,
    SEMANTIC_STRATEGY,
    RouletteProvider,
)

pytestmark = pytest.mark.small


@dataclass(frozen=True)
class _FakeRecord:
    id: str
    title: str
    content: str
    type: str = "fact"
    read_count: int = 0
    access_score: float = 0.0
    updated_at: str = "2026-03-17T00:00:00+00:00"
    last_accessed_at: str | None = None
    last_surfaced_at: str | None = None
    tags: list[str] = field(default_factory=list)


def test_roulette_anomaly_strategy_surfaces_size_extremes() -> None:
    candidates = [
        _FakeRecord(id="small", title="Small", content="tiny"),
        _FakeRecord(id="medium", title="Medium", content="m" * 40),
        _FakeRecord(id="large", title="Large", content="l" * 400),
    ]

    batch = RouletteProvider(task_name="memory-curator", task_id="task-1", candidates=candidates).get_batch(
        strategy=ANOMALY_STRATEGY,
        allowed_strategies=(ANOMALY_STRATEGY,),
        limit=2,
    )

    assert batch.strategy_used == ANOMALY_STRATEGY
    assert {record.id for record in batch.records} == {"small", "large"}


def test_roulette_strategy_choice_is_seeded_and_reproducible() -> None:
    candidates = [
        _FakeRecord(id="a", title="Alpha", content="alpha"),
        _FakeRecord(id="b", title="Beta", content="beta"),
    ]

    first = RouletteProvider(task_name="deduplicator", task_id="repeatable-task", candidates=candidates).get_batch(
        strategy=None,
        allowed_strategies=(SEMANTIC_STRATEGY, COLD_STORAGE_STRATEGY, BOUNDED_NOISE_STRATEGY),
        limit=1,
    )
    second = RouletteProvider(task_name="deduplicator", task_id="repeatable-task", candidates=candidates).get_batch(
        strategy=None,
        allowed_strategies=(SEMANTIC_STRATEGY, COLD_STORAGE_STRATEGY, BOUNDED_NOISE_STRATEGY),
        limit=1,
    )

    assert first.strategy_used == second.strategy_used
    assert [record.id for record in first.records] == [record.id for record in second.records]


def test_automatic_selection_uses_bounded_exploration_without_feedback() -> None:
    candidates = [_FakeRecord(id="a", title="Alpha", content="alpha")]

    batch = RouletteProvider(task_name="memory-curator", task_id="task-explore", candidates=candidates).get_batch(
        strategy=None,
        allowed_strategies=(SEMANTIC_STRATEGY, BOUNDED_NOISE_STRATEGY),
        limit=1,
    )

    assert batch.strategy_used == BOUNDED_NOISE_STRATEGY
    assert batch.strategy_selection_mode == "bounded_exploration"
    assert batch.strategy_selection_scores == {SEMANTIC_STRATEGY: 0.02, BOUNDED_NOISE_STRATEGY: 0.02}
    assert batch.strategy_selection_reason == (
        "selected=bounded-noise; fallback=no_priority_feedback; exploration=bounded"
    )


def test_priority_scores_select_the_best_sampler_and_explore_missing_feedback() -> None:
    candidates = [_FakeRecord(id="a", title="Alpha", content="alpha")]

    scored = RouletteProvider(
        task_name="memory-curator",
        task_id="task-priority",
        candidates=candidates,
        strategy_prior_scores={SEMANTIC_STRATEGY: 0.25, ANOMALY_STRATEGY: 0.8},
    ).get_batch(
        strategy=None,
        allowed_strategies=(SEMANTIC_STRATEGY, ANOMALY_STRATEGY),
        limit=1,
    )
    exploring = RouletteProvider(
        task_name="memory-curator",
        task_id="task-untested",
        candidates=candidates,
        strategy_prior_scores={SEMANTIC_STRATEGY: 0.0},
    ).get_batch(
        strategy=None,
        allowed_strategies=(SEMANTIC_STRATEGY, ANOMALY_STRATEGY),
        limit=1,
    )

    assert scored.strategy_used == ANOMALY_STRATEGY
    assert scored.strategy_selection_mode == "priority_scores"
    assert scored.strategy_selection_scores == {SEMANTIC_STRATEGY: 0.25, ANOMALY_STRATEGY: 0.8}
    assert "priority_score=0.8000" in (scored.strategy_selection_reason or "")
    assert exploring.strategy_used == ANOMALY_STRATEGY
    assert exploring.strategy_selection_mode == "priority_scores_with_exploration"
    assert exploring.strategy_selection_scores == {SEMANTIC_STRATEGY: 0.0, ANOMALY_STRATEGY: 0.02}


def test_curator_low_prior_explores_unmeasured_strategy() -> None:
    candidates = [_FakeRecord(id="a", title="Alpha", content="alpha")]

    batch = RouletteProvider(
        task_name="memory-curator",
        task_id="task-low-prior",
        candidates=candidates,
        strategy_prior_scores={SEMANTIC_STRATEGY: 0.4},
    ).get_batch(
        strategy=None,
        allowed_strategies=(SEMANTIC_STRATEGY, ANOMALY_STRATEGY, COLD_STORAGE_STRATEGY),
        limit=1,
    )

    assert batch.strategy_used != SEMANTIC_STRATEGY
    assert batch.strategy_selection_mode == "priority_scores_with_exploration"
    assert batch.strategy_selection_reason is not None
    assert "exploration=bounded_untested" in batch.strategy_selection_reason


def test_curator_quality_signal_exploration_uses_live_friction() -> None:
    candidates = [_FakeRecord(id="a", title="Alpha", content="alpha", type="observation")]

    batch = RouletteProvider(
        task_name="memory-curator",
        task_id="task-quality-explore",
        candidates=candidates,
        strategy_prior_scores={SEMANTIC_STRATEGY: 0.4},
    ).get_batch(
        strategy=None,
        allowed_strategies=(SEMANTIC_STRATEGY, QUALITY_SIGNAL_STRATEGY, ANOMALY_STRATEGY),
        limit=1,
    )

    assert batch.strategy_used == QUALITY_SIGNAL_STRATEGY
    assert "exploration=bounded_quality_signal" in (batch.strategy_selection_reason or "")


def test_roulette_selector_snapshot_captures_candidate_and_selected_feature_stats() -> None:
    candidates = [
        _FakeRecord(
            id="cold-large",
            title="Cold large",
            content="x" * 100,
            read_count=1,
            updated_at="2026-03-01T00:00:00+00:00",
            last_accessed_at="2026-02-01T00:00:00+00:00",
        ),
        _FakeRecord(
            id="warm-small",
            title="Warm small",
            content="tiny",
            read_count=9,
            updated_at="2026-03-30T00:00:00+00:00",
            last_accessed_at="2026-03-29T00:00:00+00:00",
            last_surfaced_at="2026-03-28T00:00:00+00:00",
        ),
    ]

    batch = RouletteProvider(
        task_name="memory-curator",
        task_id="task-snapshot",
        candidates=candidates,
        support_counts={"cold-large": 0, "warm-small": 3},
        now_timestamp=1_745_280_000.0,
    ).get_batch(
        strategy=COLD_STORAGE_STRATEGY,
        allowed_strategies=(COLD_STORAGE_STRATEGY,),
        limit=1,
    )

    assert batch.selector_feature_snapshot is not None
    snapshot = batch.selector_feature_snapshot
    assert snapshot["strategy_signals"] == {}
    assert snapshot["candidate_population"]["metrics"]["content_chars"]["min"] == 4.0
    assert snapshot["candidate_population"]["metrics"]["content_chars"]["max"] == 100.0
    assert snapshot["candidate_population"]["metrics"]["read_count"]["mean"] == 5.0
    assert snapshot["candidate_population"]["metrics"]["support_count"]["p90"] == 3.0
    assert snapshot["candidate_population"]["shares"]["never_accessed_share"] == 0.0
    assert snapshot["candidate_population"]["shares"]["low_support_share"] == 0.5
    assert snapshot["selected_population"]["count"] == 1
    assert snapshot["selected_population"]["metrics"]["support_count"]["mean"] == 0.0
    assert snapshot["selected_population"]["shares"]["never_surfaced_share"] == 1.0


def test_roulette_cold_and_never_surfaced_treat_missing_timestamps_as_oldest() -> None:
    candidates = [
        _FakeRecord(id="missing", title="Missing", content="missing timestamps"),
        _FakeRecord(
            id="recent",
            title="Recent",
            content="recent timestamps",
            last_accessed_at="2026-03-17T09:00:00+00:00",
            last_surfaced_at="2026-03-17T09:00:00+00:00",
        ),
    ]

    cold_batch = RouletteProvider(task_name="curator", task_id="task-cold", candidates=candidates).get_batch(
        strategy=COLD_STORAGE_STRATEGY,
        allowed_strategies=(COLD_STORAGE_STRATEGY,),
        limit=2,
    )
    surfaced_batch = RouletteProvider(task_name="curator", task_id="task-surfaced", candidates=candidates).get_batch(
        strategy=NEVER_SURFACED_STRATEGY,
        allowed_strategies=(NEVER_SURFACED_STRATEGY,),
        limit=2,
    )

    assert [record.id for record in cold_batch.records] == ["missing", "recent"]
    assert [record.id for record in surfaced_batch.records] == ["missing", "recent"]


def test_roulette_orphan_low_support_prefers_low_support_records() -> None:
    candidates = [
        _FakeRecord(id="well-supported", title="Canonical", content="canonical fact", read_count=10, access_score=5.0),
        _FakeRecord(id="orphan", title="Orphan", content="orphan fact", read_count=0, access_score=0.2),
    ]

    batch = RouletteProvider(
        task_name="memory-curator",
        task_id="task-orphan",
        candidates=candidates,
        support_counts={"well-supported": 3, "orphan": 0},
    ).get_batch(
        strategy=ORPHAN_LOW_SUPPORT_STRATEGY,
        allowed_strategies=(ORPHAN_LOW_SUPPORT_STRATEGY,),
        limit=1,
    )

    assert [record.id for record in batch.records] == ["orphan"]


def test_roulette_invalid_requested_strategy_falls_back_without_error() -> None:
    candidates = [
        _FakeRecord(id="a", title="Alpha", content="alpha"),
        _FakeRecord(id="b", title="Beta", content="beta"),
    ]

    batch = RouletteProvider(task_name="memory-curator", task_id="task-invalid", candidates=candidates).get_batch(
        strategy="totally-not-real",
        allowed_strategies=(COLD_STORAGE_STRATEGY, BOUNDED_NOISE_STRATEGY),
        limit=1,
    )

    assert batch.requested_strategy == "totally-not-real"
    assert batch.strategy_used in {COLD_STORAGE_STRATEGY, BOUNDED_NOISE_STRATEGY}
    assert batch.strategy_fallback_reason == "unknown_requested_strategy"


def test_curator_explicit_requested_strategy_still_wins_when_valid() -> None:
    candidates = [
        _FakeRecord(id="a", title="Alpha", content="a" * 1500),
        _FakeRecord(id="b", title="Beta", content="b" * 40),
    ]

    batch = RouletteProvider(task_name="memory-curator", task_id="task-requested", candidates=candidates).get_batch(
        strategy=ANOMALY_STRATEGY,
        allowed_strategies=(ANOMALY_STRATEGY, NEVER_SURFACED_STRATEGY),
        limit=1,
    )

    assert batch.requested_strategy == ANOMALY_STRATEGY
    assert batch.strategy_used == ANOMALY_STRATEGY
    assert batch.strategy_fallback_reason is None
    assert batch.strategy_selection_mode == "requested_strategy"


def test_invalid_requested_strategy_falls_back_to_bounded_exploration() -> None:
    candidates = [_FakeRecord(id="a", title="Alpha", content="alpha")]

    batch = RouletteProvider(task_name="memory-curator", task_id="task-invalid", candidates=candidates).get_batch(
        strategy="totally-not-real",
        allowed_strategies=(SEMANTIC_STRATEGY, BOUNDED_NOISE_STRATEGY),
        limit=1,
    )

    assert batch.requested_strategy == "totally-not-real"
    assert batch.strategy_used == BOUNDED_NOISE_STRATEGY
    assert batch.strategy_fallback_reason == "unknown_requested_strategy"
    assert batch.strategy_selection_mode == "bounded_exploration"


def test_taxonomist_explicit_requested_strategy_still_wins_when_valid() -> None:
    candidates = [
        _FakeRecord(
            id="never-1",
            title="Needs tags one",
            content="alpha",
            last_accessed_at="2026-03-31T00:00:00+00:00",
        ),
        _FakeRecord(
            id="never-2",
            title="Needs tags two",
            content="beta",
            last_accessed_at="2026-03-31T00:00:00+00:00",
        ),
    ]

    batch = RouletteProvider(task_name="taxonomist", task_id="task-requested", candidates=candidates).get_batch(
        strategy=BOUNDED_NOISE_STRATEGY,
        allowed_strategies=(COLD_STORAGE_STRATEGY, NEVER_SURFACED_STRATEGY, BOUNDED_NOISE_STRATEGY),
        limit=1,
    )

    assert batch.requested_strategy == BOUNDED_NOISE_STRATEGY
    assert batch.strategy_used == BOUNDED_NOISE_STRATEGY
    assert batch.strategy_fallback_reason is None
    assert batch.strategy_selection_mode == "requested_strategy"


def test_deduplicator_explicit_requested_strategy_still_wins_when_valid() -> None:
    candidates = [
        _FakeRecord(id="hot", title="Hot", content="hot", updated_at="1970-01-12T13:45:00+00:00"),
        _FakeRecord(id="cool", title="Cool", content="cool", updated_at="1970-01-01T00:00:00+00:00"),
    ]

    batch = RouletteProvider(task_name="deduplicator", task_id="task-requested", candidates=candidates).get_batch(
        strategy=COOLDOWN_ESCAPE_STRATEGY,
        allowed_strategies=(SEMANTIC_STRATEGY, COOLDOWN_ESCAPE_STRATEGY),
        limit=1,
    )

    assert batch.requested_strategy == COOLDOWN_ESCAPE_STRATEGY
    assert batch.strategy_used == COOLDOWN_ESCAPE_STRATEGY
    assert batch.strategy_fallback_reason is None
    assert batch.strategy_selection_mode == "requested_strategy"


def test_roulette_cooldown_escape_deprioritizes_recently_updated_records() -> None:
    now_timestamp = 1_000_000.0
    candidates = [
        _FakeRecord(id="hot", title="Hot", content="hot", updated_at="1970-01-12T13:45:00+00:00"),
        _FakeRecord(id="cool", title="Cool", content="cool", updated_at="1970-01-01T00:00:00+00:00"),
    ]

    batch = RouletteProvider(
        task_name="deduplicator",
        task_id="task-cooldown",
        candidates=candidates,
        cooldown_window_seconds=10_000.0,
        now_timestamp=now_timestamp,
    ).get_batch(
        strategy=COOLDOWN_ESCAPE_STRATEGY,
        allowed_strategies=(COOLDOWN_ESCAPE_STRATEGY,),
        limit=2,
    )

    assert [record.id for record in batch.records] == ["cool", "hot"]