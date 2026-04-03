from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from mcp_memory.core.sampling import (
    ANOMALY_STRATEGY,
    BOUNDED_NOISE_STRATEGY,
    CONFLICT_FRONTIER_STRATEGY,
    COLD_STORAGE_STRATEGY,
    COOLDOWN_ESCAPE_STRATEGY,
    GRAPH_BRIDGE_STRATEGY,
    NEVER_SURFACED_STRATEGY,
    ORPHAN_LOW_SUPPORT_STRATEGY,
    RouletteProvider,
    SEMANTIC_STRATEGY,
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


def test_curator_strategy_selection_is_deterministic_and_signal_driven_without_request() -> None:
    candidates = [
        _FakeRecord(
            id="never-1",
            title="Alpha One",
            content="alpha",
            last_accessed_at="2026-03-31T00:00:00+00:00",
        ),
        _FakeRecord(
            id="never-2",
            title="Zeta Two",
            content="beta",
            last_accessed_at="2026-03-31T00:00:00+00:00",
        ),
    ]

    first = RouletteProvider(task_name="memory-curator", task_id="task-a", candidates=candidates).get_batch(
        strategy=None,
        allowed_strategies=(
            SEMANTIC_STRATEGY,
            ANOMALY_STRATEGY,
            COLD_STORAGE_STRATEGY,
            NEVER_SURFACED_STRATEGY,
            ORPHAN_LOW_SUPPORT_STRATEGY,
            BOUNDED_NOISE_STRATEGY,
        ),
        limit=1,
    )
    second = RouletteProvider(task_name="memory-curator", task_id="task-b", candidates=candidates).get_batch(
        strategy=None,
        allowed_strategies=(
            SEMANTIC_STRATEGY,
            ANOMALY_STRATEGY,
            COLD_STORAGE_STRATEGY,
            NEVER_SURFACED_STRATEGY,
            ORPHAN_LOW_SUPPORT_STRATEGY,
            BOUNDED_NOISE_STRATEGY,
        ),
        limit=1,
    )

    assert first.strategy_used == NEVER_SURFACED_STRATEGY
    assert second.strategy_used == NEVER_SURFACED_STRATEGY
    assert first.strategy_selection_mode == "deterministic_scores"
    assert second.strategy_selection_mode == "deterministic_scores"
    assert first.strategy_selection_scores is not None
    assert first.strategy_selection_scores[NEVER_SURFACED_STRATEGY] > first.strategy_selection_scores[COLD_STORAGE_STRATEGY]
    assert first.strategy_selection_reason is not None
    assert "never_surfaced_share" in first.strategy_selection_reason


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


def test_curator_invalid_requested_strategy_falls_back_to_deterministic_selection() -> None:
    candidates = [
        _FakeRecord(
            id="never-1",
            title="Alpha One",
            content="alpha",
            last_accessed_at="2026-03-31T00:00:00+00:00",
        ),
        _FakeRecord(
            id="never-2",
            title="Zeta Two",
            content="beta",
            last_accessed_at="2026-03-31T00:00:00+00:00",
        ),
    ]

    batch = RouletteProvider(task_name="memory-curator", task_id="task-invalid-curator", candidates=candidates).get_batch(
        strategy="totally-not-real",
        allowed_strategies=(COLD_STORAGE_STRATEGY, NEVER_SURFACED_STRATEGY, BOUNDED_NOISE_STRATEGY),
        limit=1,
    )

    assert batch.requested_strategy == "totally-not-real"
    assert batch.strategy_used == NEVER_SURFACED_STRATEGY
    assert batch.strategy_fallback_reason == "unknown_requested_strategy"
    assert batch.strategy_selection_mode == "deterministic_scores"


def test_curator_utility_priors_can_shift_strategy_choice_when_live_scores_are_close() -> None:
    candidates = [
        _FakeRecord(
            id="mixed-1",
            title="Alpha One",
            content="alpha",
            last_accessed_at="2026-03-31T00:00:00+00:00",
            last_surfaced_at=None,
            read_count=5,
        ),
        _FakeRecord(
            id="mixed-2",
            title="Beta Two",
            content="beta",
            last_accessed_at="2026-02-01T00:00:00+00:00",
            last_surfaced_at="2026-03-20T00:00:00+00:00",
            read_count=5,
        ),
    ]

    without_priors = RouletteProvider(task_name="memory-curator", task_id="task-live-only", candidates=candidates).get_batch(
        strategy=None,
        allowed_strategies=(COLD_STORAGE_STRATEGY, NEVER_SURFACED_STRATEGY),
        limit=1,
    )
    with_priors = RouletteProvider(
        task_name="memory-curator",
        task_id="task-live-with-priors",
        candidates=candidates,
        strategy_prior_scores={COLD_STORAGE_STRATEGY: 1.0, NEVER_SURFACED_STRATEGY: 0.0},
    ).get_batch(
        strategy=None,
        allowed_strategies=(COLD_STORAGE_STRATEGY, NEVER_SURFACED_STRATEGY),
        limit=1,
    )

    assert without_priors.strategy_used == NEVER_SURFACED_STRATEGY
    assert with_priors.strategy_used == COLD_STORAGE_STRATEGY
    assert with_priors.strategy_selection_mode == "deterministic_scores_with_utility_priors"
    assert with_priors.strategy_selection_reason is not None
    assert "utility_priors=" in with_priors.strategy_selection_reason


def test_taxonomist_prefers_never_surfaced_when_signal_dominates() -> None:
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
        _FakeRecord(
            id="surfaced-old",
            title="Old surfaced",
            content="gamma",
            last_accessed_at="2026-01-01T00:00:00+00:00",
            last_surfaced_at="2026-03-20T00:00:00+00:00",
        ),
    ]

    batch = RouletteProvider(task_name="taxonomist", task_id="task-never", candidates=candidates).get_batch(
        strategy=None,
        allowed_strategies=(COLD_STORAGE_STRATEGY, NEVER_SURFACED_STRATEGY, BOUNDED_NOISE_STRATEGY),
        limit=1,
    )

    assert batch.strategy_used == NEVER_SURFACED_STRATEGY
    assert batch.strategy_selection_mode == "deterministic_scores"
    assert batch.strategy_selection_reason is not None
    assert "never_surfaced_share" in batch.strategy_selection_reason


def test_taxonomist_utility_priors_can_shift_close_strategy_choice() -> None:
    candidates = [
        _FakeRecord(
            id="mixed-1",
            title="Needs tags one",
            content="alpha",
            last_accessed_at="2026-03-31T00:00:00+00:00",
            last_surfaced_at=None,
        ),
        _FakeRecord(
            id="mixed-2",
            title="Old taxonomy",
            content="beta",
            last_accessed_at="2026-02-01T00:00:00+00:00",
            last_surfaced_at="2026-03-20T00:00:00+00:00",
        ),
    ]

    without_priors = RouletteProvider(task_name="taxonomist", task_id="task-live-only", candidates=candidates).get_batch(
        strategy=None,
        allowed_strategies=(COLD_STORAGE_STRATEGY, NEVER_SURFACED_STRATEGY, BOUNDED_NOISE_STRATEGY),
        limit=1,
    )
    with_priors = RouletteProvider(
        task_name="taxonomist",
        task_id="task-live-with-priors",
        candidates=candidates,
        strategy_prior_scores={COLD_STORAGE_STRATEGY: 1.0, NEVER_SURFACED_STRATEGY: 0.0},
    ).get_batch(
        strategy=None,
        allowed_strategies=(COLD_STORAGE_STRATEGY, NEVER_SURFACED_STRATEGY, BOUNDED_NOISE_STRATEGY),
        limit=1,
    )

    assert without_priors.strategy_used == NEVER_SURFACED_STRATEGY
    assert with_priors.strategy_used == COLD_STORAGE_STRATEGY
    assert with_priors.strategy_selection_mode == "deterministic_scores_with_utility_priors"
    assert with_priors.strategy_selection_reason is not None
    assert "utility_priors=" in with_priors.strategy_selection_reason


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


def test_graph_linker_prefers_graph_bridge_for_low_support_adjacent_clusters() -> None:
    candidates = [
        _FakeRecord(id="a", title="Alpha service", content="alpha dependency latency service"),
        _FakeRecord(id="b", title="Alpha client", content="alpha dependency client latency"),
        _FakeRecord(id="c", title="Gamma topic", content="gamma unrelated topic"),
    ]

    batch = RouletteProvider(
        task_name="graph-linker",
        task_id="task-bridge",
        candidates=candidates,
        support_counts={"a": 0, "b": 0, "c": 4},
    ).get_batch(
        strategy=None,
        allowed_strategies=(SEMANTIC_STRATEGY, GRAPH_BRIDGE_STRATEGY, BOUNDED_NOISE_STRATEGY),
        limit=1,
    )

    assert batch.strategy_used == GRAPH_BRIDGE_STRATEGY
    assert batch.strategy_selection_mode == "deterministic_scores"
    assert batch.strategy_selection_reason is not None
    assert "low_support_adjacent_share" in batch.strategy_selection_reason


def test_defragmenter_prefers_orphan_low_support_for_thin_low_support_records() -> None:
    candidates = [
        _FakeRecord(id="a", title="Auth note", content="auth cleanup note"),
        _FakeRecord(id="b", title="Auth follow-up", content="auth cleanup follow up"),
        _FakeRecord(id="c", title="Large stable doc", content="stable " * 200, read_count=9),
    ]

    batch = RouletteProvider(
        task_name="defragmenter",
        task_id="task-defrag",
        candidates=candidates,
        support_counts={"a": 0, "b": 0, "c": 3},
    ).get_batch(
        strategy=None,
        allowed_strategies=(COLD_STORAGE_STRATEGY, SEMANTIC_STRATEGY, ORPHAN_LOW_SUPPORT_STRATEGY),
        limit=1,
    )

    assert batch.strategy_used == ORPHAN_LOW_SUPPORT_STRATEGY
    assert batch.strategy_selection_mode == "deterministic_scores"
    assert batch.strategy_selection_reason is not None
    assert "low_support_share" in batch.strategy_selection_reason


def test_conflict_detector_uses_deterministic_signal_scoring_without_request() -> None:
    candidates = [
        _FakeRecord(
            id="a",
            title="Feature flag enabled",
            content="feature flag enabled by default for rollout",
            last_surfaced_at="2026-03-20T00:00:00+00:00",
        ),
        _FakeRecord(
            id="b",
            title="Feature flag disabled",
            content="feature flag disabled by default for rollout",
            last_surfaced_at="2026-03-20T00:00:00+00:00",
        ),
        _FakeRecord(
            id="c",
            title="Billing note",
            content="billing reminder for invoices",
            last_surfaced_at="2026-03-20T00:00:00+00:00",
        ),
    ]

    batch = RouletteProvider(task_name="conflict-detector", task_id="task-conflict", candidates=candidates).get_batch(
        strategy=None,
        allowed_strategies=(SEMANTIC_STRATEGY, CONFLICT_FRONTIER_STRATEGY, NEVER_SURFACED_STRATEGY),
        limit=1,
    )

    assert batch.strategy_used in {SEMANTIC_STRATEGY, CONFLICT_FRONTIER_STRATEGY}
    assert batch.strategy_selection_mode == "deterministic_scores"
    assert batch.strategy_selection_reason is not None
    assert "conflict_frontier_share" in batch.strategy_selection_reason


def test_deduplicator_prefers_semantic_when_overlap_signal_dominates() -> None:
    candidates = [
        _FakeRecord(id="a", title="Alpha note", content="alpha beta gamma duplicate cluster"),
        _FakeRecord(id="b", title="Alpha duplicate", content="alpha beta gamma duplicate cluster extra"),
        _FakeRecord(id="c", title="Gamma note", content="gamma delta"),
    ]

    batch = RouletteProvider(task_name="deduplicator", task_id="task-overlap", candidates=candidates).get_batch(
        strategy=None,
        allowed_strategies=(SEMANTIC_STRATEGY, ANOMALY_STRATEGY, COOLDOWN_ESCAPE_STRATEGY),
        limit=1,
    )

    assert batch.strategy_used == SEMANTIC_STRATEGY
    assert batch.strategy_selection_mode == "deterministic_scores"
    assert batch.strategy_selection_reason is not None
    assert "semantic_overlap_share" in batch.strategy_selection_reason


def test_deduplicator_utility_priors_can_shift_close_strategy_choice() -> None:
    candidates = [
        _FakeRecord(id="small-a", title="Alpha", content="alpha beta gamma duplicate cluster"),
        _FakeRecord(id="small-b", title="Alpha copy", content="alpha beta gamma duplicate cluster copy"),
        _FakeRecord(id="large-a", title="Outlier one", content="x" * 1600),
        _FakeRecord(id="large-b", title="Monster two", content="y" * 1700),
    ]

    without_priors = RouletteProvider(task_name="deduplicator", task_id="task-live-only", candidates=candidates).get_batch(
        strategy=None,
        allowed_strategies=(ANOMALY_STRATEGY, SEMANTIC_STRATEGY),
        limit=1,
    )
    with_priors = RouletteProvider(
        task_name="deduplicator",
        task_id="task-live-with-priors",
        candidates=candidates,
        strategy_prior_scores={ANOMALY_STRATEGY: 0.0, SEMANTIC_STRATEGY: 1.0},
    ).get_batch(
        strategy=None,
        allowed_strategies=(ANOMALY_STRATEGY, SEMANTIC_STRATEGY),
        limit=1,
    )

    assert without_priors.strategy_used == ANOMALY_STRATEGY
    assert with_priors.strategy_used == SEMANTIC_STRATEGY
    assert with_priors.strategy_selection_mode == "deterministic_scores_with_utility_priors"
    assert with_priors.strategy_selection_reason is not None
    assert "utility_priors=" in with_priors.strategy_selection_reason


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


def test_non_curator_and_non_deduplicator_strategy_selection_ignores_utility_priors() -> None:
    candidates = [_FakeRecord(id="a", title="Alpha", content="alpha")]

    without_priors = RouletteProvider(task_name="graph-linker", task_id="same-task", candidates=candidates).get_batch(
        strategy=None,
        allowed_strategies=(COLD_STORAGE_STRATEGY, BOUNDED_NOISE_STRATEGY),
        limit=1,
    )
    with_priors = RouletteProvider(
        task_name="graph-linker",
        task_id="same-task",
        candidates=candidates,
        strategy_prior_scores={COLD_STORAGE_STRATEGY: 0.0, BOUNDED_NOISE_STRATEGY: 1.0},
    ).get_batch(
        strategy=None,
        allowed_strategies=(COLD_STORAGE_STRATEGY, BOUNDED_NOISE_STRATEGY),
        limit=1,
    )

    assert with_priors.strategy_used == without_priors.strategy_used
    assert with_priors.strategy_selection_mode == without_priors.strategy_selection_mode


def test_roulette_weighted_strategy_mix_can_force_one_strategy() -> None:
    candidates = [_FakeRecord(id="a", title="Alpha", content="alpha")]

    batch = RouletteProvider(task_name="deduplicator", task_id="task-weighted", candidates=candidates).get_batch(
        strategy=None,
        allowed_strategies=(COLD_STORAGE_STRATEGY, BOUNDED_NOISE_STRATEGY),
        strategy_weights={COLD_STORAGE_STRATEGY: 5, BOUNDED_NOISE_STRATEGY: 0},
        limit=1,
    )

    assert batch.strategy_used == COLD_STORAGE_STRATEGY


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