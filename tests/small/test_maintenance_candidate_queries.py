from __future__ import annotations

import pytest

from mcp_memory.relational.repository import RelationalMemoryRepository
from tests.small.maintenance_candidate_query_contract import (
    assert_maintenance_candidate_query_contract,
)


pytestmark = pytest.mark.small


def _create(repository, memory_id: str, *, updated_at: str, workspace_id: str = "workspace-a", **kwargs):
    record = repository.create_memory(
        title=kwargs.pop("title", memory_id),
        content=kwargs.pop("content", f"Content for {memory_id}"),
        workspace_ids=[workspace_id],
        memory_id=memory_id,
        created_at=updated_at,
        updated_at=updated_at,
        **kwargs,
    )
    assert record is not None
    return record


def test_candidate_queries_are_global_bounded_and_skip_recent_window(db_manager, monkeypatch) -> None:
    repository = RelationalMemoryRepository(db_manager)
    _create(
        repository,
        "cold-global-old",
        updated_at="2020-01-01T00:00:00+00:00",
        workspace_id="workspace-other",
    )
    for index in range(55):
        _create(
            repository,
            f"recent-{index:02d}",
            updated_at=f"2026-01-{index + 1:02d}T00:00:00+00:00",
        )

    def fail_if_recent_window_is_loaded(*args, **kwargs):
        raise AssertionError("candidate queries must not load list_memories first")

    monkeypatch.setattr(repository, "list_memories", fail_if_recent_window_is_loaded)

    cold = repository.query_cold_candidates(limit=1)
    random_records = repository.query_seeded_random_candidates(17, limit=3)

    assert [record.id for record in cold] == ["cold-global-old"]
    assert len(random_records) == 3


def test_never_surfaced_candidates_have_explicit_deterministic_order(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    _create(repository, "never-b", updated_at="2025-01-02T00:00:00+00:00")
    _create(repository, "never-a", updated_at="2025-01-01T00:00:00+00:00")
    surfaced = _create(repository, "surfaced", updated_at="2020-01-01T00:00:00+00:00")
    repository.touch_last_surfaced([surfaced.id], "2020-01-01T00:00:00+00:00")

    first = repository.query_never_surfaced_candidates(limit=10)
    second = repository.query_maintenance_candidates("never-surfaced", limit=10)

    assert [record.id for record in first] == ["never-a", "never-b"]
    assert [record.id for record in second] == [record.id for record in first]


def test_oversized_thin_and_orphan_queries_apply_strategy_predicates(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    _create(repository, "normal", updated_at="2026-01-01T00:00:00+00:00")
    _create(
        repository,
        "thin-child",
        updated_at="2026-01-02T00:00:00+00:00",
        content="thin",
        metadata={"split_from_memory_id": "parent"},
    )
    _create(
        repository,
        "oversized",
        updated_at="2026-01-03T00:00:00+00:00",
        content="x" * 3_001,
    )

    oversized_or_thin = repository.query_oversized_thin_candidates(limit=10)

    support_source = _create(repository, "support-source", updated_at="2026-01-04T00:00:00+00:00")
    supported = _create(repository, "supported", updated_at="2026-01-05T00:00:00+00:00")
    repository.add_link(support_source.id, supported.id, "DEPENDS_ON")
    orphan_low_support = repository.query_orphan_low_support_candidates(limit=10, low_support_max=1)

    assert [record.id for record in oversized_or_thin] == ["oversized", "thin-child"]
    assert [record.id for record in orphan_low_support] == [
        "normal",
        "oversized",
        "support-source",
        "thin-child",
        "supported",
    ]


def test_quality_signal_queries_cover_each_persisted_signal(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    _create(
        repository,
        "trace",
        title="task_complete_123",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    _create(
        repository,
        "generic-summary",
        summary="Covers a broad thing.",
        updated_at="2026-01-02T00:00:00+00:00",
    )
    _create(
        repository,
        "untagged-observation",
        memory_type="observation",
        tags=[],
        updated_at="2026-01-03T00:00:00+00:00",
    )
    _create(
        repository,
        "quality-oversized",
        content="x" * 4_000,
        updated_at="2026-01-04T00:00:00+00:00",
    )
    _create(
        repository,
        "clean-observation",
        memory_type="observation",
        tags=["specific"],
        updated_at="2026-01-05T00:00:00+00:00",
    )

    all_signals = repository.query_quality_signal_candidates(limit=10)

    assert {record.id for record in all_signals} == {
        "trace",
        "generic-summary",
        "untagged-observation",
        "quality-oversized",
    }
    assert [record.id for record in repository.query_quality_signal_candidates(
        quality_signal="untagged_observation_count"
    )] == ["untagged-observation"]
    assert [record.id for record in repository.query_quality_signal_candidates(
        quality_signal="generic_summary"
    )] == ["generic-summary"]


def test_seeded_random_candidates_are_reproducible_and_tie_broken(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    for index in range(8):
        _create(
            repository,
            f"random-{index}",
            updated_at="2026-01-01T00:00:00+00:00",
        )

    first = repository.query_seeded_random_candidates("stable-seed", limit=8)
    second = repository.query_seeded_random_candidates("stable-seed", limit=8)
    different_seed = repository.query_seeded_random_candidates("different-seed", limit=8)

    assert [record.id for record in first] == [record.id for record in second]
    assert [record.id for record in first] != [record.id for record in different_seed]
    assert len({record.id for record in first}) == 8


def test_shared_candidate_query_contract(db_manager) -> None:
    assert_maintenance_candidate_query_contract(RelationalMemoryRepository(db_manager))
