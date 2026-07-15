from __future__ import annotations

from typing import Any


def assert_maintenance_candidate_query_contract(repository: Any) -> None:
    """Assert the backend-neutral whole-corpus candidate-query contract."""

    def create(
        memory_id: str,
        *,
        updated_at: str,
        workspace_id: str = "workspace-a",
        **kwargs: Any,
    ) -> Any:
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

    old = create(
        "contract-global-old",
        updated_at="2020-01-01T00:00:00+00:00",
        workspace_id="workspace-other",
    )
    for index in range(55):
        create(
            f"contract-recent-{index:02d}",
            updated_at=f"2026-02-{index + 1:02d}T00:00:00+00:00",
        )

    def fail_if_recent_window_is_loaded(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("candidate queries must not load list_memories first")

    repository.list_memories = fail_if_recent_window_is_loaded
    assert [record.id for record in repository.query_cold_candidates(limit=1)] == [old.id]
    assert len(repository.query_seeded_random_candidates(17, limit=3)) == 3

    create("contract-never-b", updated_at="2025-01-02T00:00:00+00:00", status="stale")
    create("contract-never-a", updated_at="2025-01-01T00:00:00+00:00", status="stale")
    surfaced = create("contract-surfaced", updated_at="2020-01-01T00:00:00+00:00", status="stale")
    repository.touch_last_surfaced([surfaced.id], "2020-01-01T00:00:00+00:00")
    never_ids = [
        record.id
        for record in repository.query_never_surfaced_candidates(limit=10, status="stale")
    ]
    assert never_ids == ["contract-never-a", "contract-never-b"]

    create(
        "contract-thin-child",
        updated_at="2026-03-02T00:00:00+00:00",
        status="degraded",
        content="thin",
        metadata={"split_from_memory_id": "parent"},
    )
    create(
        "contract-oversized",
        updated_at="2026-03-03T00:00:00+00:00",
        status="degraded",
        content="x" * 3_001,
    )
    oversized_ids = [
        record.id
        for record in repository.query_oversized_thin_candidates(limit=10, status="degraded")
    ]
    assert oversized_ids == ["contract-oversized", "contract-thin-child"]

    support_source = create(
        "contract-support-source",
        updated_at="2026-04-04T00:00:00+00:00",
        status="archived",
    )
    supported = create(
        "contract-supported",
        updated_at="2026-04-05T00:00:00+00:00",
        status="archived",
    )
    create("contract-orphan", updated_at="2026-04-01T00:00:00+00:00", status="archived")
    repository.add_link(support_source.id, supported.id, "DEPENDS_ON")
    orphan_ids = [
        record.id
        for record in repository.query_orphan_low_support_candidates(limit=10, status="archived")
    ]
    assert orphan_ids == [
        "contract-orphan",
        "contract-support-source",
        "contract-supported",
    ]

    create(
        "contract-trace",
        title="task_complete_123",
        updated_at="2026-05-01T00:00:00+00:00",
    )
    create(
        "contract-generic-summary",
        summary="Covers a broad thing.",
        updated_at="2026-05-02T00:00:00+00:00",
    )
    create(
        "contract-untagged-observation",
        memory_type="observation",
        tags=[],
        updated_at="2026-05-03T00:00:00+00:00",
    )
    create(
        "contract-quality-oversized",
        content="x" * 4_000,
        updated_at="2026-05-04T00:00:00+00:00",
    )
    quality_ids = {
        record.id for record in repository.query_quality_signal_candidates(limit=10)
    }
    assert quality_ids == {
        "contract-trace",
        "contract-generic-summary",
        "contract-untagged-observation",
        "contract-quality-oversized",
    }, quality_ids
    untagged_ids = [
        record.id
        for record in repository.query_quality_signal_candidates(
            quality_signal="untagged_observation_count"
        )
    ]
    assert untagged_ids == ["contract-untagged-observation"], untagged_ids

    for index in range(8):
        create(
            f"contract-random-{index}",
            updated_at="2026-06-01T00:00:00+00:00",
            workspace_id="contract-random-workspace",
        )
    first = repository.query_seeded_random_candidates(
        "stable-seed", workspace_id="contract-random-workspace", limit=8
    )
    second = repository.query_seeded_random_candidates(
        "stable-seed", workspace_id="contract-random-workspace", limit=8
    )
    different_seed = repository.query_seeded_random_candidates(
        "different-seed", workspace_id="contract-random-workspace", limit=8
    )
    assert [record.id for record in first] == [record.id for record in second]
    assert [record.id for record in first] != [record.id for record in different_seed]
    assert len({record.id for record in first}) == 8
