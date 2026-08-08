from __future__ import annotations

from types import SimpleNamespace

import pytest

from mcp_memory.core.task_handlers.curator_seed_resolution import (
    CuratorSeedResolutionState,
    resolve_curator_seed_packet,
)


def test_ready_packet_returns_active_records_and_resolution_metadata() -> None:
    """Resolve a well-formed packet while preserving missing-record diagnostics."""
    active = SimpleNamespace(id="active", status="active")
    ctx = SimpleNamespace(
        repository=SimpleNamespace(
            get_memory=lambda memory_id: active if memory_id == "active" else None,
        )
    )

    resolution = resolve_curator_seed_packet(
        ctx,
        {"seed_memory_ids": ["active", "missing"]},
    )

    assert resolution.state == CuratorSeedResolutionState.READY
    assert resolution.records == (active,)
    assert resolution.missing_memory_ids == ("missing",)
    assert resolution.metadata()["seed_resolution_fallback_allowed"] is False


def test_stale_packet_is_fallback_eligible() -> None:
    """Classify a packet whose records are no longer active as stale."""
    ctx = SimpleNamespace(repository=SimpleNamespace(get_memory=lambda _memory_id: None))

    resolution = resolve_curator_seed_packet(ctx, {"seed_memory_ids": ["gone"]})

    assert resolution.state == CuratorSeedResolutionState.STALE
    assert resolution.fallback_allowed is True
    assert resolution.reason == "no_active_records"


def test_repository_failure_is_temporarily_unavailable() -> None:
    """Keep lookup failures retryable instead of treating them as stale packets."""
    def get_memory(_memory_id: str) -> None:
        raise RuntimeError("database unavailable")

    ctx = SimpleNamespace(repository=SimpleNamespace(get_memory=get_memory))

    resolution = resolve_curator_seed_packet(ctx, {"seed_memory_ids": ["pending"]})

    assert resolution.state == CuratorSeedResolutionState.TEMPORARILY_UNAVAILABLE
    assert resolution.fallback_allowed is False
    assert resolution.reason == "repository_lookup_failed:RuntimeError"


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({}, "missing_memory_id_list"),
        ({"seed_memory_ids": []}, "empty_memory_id_lists"),
        ({"seed_memory_ids": "not-a-list"}, "seed_memory_ids_not_list"),
        ({"seed_memory_ids": [""], "support_memory_ids": []}, "seed_memory_ids_contains_invalid_id"),
    ],
)
def test_malformed_packet_is_fallback_eligible(payload: object, reason: str) -> None:
    """Reject packets that cannot identify a deterministic review seed."""
    resolution = resolve_curator_seed_packet(SimpleNamespace(repository=object()), payload)

    assert resolution.state == CuratorSeedResolutionState.MALFORMED
    assert resolution.fallback_allowed is True
    assert resolution.reason == reason
