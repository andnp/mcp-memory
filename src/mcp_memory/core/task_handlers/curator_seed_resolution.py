from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping


class CuratorSeedResolutionState(StrEnum):
    READY = "ready"
    STALE = "stale"
    TEMPORARILY_UNAVAILABLE = "temporarily_unavailable"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class CuratorSeedResolution:
    state: CuratorSeedResolutionState
    records: tuple[Any, ...] = ()
    requested_memory_ids: tuple[str, ...] = ()
    resolved_memory_ids: tuple[str, ...] = ()
    missing_memory_ids: tuple[str, ...] = ()
    reason: str | None = None

    @property
    def fallback_allowed(self) -> bool:
        return self.state in {
            CuratorSeedResolutionState.STALE,
            CuratorSeedResolutionState.MALFORMED,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "seed_resolution_state": self.state.value,
            "seed_resolution_reason": self.reason,
            "seed_resolution_fallback_allowed": self.fallback_allowed,
            "seed_resolution_requested_count": len(self.requested_memory_ids),
            "seed_resolution_resolved_count": len(self.resolved_memory_ids),
            "seed_resolution_missing_count": len(self.missing_memory_ids),
            "seed_resolution_missing_memory_ids": list(self.missing_memory_ids),
        }


_PACKET_MEMORY_ID_KEYS = ("seed_memory_ids", "candidate_memory_ids", "support_memory_ids")


def resolve_curator_seed_packet(ctx: Any, payload: Mapping[str, Any] | Any) -> CuratorSeedResolution:
    if not isinstance(payload, Mapping):
        return CuratorSeedResolution(
            CuratorSeedResolutionState.MALFORMED,
            reason="payload_not_mapping",
        )

    present_keys = [key for key in _PACKET_MEMORY_ID_KEYS if key in payload]
    if not present_keys:
        return CuratorSeedResolution(
            CuratorSeedResolutionState.MALFORMED,
            reason="missing_memory_id_list",
        )

    memory_ids: list[str] = []
    for key in present_keys:
        values = payload[key]
        if not isinstance(values, list):
            return CuratorSeedResolution(
                CuratorSeedResolutionState.MALFORMED,
                reason=f"{key}_not_list",
            )
        for memory_id in values:
            if not isinstance(memory_id, str) or not memory_id.strip():
                return CuratorSeedResolution(
                    CuratorSeedResolutionState.MALFORMED,
                    reason=f"{key}_contains_invalid_id",
                )
            if memory_id not in memory_ids:
                memory_ids.append(memory_id)

    if not memory_ids:
        return CuratorSeedResolution(
            CuratorSeedResolutionState.MALFORMED,
            reason="empty_memory_id_lists",
        )

    repository = getattr(ctx, "repository", None)
    if repository is None:
        return CuratorSeedResolution(
            CuratorSeedResolutionState.TEMPORARILY_UNAVAILABLE,
            requested_memory_ids=tuple(memory_ids),
            reason="repository_unavailable",
        )

    records: list[Any] = []
    missing_ids: list[str] = []
    try:
        for memory_id in memory_ids:
            record = repository.get_memory(memory_id)
            if record is None or getattr(record, "status", None) != "active":
                missing_ids.append(memory_id)
            else:
                records.append(record)
    except Exception as exc:
        return CuratorSeedResolution(
            CuratorSeedResolutionState.TEMPORARILY_UNAVAILABLE,
            requested_memory_ids=tuple(memory_ids),
            reason=f"repository_lookup_failed:{type(exc).__name__}",
        )

    state = CuratorSeedResolutionState.READY if records else CuratorSeedResolutionState.STALE
    return CuratorSeedResolution(
        state,
        records=tuple(records),
        requested_memory_ids=tuple(memory_ids),
        resolved_memory_ids=tuple(record.id for record in records),
        missing_memory_ids=tuple(missing_ids),
        reason="active_records_found" if records else "no_active_records",
    )
