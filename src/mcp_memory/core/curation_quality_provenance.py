"""Deterministic query provenance for curation quality replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping


class QueryProvenance(StrEnum):
    CAMPAIGN_HYPOTHESIS = "campaign_hypothesis"
    REAL_USER_SEARCH = "real_user_search"
    SYNTHETIC_PROBE = "synthetic_probe"

    @property
    def trusted(self) -> bool:
        return self is not QueryProvenance.SYNTHETIC_PROBE


@dataclass(frozen=True, slots=True)
class QualityQuery:
    query_id: str
    query_text: str
    before_memory_ids: tuple[str, ...]
    replay_complete: bool
    provenance: QueryProvenance
    query_context: Mapping[str, object] = field(default_factory=dict)
    before_search_epochs: Mapping[str, int] = field(default_factory=dict)

    @property
    def trusted(self) -> bool:
        return self.provenance.trusted
