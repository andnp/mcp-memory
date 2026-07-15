"""Deterministic ownership routing for curation operations and findings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class MaintenanceFamily(StrEnum):
    CURATOR = "curator"
    SUMMARIZER = "summarizer"
    TAXONOMIST = "taxonomist"
    GRAPH_LINKER = "graph_linker"
    CONFLICT_REVIEW = "conflict_review"
    DEDUPLICATOR = "deduplicator"
    DEFRAGMENTER = "defragmenter"
    PROJECT_MANAGER = "project_manager"
    FACT_CHECKER = "fact_checker"


class CurationOperation(StrEnum):
    NORMALIZE_MEMORY = "normalize_memory"
    REWRITE_MEMORY = "rewrite_memory"
    CREATE_LINK = "create_link"
    REMOVE_LINK = "remove_link"
    MERGE_MEMORIES = "merge_memories"
    SPLIT_MEMORY = "split_memory"
    ARCHIVE_MEMORY = "archive_memory"


class CurationFinding(StrEnum):
    SUMMARY = "summary"
    TAXONOMY = "taxonomy"
    GRAPH = "graph"
    CONFLICT = "conflict"
    DUPLICATE = "duplicate"
    DEFRAGMENTATION = "defragmentation"
    PLAN_AGING = "plan_aging"
    FACT_DEGRADATION = "fact_degradation"
    CURATOR_OWNED_SPLIT = "curator_owned_split"


@dataclass(frozen=True)
class NeedsDifferentSpecialist:
    """A pure routing decision for work owned by another maintenance family."""

    primary_family: MaintenanceFamily
    finding: CurationFinding
    disposition: Literal["needs_different_specialist"] = "needs_different_specialist"

    @property
    def result(self) -> Literal["needs_different_specialist"]:
        return self.disposition


_OPERATION_PRIMARY_FAMILIES: dict[CurationOperation, MaintenanceFamily] = {
    CurationOperation.NORMALIZE_MEMORY: MaintenanceFamily.CURATOR,
    CurationOperation.REWRITE_MEMORY: MaintenanceFamily.CURATOR,
    CurationOperation.CREATE_LINK: MaintenanceFamily.GRAPH_LINKER,
    CurationOperation.REMOVE_LINK: MaintenanceFamily.GRAPH_LINKER,
    CurationOperation.MERGE_MEMORIES: MaintenanceFamily.DEDUPLICATOR,
    CurationOperation.SPLIT_MEMORY: MaintenanceFamily.CURATOR,
    CurationOperation.ARCHIVE_MEMORY: MaintenanceFamily.CURATOR,
}

_FINDING_PRIMARY_FAMILIES: dict[CurationFinding, MaintenanceFamily] = {
    CurationFinding.SUMMARY: MaintenanceFamily.SUMMARIZER,
    CurationFinding.TAXONOMY: MaintenanceFamily.TAXONOMIST,
    CurationFinding.GRAPH: MaintenanceFamily.GRAPH_LINKER,
    CurationFinding.CONFLICT: MaintenanceFamily.CONFLICT_REVIEW,
    CurationFinding.DUPLICATE: MaintenanceFamily.DEDUPLICATOR,
    CurationFinding.DEFRAGMENTATION: MaintenanceFamily.DEFRAGMENTER,
    CurationFinding.PLAN_AGING: MaintenanceFamily.PROJECT_MANAGER,
    CurationFinding.FACT_DEGRADATION: MaintenanceFamily.FACT_CHECKER,
    CurationFinding.CURATOR_OWNED_SPLIT: MaintenanceFamily.CURATOR,
}


def primary_family_for_operation(operation: CurationOperation | str) -> MaintenanceFamily:
    """Return the family that owns a typed curation operation."""
    return _OPERATION_PRIMARY_FAMILIES[CurationOperation(operation)]


def primary_family_for_finding(finding: CurationFinding | str) -> MaintenanceFamily:
    """Return the family that owns a maintenance finding."""
    return _FINDING_PRIMARY_FAMILIES[CurationFinding(finding)]


def route_finding(finding: CurationFinding | str) -> NeedsDifferentSpecialist:
    """Return a typed specialist route without scheduling or enqueueing work."""
    normalized = CurationFinding(finding)
    return NeedsDifferentSpecialist(
        primary_family=_FINDING_PRIMARY_FAMILIES[normalized],
        finding=normalized,
    )
