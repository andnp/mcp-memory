"""Exclude labels no replay could satisfy.

A label records that a memory was read after a live search, but the corpus has
moved on: memories get archived and leave workspaces. Replaying those labels
against a later snapshot counts a guaranteed miss as a ranking failure, so they
are dropped before the comparison rather than depressing every variant equally.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from .labels import RelevanceLabel


@dataclass(frozen=True, slots=True)
class CorpusReach:
    """What the snapshot can still return, by memory."""

    active_ids: Collection[str]
    workspaces: Mapping[str, Collection[str]]

    def can_return(self, label: RelevanceLabel) -> bool:
        """Report whether a search could place this label's memory at all."""
        if label.memory_id not in self.active_ids:
            return False
        if label.workspace_id is None:
            return True
        return label.workspace_id in self.workspaces.get(label.memory_id, ())


@dataclass(frozen=True, slots=True)
class ReachPartition:
    """The replayable labels, and why the rest were excluded."""

    retrievable: tuple[RelevanceLabel, ...]
    inactive: int
    out_of_workspace: int

    def to_mapping(self) -> dict[str, object]:
        """Serialize so a report states what it excluded."""
        return {
            "retrievable": len(self.retrievable),
            "excluded_inactive": self.inactive,
            "excluded_out_of_workspace": self.out_of_workspace,
        }


def partition_by_reach(
    labels: Sequence[RelevanceLabel], reach: CorpusReach
) -> ReachPartition:
    """Split labels into those the snapshot can still satisfy and those it cannot."""
    retrievable: list[RelevanceLabel] = []
    inactive = 0
    out_of_workspace = 0
    for label in labels:
        if label.memory_id not in reach.active_ids:
            inactive += 1
        elif not reach.can_return(label):
            out_of_workspace += 1
        else:
            retrievable.append(label)
    return ReachPartition(
        retrievable=tuple(retrievable),
        inactive=inactive,
        out_of_workspace=out_of_workspace,
    )
