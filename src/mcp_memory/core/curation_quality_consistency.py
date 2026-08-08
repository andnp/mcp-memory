"""Consistency checks for before/after curation quality replay."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class ConsistencyAssessment:
    flags: tuple[str, ...] = ()

    @property
    def verified(self) -> bool:
        return not self.flags


def assess_replay_consistency(
    *,
    before_context: Mapping[str, object],
    after_context: Mapping[str, object],
    before_epochs: Mapping[str, int],
    after_epochs: Mapping[str, int],
    replay_complete: bool,
    unrelated_write_count: int = 0,
    write_pollution: bool = False,
    index_lag: bool = False,
) -> ConsistencyAssessment:
    """Return bounded reasons a replay cannot support a quality claim."""
    flags: list[str] = []
    if not replay_complete:
        flags.append("incomplete_replay")
    if dict(before_context) != dict(after_context):
        flags.append("query_context_changed")
    for lane, before_epoch in before_epochs.items():
        after_epoch = after_epochs.get(lane)
        if after_epoch is not None and after_epoch < before_epoch:
            flags.append("index_lag")
            break
    if index_lag and "index_lag" not in flags:
        flags.append("index_lag")
    if unrelated_write_count > 0:
        flags.append("unrelated_writes")
    if write_pollution:
        flags.append("write_pollution")
    return ConsistencyAssessment(tuple(dict.fromkeys(flags)))
