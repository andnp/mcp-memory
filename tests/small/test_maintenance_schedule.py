from __future__ import annotations

import pytest

from mcp_memory.core.maintenance_schedule import (
    CONFLICT_DETECTOR_TASK_NAME,
    CURATOR_TASK_NAME,
    DEDUPLICATOR_TASK_NAME,
    MAINTENANCE_FAMILY_REGISTRY,
    SWEEPER_TASK_NAME,
    TAXONOMIST_TASK_NAME,
)
from mcp_memory.core.task_handlers import (
    AUTONOMOUS_RECURRING_MAINTENANCE_TASK_NAMES,
    AUTONOMOUS_RECURRING_TASK_INTERVAL_SECONDS,
    RECURRING_TASK_INTERVAL_SECONDS,
    TASK_PRIORITIES,
    task_priority,
)
from mcp_memory.core.task_handlers.constants import (
    CONFLICT_DETECTOR_PRIORITY,
    DEDUPLICATOR_PRIORITY,
    TAXONOMIST_PRIORITY,
)
from mcp_memory.core.task_policy import DEFAULT_TASK_CLASS_BY_NAME

pytestmark = pytest.mark.small


REMOVED_MAINTENANCE_TASK_NAMES = (
    "conflict-screening",
    "dedup-prep",
    "tag-normalizer",
)


def test_maintenance_family_registry_captures_the_canonical_trio_contract() -> None:
    assert list(MAINTENANCE_FAMILY_REGISTRY) == [
        CONFLICT_DETECTOR_TASK_NAME,
        DEDUPLICATOR_TASK_NAME,
        TAXONOMIST_TASK_NAME,
    ]

    assert MAINTENANCE_FAMILY_REGISTRY[CONFLICT_DETECTOR_TASK_NAME].recurring_interval_seconds == 21600.0
    assert MAINTENANCE_FAMILY_REGISTRY[CONFLICT_DETECTOR_TASK_NAME].autonomous_recurring is True
    assert MAINTENANCE_FAMILY_REGISTRY[CONFLICT_DETECTOR_TASK_NAME].default_task_class == "deterministic"
    assert MAINTENANCE_FAMILY_REGISTRY[CONFLICT_DETECTOR_TASK_NAME].default_priority == 60

    assert MAINTENANCE_FAMILY_REGISTRY[DEDUPLICATOR_TASK_NAME].recurring_interval_seconds == 21600.0
    assert MAINTENANCE_FAMILY_REGISTRY[DEDUPLICATOR_TASK_NAME].autonomous_recurring is True
    assert MAINTENANCE_FAMILY_REGISTRY[DEDUPLICATOR_TASK_NAME].default_task_class == "deterministic"
    assert MAINTENANCE_FAMILY_REGISTRY[DEDUPLICATOR_TASK_NAME].default_priority == 70

    assert MAINTENANCE_FAMILY_REGISTRY[TAXONOMIST_TASK_NAME].recurring_interval_seconds == 7200.0
    assert MAINTENANCE_FAMILY_REGISTRY[TAXONOMIST_TASK_NAME].autonomous_recurring is True
    assert MAINTENANCE_FAMILY_REGISTRY[TAXONOMIST_TASK_NAME].default_task_class == "cheap_json"
    assert MAINTENANCE_FAMILY_REGISTRY[TAXONOMIST_TASK_NAME].default_priority == 60

    for removed_task_name in REMOVED_MAINTENANCE_TASK_NAMES:
        assert removed_task_name not in MAINTENANCE_FAMILY_REGISTRY
        assert removed_task_name not in RECURRING_TASK_INTERVAL_SECONDS
        assert removed_task_name not in AUTONOMOUS_RECURRING_TASK_INTERVAL_SECONDS
        assert removed_task_name not in AUTONOMOUS_RECURRING_MAINTENANCE_TASK_NAMES
        assert removed_task_name not in TASK_PRIORITIES
        assert removed_task_name not in DEFAULT_TASK_CLASS_BY_NAME

    for specialist_task_name in (
        CONFLICT_DETECTOR_TASK_NAME,
        DEDUPLICATOR_TASK_NAME,
        TAXONOMIST_TASK_NAME,
    ):
        assert specialist_task_name not in AUTONOMOUS_RECURRING_MAINTENANCE_TASK_NAMES
        assert specialist_task_name not in AUTONOMOUS_RECURRING_TASK_INTERVAL_SECONDS

    for legacy_cleanup_task_name in (
        "project-manager",
        "fact-checker",
        "graph-linker",
        "defragmenter",
    ):
        assert legacy_cleanup_task_name not in AUTONOMOUS_RECURRING_MAINTENANCE_TASK_NAMES
        assert legacy_cleanup_task_name not in AUTONOMOUS_RECURRING_TASK_INTERVAL_SECONDS

    assert SWEEPER_TASK_NAME in AUTONOMOUS_RECURRING_TASK_INTERVAL_SECONDS
    assert CURATOR_TASK_NAME in AUTONOMOUS_RECURRING_MAINTENANCE_TASK_NAMES
    assert CURATOR_TASK_NAME in AUTONOMOUS_RECURRING_TASK_INTERVAL_SECONDS


def test_trio_consumers_derive_schedule_priority_and_task_class_from_registry() -> None:
    public_priority_constants = {
        CONFLICT_DETECTOR_TASK_NAME: CONFLICT_DETECTOR_PRIORITY,
        DEDUPLICATOR_TASK_NAME: DEDUPLICATOR_PRIORITY,
        TAXONOMIST_TASK_NAME: TAXONOMIST_PRIORITY,
    }

    for task_name, entry in MAINTENANCE_FAMILY_REGISTRY.items():
        assert RECURRING_TASK_INTERVAL_SECONDS[task_name] == entry.recurring_interval_seconds
        assert task_name not in AUTONOMOUS_RECURRING_MAINTENANCE_TASK_NAMES
        assert task_name not in AUTONOMOUS_RECURRING_TASK_INTERVAL_SECONDS
        assert public_priority_constants[task_name] == entry.default_priority
        assert TASK_PRIORITIES[task_name] == entry.default_priority
        assert task_priority(task_name) == entry.default_priority
        assert DEFAULT_TASK_CLASS_BY_NAME[task_name] == entry.default_task_class