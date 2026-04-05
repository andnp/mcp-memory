from __future__ import annotations

import pytest

from mcp_memory.core.maintenance_schedule import (
    CONFLICT_DETECTOR_TASK_NAME,
    CONFLICT_SCREENING_TASK_NAME,
    DEDUPLICATOR_TASK_NAME,
    DEDUP_PREP_TASK_NAME,
    LEGACY_MAINTENANCE_TASK_NAME_ALIASES,
    MAINTENANCE_FAMILY_REGISTRY,
    TAG_NORMALIZER_TASK_NAME,
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


def test_maintenance_family_registry_captures_the_trio_contract() -> None:
    assert list(MAINTENANCE_FAMILY_REGISTRY) == [
        CONFLICT_DETECTOR_TASK_NAME,
        DEDUPLICATOR_TASK_NAME,
        TAXONOMIST_TASK_NAME,
    ]
    assert LEGACY_MAINTENANCE_TASK_NAME_ALIASES == {
        CONFLICT_SCREENING_TASK_NAME: CONFLICT_DETECTOR_TASK_NAME,
        DEDUP_PREP_TASK_NAME: DEDUPLICATOR_TASK_NAME,
        TAG_NORMALIZER_TASK_NAME: TAXONOMIST_TASK_NAME,
    }

    assert MAINTENANCE_FAMILY_REGISTRY[CONFLICT_DETECTOR_TASK_NAME].legacy_aliases == (CONFLICT_SCREENING_TASK_NAME,)
    assert MAINTENANCE_FAMILY_REGISTRY[CONFLICT_DETECTOR_TASK_NAME].recurring_interval_seconds == 21600.0
    assert MAINTENANCE_FAMILY_REGISTRY[CONFLICT_DETECTOR_TASK_NAME].autonomous_recurring is True
    assert MAINTENANCE_FAMILY_REGISTRY[CONFLICT_DETECTOR_TASK_NAME].default_task_class == "cheap_json"
    assert MAINTENANCE_FAMILY_REGISTRY[CONFLICT_DETECTOR_TASK_NAME].default_priority == 60

    assert MAINTENANCE_FAMILY_REGISTRY[DEDUPLICATOR_TASK_NAME].legacy_aliases == (DEDUP_PREP_TASK_NAME,)
    assert MAINTENANCE_FAMILY_REGISTRY[DEDUPLICATOR_TASK_NAME].recurring_interval_seconds == 21600.0
    assert MAINTENANCE_FAMILY_REGISTRY[DEDUPLICATOR_TASK_NAME].autonomous_recurring is True
    assert MAINTENANCE_FAMILY_REGISTRY[DEDUPLICATOR_TASK_NAME].default_task_class == "cheap_agentic"
    assert MAINTENANCE_FAMILY_REGISTRY[DEDUPLICATOR_TASK_NAME].default_priority == 70

    assert MAINTENANCE_FAMILY_REGISTRY[TAXONOMIST_TASK_NAME].legacy_aliases == (TAG_NORMALIZER_TASK_NAME,)
    assert MAINTENANCE_FAMILY_REGISTRY[TAXONOMIST_TASK_NAME].recurring_interval_seconds == 7200.0
    assert MAINTENANCE_FAMILY_REGISTRY[TAXONOMIST_TASK_NAME].autonomous_recurring is True
    assert MAINTENANCE_FAMILY_REGISTRY[TAXONOMIST_TASK_NAME].default_task_class == "cheap_json"
    assert MAINTENANCE_FAMILY_REGISTRY[TAXONOMIST_TASK_NAME].default_priority == 60


def test_trio_consumers_derive_schedule_priority_and_task_class_from_registry() -> None:
    public_priority_constants = {
        CONFLICT_DETECTOR_TASK_NAME: CONFLICT_DETECTOR_PRIORITY,
        DEDUPLICATOR_TASK_NAME: DEDUPLICATOR_PRIORITY,
        TAXONOMIST_TASK_NAME: TAXONOMIST_PRIORITY,
    }

    for task_name, entry in MAINTENANCE_FAMILY_REGISTRY.items():
        assert RECURRING_TASK_INTERVAL_SECONDS[task_name] == entry.recurring_interval_seconds
        assert AUTONOMOUS_RECURRING_TASK_INTERVAL_SECONDS[task_name] == entry.recurring_interval_seconds
        assert task_name in AUTONOMOUS_RECURRING_MAINTENANCE_TASK_NAMES
        assert public_priority_constants[task_name] == entry.default_priority
        assert TASK_PRIORITIES[task_name] == entry.default_priority
        assert task_priority(task_name) == entry.default_priority
        assert DEFAULT_TASK_CLASS_BY_NAME[task_name] == entry.default_task_class