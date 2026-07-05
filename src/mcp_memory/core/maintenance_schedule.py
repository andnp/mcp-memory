from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaintenanceFamilyRegistryEntry:
    canonical_task_name: str
    recurring_interval_seconds: float
    autonomous_recurring: bool
    default_task_class: str
    default_priority: int


PROJECT_MANAGER_TASK_NAME = "project-manager"
FACT_CHECKER_TASK_NAME = "fact-checker"
GRAPH_LINKER_TASK_NAME = "graph-linker"
CONFLICT_DETECTOR_TASK_NAME = "conflict-detector"
DEFRAGMENTER_TASK_NAME = "defragmenter"
DEDUPLICATOR_TASK_NAME = "deduplicator"
TAXONOMIST_TASK_NAME = "taxonomist"
SWEEPER_TASK_NAME = "sweeper"
CURATOR_TASK_NAME = "memory-curator"

# Trio-only canonical family registry for the post-wrapper maintenance families.
MAINTENANCE_FAMILY_REGISTRY = {
    CONFLICT_DETECTOR_TASK_NAME: MaintenanceFamilyRegistryEntry(
        canonical_task_name=CONFLICT_DETECTOR_TASK_NAME,
        recurring_interval_seconds=21600.0,
        autonomous_recurring=True,
        default_task_class="deterministic",
        default_priority=60,
    ),
    DEDUPLICATOR_TASK_NAME: MaintenanceFamilyRegistryEntry(
        canonical_task_name=DEDUPLICATOR_TASK_NAME,
        recurring_interval_seconds=21600.0,
        autonomous_recurring=True,
        default_task_class="deterministic",
        default_priority=70,
    ),
    TAXONOMIST_TASK_NAME: MaintenanceFamilyRegistryEntry(
        canonical_task_name=TAXONOMIST_TASK_NAME,
        recurring_interval_seconds=7200.0,
        autonomous_recurring=True,
        default_task_class="cheap_json",
        default_priority=60,
    ),
}

MAINTENANCE_TASK_NAMES = (
    PROJECT_MANAGER_TASK_NAME,
    FACT_CHECKER_TASK_NAME,
    GRAPH_LINKER_TASK_NAME,
    CONFLICT_DETECTOR_TASK_NAME,
    DEFRAGMENTER_TASK_NAME,
    DEDUPLICATOR_TASK_NAME,
    TAXONOMIST_TASK_NAME,
    SWEEPER_TASK_NAME,
    CURATOR_TASK_NAME,
)

AUTONOMOUS_RECURRING_MAINTENANCE_TASK_NAMES = (
    PROJECT_MANAGER_TASK_NAME,
    FACT_CHECKER_TASK_NAME,
    GRAPH_LINKER_TASK_NAME,
    DEFRAGMENTER_TASK_NAME,
    *(
        task_name
        for task_name in (
            CONFLICT_DETECTOR_TASK_NAME,
            DEDUPLICATOR_TASK_NAME,
            TAXONOMIST_TASK_NAME,
        )
        if MAINTENANCE_FAMILY_REGISTRY[task_name].autonomous_recurring
    ),
    SWEEPER_TASK_NAME,
    CURATOR_TASK_NAME,
)

DEFAULT_RECURRING_TASK_INTERVAL_SECONDS = 300.0

RECURRING_TASK_INTERVAL_SECONDS = {
    PROJECT_MANAGER_TASK_NAME: 1800.0,
    FACT_CHECKER_TASK_NAME: 1800.0,
    GRAPH_LINKER_TASK_NAME: 3600.0,
    CONFLICT_DETECTOR_TASK_NAME: MAINTENANCE_FAMILY_REGISTRY[CONFLICT_DETECTOR_TASK_NAME].recurring_interval_seconds,
    DEFRAGMENTER_TASK_NAME: 21600.0,
    DEDUPLICATOR_TASK_NAME: MAINTENANCE_FAMILY_REGISTRY[DEDUPLICATOR_TASK_NAME].recurring_interval_seconds,
    TAXONOMIST_TASK_NAME: MAINTENANCE_FAMILY_REGISTRY[TAXONOMIST_TASK_NAME].recurring_interval_seconds,
    SWEEPER_TASK_NAME: 21600.0,
    CURATOR_TASK_NAME: 300.0,
}

AUTONOMOUS_RECURRING_TASK_INTERVAL_SECONDS = {
    task_name: RECURRING_TASK_INTERVAL_SECONDS[task_name]
    for task_name in AUTONOMOUS_RECURRING_MAINTENANCE_TASK_NAMES
}
