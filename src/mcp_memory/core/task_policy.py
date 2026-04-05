from __future__ import annotations

from mcp_memory.config import Config
from mcp_memory.core.maintenance_schedule import MAINTENANCE_FAMILY_REGISTRY
from mcp_memory.core.task_handlers.constants import (
    CONFLICT_DETECTOR_TASK_NAME,
    CURATOR_TASK_NAME,
    DEDUPLICATOR_TASK_NAME,
    DEFRAGMENTER_TASK_NAME,
    FACT_CHECKER_TASK_NAME,
    GRAPH_LINK_DISCOVERY_TASK_NAME,
    GRAPH_LINKER_TASK_NAME,
    PROJECT_MANAGER_TASK_NAME,
    SUMMARIZE_MEMORY_TASK_NAME,
    SWEEPER_TASK_NAME,
    SYSTEM1_INGEST_TASK_NAME,
    TAXONOMIST_TASK_NAME,
)


TASK_CLASS_PREMIUM_AGENTIC = "premium_agentic"
TASK_CLASS_CHEAP_AGENTIC = "cheap_agentic"
TASK_CLASS_CHEAP_JSON = "cheap_json"
TASK_CLASS_DETERMINISTIC = "deterministic"

DEFAULT_TASK_CLASS_BY_NAME = {
    SYSTEM1_INGEST_TASK_NAME: TASK_CLASS_CHEAP_AGENTIC,
    CURATOR_TASK_NAME: TASK_CLASS_PREMIUM_AGENTIC,
    GRAPH_LINK_DISCOVERY_TASK_NAME: TASK_CLASS_DETERMINISTIC,
    GRAPH_LINKER_TASK_NAME: TASK_CLASS_CHEAP_JSON,
    DEFRAGMENTER_TASK_NAME: TASK_CLASS_CHEAP_JSON,
    FACT_CHECKER_TASK_NAME: TASK_CLASS_DETERMINISTIC,
    PROJECT_MANAGER_TASK_NAME: TASK_CLASS_DETERMINISTIC,
    SUMMARIZE_MEMORY_TASK_NAME: TASK_CLASS_DETERMINISTIC,
    SWEEPER_TASK_NAME: TASK_CLASS_DETERMINISTIC,
    **{
        task_name: MAINTENANCE_FAMILY_REGISTRY[task_name].default_task_class
        for task_name in (
            CONFLICT_DETECTOR_TASK_NAME,
            DEDUPLICATOR_TASK_NAME,
            TAXONOMIST_TASK_NAME,
        )
    },
}

DEFAULT_AGENTIC_TASK_NAMES = frozenset(
    {
        task_name
        for task_name, task_class in DEFAULT_TASK_CLASS_BY_NAME.items()
        if task_class in {TASK_CLASS_CHEAP_AGENTIC, TASK_CLASS_PREMIUM_AGENTIC}
    }
)

DEFAULT_LOW_PRIORITY_TASK_NAMES = frozenset(
    {
        GRAPH_LINKER_TASK_NAME,
        CONFLICT_DETECTOR_TASK_NAME,
        DEFRAGMENTER_TASK_NAME,
        TAXONOMIST_TASK_NAME,
    }
)


def task_class_for_task(config: Config | None, task_name: str) -> str:
    routing = None if config is None else config.provider_routing
    if routing is not None and task_name in routing.task_classes:
        return routing.task_classes[task_name]
    return DEFAULT_TASK_CLASS_BY_NAME.get(task_name, TASK_CLASS_CHEAP_JSON)


def is_deterministic_task(config: Config | None, task_name: str) -> bool:
    return task_class_for_task(config, task_name) == TASK_CLASS_DETERMINISTIC
