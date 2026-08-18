"""Compatibility exports for the task queue boundary.

Application code should import records and protocols from
``mcp_memory.core.ports.tasks`` and concrete repositories from storage.
"""

from __future__ import annotations

import json
from importlib import import_module
from typing import TYPE_CHECKING, Any

from mcp_memory.core.ports.tasks import TaskQueue, TaskRecord, TaskRunRecord, TaskRunSummary
from mcp_memory.core.ports.tasks import is_process_alive as _is_process_alive  # noqa: F401

if TYPE_CHECKING:
    from mcp_memory.storage.sqlite_task_queue import SQLiteTaskQueue


def _decode_json_object(value: object) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        return {}
    decoded = json.loads(value)
    return decoded if isinstance(decoded, dict) else {}


__all__ = ["SQLiteTaskQueue", "TaskQueue", "TaskRecord", "TaskRunRecord", "TaskRunSummary"]


def __getattr__(name: str) -> Any:
    if name == "SQLiteTaskQueue":
        return getattr(import_module("mcp_memory.storage.sqlite_task_queue"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
