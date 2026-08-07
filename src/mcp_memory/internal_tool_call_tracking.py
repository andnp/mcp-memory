from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any


_READ_ONLY_INTERNAL_TOOL_NAMES = frozenset(
    {
        "task_complete",
        "internal_task_complete",
        "internal_search_memory_records",
        "internal_read_memory_record",
        "internal_peek_record",
        "internal_maintenance_search",
        "internal_list_relationships",
        "internal_bounded_adjacency",
        "internal_list_memory_records",
        "internal_get_next_dedup_batch",
        "internal_get_next_curator_batch",
        "internal_get_next_ingest_batch",
        "internal_get_work_batch",
        "internal_get_compatible_work_batch",
    }
)


def internal_tool_is_mutating(tool_name: str) -> bool:
    return tool_name not in _READ_ONLY_INTERNAL_TOOL_NAMES


@dataclass(frozen=True)
class InternalToolCallSnapshot:
    task_id: str
    total_calls: int = 0
    mutating_calls: int = 0
    by_name: dict[str, int] = field(default_factory=dict)
    tool_call_ledger: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tool_names_used(self) -> list[str]:
        return sorted(name for name, count in self.by_name.items() if count > 0)


@dataclass
class _MutableTaskToolCallState:
    total_calls: int = 0
    mutating_calls: int = 0
    by_name: dict[str, int] = field(default_factory=dict)
    tool_call_ledger: list[dict[str, Any]] = field(default_factory=list)


class InternalToolCallTracker:
    def __init__(self) -> None:
        self._lock = Lock()
        self._task_state: dict[str, _MutableTaskToolCallState] = {}
        self._session_to_task: dict[str, str] = {}

    def reset_task(self, task_id: str, *, session_id: str | None = None) -> None:
        normalized_task_id = _normalized_string(task_id)
        if normalized_task_id is None:
            return
        normalized_session_id = _normalized_string(session_id)
        with self._lock:
            self._task_state[normalized_task_id] = _MutableTaskToolCallState()
            if normalized_session_id is not None:
                self._session_to_task[normalized_session_id] = normalized_task_id

    def bind_session_to_task(self, session_id: str | None, task_id: str | None) -> None:
        normalized_session_id = _normalized_string(session_id)
        normalized_task_id = _normalized_string(task_id)
        if normalized_session_id is None or normalized_task_id is None:
            return
        with self._lock:
            self._session_to_task[normalized_session_id] = normalized_task_id

    def record_call(
        self,
        tool_name: str,
        *,
        task_id: str | None = None,
        session_id: str | None = None,
        success: bool = True,
        arguments: dict[str, object] | None = None,
    ) -> str | None:
        normalized_tool_name = _normalized_string(tool_name)
        if normalized_tool_name is None:
            return None
        normalized_task_id = _normalized_string(task_id)
        normalized_session_id = _normalized_string(session_id)

        with self._lock:
            resolved_task_id = self._resolve_task_id_locked(
                task_id=normalized_task_id,
                session_id=normalized_session_id,
            )
            if resolved_task_id is None:
                return None
            state = self._task_state.setdefault(resolved_task_id, _MutableTaskToolCallState())
            state.total_calls += 1
            state.by_name[normalized_tool_name] = state.by_name.get(normalized_tool_name, 0) + 1
            is_mutating = internal_tool_is_mutating(normalized_tool_name)
            if success and is_mutating:
                state.mutating_calls += 1
            state.tool_call_ledger.append(
                {
                    "sequence": state.total_calls,
                    "tool_name": normalized_tool_name,
                    "kind": "mutation" if is_mutating else "read",
                    "status": "success" if success else "error",
                    "argument_keys": sorted(arguments or {}),
                    "memory_ids": _extract_memory_ids(arguments),
                }
            )
            return resolved_task_id

    def snapshot_task(self, task_id: str) -> InternalToolCallSnapshot | None:
        normalized_task_id = _normalized_string(task_id)
        if normalized_task_id is None:
            return None
        with self._lock:
            return self._snapshot_task_locked(normalized_task_id)

    def finalize_task(self, task_id: str) -> InternalToolCallSnapshot | None:
        normalized_task_id = _normalized_string(task_id)
        if normalized_task_id is None:
            return None
        with self._lock:
            snapshot = self._snapshot_task_locked(normalized_task_id)
            self._task_state.pop(normalized_task_id, None)
            stale_sessions = [
                session_id
                for session_id, mapped_task_id in self._session_to_task.items()
                if mapped_task_id == normalized_task_id
            ]
            for session_id in stale_sessions:
                self._session_to_task.pop(session_id, None)
            return snapshot

    def _resolve_task_id_locked(self, *, task_id: str | None, session_id: str | None) -> str | None:
        if task_id is not None:
            if session_id is not None:
                self._session_to_task[session_id] = task_id
            return task_id
        if session_id is not None:
            mapped_task_id = self._session_to_task.get(session_id)
            if mapped_task_id is not None:
                return mapped_task_id
        if len(self._task_state) == 1:
            return next(iter(self._task_state))
        return None

    def _snapshot_task_locked(self, task_id: str) -> InternalToolCallSnapshot | None:
        state = self._task_state.get(task_id)
        if state is None:
            return None
        return InternalToolCallSnapshot(
            task_id=task_id,
            total_calls=state.total_calls,
            mutating_calls=state.mutating_calls,
            by_name=dict(state.by_name),
            tool_call_ledger=[dict(call) for call in state.tool_call_ledger],
        )


def _normalized_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


_MEMORY_ID_ARGUMENT_KEYS = frozenset(
    {
        "memory_id",
        "memory_ids",
        "record_id",
        "record_ids",
        "source_id",
        "target_id",
        "canonical_id",
        "source_memory_id",
        "target_memory_id",
        "affected_memory_ids",
    }
)


def _extract_memory_ids(arguments: dict[str, object] | None) -> list[str]:
    if not arguments:
        return []
    memory_ids: list[str] = []
    for key, value in arguments.items():
        if key not in _MEMORY_ID_ARGUMENT_KEYS:
            continue
        values = value if isinstance(value, (list, tuple, set, frozenset)) else (value,)
        for item in values:
            if isinstance(item, (str, int)) and str(item).strip():
                memory_ids.append(str(item))
    return list(dict.fromkeys(memory_ids))
