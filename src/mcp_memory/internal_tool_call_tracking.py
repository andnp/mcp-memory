from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


_READ_ONLY_INTERNAL_TOOL_NAMES = frozenset(
    {
        "task_complete",
        "internal_task_complete",
        "internal_search_memory_records",
        "internal_read_memory_record",
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

    @property
    def tool_names_used(self) -> list[str]:
        return sorted(name for name, count in self.by_name.items() if count > 0)


@dataclass
class _MutableTaskToolCallState:
    total_calls: int = 0
    mutating_calls: int = 0
    by_name: dict[str, int] = field(default_factory=dict)


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
            if internal_tool_is_mutating(normalized_tool_name):
                state.mutating_calls += 1
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
        )


def _normalized_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None