"""Execution identity primitives for direct curator tool correlation."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from uuid import UUID, uuid4


@dataclass(frozen=True)
class CuratorExecutionIdentity:
    task_id: str
    execution_epoch: int
    session_id: str | None = None
    run_id: UUID | None = None


@dataclass(frozen=True)
class CuratorExecutionContext:
    identity: CuratorExecutionIdentity
    call_id: str | None = None
    sequence: int = 0

    @property
    def task_id(self) -> str:
        return self.identity.task_id

    @property
    def execution_epoch(self) -> int:
        return self.identity.execution_epoch

    @property
    def session_id(self) -> str | None:
        return self.identity.session_id

    @property
    def run_id(self) -> UUID | None:
        return self.identity.run_id


_current_context: ContextVar[CuratorExecutionContext | None] = ContextVar(
    "mcp_memory_curator_execution_context", default=None
)


def reset_curator_execution(
    task_id: str,
    *,
    execution_epoch: int = 0,
    session_id: str | None = None,
    run_id: UUID | None = None,
) -> CuratorExecutionContext:
    context = CuratorExecutionContext(
        identity=CuratorExecutionIdentity(task_id, execution_epoch, session_id, run_id)
    )
    _current_context.set(context)
    return context


def current_curator_execution() -> CuratorExecutionContext | None:
    return _current_context.get()


def begin_tool_call(
    *,
    task_id: str | None = None,
    execution_epoch: int | None = None,
    session_id: str | None = None,
) -> Token[CuratorExecutionContext | None]:
    current = _current_context.get()
    if current is None:
        if task_id is None:
            raise ValueError("task_id is required without an active curator execution")
        current = CuratorExecutionContext(
            CuratorExecutionIdentity(
                task_id,
                execution_epoch or 0,
                session_id,
                None,
            )
        )
    elif task_id is not None and task_id != current.task_id:
        current = CuratorExecutionContext(
            CuratorExecutionIdentity(
                task_id,
                0 if execution_epoch is None else execution_epoch,
                session_id if session_id is not None else current.session_id,
                None,
            ),
        )
    elif session_id is not None and session_id != current.session_id:
        current = replace(
            current,
            identity=replace(current.identity, session_id=session_id),
        )
    call = replace(current, call_id=uuid4().hex, sequence=current.sequence + 1)
    return _current_context.set(call)


def finish_tool_call(token: Token[CuratorExecutionContext | None]) -> None:
    completed = _current_context.get()
    _current_context.reset(token)
    parent = _current_context.get()
    if completed is not None and parent is not None and completed.task_id == parent.task_id:
        _current_context.set(replace(parent, sequence=completed.sequence))


def finalize_curator_execution() -> CuratorExecutionContext | None:
    context = _current_context.get()
    _current_context.set(None)
    return context
