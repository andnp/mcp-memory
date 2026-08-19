"""Generic run/catch/reopen-and-retry guard for embedding-maintenance health checks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from mcp_memory.core.ports.embedding_maintenance import EmbeddingDatabaseHealthError

_ExcT = TypeVar("_ExcT", bound=BaseException)


class GuardedHealthCheck(Generic[_ExcT]):
    """Runs a probe/operation, wraps driver errors as domain errors, and retries once after reopening."""

    def __init__(
        self,
        *,
        exception_type: type[_ExcT] | tuple[type[_ExcT], ...],
        reopen: Callable[[], None],
        probe: Callable[[], None] | None = None,
    ) -> None:
        self._exception_type = exception_type
        self._reopen = reopen
        self._probe = probe

    def run_integrity_check(self, operation: Callable[[], None]) -> None:
        try:
            if self._probe is not None:
                self._probe()
            operation()
        except self._exception_type as exc:
            raise EmbeddingDatabaseHealthError(str(exc)) from exc

    def retry_after_reopen(self, operation: Callable[[], None]) -> bool:
        try:
            self._reopen()
            self.run_integrity_check(operation)
        except (EmbeddingDatabaseHealthError, OSError, ValueError):
            return False
        return True
