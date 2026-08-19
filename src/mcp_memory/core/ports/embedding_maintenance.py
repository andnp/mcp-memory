"""Provider-neutral persistence capabilities for embedding maintenance."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable


class EmbeddingDatabaseHealthError(RuntimeError):
    """Backend failures translated at the embedding-maintenance boundary."""


@runtime_checkable
class EmbeddingDatabaseHealthPort(Protocol):
    """Persistence health operations required by embedding maintenance."""

    def run_integrity_check(self, operation: Callable[[], None]) -> None: ...

    def retry_after_reopen(self, operation: Callable[[], None]) -> bool: ...


__all__ = ["EmbeddingDatabaseHealthError", "EmbeddingDatabaseHealthPort"]
