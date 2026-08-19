from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class RecordThoughtWritebackEntry(Protocol):
    @property
    def outbox_id(self) -> int: ...

    @property
    def content(self) -> str: ...

    @property
    def workspace_id(self) -> str | None: ...

    @property
    def timestamp(self) -> float: ...


class RecordThoughtWritebackPort(Protocol):
    def enqueue_record_thought_outbox_entry(
        self,
        *,
        content: str,
        workspace_id: str | None,
        timestamp: float,
        max_entries: int,
    ) -> RecordThoughtWritebackEntry | None: ...

    def list_record_thought_outbox_entries(
        self,
        *,
        limit: int,
    ) -> Sequence[RecordThoughtWritebackEntry]: ...

    def delete_record_thought_outbox_entries(self, outbox_ids: list[int]) -> None: ...

    def count_record_thought_outbox_entries(self) -> int: ...
