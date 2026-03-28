from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
import logging
import threading
import time
from typing import Generic, TypeVar


logger = logging.getLogger(__name__)


ItemT = TypeVar("ItemT")


class BufferedWriter(Generic[ItemT]):
    def __init__(
        self,
        write_batch: Callable[[list[ItemT]], None],
        *,
        name: str,
        low_watermark: int = 1,
        high_watermark: int = 64,
        flush_interval_seconds: float = 0.05,
        max_queue_size: int | None = None,
    ) -> None:
        self._write_batch = write_batch
        self._name = name
        self._low_watermark = max(low_watermark, 1)
        self._high_watermark = max(high_watermark, self._low_watermark)
        self._flush_interval_seconds = max(flush_interval_seconds, 0.0)
        self._max_queue_size = None if max_queue_size is None else max(max_queue_size, self._high_watermark)
        self._condition = threading.Condition()
        self._pending: deque[ItemT] = deque()
        self._inflight_batches = 0
        self._flush_requested = False
        self._closed = False
        self._worker_error: Exception | None = None
        self._worker = threading.Thread(
            target=self._run,
            name=f"buffered-writer:{name}",
            daemon=True,
        )
        self._worker.start()

    def write(self, item: ItemT) -> None:
        self.write_many([item])

    def write_many(self, items: Sequence[ItemT]) -> None:
        if not items:
            return
        with self._condition:
            self._raise_if_closed_locked()
            self._raise_worker_error_locked()
            if self._max_queue_size is not None:
                while len(self._pending) + len(items) > self._max_queue_size and not self._closed:
                    self._condition.wait(timeout=self._flush_interval_seconds or None)
                    self._raise_if_closed_locked()
                    self._raise_worker_error_locked()
            self._pending.extend(items)
            if len(self._pending) >= self._low_watermark:
                self._condition.notify_all()

    def flush(self, *, timeout_seconds: float | None = None) -> None:
        deadline = None if timeout_seconds is None else time.monotonic() + max(timeout_seconds, 0.0)
        with self._condition:
            self._raise_worker_error_locked()
            if not self._pending and self._inflight_batches == 0:
                return
            self._flush_requested = True
            self._condition.notify_all()
            while (self._pending or self._inflight_batches > 0) and self._worker_error is None:
                remaining = None if deadline is None else max(deadline - time.monotonic(), 0.0)
                if remaining == 0.0:
                    raise TimeoutError(f"Timed out flushing buffered writer {self._name}")
                self._condition.wait(timeout=remaining)
            self._raise_worker_error_locked()

    def close(self) -> None:
        worker: threading.Thread | None = None
        with self._condition:
            if self._closed:
                return
            self._flush_requested = True
            self._closed = True
            worker = self._worker
            self._condition.notify_all()
        assert worker is not None
        worker.join()
        with self._condition:
            self._raise_worker_error_locked()

    def _run(self) -> None:
        try:
            while True:
                batch = self._take_batch()
                if batch is None:
                    return
                if not batch:
                    continue
                try:
                    self._write_batch(batch)
                finally:
                    with self._condition:
                        self._inflight_batches -= 1
                        self._condition.notify_all()
        except Exception as exc:  # pragma: no cover - surfaced via flush/close callers
            logger.warning("Buffered writer %s failed: %s", self._name, exc, exc_info=True)
            with self._condition:
                self._worker_error = exc
                self._condition.notify_all()

    def _take_batch(self) -> list[ItemT] | None:
        with self._condition:
            while True:
                self._raise_worker_error_locked()
                if self._closed and not self._pending and self._inflight_batches == 0:
                    return None
                if self._pending and (
                    self._flush_requested
                    or len(self._pending) >= self._low_watermark
                    or self._flush_interval_seconds == 0.0
                ):
                    break
                if self._pending and self._flush_interval_seconds > 0.0:
                    self._condition.wait(timeout=self._flush_interval_seconds)
                    if self._pending:
                        break
                    continue
                self._condition.wait()
            batch_size = min(len(self._pending), self._high_watermark)
            batch = [self._pending.popleft() for _ in range(batch_size)]
            self._inflight_batches += 1
            self._flush_requested = self._flush_requested and bool(self._pending)
            self._condition.notify_all()
            return batch

    def _raise_if_closed_locked(self) -> None:
        if self._closed:
            raise RuntimeError(f"Buffered writer {self._name} is closed")

    def _raise_worker_error_locked(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError(f"Buffered writer {self._name} failed") from self._worker_error
