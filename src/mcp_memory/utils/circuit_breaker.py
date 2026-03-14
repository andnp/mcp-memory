from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import time
from typing import Callable, TypeVar


T = TypeVar("T")


class CircuitBreakerOpen(RuntimeError):
    pass


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int
    recovery_timeout: float
    success_threshold: int
    window_duration: float


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int,
        recovery_timeout: float,
        success_threshold: int,
        window_duration: float,
    ) -> None:
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            window_duration=window_duration,
        )
        self.state = CircuitState.CLOSED
        self._failure_timestamps: deque[float] = deque()
        self._open_time: float | None = None
        self._half_open_successes = 0

    def reset(self) -> None:
        self.state = CircuitState.CLOSED
        self._failure_timestamps.clear()
        self._open_time = None
        self._half_open_successes = 0

    def call(self, operation: Callable[[], T]) -> T:
        self._advance_state_if_needed()
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpen("circuit breaker is open")

        try:
            result = operation()
        except Exception:
            self._record_failure()
            raise

        self._record_success()
        return result

    def _advance_state_if_needed(self) -> None:
        if self.state != CircuitState.OPEN or self._open_time is None:
            return
        if time.time() - self._open_time >= self.config.recovery_timeout:
            self.state = CircuitState.HALF_OPEN
            self._half_open_successes = 0

    def _record_failure(self) -> None:
        now = time.time()
        self._failure_timestamps.append(now)
        while self._failure_timestamps and now - self._failure_timestamps[0] > self.config.window_duration:
            self._failure_timestamps.popleft()
        self._half_open_successes = 0
        if len(self._failure_timestamps) >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self._open_time = now

    def _record_success(self) -> None:
        if self.state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self.config.success_threshold:
                self.reset()
            return
        self.state = CircuitState.CLOSED