from __future__ import annotations

import time


def suspend_aware_now() -> float:
    clock_boottime = getattr(time, "CLOCK_BOOTTIME", None)
    if clock_boottime is None:
        return time.monotonic()
    return time.clock_gettime(clock_boottime)


def suspend_aware_deadline(timeout_seconds: float) -> float:
    return suspend_aware_now() + max(timeout_seconds, 0.0)


def remaining_suspend_aware_seconds(deadline: float) -> float:
    return max(deadline - suspend_aware_now(), 0.0)
