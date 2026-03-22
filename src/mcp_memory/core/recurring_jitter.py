from __future__ import annotations

import random


DEFAULT_AUTONOMOUS_RECURRING_JITTER_RATIO = 0.05
DEFAULT_AUTONOMOUS_RECURRING_MAX_SECONDS = 120.0


def compute_recurring_jitter_seconds(interval_seconds: float) -> float:
    if interval_seconds <= 0:
        return 0.0
    max_jitter_seconds = min(
        float(interval_seconds) * DEFAULT_AUTONOMOUS_RECURRING_JITTER_RATIO,
        DEFAULT_AUTONOMOUS_RECURRING_MAX_SECONDS,
    )
    if max_jitter_seconds <= 0:
        return 0.0
    return round(random.random() * max_jitter_seconds, 6)


def read_recurring_jitter_seconds(data: dict[str, object] | None) -> float | None:
    if not isinstance(data, dict):
        return None
    raw_value = data.get("jitter_seconds")
    if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
        return max(float(raw_value), 0.0)
    return None