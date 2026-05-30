from __future__ import annotations

from datetime import datetime
import math


def _bucket_starts(*, cutoff: float, generated_at: float, bucket_seconds: int) -> list[int]:
    if generated_at < cutoff:
        return []
    start_bucket = int(cutoff // bucket_seconds) * bucket_seconds
    end_bucket = int(generated_at // bucket_seconds) * bucket_seconds
    if end_bucket < start_bucket:
        return []
    return list(range(start_bucket, end_bucket + bucket_seconds, bucket_seconds))


def _datetime_to_timestamp(value: datetime | None) -> float | None:
    if value is None:
        return None
    return value.timestamp()


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(math.ceil(len(sorted_values) * ratio) - 1, 0)
    return float(sorted_values[index])
