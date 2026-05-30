from __future__ import annotations

from datetime import datetime


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
