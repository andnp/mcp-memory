from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path


def normalize_time_filters(
    after_timestamp: int | None,
    before_timestamp: int | None,
    relative_days: int | None,
) -> tuple[int | None, int | None]:
    if after_timestamp is not None and before_timestamp is not None and after_timestamp >= before_timestamp:
        raise ValueError("after_timestamp must be less than before_timestamp")

    if relative_days is not None:
        if relative_days < 0:
            raise ValueError("relative_days must be non-negative")
        cutoff = datetime.now(timezone.utc) - timedelta(days=relative_days)
        return int(cutoff.timestamp()), None

    return after_timestamp, before_timestamp


def get_filtering_timestamp(chunk_data: dict) -> datetime | None:
    metadata = chunk_data.get("metadata", {})
    if not isinstance(metadata, dict):
        return None

    created_at_str = metadata.get("memory_created_at")
    if created_at_str:
        try:
            return datetime.fromisoformat(created_at_str)
        except ValueError:
            pass

    file_path = chunk_data.get("file_path")
    if file_path and isinstance(file_path, str):
        path = Path(file_path)
        if path.exists():
            return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    return None


def passes_time_filter(
    filtering_timestamp: datetime | None,
    after_timestamp: int | None,
    before_timestamp: int | None,
) -> bool:
    if filtering_timestamp is None:
        return True

    if filtering_timestamp.tzinfo is None:
        filtering_timestamp = filtering_timestamp.replace(tzinfo=timezone.utc)

    timestamp = int(filtering_timestamp.timestamp())
    if after_timestamp is not None and timestamp < after_timestamp:
        return False
    if before_timestamp is not None and timestamp > before_timestamp:
        return False
    return True