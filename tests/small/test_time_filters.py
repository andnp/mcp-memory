from datetime import datetime, timezone
from pathlib import Path

import pytest

from mcp_memory.core.scoring import apply_recency_boost
from mcp_memory.core.time_filters import (
    get_filtering_timestamp,
    normalize_time_filters,
    passes_time_filter,
)

pytestmark = pytest.mark.small


def test_normalize_time_filters_supports_relative_days() -> None:
    after_timestamp, before_timestamp = normalize_time_filters(None, None, 0)

    assert after_timestamp is not None
    assert before_timestamp is None


def test_get_filtering_timestamp_uses_file_mtime_fallback(tmp_path: Path) -> None:
    target = tmp_path / "memory.md"
    target.write_text("body", encoding="utf-8")

    timestamp = get_filtering_timestamp({"metadata": {}, "file_path": str(target)})

    assert timestamp is not None


def test_passes_time_filter_and_recency_boost_behave_consistently() -> None:
    created_at = datetime.now(timezone.utc)

    assert passes_time_filter(created_at, int(created_at.timestamp()) - 10, None) is True
    assert apply_recency_boost(0.4, created_at, 14, 0.2, 0.95) > 0.4


def test_normalize_time_filters_rejects_invalid_range() -> None:
    with pytest.raises(ValueError, match="after_timestamp must be less than before_timestamp"):
        normalize_time_filters(20, 10, None)