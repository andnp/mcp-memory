from __future__ import annotations

from datetime import datetime, timezone


def apply_recency_boost(
    score: float,
    created_at: datetime | None,
    boost_window_days: int,
    max_boost_amount: float,
    boost_decay_rate: float,
) -> float:
    if created_at is None:
        return score

    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    age_days = (datetime.now(timezone.utc) - created_at).days
    if age_days > boost_window_days:
        return score

    bonus = (boost_decay_rate**age_days) * max_boost_amount
    return min(1.0, score + bonus)