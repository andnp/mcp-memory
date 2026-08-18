"""Paired bootstrap confidence intervals for variant comparisons."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_RESAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True, slots=True)
class PairedDelta:
    """A challenger's mean per-query gain over a baseline, with an interval."""

    delta: float
    low: float
    high: float
    confidence: float
    pair_count: int
    resamples: int

    @property
    def significant(self) -> bool:
        """Report whether the interval excludes no-change."""
        return self.low > 0.0 or self.high < 0.0

    def to_mapping(self) -> dict[str, object]:
        """Serialize with stable ordering for report diffs."""
        return {
            "delta": round(self.delta, 6),
            "low": round(self.low, 6),
            "high": round(self.high, 6),
            "confidence": self.confidence,
            "pair_count": self.pair_count,
            "resamples": self.resamples,
            "significant": self.significant,
        }


def paired_bootstrap(
    pairs: Sequence[tuple[float, float]],
    *,
    seed: int,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> PairedDelta:
    """Bootstrap the mean per-query difference between two variants.

    Pairs arrive as ``(baseline, challenger)`` per query so the two variants
    cannot silently fall out of alignment. Resampling draws whole queries, which
    keeps the pairing intact and is what makes the interval a statement about
    query-to-query variation rather than about the two score sets separately.
    """
    if not pairs:
        raise ValueError("paired_bootstrap requires at least one pair")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    differences = [challenger - baseline for baseline, challenger in pairs]
    count = len(differences)
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choices(differences, k=count)) / count for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0
    return PairedDelta(
        delta=sum(differences) / count,
        low=means[min(int(tail * resamples), resamples - 1)],
        high=means[min(int((1.0 - tail) * resamples), resamples - 1)],
        confidence=confidence,
        pair_count=count,
        resamples=resamples,
    )
