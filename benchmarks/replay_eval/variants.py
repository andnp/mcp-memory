"""Ranker variants compared by replay, including the controls."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass

CalibrationFn = Callable[[float], float]

SIGMOID_THRESHOLD = 0.035
SIGMOID_STEEPNESS = 150.0
DEFAULT_RRF_K = 60.0
DEFAULT_HALF_CONFIDENCE_RANK = 1.0
DEFAULT_NOISE_MAGNITUDE = 0.05


@dataclass(frozen=True, slots=True)
class Variant:
    """One fused-score calibration under test.

    Every variant is a replacement for the single function each fused score
    passes through, so variants differ only in how a fused score becomes the
    base of the composite score and nowhere else.
    """

    name: str
    description: str
    calibrate: CalibrationFn
    is_control: bool = False

    def __post_init__(self) -> None:
        """Reject a variant a report could not attribute."""
        if not self.name.strip():
            raise ValueError("name must not be empty")


def sigmoid(
    *, threshold: float = SIGMOID_THRESHOLD, steepness: float = SIGMOID_STEEPNESS
) -> Variant:
    """The incumbent curve, kept as the comparison baseline."""

    def calibrate(score: float) -> float:
        exponent = max(-20.0, min(20.0, -steepness * (score - threshold)))
        return max(0.0, min(1.0, 1.0 / (1.0 + math.exp(exponent))))

    return Variant(
        name="sigmoid",
        description=f"logistic, midpoint {threshold}, steepness {steepness}",
        calibrate=calibrate,
    )


def saturation(
    *,
    rrf_k: float = DEFAULT_RRF_K,
    half_confidence_rank: float = DEFAULT_HALF_CONFIDENCE_RANK,
) -> Variant:
    """A saturating curve whose midpoint is expressed in rank units.

    The even-odds point tracks ``rrf_k`` instead of being a fixed score, so
    changing the fusion constant cannot silently decalibrate the curve.
    """
    midpoint = 1.0 / (rrf_k + half_confidence_rank)

    def calibrate(score: float) -> float:
        return score / (score + midpoint) if score > 0.0 else 0.0

    return Variant(
        name="saturation",
        description=f"saturating, even odds at rank {half_confidence_rank}",
        calibrate=calibrate,
    )


def noise_control(
    *, magnitude: float = DEFAULT_NOISE_MAGNITUDE, seed: int = 0
) -> Variant:
    """A variant that reshuffles results without using any relevance signal.

    This is the harness's own test. Because the labeled set is conditioned on
    production having surfaced each memory, most labels sit mid-list with more
    room to rise than to fall, so a comparison that mistakes churn for progress
    will score this variant as an improvement. It must come back insignificant;
    if it does not, no other result from the harness can be believed.
    """
    if magnitude < 0.0:
        raise ValueError("magnitude must not be negative")
    baseline = sigmoid().calibrate

    def calibrate(score: float) -> float:
        digest = hashlib.blake2b(
            f"{seed}:{score!r}".encode(), digest_size=8
        ).digest()
        unit = int.from_bytes(digest, "big") / float(1 << 64)
        return max(0.0, min(1.0, baseline(score) + (unit * 2.0 - 1.0) * magnitude))

    return Variant(
        name="noise",
        description=f"baseline plus deterministic jitter of {magnitude}",
        calibrate=calibrate,
        is_control=True,
    )


def default_variants() -> tuple[Variant, ...]:
    """The comparison set, with the control always included."""
    return (sigmoid(), saturation(), noise_control())
