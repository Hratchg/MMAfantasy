"""Exponentially Weighted Moving Average (EWMA) computation.

Implements a 3-fight half-life EWMA (D-03, D-04) applied to all rate features.
Pure function, no side effects.
"""

from __future__ import annotations


def compute_ewma(values: list[float], alpha: float = 0.206) -> float:
    """Compute EWMA over a chronological list of per-fight values.

    Args:
        values: Oldest-first list of per-fight metric values.
        alpha: Smoothing factor. Default 0.206 = 1 - 0.5^(1/3) for
               3-fight half-life per D-03.

    Returns:
        The EWMA value after processing all values, or 0.0 if empty.
    """
    if not values:
        return 0.0

    ewma = values[0]
    for val in values[1:]:
        ewma = alpha * val + (1.0 - alpha) * ewma
    return ewma
