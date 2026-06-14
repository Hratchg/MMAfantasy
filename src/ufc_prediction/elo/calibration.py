"""Calibration validation for Elo rating predictions.

Bins predictions by Elo rating differential and compares predicted
vs observed win rates per bucket (per D-15, D-16).
"""

from __future__ import annotations

# Bucket boundaries per D-15
CALIBRATION_BUCKETS: dict[str, tuple[float, float]] = {
    "0-50": (0, 50),
    "50-100": (50, 100),
    "100-200": (100, 200),
    "200-300": (200, 300),
    "300+": (300, float("inf")),
}


def compute_calibration_buckets(
    elo_diffs: list[float],
    predictions: list[float],
    outcomes: list[int],
) -> dict[str, dict[str, float | int]]:
    """Compute calibration statistics per Elo differential bucket.

    Args:
        elo_diffs: Absolute Elo rating differentials for each fight.
        predictions: Predicted win probabilities for the favored fighter.
        outcomes: Actual outcomes (1 = favored fighter won, 0 = lost).

    Returns:
        Dict keyed by bucket name, each containing count, predicted_rate,
        observed_rate, and gap. Buckets with 0 fights are excluded.
    """
    # Initialize bucket accumulators
    bucket_data: dict[str, dict[str, list[float]]] = {
        name: {"predictions": [], "outcomes": []}
        for name in CALIBRATION_BUCKETS
    }

    for diff, pred, outcome in zip(elo_diffs, predictions, outcomes, strict=True):
        abs_diff = abs(diff)
        for name, (lo, hi) in CALIBRATION_BUCKETS.items():
            if lo <= abs_diff < hi:
                bucket_data[name]["predictions"].append(pred)
                bucket_data[name]["outcomes"].append(outcome)
                break

    result = {}
    for name, data in bucket_data.items():
        count = len(data["predictions"])
        if count == 0:
            continue  # Skip empty buckets
        predicted_rate = sum(data["predictions"]) / count
        observed_rate = sum(data["outcomes"]) / count
        gap = abs(predicted_rate - observed_rate)
        result[name] = {
            "count": count,
            "predicted_rate": predicted_rate,
            "observed_rate": observed_rate,
            "gap": gap,
        }

    return result


def check_calibration(buckets: dict[str, dict[str, float | int]], threshold: float = 0.05) -> bool:
    """Check if all calibration buckets are within threshold (per D-15).

    Returns True if all buckets have gap <= threshold, or if no buckets have data.
    """
    if not buckets:
        return True
    return all(bucket["gap"] <= threshold for bucket in buckets.values())
