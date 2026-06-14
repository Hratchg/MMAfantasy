"""Unit tests for Elo calibration module."""

from __future__ import annotations

from ufc_prediction.elo.calibration import check_calibration, compute_calibration_buckets


def test_calibration_buckets_basic():
    """Given known elo_diffs, predictions, and outcomes, buckets are correctly populated."""
    elo_diffs = [30.0, 80.0, 150.0, 250.0, 350.0]
    predictions = [0.55, 0.6, 0.7, 0.8, 0.9]
    outcomes = [1, 1, 1, 1, 1]

    buckets = compute_calibration_buckets(elo_diffs, predictions, outcomes)

    # Each diff maps to a bucket: 30->0-50, 80->50-100, 150->100-200, 250->200-300, 350->300+
    assert "0-50" in buckets
    assert "50-100" in buckets
    assert "100-200" in buckets
    assert "200-300" in buckets
    assert "300+" in buckets

    # Each bucket has 1 fight
    assert buckets["0-50"]["count"] == 1
    assert buckets["50-100"]["count"] == 1
    assert buckets["100-200"]["count"] == 1
    assert buckets["200-300"]["count"] == 1
    assert buckets["300+"]["count"] == 1

    # Predicted and observed rates match the inputs
    assert abs(buckets["0-50"]["predicted_rate"] - 0.55) < 1e-9
    assert abs(buckets["0-50"]["observed_rate"] - 1.0) < 1e-9


def test_calibration_passes_within_threshold():
    """When predicted and observed rates differ by <0.05, check_calibration returns True."""
    buckets = {
        "0-50": {"count": 10, "predicted_rate": 0.52, "observed_rate": 0.50, "gap": 0.02},
        "50-100": {"count": 10, "predicted_rate": 0.58, "observed_rate": 0.60, "gap": 0.02},
        "100-200": {"count": 10, "predicted_rate": 0.70, "observed_rate": 0.68, "gap": 0.02},
    }
    assert check_calibration(buckets) is True


def test_calibration_fails_outside_threshold():
    """When one bucket has gap > 0.05, check_calibration returns False."""
    buckets = {
        "0-50": {"count": 10, "predicted_rate": 0.52, "observed_rate": 0.50, "gap": 0.02},
        "50-100": {"count": 10, "predicted_rate": 0.58, "observed_rate": 0.80, "gap": 0.22},
    }
    assert check_calibration(buckets) is False


def test_empty_bucket_skipped():
    """Bucket with no fights is not included in results."""
    # Only provide diffs that fall into 0-50 bucket
    elo_diffs = [10.0, 20.0, 30.0]
    predictions = [0.52, 0.53, 0.54]
    outcomes = [1, 0, 1]

    buckets = compute_calibration_buckets(elo_diffs, predictions, outcomes)

    # Only 0-50 should be present; other buckets have 0 fights and are skipped
    assert "0-50" in buckets
    assert "50-100" not in buckets
    assert "100-200" not in buckets
    assert "200-300" not in buckets
    assert "300+" not in buckets
