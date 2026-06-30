"""v2.2 feature-compute helpers (REF + TRAVEL + META rich Level-1).

Single-source-of-truth math for new cols in FEATURE_COLUMNS_V22.
Both feature_matrix.py (training path) and inference_features.py (live-
prediction path) import from this submodule so train/inference column
math is byte-identical (Pitfall #12 LIVE-03 parity per CONTEXT D-10).

Banned imports per project convention: nothing under ufc_prediction.ml.train
or ufc_prediction.ml.predictor — these are downstream callers.
"""

from ufc_prediction.ml.features_v22.meta import (
    age_at_fight,
    division_finish_rate_shrunk,
    elo_velocity,
    layoff_days,
    reach_diff_normalized,
)
from ufc_prediction.ml.features_v22.ref import (
    beta_binomial_shrink,
    classify_outcome,
    compute_ref_rates_shrunk,
)
from ufc_prediction.ml.features_v22.travel import (
    EARTH_RADIUS_MILES,
    compute_travel_features,
    compute_tz_shift_signed,
    haversine_miles,
    tz_offset_hours,
)

__all__ = [
    # Plan 23-02 TRAVEL:
    "EARTH_RADIUS_MILES",
    "age_at_fight",
    # Plan 23-01 REF:
    "beta_binomial_shrink",
    "classify_outcome",
    "compute_ref_rates_shrunk",
    "compute_travel_features",
    "compute_tz_shift_signed",
    "division_finish_rate_shrunk",
    "elo_velocity",
    "haversine_miles",
    # Plan 23-03 META:
    "layoff_days",
    "reach_diff_normalized",
    "tz_offset_hours",
]
