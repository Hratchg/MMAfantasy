"""Phase 45 Plan 45-03 — meta_v3 candidate trainer invariants (META3-V25-02).

3 unit tests cover:
  1. meta_v3 metadata's meta_feature_columns == META_V22_FEATURE_COLUMNS verbatim
     (13 cols; D-CONTEXT § "same shape as META-V22" decision).
  2. meta_v3 metadata's base_model_sha256 matches sha256(models/xgb_v3.joblib)
     (lineage anchor; predictor.py load-time validates this).
  3. assemble_level1_input() substitutes xgb_v3_oof_prob into the xgb_oof_prob
     Level-1 slot — column 0 of the returned matrix is sourced from
     oof_df.xgb_v3_oof_prob, NOT from xgb_v2.

Tests use synthetic DataFrames (no DB, no on-disk artifacts) so the file runs
<1s and is hermetic. The training driver `scripts/train_meta_v3_v25.py` is
loaded via importlib so the script does not need to live inside a packaged
module (matches the Plan 45-02 test_train_xgb_v3_v25 pattern).
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ufc_prediction.ml.meta_features_v22 import META_V22_FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train_meta_v3_v25.py"
XGB_V3_PATH = PROJECT_ROOT / "models" / "xgb_v3.joblib"


def _load_module():
    """Import scripts/train_meta_v3_v25.py via importlib (script, not package)."""
    if not SCRIPT_PATH.is_file():
        raise FileNotFoundError(SCRIPT_PATH)
    spec = importlib.util.spec_from_file_location("train_meta_v3_v25", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: META_V22 13-col input set preserved verbatim (no v2.5 TRAVEL drift)
# ─────────────────────────────────────────────────────────────────────────────


def test_meta_v3_uses_13_col_input():
    """meta_feature_columns must be == META_V22_FEATURE_COLUMNS verbatim.

    Per D-CONTEXT § "TRAVEL Inclusion Strategy": conservative path locked.
    NO travel_distance_km, NO tz_shift_hours, NO new feature columns.
    """
    mod = _load_module()
    meta = mod.build_meta_v3_metadata(
        base_model_sha256="0" * 64,
        meta_oof_parquet_sha256="0" * 64,
        per_slice_metrics={
            "most_recent_12mo": {"brier_score": 0.21, "accuracy": 0.70},
            "most_recent_24mo": {"brier_score": 0.21, "accuracy": 0.70},
            "random_15pct": {"brier_score": 0.19, "accuracy": 0.78},
        },
        meta_input_distribution_hash="0" * 64,
    )
    assert meta["meta_feature_columns"] == list(META_V22_FEATURE_COLUMNS), (
        f"meta_feature_columns drift from META_V22 single-source-of-truth: "
        f"got {meta['meta_feature_columns']!r} expected {list(META_V22_FEATURE_COLUMNS)!r}"
    )
    assert len(meta["meta_feature_columns"]) == 13, (
        f"META-V22 shape (13 cols) violated; got {len(meta['meta_feature_columns'])}"
    )
    # Conservative-path lock: NO v2.5 TRAVEL sibling cols
    assert "travel_distance_km" not in meta["meta_feature_columns"], (
        "v2.5 TRAVEL sibling col leaked into Level-1 (D-CONTEXT conservative path)"
    )
    assert "tz_shift_hours" not in meta["meta_feature_columns"], (
        "v2.5 TRAVEL sibling col leaked into Level-1 (D-CONTEXT conservative path)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: base_model_sha256 lineage anchor → models/xgb_v3.joblib
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not XGB_V3_PATH.is_file(),
    reason="xgb_v3.joblib not present (Plan 45-02 dependency)",
)
def test_base_sha_matches_xgb_v3():
    """meta_v3 metadata's base_model_sha256 must equal sha256(models/xgb_v3.joblib).

    Predictor.py validates this at model-load time to detect substrate drift.
    Without this lineage anchor, meta_v3 inference could silently use stale or
    swapped base predictions.
    """
    mod = _load_module()
    actual_xgb_v3_sha = hashlib.sha256(XGB_V3_PATH.read_bytes()).hexdigest()
    meta = mod.build_meta_v3_metadata(
        base_model_sha256=actual_xgb_v3_sha,
        meta_oof_parquet_sha256="0" * 64,
        per_slice_metrics={
            "most_recent_12mo": {"brier_score": 0.21, "accuracy": 0.70},
            "most_recent_24mo": {"brier_score": 0.21, "accuracy": 0.70},
            "random_15pct": {"brier_score": 0.19, "accuracy": 0.78},
        },
        meta_input_distribution_hash="0" * 64,
    )
    assert meta["base_model_sha256"] == actual_xgb_v3_sha, (
        f"base_model_sha256 lineage drift: meta says "
        f"{meta['base_model_sha256'][:12]}..., disk says {actual_xgb_v3_sha[:12]}..."
    )
    assert meta["meta_version"] == "v3"
    assert meta["base_model_version"] == "v3"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: xgb_v3 OOF substitution at xgb_oof_prob Level-1 slot
# ─────────────────────────────────────────────────────────────────────────────


def test_no_travel_v25_sibling_cols():
    """assemble_level1_input substitutes xgb_v3_oof_prob into the xgb_oof_prob slot.

    Column 0 of the returned X_meta MUST be sourced from oof_df.xgb_v3_oof_prob
    (NOT from any xgb_v2 OOF column). Also verifies no v2.5 TRAVEL sibling cols
    are stacked into the matrix (column count = 13 verbatim).
    """
    mod = _load_module()

    # Synthetic OOF DataFrame — distinctive sentinel values in xgb_v3 column
    # so we can assert exact propagation to Level-1 col 0.
    n = 6
    oof_df = pd.DataFrame(
        {
            "fight_id": np.arange(n, dtype=np.int64),
            "xgb_v3_oof_prob": np.array([0.111, 0.222, 0.333, 0.444, 0.555, 0.666]),
            "split": ["train"] * n,
            "train_or_test": ["train"] * n,
            "event_date": pd.to_datetime(
                [
                    "2023-06-01",
                    "2023-07-01",
                    "2023-08-01",
                    "2023-09-01",
                    "2023-10-01",
                    "2023-11-01",
                ]
            ),
        }
    )

    # Synthetic Level-1 source DataFrame containing the 11 non-xgb-OOF cols
    # (elo_prob is the 2nd "external" col + the 11 v2.2 FEATURE_COLUMNS_V22 cols).
    # Use distinctive but distinct values per column so substitution drift is
    # detectable.
    level1_df = pd.DataFrame(
        {
            "fight_id": np.arange(n, dtype=np.int64),
            "elo_prob": np.linspace(0.40, 0.60, n),
            "closing_prob_diff": np.linspace(-0.10, 0.10, n),
            "stance_matchup": np.zeros(n),
            "height_diff": np.linspace(-2.0, 2.0, n),
            "reach_diff": np.linspace(-3.0, 3.0, n),
            "days_since_last_fight_diff": np.linspace(-30, 30, n),
            "age_diff": np.linspace(-5.0, 5.0, n),
            "elo_overall_diff": np.linspace(-100.0, 100.0, n),
            "elo_striking_diff": np.linspace(-50.0, 50.0, n),
            "elo_grappling_diff": np.linspace(-50.0, 50.0, n),
            "division_finish_rate_shrunk": np.linspace(0.30, 0.50, n),
            "sharp_money_signal": np.linspace(-0.05, 0.05, n),
            "y": np.array([0, 1, 0, 1, 0, 1]),
        }
    )

    X_meta, y = mod.assemble_level1_input(oof_df, level1_df)

    # Shape invariant — 13 cols, n rows
    assert X_meta.shape == (n, 13), (
        f"Level-1 shape drift: got {X_meta.shape}, expected ({n}, 13). "
        f"Any v2.5 TRAVEL sibling col leakage would inflate column count."
    )

    # Column 0 = xgb_oof_prob slot — substituted from xgb_v3_oof_prob
    np.testing.assert_array_equal(
        X_meta[:, 0],
        oof_df["xgb_v3_oof_prob"].to_numpy(),
        err_msg="Level-1 col 0 (xgb_oof_prob slot) MUST be xgb_v3_oof_prob "
        "(NOT xgb_v2 OOF, NOT default-zero). The whole point of meta_v3 is "
        "substituting xgb_v3 OOF for xgb_v2 OOF at this slot.",
    )

    # Column 1 = elo_prob (the 2nd "external" col from META_V22_FEATURE_COLUMNS)
    np.testing.assert_array_equal(
        X_meta[:, 1],
        level1_df["elo_prob"].to_numpy(),
    )

    # Targets propagate through
    np.testing.assert_array_equal(y, level1_df["y"].to_numpy())
