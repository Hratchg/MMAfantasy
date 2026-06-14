"""Phase 26 META-V22-01 unit tests: 13-col rich Level-1 stacker.

Per Plan 26-01 Task 2 behavior list:
- Test 1: META_V22_FEATURE_COLUMNS has exactly 13 entries
- Test 2: every internal col (slice [2:]) resolves in FEATURE_COLUMNS_V22
- Test 3: build_meta_features_v22 returns shape (n, 13)
- Test 4: pre-existing NaNs are PRESERVED (downstream symmetric drop handles them)
- Test 5: DEDUP grep — no duplicates; no age_diff/age_at_fight_* collision
"""

from __future__ import annotations

import numpy as np
import pytest

from ufc_prediction.ml.config import FEATURE_COLUMNS_NO_NET, FEATURE_COLUMNS_V22
from ufc_prediction.ml.meta_features_v22 import (
    META_V22_FEATURE_COLUMNS,
    build_meta_features_v22,
)


def test_meta_v22_feature_columns_has_exactly_13_entries():
    """META_V22_FEATURE_COLUMNS is locked at 13 cols (post-DEDUP)."""
    assert len(META_V22_FEATURE_COLUMNS) == 13


def test_internal_cols_all_resolve_in_feature_columns_v22():
    """Every internal col (after the 2 external ones) must be present in FEATURE_COLUMNS_V22."""
    for name in META_V22_FEATURE_COLUMNS[2:]:
        # raises ValueError if missing
        idx = FEATURE_COLUMNS_V22.index(name)
        assert 0 <= idx < len(FEATURE_COLUMNS_V22)


def test_build_meta_features_v22_returns_n_by_13_shape():
    """build_meta_features_v22 stacks to shape (n, 13)."""
    n = 20
    rng = np.random.default_rng(42)
    xgb_oof_prob = rng.uniform(0.0, 1.0, size=n)
    elo_prob = rng.uniform(0.0, 1.0, size=n)
    X_v22 = rng.standard_normal((n, len(FEATURE_COLUMNS_V22)))
    out = build_meta_features_v22(xgb_oof_prob, elo_prob, X_v22)
    assert out.shape == (n, 13), out.shape


def test_nan_preservation_in_internal_cols():
    """Pre-existing NaNs in X_v22 cols flow through to the output (Pattern 1 contract)."""
    n = 20
    rng = np.random.default_rng(0)
    xgb_oof_prob = rng.uniform(0.0, 1.0, size=n)
    elo_prob = rng.uniform(0.0, 1.0, size=n)
    X_v22 = rng.standard_normal((n, len(FEATURE_COLUMNS_V22)))
    # Inject NaN into rows {3, 7, 11} at one of the internal-col indices
    # (use closing_prob_diff which is in FEATURE_COLUMNS_NO_NET).
    nan_rows = [3, 7, 11]
    closing_idx = FEATURE_COLUMNS_V22.index("closing_prob_diff")
    for r in nan_rows:
        X_v22[r, closing_idx] = np.nan
    out = build_meta_features_v22(xgb_oof_prob, elo_prob, X_v22)
    for r in nan_rows:
        assert np.isnan(out[r]).any(), (
            f"row {r} should have at least one NaN preserved; got {out[r]!r}"
        )


def test_dedup_grep_no_duplicates_and_no_age_collision():
    """META_V22_FEATURE_COLUMNS contains no duplicates and no age_diff/age_at_fight_* collision."""
    # No duplicates.
    assert len(set(META_V22_FEATURE_COLUMNS)) == len(META_V22_FEATURE_COLUMNS), (
        f"duplicate entries in META_V22_FEATURE_COLUMNS: "
        f"{[c for c in META_V22_FEATURE_COLUMNS if META_V22_FEATURE_COLUMNS.count(c) > 1]}"
    )
    # age_diff is the chosen representation; the per-fighter age_at_fight_{red,blue}
    # cols (Phase 23 additions) must NOT appear in META_V22 (DEDUP per D-03).
    assert "age_diff" in META_V22_FEATURE_COLUMNS
    assert "age_at_fight_red" not in META_V22_FEATURE_COLUMNS
    assert "age_at_fight_blue" not in META_V22_FEATURE_COLUMNS
    # FEATURE_COLUMNS_NO_NET contains age_diff (so the per-fighter forms in
    # FEATURE_COLUMNS_V22 are intentionally redundant by design — DEDUP picks
    # the differential).
    assert "age_diff" in FEATURE_COLUMNS_NO_NET
