"""Tests for ModelPredictor.__init__ FEATURE_COLUMNS hard assertion (LIVE-03).

Per CONTEXT.md ``<plan_specific_requirements>``: assertion RAISES
``RuntimeError`` (NOT AssertionError, NOT log-and-continue) on any drift
between code-level FEATURE_COLUMNS and ``meta["feature_columns"]``.

This is the Pitfall #12 silent-garbage-prediction safeguard: if a model
was trained against a different column space than the running code knows
about, ``predict_proba`` would receive misordered or wrong-cardinality
features and silently emit garbage.

Tests:
  1. happy path — matching columns, no raise
  2. drift on length — shorter feature_columns raises
  3. drift on order — same length, reversed order raises
  4. xgb_v2 still loads (regression — Phase 15.1 model has matching cols)
  5. xgb_v1 documented behavior — 28 features vs 72; raises (intended)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

from ufc_prediction.ml.config import FEATURE_COLUMNS
from ufc_prediction.ml.persistence import save_model

predictor = pytest.importorskip("ufc_prediction.ml.predictor")


# ── Shared fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def trained_calibrated_model():
    """Build a tiny calibrated CalibratedClassifierCV that matches the
    persistence-layer's expected shape.
    """
    rng = np.random.RandomState(42)
    n = len(FEATURE_COLUMNS)
    X = rng.randn(60, n)
    y = rng.randint(0, 2, size=60)

    base = XGBClassifier(
        n_estimators=5, max_depth=2, objective="binary:logistic",
        random_state=42, verbosity=0,
    )
    base.fit(X[:48], y[:48])
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    cal.fit(X[48:], y[48:])
    return cal


# ── Test 1: happy path ──────────────────────────────────────────────────────


def test_predictor_load_passes_when_feature_columns_match(
    tmp_path, trained_calibrated_model,
):
    """Test 1: save_model with FEATURE_COLUMNS → ModelPredictor() does NOT raise."""
    save_model(
        model=trained_calibrated_model,
        metrics={"brier_score": 0.22},
        feature_columns=FEATURE_COLUMNS,
        best_params={},
        model_dir=str(tmp_path),
        version="vtest",
    )
    pred = predictor.ModelPredictor(model_dir=str(tmp_path), version="vtest")
    assert pred.metadata["feature_columns"] == list(FEATURE_COLUMNS)


# ── Test 2: drift on length ─────────────────────────────────────────────────


def test_predictor_load_raises_on_length_mismatch(
    tmp_path, trained_calibrated_model,
):
    """Test 2: shorter feature_columns → RuntimeError with column counts."""
    short_cols = list(FEATURE_COLUMNS)[:-1]
    save_model(
        model=trained_calibrated_model,
        metrics={},
        feature_columns=short_cols,
        best_params={},
        model_dir=str(tmp_path),
        version="vshort",
    )
    with pytest.raises(RuntimeError) as exc_info:
        predictor.ModelPredictor(model_dir=str(tmp_path), version="vshort")
    msg = str(exc_info.value)
    assert "feature_columns" in msg.lower() or "feature column" in msg.lower()
    assert str(len(FEATURE_COLUMNS)) in msg
    assert str(len(short_cols)) in msg


# ── Test 3: drift on order ──────────────────────────────────────────────────


def test_predictor_load_raises_on_order_mismatch(
    tmp_path, trained_calibrated_model,
):
    """Test 3: same length, reversed order → RuntimeError."""
    reversed_cols = list(reversed(FEATURE_COLUMNS))
    save_model(
        model=trained_calibrated_model,
        metrics={},
        feature_columns=reversed_cols,
        best_params={},
        model_dir=str(tmp_path),
        version="vrev",
    )
    with pytest.raises(RuntimeError) as exc_info:
        predictor.ModelPredictor(model_dir=str(tmp_path), version="vrev")
    assert "feature" in str(exc_info.value).lower()


# ── Test 4: xgb_v2 still loads ──────────────────────────────────────────────


def test_xgb_v2_still_loads():
    """Test 4: real production xgb_v2 model loads cleanly post-LIVE-03.

    Phase 18 NET-V2-01 evolved this assertion: xgb_v2 was saved BEFORE
    Phase 16-03 appended the 3 NET-* cols, so its meta has 72 columns
    (n_features=72). It must continue to load cleanly via the
    include_net=False dispatch path. Any change that breaks this test
    is a Pitfall #12 regression on the rollback path.
    """
    from ufc_prediction.ml.config import FEATURE_COLUMNS_NO_NET

    models_dir = Path("models")
    if not (models_dir / "xgb_v2.joblib").exists():
        pytest.skip("xgb_v2.joblib not present in this worktree")
    pred = predictor.ModelPredictor(model_dir="models", version="v2")
    # xgb_v2 was trained on the 72-col view (pre-Phase 16-03 NET-* append).
    assert pred.metadata["feature_columns"] == list(FEATURE_COLUMNS_NO_NET)
    # Phase 18 dispatch: 72 -> include_net=False.
    assert pred._include_net is False


# ── Test 5: documented behaviour for xgb_v1 ────────────────────────────────


def test_xgb_v1_documented_behavior_raises():
    """Test 5: xgb_v1 has 28 features vs current FEATURE_COLUMNS (72) →
    intentionally raises RuntimeError.

    Documents that rollback to xgb_v1 in the production tree requires
    pinning FEATURE_COLUMNS at the v1 snapshot (NOT in scope for Phase 16).
    """
    models_dir = Path("models")
    if not (models_dir / "xgb_v1.joblib").exists():
        pytest.skip("xgb_v1.joblib not present in this worktree")
    with pytest.raises(RuntimeError):
        predictor.ModelPredictor(model_dir="models", version="v1")
