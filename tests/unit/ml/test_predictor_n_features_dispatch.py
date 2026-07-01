"""Phase 18 NET-V2-01 Wave-0 RED — predictor.py auto-derives include_net from n_features.

Goal: ModelPredictor(__init__) reads `metadata["n_features"]` and validates
it `in (72, 75)`. Stores `self._include_net = (n_features == 75)`. LIVE-03
compares against `get_feature_columns(include_net=self._include_net)`.

Pitfall B (NET-V2-01 + Pitfall #12 carry-forward): n_features outside (72, 75)
is a hard halt — the model was trained against an unknown column space; we
cannot dispatch the correct inference_features.build() variant.

These tests RED on import (Wave 0) — `FEATURE_COLUMNS_NO_NET` does not yet
exist in `config.py`; `self._include_net` does not yet exist on ModelPredictor.
Goes GREEN at Wave 1 Task 11.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

from ufc_prediction.ml.config import FEATURE_COLUMNS, FEATURE_COLUMNS_NO_NET
from ufc_prediction.ml.persistence import save_model

predictor = pytest.importorskip("ufc_prediction.ml.predictor")


# ── Shared fixture (lifted from tests/unit/ml/test_predictor_load_assertion.py) ──


def _build_calibrated_model(n_features: int) -> CalibratedClassifierCV:
    """Tiny calibrated XGBClassifier sized to `n_features`."""
    rng = np.random.RandomState(42)
    X = rng.randn(60, n_features)
    y = rng.randint(0, 2, size=60)
    base = XGBClassifier(
        n_estimators=5,
        max_depth=2,
        objective="binary:logistic",
        random_state=42,
        verbosity=0,
    )
    base.fit(X[:48], y[:48])
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    cal.fit(X[48:], y[48:])
    return cal


# ── NET-V2-01 dispatch tests ────────────────────────────────────────────────


def test_predictor_loads_72_col_ablation_model(tmp_path):
    """Persist a 72-col toy model; ModelPredictor loads it and sets
    `_include_net = False`. Metadata contains n_features=72.
    """
    model = _build_calibrated_model(72)
    save_model(
        model=model,
        metrics={"brier_score": 0.22},
        feature_columns=FEATURE_COLUMNS_NO_NET,
        best_params={},
        model_dir=str(tmp_path),
        version="vablation",
    )
    p = predictor.ModelPredictor(model_dir=str(tmp_path), version="vablation", meta_dir=None)
    assert p.metadata["n_features"] == 72
    assert p._include_net is False


def test_predictor_loads_75_col_timedecayed_model(tmp_path):
    """Persist a 75-col toy model; ModelPredictor loads it and sets
    `_include_net = True`. Metadata contains n_features=75.
    """
    model = _build_calibrated_model(75)
    save_model(
        model=model,
        metrics={"brier_score": 0.22},
        feature_columns=FEATURE_COLUMNS,
        best_params={},
        model_dir=str(tmp_path),
        version="vtimedecayed",
    )
    p = predictor.ModelPredictor(model_dir=str(tmp_path), version="vtimedecayed", meta_dir=None)
    assert p.metadata["n_features"] == 75
    assert p._include_net is True


def test_predictor_raises_on_invalid_n_features(tmp_path):
    """Pitfall B: n_features=50 is neither 72 nor 75 → ModelPredictor raises
    RuntimeError before any predict_proba can run.
    """
    model = _build_calibrated_model(50)
    save_model(
        model=model,
        metrics={"brier_score": 0.5},
        feature_columns=[f"col_{i}" for i in range(50)],
        best_params={},
        model_dir=str(tmp_path),
        version="vinvalid",
    )
    with pytest.raises(RuntimeError) as exc_info:
        predictor.ModelPredictor(model_dir=str(tmp_path), version="vinvalid")
    msg = str(exc_info.value)
    # Must reference the bad value AND the expected set
    assert "50" in msg
    assert "72" in msg
    assert "75" in msg
