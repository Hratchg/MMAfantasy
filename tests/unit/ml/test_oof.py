"""Phase 19 META-03 Wave-0 RED — oof.py generates leakage-free OOF predictions via TimeSeriesSplit.

These tests RED on import (Wave 0) — `ufc_prediction.ml.oof.generate_oof_predictions`
does not yet exist. They go GREEN at Wave 1 when oof.py lands.

Per CONTEXT.md D-06(P19): n_jobs=1 NON-NEGOTIABLE (Py 3.14 spawn pickling).
Per RESEARCH.md OQ-2: raw XGBClassifier per fold (META-01 absorbs recalibration).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from ufc_prediction.ml.config import FEATURE_COLUMNS_NO_NET

oof = pytest.importorskip("ufc_prediction.ml.oof")


def _make_synthetic_data(n: int = 200, n_features: int = 72, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 2, size=n)
    fight_dates = np.array([
        np.datetime64("2023-01-01") + np.timedelta64(i, "D") for i in range(n)
    ])
    return X, y, fight_dates


def _make_base_trainer():
    """Tiny stub trainer exposing _make_estimator() returning XGBClassifier."""
    trainer = MagicMock()
    trainer._make_estimator = MagicMock(return_value=XGBClassifier(
        n_estimators=10, max_depth=2, objective="binary:logistic",
        random_state=42, verbosity=0,
    ))
    return trainer


def test_oof_returns_correct_shape():
    """Smoke: len(oof_proba) == len(y_train)."""
    X, y, dates = _make_synthetic_data(n=120, n_features=72)
    trainer = _make_base_trainer()
    oof_proba, meta = oof.generate_oof_predictions(X, y, dates, trainer, n_splits=5)
    assert oof_proba.shape == (120,)
    assert "training_accuracy" in meta


def test_oof_uses_timeseries_split():
    """Pitfall #11: cv must be TimeSeriesSplit instance, n_jobs must be 1."""
    X, y, dates = _make_synthetic_data(n=120)
    trainer = _make_base_trainer()
    with patch("ufc_prediction.ml.oof.cross_val_predict") as mock_cvp:
        # Return a fake (n,2) probability matrix shaped like predict_proba.
        # Use noisy probs that don't perfectly track y — otherwise the
        # Pitfall #11 OOFLeakageError sanity check fires before kwargs
        # assertions are evaluated. 0.5 + (y-0.5)*0.05 → range [0.475, 0.525]
        # for noise-floor accuracy ~50% which is below the 0.75 leakage gate.
        rng_local = np.random.default_rng(0)
        noise = rng_local.uniform(-0.05, 0.05, size=len(y))
        prob_pos = np.clip(0.5 + noise, 0.0, 1.0)
        mock_cvp.return_value = np.column_stack([1 - prob_pos, prob_pos])
        oof.generate_oof_predictions(X, y, dates, trainer, n_splits=5)
        kwargs = mock_cvp.call_args.kwargs
        assert isinstance(kwargs["cv"], TimeSeriesSplit), (
            f"cv must be TimeSeriesSplit instance; got {type(kwargs['cv']).__name__}"
        )
        assert kwargs["n_jobs"] == 1, (
            f"n_jobs must be 1 (Py 3.14 spawn safety); got {kwargs['n_jobs']!r}"
        )
        assert kwargs["method"] == "predict_proba"


def test_oof_cache_invariant_check_xgb_sha(tmp_path):
    """D-06(P19): cache with stale xgb_v2_sha256 → InvariantCheckError."""
    cache_path = tmp_path / "oof_predictions.parquet"
    sidecar = tmp_path / "oof_predictions.meta.json"
    # Write a fake parquet + sidecar with stale SHA
    import pandas as pd
    pd.DataFrame({"fight_id": [1], "xgb_oof_prob": [0.5]}).to_parquet(cache_path)
    sidecar.write_text(json.dumps({
        "xgb_v2_sha256": "deadbeef" * 8,  # stale
        "n_features": 72, "cutoff_date": "2023-01-01",
        "event_date_min": "2023-01-01", "event_date_max": "2024-01-01",
        "n_splits": 5, "training_accuracy": 0.65,
        "trained_at": "2026-05-09", "cv_kind": "TimeSeriesSplit",
    }))
    X, y, dates = _make_synthetic_data(n=120)
    trainer = _make_base_trainer()
    with pytest.raises(oof.InvariantCheckError, match="xgb_v2_sha256"):
        oof.generate_oof_predictions(
            X, y, dates, trainer, n_splits=5,
            cache_path=cache_path, force_rebuild=False,
        )


def test_oof_cache_invariant_check_n_features(tmp_path):
    """D-06(P19): cache with mismatched n_features → InvariantCheckError."""
    cache_path = tmp_path / "oof_predictions.parquet"
    sidecar = tmp_path / "oof_predictions.meta.json"
    import pandas as pd
    pd.DataFrame({"fight_id": [1], "xgb_oof_prob": [0.5]}).to_parquet(cache_path)
    # Write the LIVE xgb_v2 SHA so the SHA check passes; n_features mismatch fires
    actual_sha = hashlib.sha256(Path("models/xgb_v2.joblib").read_bytes()).hexdigest()
    sidecar.write_text(json.dumps({
        "xgb_v2_sha256": actual_sha,
        "n_features": 75,  # mismatch — code expects 72
        "cutoff_date": "2023-01-01",
        "event_date_min": "2023-01-01", "event_date_max": "2024-01-01",
        "n_splits": 5, "training_accuracy": 0.65,
        "trained_at": "2026-05-09", "cv_kind": "TimeSeriesSplit",
    }))
    X, y, dates = _make_synthetic_data(n=120)
    trainer = _make_base_trainer()
    with pytest.raises(oof.InvariantCheckError, match="n_features"):
        oof.generate_oof_predictions(
            X, y, dates, trainer, n_splits=5,
            cache_path=cache_path, force_rebuild=False,
        )


def test_oof_training_accuracy_assertion():
    """Pitfall #11 sanity: training_accuracy >= 0.75 → OOFLeakageError."""
    X, y, dates = _make_synthetic_data(n=120)
    trainer = _make_base_trainer()
    with patch("ufc_prediction.ml.oof.cross_val_predict") as mock_cvp:
        # Return probs that perfectly match y → training_accuracy ~ 1.0
        perfect_probs = np.where(y == 1, 0.99, 0.01)
        mock_cvp.return_value = np.column_stack([1 - perfect_probs, perfect_probs])
        with pytest.raises(oof.OOFLeakageError, match="in-sample"):
            oof.generate_oof_predictions(X, y, dates, trainer, n_splits=5)
