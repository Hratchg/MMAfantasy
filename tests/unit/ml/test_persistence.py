"""HOUSE-01: assert save_model writes n_features to *-meta.json.

Plan 16-01 Task 1. Pitfall #12 corroboration test — `n_features` must equal
`len(feature_columns)` in every meta JSON. xgb_v2_meta.json already happens
to have the field (manual edit per v1.1-MILESTONE-AUDIT.md), so this fix
only affects FUTURE save_model calls (Gotcha 6).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

persistence = pytest.importorskip("ufc_prediction.ml.persistence")
from ufc_prediction.ml.config import FEATURE_COLUMNS  # noqa: E402


@pytest.fixture()
def trained_model():
    """Lightweight calibrated model for round-tripping save_model.

    Mirrors the fixture in tests/ml/test_persistence.py:21-42 so behaviour
    matches what the existing suite exercises.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from xgboost import XGBClassifier

    rng = np.random.RandomState(42)
    X = rng.randn(60, len(FEATURE_COLUMNS))
    y = rng.randint(0, 2, size=60)

    base_model = XGBClassifier(
        n_estimators=10,
        max_depth=3,
        objective="binary:logistic",
        random_state=42,
        verbosity=0,
    )
    base_model.fit(X[:48], y[:48])

    calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
    calibrated.fit(X[48:], y[48:])
    return calibrated


def _save_and_load(tmp_path, model, feature_columns):
    persistence.save_model(
        model=model,
        feature_columns=feature_columns,
        best_params={},
        metrics={"brier_score": 0.25, "auc_roc": 0.5, "accuracy": 0.5},
        cutoff_date="2024-01-01",
        n_training_fights=48,
        n_test_fights=12,
        model_dir=str(tmp_path),
        version="house01",
    )
    return json.loads((tmp_path / "xgb_house01_meta.json").read_text())


def test_save_model_writes_n_features(tmp_path, trained_model):
    """HOUSE-01 contract: meta["n_features"] is recorded as int(len(feature_columns))."""
    meta = _save_and_load(tmp_path, trained_model, list(FEATURE_COLUMNS))
    assert "n_features" in meta, "save_model must record n_features after HOUSE-01"
    assert meta["n_features"] == len(FEATURE_COLUMNS)


def test_n_features_equals_feature_columns_length(tmp_path, trained_model):
    """Pitfall #12 corroboration: n_features must always equal len(feature_columns)."""
    meta = _save_and_load(tmp_path, trained_model, list(FEATURE_COLUMNS))
    assert meta["n_features"] == len(meta["feature_columns"])


def test_n_features_int_serializable(tmp_path, trained_model):
    """Regression: n_features is plain Python int, not numpy.int64.

    JSON serialization fails on numpy scalars; the implementation must wrap
    the length in `int(...)` per existing _convert_numpy_types convention.
    """
    meta = _save_and_load(tmp_path, trained_model, list(FEATURE_COLUMNS))
    assert isinstance(meta["n_features"], int)
    # Round-trip — if the field were numpy-typed the prior read would have raised.
    assert meta["n_features"] > 0
