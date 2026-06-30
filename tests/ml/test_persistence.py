"""Tests for model persistence (save/load with joblib + JSON metadata).

Covers save_model, load_model, load_metadata, and get_latest_version
per D-07: joblib file with companion JSON metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

from ufc_prediction.ml.config import FEATURE_COLUMNS


@pytest.fixture()
def trained_model():
    """Create a simple trained and calibrated model for persistence tests."""
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


@pytest.fixture()
def sample_metrics():
    """Sample evaluation metrics."""
    return {
        "brier_score": 0.235,
        "auc_roc": 0.62,
        "accuracy": 0.58,
    }


@pytest.fixture()
def sample_params():
    """Sample best hyperparameters."""
    return {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
    }


class TestSaveModel:
    """Tests for save_model function."""

    def test_save_creates_joblib_file(self, tmp_path, trained_model, sample_metrics, sample_params):
        """save_model creates a .joblib file in model_dir."""
        from ufc_prediction.ml.persistence import save_model

        save_model(
            model=trained_model,
            metrics=sample_metrics,
            feature_columns=FEATURE_COLUMNS,
            best_params=sample_params,
            model_dir=str(tmp_path),
            version="v1",
        )

        assert (tmp_path / "xgb_v1.joblib").exists()

    def test_save_creates_meta_json(self, tmp_path, trained_model, sample_metrics, sample_params):
        """save_model creates a _meta.json file in model_dir."""
        from ufc_prediction.ml.persistence import save_model

        save_model(
            model=trained_model,
            metrics=sample_metrics,
            feature_columns=FEATURE_COLUMNS,
            best_params=sample_params,
            model_dir=str(tmp_path),
            version="v1",
        )

        assert (tmp_path / "xgb_v1_meta.json").exists()

    def test_metadata_content(self, tmp_path, trained_model, sample_metrics, sample_params):
        """Metadata JSON contains all required fields."""
        from ufc_prediction.ml.persistence import save_model

        save_model(
            model=trained_model,
            metrics=sample_metrics,
            feature_columns=FEATURE_COLUMNS,
            best_params=sample_params,
            model_dir=str(tmp_path),
            version="v1",
            cutoff_date="2023-01-01",
            n_training_fights=10000,
            n_test_fights=3000,
        )

        meta_path = tmp_path / "xgb_v1_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        assert meta["version"] == "v1"
        assert "trained_at" in meta
        assert "xgboost_version" in meta
        assert "sklearn_version" in meta
        assert "python_version" in meta
        assert meta["feature_columns"] == list(FEATURE_COLUMNS)
        assert meta["best_params"] == sample_params
        assert meta["metrics"] == sample_metrics
        assert meta["cutoff_date"] == "2023-01-01"
        assert meta["n_training_fights"] == 10000
        assert meta["n_test_fights"] == 3000
        # HOUSE-01: n_features mirrors len(feature_columns) (Pitfall #12 corroboration)
        assert meta["n_features"] == len(FEATURE_COLUMNS)


class TestLoadModel:
    """Tests for load_model function."""

    def test_load_model_round_trip(self, tmp_path, trained_model, sample_metrics, sample_params):
        """save_model + load_model round-trip preserves model predictions."""
        from ufc_prediction.ml.persistence import save_model, load_model

        rng = np.random.RandomState(99)
        test_X = rng.randn(5, len(FEATURE_COLUMNS))

        save_model(
            model=trained_model,
            metrics=sample_metrics,
            feature_columns=FEATURE_COLUMNS,
            best_params=sample_params,
            model_dir=str(tmp_path),
            version="v1",
        )

        loaded = load_model(str(tmp_path), "v1")

        original_probs = trained_model.predict_proba(test_X)
        loaded_probs = loaded.predict_proba(test_X)

        np.testing.assert_array_almost_equal(original_probs, loaded_probs)

    def test_load_nonexistent_raises(self, tmp_path):
        """load_model raises FileNotFoundError for nonexistent version."""
        from ufc_prediction.ml.persistence import load_model

        with pytest.raises(FileNotFoundError):
            load_model(str(tmp_path), "v999")

    def test_load_metadata(self, tmp_path, trained_model, sample_metrics, sample_params):
        """load_metadata returns the JSON metadata dict."""
        from ufc_prediction.ml.persistence import save_model, load_metadata

        save_model(
            model=trained_model,
            metrics=sample_metrics,
            feature_columns=FEATURE_COLUMNS,
            best_params=sample_params,
            model_dir=str(tmp_path),
            version="v1",
        )

        meta = load_metadata(str(tmp_path), "v1")

        assert meta["version"] == "v1"
        assert meta["metrics"] == sample_metrics


class TestGetLatestVersion:
    """Tests for get_latest_version function."""

    def test_get_latest_version_multiple(self, tmp_path):
        """get_latest_version returns highest version number from existing files."""
        from ufc_prediction.ml.persistence import get_latest_version

        # Create fake model files
        (tmp_path / "xgb_v1.joblib").touch()
        (tmp_path / "xgb_v2.joblib").touch()

        result = get_latest_version(str(tmp_path))
        assert result == "v2"

    def test_get_latest_version_empty(self, tmp_path):
        """get_latest_version returns 'v1' for empty directory."""
        from ufc_prediction.ml.persistence import get_latest_version

        result = get_latest_version(str(tmp_path))
        assert result == "v1"

    def test_get_latest_version_single(self, tmp_path):
        """get_latest_version returns correct version with one file."""
        from ufc_prediction.ml.persistence import get_latest_version

        (tmp_path / "xgb_v5.joblib").touch()

        result = get_latest_version(str(tmp_path))
        assert result == "v5"
