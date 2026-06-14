"""Phase 19 META-05 Wave-0 RED — predictor.py auto-loads meta + dispatches per D-04/D-05.

These tests focus on the predictor BOUNDARY behavior. The session is mocked
(per OQ-5 — no live PostgreSQL) so we can validate dispatch without DB.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

from ufc_prediction.ml.config import FEATURE_COLUMNS_NO_NET
from ufc_prediction.ml.persistence import save_model

predictor = pytest.importorskip("ufc_prediction.ml.predictor")
meta_learner = pytest.importorskip("ufc_prediction.ml.meta_learner")
meta_persistence = pytest.importorskip("ufc_prediction.ml.meta_persistence")


def _build_calibrated_model(n_features: int = 72):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, n_features))
    y = rng.integers(0, 2, size=60)
    base = XGBClassifier(n_estimators=5, max_depth=2, objective="binary:logistic",
                         random_state=42, verbosity=0)
    base.fit(X[:48], y[:48])
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    cal.fit(X[48:], y[48:])
    return cal


def _save_xgb_v2(tmp_path: Path) -> Path:
    """Persist a tiny calibrated 72-col model named xgb_vmeta.joblib (so version='vmeta' loads)."""
    model = _build_calibrated_model(72)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    save_model(
        model=model,
        metrics={"brier_score": 0.22, "auc_roc": 0.69, "accuracy": 0.65},
        feature_columns=list(FEATURE_COLUMNS_NO_NET),
        best_params={},
        model_dir=str(model_dir),
        version="vmeta",
        cutoff_date="2023-01-01",
        n_training_fights=48,
        n_test_fights=12,
    )
    return model_dir


def _save_meta_v1(tmp_path: Path, base_sha: str | None = None,
                  feature_columns: list[str] | None = None) -> Path:
    """Persist a tiny MetaLearnerLogistic at meta_dir/meta_v1.joblib + meta_v1_meta.json."""
    rng = np.random.default_rng(42)
    Xm = rng.uniform(0.1, 0.9, size=(80, 3))
    ym = rng.integers(0, 2, size=80)
    m = meta_learner.MetaLearnerLogistic().fit(Xm, ym)
    meta_dir = tmp_path / "models" / "meta"
    meta_dir.mkdir(parents=True)
    if base_sha is None:
        import hashlib
        base_sha = hashlib.sha256((tmp_path / "models" / "xgb_vmeta.joblib").read_bytes()).hexdigest()
    meta_persistence.save_meta_model(
        m,
        meta_kind="logistic", meta_version="v1",
        base_model_version="vmeta", base_model_sha256=base_sha,
        meta_feature_columns=feature_columns or ["xgb_oof_prob", "elo_prob", "closing_prob_diff"],
        meta_input_distribution_hash="d" * 64,
        meta_oof_parquet_sha256="c" * 64,
        meta_learner_brier_delta_vs_logistic=0.0,
        best_params={"C": 1.0},
        metrics={"per_slice": {}, "median_brier_overall": 0.21},
        meta_dir=str(meta_dir),
    )
    return meta_dir


def test_predict_with_meta_loaded(tmp_path):
    """Mocked meta returns 0.7 → win_probability=0.7, meta_skipped=False."""
    model_dir = _save_xgb_v2(tmp_path)
    meta_dir = _save_meta_v1(tmp_path)
    p = predictor.ModelPredictor(model_dir=str(model_dir), version="vmeta", meta_dir=str(meta_dir))
    # Mock the inference path so we don't need a DB session
    with patch.object(p, "model") as mock_xgb, \
         patch.object(p, "meta_model") as mock_meta, \
         patch("ufc_prediction.ml.predictor._resolve_fighter") as mock_resolve, \
         patch("ufc_prediction.ml.predictor.build_inference_features") as mock_build, \
         patch("ufc_prediction.ml.predictor.fetch_matchup_odds") as mock_odds, \
         patch("ufc_prediction.ml.predictor._get_latest_elo") as mock_elo:
        mock_xgb.predict_proba.return_value = np.array([[0.4, 0.6]])
        mock_meta.predict_proba.return_value = np.array([[0.3, 0.7]])
        fa = MagicMock(name="fa", id=1); fa.name = "A"
        fb = MagicMock(name="fb", id=2); fb.name = "B"
        mock_resolve.side_effect = [fa, fb]
        # Build a 72-col feature vector with closing_prob_diff at the canonical index
        idx = FEATURE_COLUMNS_NO_NET.index("closing_prob_diff")
        feature_vec = np.zeros((1, 72))
        feature_vec[0, idx] = 0.05
        mock_build.return_value = feature_vec
        mock_odds.return_value = None
        mock_elo.return_value = 1500
        result = p.predict(MagicMock(), "A", "B")
    assert result["meta_skipped"] is False
    assert result["meta_prob"] == pytest.approx(0.7)
    assert result["win_probability"] == pytest.approx(0.7)
    assert result["base_prob"] == pytest.approx(0.6)
    assert result["meta_kind"] == "logistic"
    assert result["meta_learner_version"] == "v1"
    assert result["meta_skipped_reason"] is None


def test_predict_with_no_meta_artifact(tmp_path):
    """meta_dir=None → meta_skipped=True, reason='no_meta_artifact', win_probability=base_prob."""
    model_dir = _save_xgb_v2(tmp_path)
    p = predictor.ModelPredictor(model_dir=str(model_dir), version="vmeta", meta_dir=None)
    assert p.meta_model is None
    with patch.object(p, "model") as mock_xgb, \
         patch("ufc_prediction.ml.predictor._resolve_fighter") as mock_resolve, \
         patch("ufc_prediction.ml.predictor.build_inference_features") as mock_build, \
         patch("ufc_prediction.ml.predictor.fetch_matchup_odds") as mock_odds, \
         patch("ufc_prediction.ml.predictor._get_latest_elo") as mock_elo:
        mock_xgb.predict_proba.return_value = np.array([[0.45, 0.55]])
        fa = MagicMock(name="fa", id=1); fa.name = "A"
        fb = MagicMock(name="fb", id=2); fb.name = "B"
        mock_resolve.side_effect = [fa, fb]
        mock_build.return_value = np.zeros((1, 72))
        mock_odds.return_value = None
        mock_elo.return_value = 1500
        result = p.predict(MagicMock(), "A", "B")
    assert result["meta_skipped"] is True
    assert result["meta_skipped_reason"] == "no_meta_artifact"
    assert result["meta_prob"] is None
    assert result["win_probability"] == pytest.approx(0.55)


def test_predict_with_nan_closing_prob_diff(tmp_path):
    """NaN closing_prob_diff → meta_skipped=True, reason='nan_closing_prob_diff'."""
    model_dir = _save_xgb_v2(tmp_path)
    meta_dir = _save_meta_v1(tmp_path)
    p = predictor.ModelPredictor(model_dir=str(model_dir), version="vmeta", meta_dir=str(meta_dir))
    with patch.object(p, "model") as mock_xgb, \
         patch("ufc_prediction.ml.predictor._resolve_fighter") as mock_resolve, \
         patch("ufc_prediction.ml.predictor.build_inference_features") as mock_build, \
         patch("ufc_prediction.ml.predictor.fetch_matchup_odds") as mock_odds, \
         patch("ufc_prediction.ml.predictor._get_latest_elo") as mock_elo:
        mock_xgb.predict_proba.return_value = np.array([[0.4, 0.6]])
        fa = MagicMock(name="fa", id=1); fa.name = "A"
        fb = MagicMock(name="fb", id=2); fb.name = "B"
        mock_resolve.side_effect = [fa, fb]
        idx = FEATURE_COLUMNS_NO_NET.index("closing_prob_diff")
        feature_vec = np.zeros((1, 72))
        feature_vec[0, idx] = np.nan  # missing live odds
        mock_build.return_value = feature_vec
        mock_odds.return_value = None
        mock_elo.return_value = 1500
        result = p.predict(MagicMock(), "A", "B")
    assert result["meta_skipped"] is True
    assert result["meta_skipped_reason"] == "nan_closing_prob_diff"
    assert result["meta_prob"] is None
    assert result["win_probability"] == pytest.approx(0.6)


def test_predict_meta_base_sha_mismatch_halts(tmp_path):
    """LIVE-03 evolution: meta with wrong base_model_sha256 → RuntimeError on __init__."""
    model_dir = _save_xgb_v2(tmp_path)
    # Save meta with a fake (mismatched) base SHA
    _save_meta_v1(tmp_path, base_sha="d" * 64)
    with pytest.raises(RuntimeError, match="base_model_sha256"):
        predictor.ModelPredictor(model_dir=str(model_dir), version="vmeta",
                                 meta_dir=str(tmp_path / "models" / "meta"))


def test_predict_meta_feature_columns_drift_halts(tmp_path):
    """LIVE-03 evolution: meta_feature_columns drift → RuntimeError on __init__."""
    model_dir = _save_xgb_v2(tmp_path)
    _save_meta_v1(tmp_path, feature_columns=["xgb_oof_prob", "elo_prob", "OTHER_FEATURE"])
    with pytest.raises(RuntimeError, match="meta_feature_columns"):
        predictor.ModelPredictor(model_dir=str(model_dir), version="vmeta",
                                 meta_dir=str(tmp_path / "models" / "meta"))
