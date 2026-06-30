"""Phase 19 META-05 ship-criterion — `ufc predict matchup` CLI returns meta-aware JSON.

Per OQ-5 (RESEARCH.md): mocked DB session — NO live PostgreSQL required.
Mirrors tests/integration/test_live_odds_acceptance.py fixture pattern.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

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

pytestmark = [pytest.mark.integration]


@pytest.fixture
def upcoming_event_date() -> date:
    return date.today() + timedelta(days=7)


@pytest.fixture
def predict_log_path(tmp_path, monkeypatch) -> Path:
    log_path = tmp_path / "predict-trace.jsonl"
    monkeypatch.setenv("UFC_PREDICT_LOG", str(log_path))
    return log_path


@pytest.fixture
def stub_fighters():
    fa = MagicMock(id=1, source="ufcstats", height_inches=70.0)
    fa.name = "Khabib Nurmagomedov"
    fb = MagicMock(id=2, source="ufcstats", height_inches=72.0)
    fb.name = "Conor McGregor"
    return fa, fb


@pytest.fixture
def models_with_meta(tmp_path):
    """Persist a tiny xgb_vmeta + meta_v1 in tmp_path/models/."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 72))
    y = rng.integers(0, 2, size=60)
    base = XGBClassifier(
        n_estimators=5, max_depth=2, objective="binary:logistic", random_state=42, verbosity=0
    )
    base.fit(X[:48], y[:48])
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    cal.fit(X[48:], y[48:])
    save_model(
        model=cal,
        metrics={"brier_score": 0.22, "auc_roc": 0.69, "accuracy": 0.65},
        feature_columns=list(FEATURE_COLUMNS_NO_NET),
        best_params={},
        model_dir=str(model_dir),
        version="vmeta",
        cutoff_date="2023-01-01",
        n_training_fights=48,
        n_test_fights=12,
    )
    meta_dir = model_dir / "meta"
    meta_dir.mkdir()
    Xm = rng.uniform(0.1, 0.9, size=(80, 3))
    ym = rng.integers(0, 2, size=80)
    m = meta_learner.MetaLearnerLogistic().fit(Xm, ym)
    import hashlib

    base_sha = hashlib.sha256((model_dir / "xgb_vmeta.joblib").read_bytes()).hexdigest()
    meta_persistence.save_meta_model(
        m,
        meta_kind="logistic",
        meta_version="v1",
        base_model_version="vmeta",
        base_model_sha256=base_sha,
        meta_feature_columns=["xgb_oof_prob", "elo_prob", "closing_prob_diff"],
        meta_input_distribution_hash="d" * 64,
        meta_oof_parquet_sha256="c" * 64,
        meta_learner_brier_delta_vs_logistic=0.0,
        best_params={"C": 1.0},
        metrics={"per_slice": {}, "median_brier_overall": 0.21},
        meta_dir=str(meta_dir),
    )
    return model_dir, meta_dir


def _patched_predict(
    model_dir,
    meta_dir,
    stub_fighters,
    *,
    closing_prob_diff: float = 0.05,
    no_use_meta: bool = False,
):
    fa, fb = stub_fighters
    p = predictor.ModelPredictor(
        model_dir=str(model_dir),
        version="vmeta",
        meta_dir=None if no_use_meta else str(meta_dir),
    )
    idx = FEATURE_COLUMNS_NO_NET.index("closing_prob_diff")
    feature_vec = np.zeros((1, 72))
    feature_vec[0, idx] = closing_prob_diff
    with (
        patch("ufc_prediction.ml.predictor._resolve_fighter") as mock_resolve,
        patch("ufc_prediction.ml.predictor.build_inference_features", return_value=feature_vec),
        patch("ufc_prediction.ml.predictor.fetch_matchup_odds", return_value=None),
        patch("ufc_prediction.ml.predictor._get_latest_elo", return_value=1500),
    ):
        mock_resolve.side_effect = [fa, fb]
        return p.predict(MagicMock(), fa.name, fb.name)


def test_predict_matchup_includes_meta_fields(models_with_meta, stub_fighters, predict_log_path):
    model_dir, meta_dir = models_with_meta
    result = _patched_predict(model_dir, meta_dir, stub_fighters)
    for field in (
        "win_probability",
        "base_prob",
        "meta_prob",
        "meta_kind",
        "meta_learner_version",
        "meta_skipped",
        "meta_skipped_reason",
    ):
        assert field in result, f"missing field {field!r} in predictor output"
    assert result["meta_kind"] == "logistic"
    assert result["meta_learner_version"] == "v1"
    assert result["meta_skipped"] is False


def test_predict_matchup_no_use_meta_flag(models_with_meta, stub_fighters):
    model_dir, meta_dir = models_with_meta
    result = _patched_predict(model_dir, meta_dir, stub_fighters, no_use_meta=True)
    assert result["meta_skipped"] is True
    assert result["meta_prob"] is None
    assert result["win_probability"] == pytest.approx(result["base_prob"])


def test_predict_matchup_skips_on_nan_closing_prob_diff(models_with_meta, stub_fighters):
    model_dir, meta_dir = models_with_meta
    result = _patched_predict(model_dir, meta_dir, stub_fighters, closing_prob_diff=float("nan"))
    assert result["meta_skipped"] is True
    assert result["meta_skipped_reason"] == "nan_closing_prob_diff"
    assert result["meta_prob"] is None
