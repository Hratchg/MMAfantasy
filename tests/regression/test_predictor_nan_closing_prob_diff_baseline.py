"""
Characterization test (per Phase 34 revision m1). Captures the CURRENT (pre-Phase-34) behavior of predictor.py
when BFO is unreachable AND DB odds cache misses (closing_prob_diff = NaN at inference).

NOT a TDD red-green test — this captures current state pre-patch. The Wave 5 integration
test (tests/integration/test_predictor_fallback_no_odds.py) is the actual assertion of
new behavior. This file will be archived or xfail-strict'd by 34-06 Task 3.

Locked bucket: **B** — graceful meta-skip with fallback to base xgb_v2 probability.

When ``closing_prob_diff`` is NaN at the canonical column index of the inference feature
vector, ``ModelPredictor.predict()`` does NOT crash. Instead the META block is
short-circuited and the response carries:

    result["meta_skipped"]         == True
    result["meta_skipped_reason"]  == "nan_closing_prob_diff"
    result["meta_prob"]            is None
    result["win_probability"]      == result["base_prob"]   # raw xgb_v2 probability

The implication for FALLBACK-V24-02 (Wave 5): the pre-fix path silently serves a
degraded prediction (raw 72-col xgb_v2 inference with a NaN cell injected). The
fix introduces a new code path — load ``xgb_v2_no_odds`` and predict from a 67-col
vector — that bypasses both META AND the NaN-carrying xgb_v2 inference call. This
characterization test exists to make that change visible: when Wave 5 lands, this
test's assertions will no longer hold, and it will be archived (since the entire
Bucket-B path is obsoleted, not merely replaced — see 34-06 Task 3).

Per Phase 34 RESEARCH Open Q1.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

from ufc_prediction.ml.config import FEATURE_COLUMNS_NO_NET
from ufc_prediction.ml.persistence import save_model

predictor_module = pytest.importorskip("ufc_prediction.ml.predictor")
meta_learner = pytest.importorskip("ufc_prediction.ml.meta_learner")
meta_persistence = pytest.importorskip("ufc_prediction.ml.meta_persistence")


def _build_calibrated_model(n_features: int = 72):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, n_features))
    y = rng.integers(0, 2, size=60)
    base = XGBClassifier(
        n_estimators=5, max_depth=2, objective="binary:logistic", random_state=42, verbosity=0
    )
    base.fit(X[:48], y[:48])
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    cal.fit(X[48:], y[48:])
    return cal


def _save_xgb_vmeta(tmp_path: Path) -> Path:
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


def _save_meta_v1(tmp_path: Path) -> Path:
    import hashlib

    rng = np.random.default_rng(42)
    Xm = rng.uniform(0.1, 0.9, size=(80, 3))
    ym = rng.integers(0, 2, size=80)
    m = meta_learner.MetaLearnerLogistic().fit(Xm, ym)
    meta_dir = tmp_path / "models" / "meta"
    meta_dir.mkdir(parents=True)
    base_sha = hashlib.sha256((tmp_path / "models" / "xgb_vmeta.joblib").read_bytes()).hexdigest()
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
    return meta_dir


def test_nan_closing_prob_diff_currently_falls_back_to_base_prob(tmp_path):
    """Bucket B characterization — current behavior on main (pre-FALLBACK-V24-02).

    When BFO is unreachable (fetch_matchup_odds returns None) AND the built feature
    vector has NaN at the closing_prob_diff column, the predictor:
      - does NOT crash
      - sets meta_skipped=True with reason="nan_closing_prob_diff"
      - returns win_probability == base_prob (raw xgb_v2 inference)

    Wave 5 will change this — the predictor will route to xgb_v2_no_odds and the
    win_probability will derive from the 67-col fallback model, not the 72-col
    base model with a NaN cell. This test will then be archived by 34-06 Task 3.
    """
    model_dir = _save_xgb_vmeta(tmp_path)
    meta_dir = _save_meta_v1(tmp_path)
    p = predictor_module.ModelPredictor(
        model_dir=str(model_dir), version="vmeta", meta_dir=str(meta_dir)
    )
    # This test characterizes the nan_closing_prob_diff meta-skip path; the
    # global META_DISABLED_NO_LIFT guard (which would short-circuit before the
    # dispatch) is covered separately in test_predictor_meta_dispatch.py.
    p.META_DISABLED_NO_LIFT = False

    with (
        patch.object(p, "model") as mock_xgb,
        patch("ufc_prediction.ml.predictor._resolve_fighter") as mock_resolve,
        patch("ufc_prediction.ml.predictor.build_inference_features") as mock_build,
        patch("ufc_prediction.ml.predictor.fetch_matchup_odds") as mock_odds,
        patch("ufc_prediction.ml.predictor._get_latest_elo") as mock_elo,
    ):
        mock_xgb.predict_proba.return_value = np.array([[0.4, 0.6]])
        fa = MagicMock(name="fa", id=1)
        fa.name = "A"
        fb = MagicMock(name="fb", id=2)
        fb.name = "B"
        mock_resolve.side_effect = [fa, fb]
        idx = FEATURE_COLUMNS_NO_NET.index("closing_prob_diff")
        feature_vec = np.zeros((1, 72))
        feature_vec[0, idx] = np.nan  # simulate BFO unreachable + cache miss
        mock_build.return_value = feature_vec
        mock_odds.return_value = None  # bfo_live returns None when both cache + live miss
        mock_elo.return_value = 1500

        result = p.predict(MagicMock(), "A", "B")

    # === Bucket B assertions (current behavior pre-Wave-5) ===
    assert result["meta_skipped"] is True, (
        "Expected meta_skipped=True; Bucket B (graceful skip) is the observed pre-fix path"
    )
    assert result["meta_skipped_reason"] == "nan_closing_prob_diff", (
        "Bucket B reason should be 'nan_closing_prob_diff' per predictor.py:268"
    )
    assert result["meta_prob"] is None, "Bucket B: meta_prob is unset when meta is skipped"
    assert result["win_probability"] == pytest.approx(0.6), (
        "Bucket B: win_probability falls back to raw xgb_v2 base_prob (0.6 from mocked predict_proba)"
    )
    assert result["base_prob"] == pytest.approx(0.6), "Base prob matches mocked predict_proba[0,1]"


# TODO Wave 5: replace this test with tests/integration/test_predictor_fallback_no_odds.py
# per FALLBACK-V24-02. The new behavior is: route to xgb_v2_no_odds (67-col) and emit
# response["_meta"]["win_probability_source"] = "xgb_v2_no_odds" + response["odds_source"] = "nan".
# This file will be archived to tests/regression/_archive/ at that time (per 34-06 Task 3).
