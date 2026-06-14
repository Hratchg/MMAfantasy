"""Plan 34-06 Task 2 — Unit tests for FALLBACK-V24-02 fallback routing.

Per revision m3 Option B + revision n1: assertions exercise the nested
`_meta.win_probability_source` path via real predict() invocation. Snapshot the
pre-Wave-5 response key set as a constant; the new `_meta` key is ADDITIVE.
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

predictor_module = pytest.importorskip("ufc_prediction.ml.predictor")


# Snapshot of pre-Wave-5 response keys (Phase 16-19 lineage). The new `_meta` key
# (containing `win_probability_source`) is ADDITIVE per revision m3 Option B.
# Phase 35 CONTRACT-V24-02 will rename `_meta` → `prediction_metadata`.
PRE_WAVE_5_KEYS = frozenset({
    "fighter_a",
    "fighter_b",
    "win_probability",
    "base_prob",
    "meta_prob",
    "meta_kind",
    "meta_learner_version",
    "meta_skipped",
    "meta_skipped_reason",
    "model_probability_a",
    "model_probability_b",
    "elo_probability_a",
    "elo_probability_b",
    "elo_a",
    "elo_b",
    "feature_importances",
    "odds_source",
})


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


def _make_fallback_artifacts(tmp_path: Path, no_odds_cols: list[str]) -> None:
    """Persist a tiny 67-col fallback model + meta JSON in `tmp_path/models/`."""
    fallback_model = _build_calibrated_model(67)
    models_dir = tmp_path / "models"
    fallback_joblib = models_dir / "xgb_v2_no_odds.joblib"
    fallback_meta = models_dir / "xgb_v2_no_odds_meta.json"
    joblib.dump(fallback_model, fallback_joblib)
    fallback_meta.write_text(
        json.dumps(
            {
                "model_name": "xgb_v2_no_odds",
                "n_features": 67,
                "feature_columns": no_odds_cols,
                "parent_model": "xgb_v2",
                "parent_model_sha256": "test-sha",
                "dropped_columns": [
                    "opening_prob_diff", "closing_prob_diff", "line_movement_diff",
                    "sharp_money_signal", "odds_elo_divergence",
                ],
                "train_seed": 42,
                "cutoff_date": "2023-01-01",
                "best_params": {},
            }
        )
    )


def _build_predictor_with_fallback(tmp_path: Path):
    """Build a tmp_path-rooted Predictor with xgb_vtest + xgb_v2_no_odds + no meta."""
    base_model = _build_calibrated_model(72)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    save_model(
        model=base_model,
        metrics={"brier_score": 0.22, "auc_roc": 0.7, "accuracy": 0.65},
        feature_columns=list(FEATURE_COLUMNS_NO_NET),
        best_params={},
        model_dir=str(models_dir),
        version="vtest",
        cutoff_date="2023-01-01",
        n_training_fights=48,
        n_test_fights=12,
    )
    odds_cols = {
        "opening_prob_diff", "closing_prob_diff", "line_movement_diff",
        "sharp_money_signal", "odds_elo_divergence",
    }
    no_odds_cols = [c for c in FEATURE_COLUMNS_NO_NET if c not in odds_cols]
    assert len(no_odds_cols) == 67
    _make_fallback_artifacts(tmp_path, no_odds_cols)
    return predictor_module.ModelPredictor(
        model_dir=str(models_dir), version="vtest", meta_dir=None
    )


def test_predictor_loads_fallback_model_on_init(tmp_path: Path) -> None:
    """When xgb_v2_no_odds.joblib + meta JSON are present in model_dir, the
    Predictor.__init__ lazy-loads them and exposes them via attributes."""
    p = _build_predictor_with_fallback(tmp_path)
    assert p.fallback_model is not None
    assert p.fallback_meta is not None
    assert p.fallback_keep_idx is not None
    assert len(p.fallback_keep_idx) == 67


def test_routes_to_fallback_when_live_odds_none(tmp_path: Path) -> None:
    """Mock bfo_live=None + force closing_prob_diff NaN → response routes to fallback."""
    p = _build_predictor_with_fallback(tmp_path)
    with (
        patch("ufc_prediction.ml.predictor._resolve_fighter") as mock_resolve,
        patch("ufc_prediction.ml.predictor.build_inference_features") as mock_build,
        patch("ufc_prediction.ml.predictor.fetch_matchup_odds") as mock_odds,
        patch("ufc_prediction.ml.predictor._get_latest_elo") as mock_elo,
    ):
        fa = MagicMock(name="fa", id=1); fa.name = "A"
        fb = MagicMock(name="fb", id=2); fb.name = "B"
        mock_resolve.side_effect = [fa, fb]
        idx = FEATURE_COLUMNS_NO_NET.index("closing_prob_diff")
        feature_vec = np.zeros((1, 72))
        feature_vec[0, idx] = np.nan
        mock_build.return_value = feature_vec
        mock_odds.return_value = None
        mock_elo.return_value = 1500
        result = p.predict(MagicMock(), "A", "B")

    assert "prediction_metadata" in result, (
        "response missing top-level `prediction_metadata` dict (Phase 35 CONTRACT-V24-02 rename of `_meta`)"
    )
    assert result["prediction_metadata"]["win_probability_source"] == "xgb_v2_no_odds", (
        f"expected fallback route; got {result['prediction_metadata']['win_probability_source']!r}"
    )
    assert result["odds_source"] == "nan"
    # The Wave-0 baseline test asserted win_probability == base_prob (raw xgb_v2);
    # Wave 5 changes this to come from the fallback model.
    assert isinstance(result["win_probability"], float)
    assert 0.0 <= result["win_probability"] <= 1.0


def test_no_route_when_live_odds_present(tmp_path: Path) -> None:
    """When live_odds is a real OddsRecord, fallback is NOT used."""
    p = _build_predictor_with_fallback(tmp_path)
    # Build a fake OddsRecord-like object with .source
    fake_odds = MagicMock(source="live", fetched_at=MagicMock())
    fake_odds.fetched_at.isoformat.return_value = "2024-01-01T00:00:00+00:00"
    with (
        patch("ufc_prediction.ml.predictor._resolve_fighter") as mock_resolve,
        patch("ufc_prediction.ml.predictor.build_inference_features") as mock_build,
        patch("ufc_prediction.ml.predictor.fetch_matchup_odds") as mock_odds,
        patch("ufc_prediction.ml.predictor._get_latest_elo") as mock_elo,
    ):
        fa = MagicMock(name="fa", id=1); fa.name = "A"
        fb = MagicMock(name="fb", id=2); fb.name = "B"
        mock_resolve.side_effect = [fa, fb]
        idx = FEATURE_COLUMNS_NO_NET.index("closing_prob_diff")
        feature_vec = np.zeros((1, 72))
        feature_vec[0, idx] = 0.05  # non-NaN
        mock_build.return_value = feature_vec
        mock_odds.return_value = fake_odds
        mock_elo.return_value = 1500
        result = p.predict(MagicMock(), "A", "B")
    assert result["prediction_metadata"]["win_probability_source"] in {"meta_v2", "xgb_v2"}
    assert result["odds_source"] == "live"


def test_fallback_probability_matches_direct_predict(tmp_path: Path) -> None:
    """Capture predict's win_probability; separately call fallback.predict_proba on the
    67-col vector; assert match within 1e-9."""
    p = _build_predictor_with_fallback(tmp_path)
    feature_vec_full = np.random.RandomState(7).standard_normal((1, 72))
    idx = FEATURE_COLUMNS_NO_NET.index("closing_prob_diff")
    feature_vec_full[0, idx] = np.nan  # trigger fallback
    with (
        patch("ufc_prediction.ml.predictor._resolve_fighter") as mock_resolve,
        patch("ufc_prediction.ml.predictor.build_inference_features") as mock_build,
        patch("ufc_prediction.ml.predictor.fetch_matchup_odds") as mock_odds,
        patch("ufc_prediction.ml.predictor._get_latest_elo") as mock_elo,
    ):
        fa = MagicMock(name="fa", id=1); fa.name = "A"
        fb = MagicMock(name="fb", id=2); fb.name = "B"
        mock_resolve.side_effect = [fa, fb]
        mock_build.return_value = feature_vec_full
        mock_odds.return_value = None
        mock_elo.return_value = 1500
        result = p.predict(MagicMock(), "A", "B")

    # Independent call to the same fallback model on the 67-col view
    X_no_odds = feature_vec_full[:, p.fallback_keep_idx]
    expected_prob = float(p.fallback_model.predict_proba(X_no_odds)[0, 1])
    assert result["win_probability"] == pytest.approx(expected_prob, abs=1e-9)
    assert result["prediction_metadata"]["win_probability_source"] == "xgb_v2_no_odds"


def test_response_shape_backward_compatible(tmp_path: Path) -> None:
    """All pre-existing response keys present + `prediction_metadata` key ADDED.

    Snapshot taken from main pre-Phase-34; Phase 34 introduced a `_meta`
    transitional nest carrying `win_probability_source`. Phase 35
    CONTRACT-V24-02 renamed `_meta` → `prediction_metadata` as the single
    source of truth for the v1.2.0 API contract. The rename is a key-rename
    only; downstream pre-Wave-5 keys remain in place.
    """
    p = _build_predictor_with_fallback(tmp_path)
    with (
        patch("ufc_prediction.ml.predictor._resolve_fighter") as mock_resolve,
        patch("ufc_prediction.ml.predictor.build_inference_features") as mock_build,
        patch("ufc_prediction.ml.predictor.fetch_matchup_odds") as mock_odds,
        patch("ufc_prediction.ml.predictor._get_latest_elo") as mock_elo,
    ):
        fa = MagicMock(name="fa", id=1); fa.name = "A"
        fb = MagicMock(name="fb", id=2); fb.name = "B"
        mock_resolve.side_effect = [fa, fb]
        idx = FEATURE_COLUMNS_NO_NET.index("closing_prob_diff")
        feature_vec = np.zeros((1, 72))
        feature_vec[0, idx] = np.nan
        mock_build.return_value = feature_vec
        mock_odds.return_value = None
        mock_elo.return_value = 1500
        result = p.predict(MagicMock(), "A", "B")

    # All pre-Wave-5 keys still present
    for k in PRE_WAVE_5_KEYS:
        assert k in result, f"pre-Wave-5 key {k} dropped from response"
    # Phase 35 CONTRACT-V24-02 — `_meta` was renamed to `prediction_metadata`
    # (single key-rename, not duplicated). `_meta` MUST NOT appear in the
    # return dict (the v1.2.0 contract is the only metadata surface).
    assert "_meta" not in result
    assert "prediction_metadata" in result
    assert isinstance(result["prediction_metadata"], dict)
    assert "win_probability_source" in result["prediction_metadata"]
