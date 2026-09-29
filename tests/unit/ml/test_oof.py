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

oof = pytest.importorskip("ufc_prediction.ml.oof")


def _make_synthetic_data(n: int = 200, n_features: int = 72, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 2, size=n)
    fight_dates = np.array([np.datetime64("2023-01-01") + np.timedelta64(i, "D") for i in range(n)])
    return X, y, fight_dates


def _make_base_trainer():
    """Tiny stub trainer exposing _make_estimator() returning XGBClassifier."""
    trainer = MagicMock()
    trainer._make_estimator = MagicMock(
        return_value=XGBClassifier(
            n_estimators=10,
            max_depth=2,
            objective="binary:logistic",
            random_state=42,
            verbosity=0,
        )
    )
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
    sidecar.write_text(
        json.dumps(
            {
                "xgb_v2_sha256": "deadbeef" * 8,  # stale
                "n_features": 72,
                "cutoff_date": "2023-01-01",
                "event_date_min": "2023-01-01",
                "event_date_max": "2024-01-01",
                "n_splits": 5,
                "training_accuracy": 0.65,
                "trained_at": "2026-05-09",
                "cv_kind": "TimeSeriesSplit",
            }
        )
    )
    X, y, dates = _make_synthetic_data(n=120)
    trainer = _make_base_trainer()
    with pytest.raises(oof.InvariantCheckError, match="xgb_v2_sha256"):
        oof.generate_oof_predictions(
            X,
            y,
            dates,
            trainer,
            n_splits=5,
            cache_path=cache_path,
            force_rebuild=False,
        )


def test_oof_cache_invariant_check_n_features(tmp_path):
    """D-06(P19): cache with mismatched n_features → InvariantCheckError."""
    cache_path = tmp_path / "oof_predictions.parquet"
    sidecar = tmp_path / "oof_predictions.meta.json"
    import pandas as pd

    pd.DataFrame({"fight_id": [1], "xgb_oof_prob": [0.5]}).to_parquet(cache_path)
    # Write the LIVE xgb_v2 SHA so the SHA check passes; n_features mismatch fires
    actual_sha = hashlib.sha256(Path("models/xgb_v2.joblib").read_bytes()).hexdigest()
    sidecar.write_text(
        json.dumps(
            {
                "xgb_v2_sha256": actual_sha,
                "n_features": 75,  # mismatch — code expects 72
                "cutoff_date": "2023-01-01",
                "event_date_min": "2023-01-01",
                "event_date_max": "2024-01-01",
                "n_splits": 5,
                "training_accuracy": 0.65,
                "trained_at": "2026-05-09",
                "cv_kind": "TimeSeriesSplit",
            }
        )
    )
    X, y, dates = _make_synthetic_data(n=120)
    trainer = _make_base_trainer()
    with pytest.raises(oof.InvariantCheckError, match="n_features"):
        oof.generate_oof_predictions(
            X,
            y,
            dates,
            trainer,
            n_splits=5,
            cache_path=cache_path,
            force_rebuild=False,
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


# ── Cache row-identity invariants (code-review finding: OOF cache returned a
# stale array for a different row set; callers re-align it positionally) ──


def _write_cache(tmp_path, *, n=120, n_splits=5, fight_ids=None, dates=None, y=None, X=None):
    """Populate an OOF cache via the real generator and return its inputs."""
    X0, y0, dates0 = _make_synthetic_data(n=n)
    X = X0 if X is None else X
    y = y0 if y is None else y
    dates = dates0 if dates is None else dates
    fight_ids = list(range(1000, 1000 + n)) if fight_ids is None else fight_ids
    cache_path = tmp_path / "oof_predictions.parquet"
    proba, _ = oof.generate_oof_predictions(
        X,
        y,
        dates,
        _make_base_trainer(),
        n_splits=n_splits,
        cache_path=cache_path,
        fight_ids=fight_ids,
    )
    return cache_path, X, y, dates, fight_ids, proba


def test_oof_cache_hit_same_inputs_returns_cached(tmp_path):
    """Sanity: an identical call is a cache hit returning the cached array."""
    cache_path, X, y, dates, ids, proba = _write_cache(tmp_path)
    trainer = _make_base_trainer()
    with patch("ufc_prediction.ml.oof.cross_val_predict") as mock_cvp:
        got, _ = oof.generate_oof_predictions(
            X, y, dates, trainer, n_splits=5, cache_path=cache_path, fight_ids=ids
        )
        mock_cvp.assert_not_called()
    np.testing.assert_array_equal(got, proba)


def test_oof_cache_rejects_row_subset_within_cached_date_range(tmp_path):
    """A 730d-window caller hitting a 365d-window cache (or vice versa) must not
    receive an array for a different row set. The date-range containment check
    alone passed here and an N_cached array came back for N_input rows."""
    cache_path, X, y, dates, ids, _ = _write_cache(tmp_path)
    sub = slice(10, 100)  # dates strictly inside the cached range
    with pytest.raises(oof.InvariantCheckError, match="row"):
        oof.generate_oof_predictions(
            X[sub],
            y[sub],
            dates[sub],
            _make_base_trainer(),
            n_splits=5,
            cache_path=cache_path,
            fight_ids=ids[sub],
        )


def test_oof_cache_rejects_same_count_different_membership(tmp_path):
    """Same row count but a different fight set (re-link / dedup) → error,
    not silently mis-attached OOF probabilities."""
    cache_path, X, y, dates, ids, _ = _write_cache(tmp_path)
    ids2 = list(ids)
    ids2[50] = 999_999
    with pytest.raises(oof.InvariantCheckError, match="fight_id"):
        oof.generate_oof_predictions(
            X, y, dates, _make_base_trainer(), n_splits=5, cache_path=cache_path, fight_ids=ids2
        )


def test_oof_cache_rekeys_predictions_by_fight_id(tmp_path):
    """Same fights, different input row order (same-date ties) → the cached
    probabilities are re-keyed by fight_id into the caller's sort order."""
    n = 120
    X, y, _ = _make_synthetic_data(n=n)
    # Four fights per event date → many ties for argsort to order arbitrarily.
    dates = np.array([np.datetime64("2023-01-01") + np.timedelta64(i // 4, "D") for i in range(n)])
    cache_path, X, y, dates, ids, proba = _write_cache(tmp_path, X=X, y=y, dates=dates)
    cached_by_id = {
        fid: p for fid, p in zip([ids[int(i)] for i in np.argsort(dates)], proba, strict=True)
    }

    perm = np.random.default_rng(3).permutation(n)
    X_p, y_p, dates_p = X[perm], y[perm], dates[perm]
    ids_p = [ids[int(i)] for i in perm]
    got, _ = oof.generate_oof_predictions(
        X_p, y_p, dates_p, _make_base_trainer(), n_splits=5, cache_path=cache_path, fight_ids=ids_p
    )
    expected = np.array([cached_by_id[ids_p[int(i)]] for i in np.argsort(dates_p)])
    np.testing.assert_array_equal(got, expected)


def test_oof_cache_rejects_n_splits_mismatch(tmp_path):
    """Cache hits must honor the requested n_splits."""
    cache_path, X, y, dates, ids, _ = _write_cache(tmp_path, n_splits=5)
    with pytest.raises(oof.InvariantCheckError, match="n_splits"):
        oof.generate_oof_predictions(
            X, y, dates, _make_base_trainer(), n_splits=3, cache_path=cache_path, fight_ids=ids
        )


def test_oof_cache_rejects_label_change_for_same_fights(tmp_path):
    """A winner_id change inside the range (same fights, different y) → stale cache."""
    cache_path, X, y, dates, ids, _ = _write_cache(tmp_path)
    y2 = y.copy()
    y2[40] = 1 - y2[40]
    with pytest.raises(oof.InvariantCheckError, match="input"):
        oof.generate_oof_predictions(
            X, y2, dates, _make_base_trainer(), n_splits=5, cache_path=cache_path, fight_ids=ids
        )
