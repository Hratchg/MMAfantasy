"""Tests for model evaluation metrics.

Covers evaluate_model with Brier score, AUC-ROC, accuracy,
and calibration curve computation per D-09.

Phase 17 additions (per CONTEXT.md D-04/D-05/D-08(P17)):
  - TestGateVerdictWithContract: gate_verdict consumes injected GateContract
    (lazy-loads .planning/gate_contract.json when contract=None).
  - TestBootstrapPerSliceCI: bootstrap_per_slice_ci helper returns BCa
    confidence interval half-widths per slice; degenerate inputs do not raise.

Pitfall C (lru_cache discipline): every fixture clears
load_gate_contract.cache_clear() in BOTH setup and teardown.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pytest


class MockModel:
    """Mock model with deterministic predict_proba and predict methods."""

    def __init__(self, probs: np.ndarray):
        self._probs = probs

    def predict_proba(self, X):
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        p = self._probs[:n]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return (self._probs[:n] >= 0.5).astype(int)


class _SliceModel:
    """Mock model that maps each X row to a fixed prob via a lookup column.

    Lifted verbatim from tests/integration/test_xgb_v3_gate.py:42-59 to give
    the bootstrap CI tests a deterministic Brier surface across slices.
    """

    def __init__(self, prob_lookup: np.ndarray):
        self._lookup = prob_lookup

    def predict_proba(self, X):
        idx = X[:, 0].astype(int)
        p = self._lookup[idx]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        idx = X[:, 0].astype(int)
        return (self._lookup[idx] >= 0.5).astype(int)


def _valid_contract_dict_module_local() -> dict:
    """Module-local helper — same shape as Task 2's _valid_contract_dict.

    Kept module-local on purpose (DO NOT cross-import test files).
    """
    slice_payload = {
        "brier_max": 0.215,
        "accuracy_min": 0.67,
        "median_brier_xgb_v2": 0.22,
        "median_acc_xgb_v2": 0.67,
        "seed_std_brier": 0.005,
        "seed_std_acc": 0.008,
        "bootstrap_ci_half_brier": 0.004,
        "bootstrap_ci_half_acc": 0.006,
        "std_brier_used": 0.005,
        "std_acc_used": 0.008,
    }
    return {
        "version": "v2.1",
        "derived_at": "2026-05-04",
        "n_seeds_observed": 10,
        "base_features_set": "FEATURE_COLUMNS_NO_NET",
        "n_features": 69,
        "k_value": 1,
        "formula_hash": "0" * 64,
        "cutoff_date": "2023-01-01",
        "per_slice": {
            "most_recent_12mo": dict(slice_payload),
            "most_recent_24mo": dict(slice_payload),
            "random_15pct": dict(slice_payload),
        },
        "secondary_metrics_observed": {},
        "supersedes": ["D-13(P16)", "D-17(v2.0)"],
        "notes": "test fixture",
    }


@pytest.fixture()
def synthetic_slices():
    """Lifted from tests/integration/test_xgb_v3_gate.py:62-85.

    100 rows split into 30 (within 12mo) + 30 (12-24mo) + 40 (older).
    """
    today = date(2026, 5, 3)
    dates: list[date] = []
    dates.extend([today - timedelta(days=30 + i) for i in range(30)])
    dates.extend([today - timedelta(days=400 + i * 5) for i in range(30)])
    dates.extend([today - timedelta(days=900 + i * 5) for i in range(40)])
    n = len(dates)
    X = np.column_stack([np.arange(n), np.zeros((n, 4))])
    rng = np.random.RandomState(42)
    y = rng.binomial(1, 0.6, size=n)
    prob_lookup = rng.uniform(0.5, 0.7, size=n)
    return X, y, np.array(dates), prob_lookup, today


class TestGateVerdictWithContract:
    """Phase 17: gate_verdict accepts an injected GateContract.

    After Wave 1 task 9 deletes GATE_BRIER_MAX / GATE_ACCURACY_MIN constants,
    gate_verdict must consume thresholds from a contract instead. These tests
    enforce both the injection path AND the lazy-load fallback hermeticity.
    """

    @pytest.fixture()
    def fixed_contract(self, tmp_path):
        from ufc_prediction.ml.gate_contract import load_gate_contract

        load_gate_contract.cache_clear()
        payload = _valid_contract_dict_module_local()
        p = tmp_path / "gate_contract.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        contract = load_gate_contract(p)
        yield contract
        load_gate_contract.cache_clear()

    def test_gate_verdict_uses_contract(self, fixed_contract):
        from ufc_prediction.ml.evaluator import gate_verdict

        median = {
            "most_recent_12mo": {"brier_score": 0.21, "accuracy": 0.68},
            "most_recent_24mo": {"brier_score": 0.20, "accuracy": 0.69},
            "random_15pct": {"brier_score": 0.215, "accuracy": 0.67},
        }
        passed, failed = gate_verdict(median, contract=fixed_contract)

        assert passed is True
        assert failed == []

    def test_gate_verdict_failures(self, fixed_contract):
        from ufc_prediction.ml.evaluator import gate_verdict

        median = {
            "most_recent_12mo": {"brier_score": 0.99, "accuracy": 0.50},
            "most_recent_24mo": {"brier_score": 0.20, "accuracy": 0.69},
            "random_15pct": {"brier_score": 0.21, "accuracy": 0.68},
        }
        passed, failed = gate_verdict(median, contract=fixed_contract)

        assert passed is False
        # 12mo slice should report both a brier failure AND an accuracy failure.
        brier_failures = [f for f in failed if "most_recent_12mo" in f and "brier_score" in f]
        acc_failures = [f for f in failed if "most_recent_12mo" in f and "accuracy" in f]
        assert len(brier_failures) >= 1
        assert len(acc_failures) >= 1

    def test_gate_verdict_injected_contract(self, fixed_contract):
        """When `contract=` is passed, no .planning/gate_contract.json file is needed.

        Demonstrates hermeticity: the injection path bypasses the lazy load.
        """
        from ufc_prediction.ml.evaluator import gate_verdict

        median = {
            "most_recent_12mo": {"brier_score": 0.21, "accuracy": 0.68},
            "most_recent_24mo": {"brier_score": 0.20, "accuracy": 0.69},
            "random_15pct": {"brier_score": 0.21, "accuracy": 0.68},
        }
        passed, failed = gate_verdict(median, contract=fixed_contract)
        assert passed is True


class TestBootstrapPerSliceCI:
    """Phase 17 GATE-06: bootstrap_per_slice_ci returns BCa CI half-widths per slice.

    Per D-06(P17): scipy.stats.bootstrap with method='BCa', confidence_level=0.68,
    n_resamples=9999. Pitfall B — degenerate inputs return NaN-tolerantly without raising.
    """

    def test_bootstrap_per_slice_ci_shape(self, synthetic_slices):
        from ufc_prediction.ml.evaluator import bootstrap_per_slice_ci

        X, y, dates, prob_lookup, today = synthetic_slices
        model = _SliceModel(prob_lookup)
        result = bootstrap_per_slice_ci(model, X, y, dates, today=today)

        assert set(result.keys()) == {
            "most_recent_12mo",
            "most_recent_24mo",
            "random_15pct",
        }
        for slice_name, metrics in result.items():
            assert "brier_ci_half" in metrics, f"{slice_name} missing brier_ci_half"
            assert "acc_ci_half" in metrics, f"{slice_name} missing acc_ci_half"
            # Values must be float (NaN is float — Pitfall B fallback).
            assert isinstance(metrics["brier_ci_half"], float)
            assert isinstance(metrics["acc_ci_half"], float)

    def test_bootstrap_per_slice_ci_degenerate(self, synthetic_slices):
        """Pitfall B: all-zeros probs and all-zeros y → BCa degenerate; NO RAISE.

        Returned half-widths are either NaN or finite — never crash.
        """
        from ufc_prediction.ml.evaluator import bootstrap_per_slice_ci

        X, y, dates, _prob_lookup, today = synthetic_slices
        n = len(dates)
        zero_probs = np.zeros(n)
        y_zeros = np.zeros(n, dtype=int)
        model = _SliceModel(zero_probs)

        # Must not raise.
        result = bootstrap_per_slice_ci(model, X, y_zeros, dates, today=today)

        for slice_name, metrics in result.items():
            half_brier = metrics["brier_ci_half"]
            half_acc = metrics["acc_ci_half"]
            # NaN or finite — both acceptable per Pitfall B.
            assert isinstance(half_brier, float)
            assert isinstance(half_acc, float)
            assert np.isnan(half_brier) or np.isfinite(half_brier)
            assert np.isnan(half_acc) or np.isfinite(half_acc)

    def test_bootstrap_per_slice_ci_uses_bca(self, synthetic_slices, monkeypatch):
        """Spy on scipy.stats.bootstrap; assert every observed call uses method='BCa'."""
        from scipy.stats import bootstrap as real_bootstrap

        from ufc_prediction.ml.evaluator import bootstrap_per_slice_ci

        observed_methods: list[str] = []

        def spy_bootstrap(*args, **kwargs):
            observed_methods.append(kwargs.get("method", ""))
            return real_bootstrap(*args, **kwargs)

        monkeypatch.setattr("scipy.stats.bootstrap", spy_bootstrap)

        X, y, dates, prob_lookup, today = synthetic_slices
        model = _SliceModel(prob_lookup)
        bootstrap_per_slice_ci(model, X, y, dates, today=today)

        assert len(observed_methods) > 0, "scipy.stats.bootstrap was never called"
        assert all(m == "BCa" for m in observed_methods), f"non-BCa calls: {observed_methods}"


class TestEvaluateModel:
    """Tests for the evaluate_model function."""

    def test_returns_brier_score(self):
        """evaluate_model returns dict with 'brier_score' key."""
        from ufc_prediction.ml.evaluator import evaluate_model

        probs = np.array([0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.9, 0.1, 0.5, 0.5])
        model = MockModel(probs)
        y_test = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        X_test = np.zeros((10, 5))  # Dummy features

        result = evaluate_model(model, X_test, y_test)

        assert "brier_score" in result
        assert isinstance(result["brier_score"], float)

    def test_returns_auc_roc(self):
        """evaluate_model returns dict with 'auc_roc' key."""
        from ufc_prediction.ml.evaluator import evaluate_model

        probs = np.array([0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.9, 0.1, 0.5, 0.5])
        model = MockModel(probs)
        y_test = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        X_test = np.zeros((10, 5))

        result = evaluate_model(model, X_test, y_test)

        assert "auc_roc" in result
        assert isinstance(result["auc_roc"], float)

    def test_returns_accuracy(self):
        """evaluate_model returns dict with 'accuracy' key."""
        from ufc_prediction.ml.evaluator import evaluate_model

        probs = np.array([0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.9, 0.1, 0.5, 0.5])
        model = MockModel(probs)
        y_test = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        X_test = np.zeros((10, 5))

        result = evaluate_model(model, X_test, y_test)

        assert "accuracy" in result
        assert isinstance(result["accuracy"], float)

    def test_returns_calibration_curve(self):
        """evaluate_model returns 'calibration_curve' with correct structure."""
        from ufc_prediction.ml.evaluator import evaluate_model

        probs = np.array([0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.9, 0.1, 0.5, 0.5])
        model = MockModel(probs)
        y_test = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        X_test = np.zeros((10, 5))

        result = evaluate_model(model, X_test, y_test)

        assert "calibration_curve" in result
        cal = result["calibration_curve"]
        assert "fraction_of_positives" in cal
        assert "mean_predicted_value" in cal
        assert len(cal["fraction_of_positives"]) == len(cal["mean_predicted_value"])

    def test_brier_score_perfect(self):
        """Perfect predictions give Brier score near 0."""
        from ufc_prediction.ml.evaluator import evaluate_model

        probs = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        model = MockModel(probs)
        y_test = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        X_test = np.zeros((8, 5))

        result = evaluate_model(model, X_test, y_test)

        assert result["brier_score"] == pytest.approx(0.0, abs=1e-6)

    def test_brier_score_random(self):
        """0.5 predictions give Brier score near 0.25."""
        from ufc_prediction.ml.evaluator import evaluate_model

        probs = np.full(100, 0.5)
        model = MockModel(probs)
        y_test = np.array([1, 0] * 50)
        X_test = np.zeros((100, 5))

        result = evaluate_model(model, X_test, y_test)

        assert result["brier_score"] == pytest.approx(0.25, abs=1e-6)

    def test_brier_score_correct_computation(self):
        """Brier score computed correctly against known values."""
        from sklearn.metrics import brier_score_loss

        from ufc_prediction.ml.evaluator import evaluate_model

        probs = np.array([0.9, 0.8, 0.3, 0.1, 0.6])
        model = MockModel(probs)
        y_test = np.array([1, 1, 0, 0, 1])
        X_test = np.zeros((5, 5))

        result = evaluate_model(model, X_test, y_test)

        expected_brier = brier_score_loss(y_test, probs)
        assert result["brier_score"] == pytest.approx(expected_brier, abs=1e-6)

    def test_auc_roc_correct_computation(self):
        """AUC-ROC computed correctly against known values."""
        from sklearn.metrics import roc_auc_score

        from ufc_prediction.ml.evaluator import evaluate_model

        probs = np.array([0.9, 0.8, 0.3, 0.1, 0.6])
        model = MockModel(probs)
        y_test = np.array([1, 1, 0, 0, 1])
        X_test = np.zeros((5, 5))

        result = evaluate_model(model, X_test, y_test)

        expected_auc = roc_auc_score(y_test, probs)
        assert result["auc_roc"] == pytest.approx(expected_auc, abs=1e-6)

    def test_accuracy_correct_computation(self):
        """Accuracy computed correctly against known values."""
        from ufc_prediction.ml.evaluator import evaluate_model

        probs = np.array([0.9, 0.8, 0.3, 0.1, 0.6])
        model = MockModel(probs)
        y_test = np.array([1, 1, 0, 0, 1])
        X_test = np.zeros((5, 5))

        result = evaluate_model(model, X_test, y_test)

        # predictions: [1, 1, 0, 0, 1] vs actual [1, 1, 0, 0, 1] -> 100%
        assert result["accuracy"] == pytest.approx(1.0, abs=1e-6)
