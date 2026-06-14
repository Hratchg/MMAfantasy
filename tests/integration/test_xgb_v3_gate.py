"""MODEL-01: 3-slice xgb_v3 promotion gate harness — Phase 17 refactored.

History:
  - v2.0 (Phase 16): hardcoded brier-max / accuracy-min module constants drove
    gate_verdict; this file pinned them via test_constants_pinned (now deleted).
  - v2.1 (Phase 17): constants deleted from evaluator.py; gate_verdict now
    consumes a GateContract. The 3 retained gate_verdict tests below use
    `fixed_v20_compatible_contract` — a fixture-injected contract whose
    per-slice thresholds match the v2.0 values (brier_max=0.215, accuracy_min=0.67)
    so the existing fixture metric values keep their pass/fail semantics.

Exercises:
  - evaluate_per_slice over synthetic fixtures with mock model
  - delegation to evaluate_model (back-compat per Gotcha 8)
  - 12mo / 24mo slice masking by event date
  - random-15% deterministic slice selection (seeded RNG)
  - back-compat: existing evaluate_model behavior preserved
  - gate_verdict logic: PASS only if all 3 slices clear thresholds
    (now contract-driven; fixture pins v2.0-compatible thresholds)

Pitfall A + Pitfall C: fixture clears load_gate_contract.cache_clear() in
BOTH setup and teardown.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pytest

from ufc_prediction.ml.gate_contract import load_gate_contract


class MockModel:
    """Mock model with deterministic predict_proba and predict methods.

    Mirrors the MockModel pattern from tests/ml/test_evaluator.py.
    """

    def __init__(self, probs: np.ndarray):
        self._probs = probs

    def predict_proba(self, X):
        # Find indices of X rows in our probs lookup. For the simple test case
        # we assume X length matches probs length (caller passes through masks).
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        # Use the first n entries — masks always preserve order.
        p = self._probs[:n]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return (self._probs[:n] >= 0.5).astype(int)


class _SliceModel:
    """Mock model that maps each X row to a fixed prob via a lookup column.

    Useful for slicing tests where masking changes which probs are visible.
    Stores probs and uses X[:, 0] (an index column) to pick the right prob.
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


@pytest.fixture()
def synthetic_slices():
    """Build a 100-row synthetic dataset with mixed event dates.

    Layout:
      - 30 rows within last 365 days  (12mo + 24mo slices include them)
      - 30 rows between 365 and 730 days  (24mo slice only)
      - 40 rows older than 730 days   (only random-15% may sample them)

    Each row has a single index feature in column 0; the test model uses
    that index to look up its prob in `prob_lookup`.
    """
    today = date(2026, 5, 3)
    dates: list[date] = []
    dates.extend([today - timedelta(days=30 + i) for i in range(30)])      # 12mo
    dates.extend([today - timedelta(days=400 + i * 5) for i in range(30)])  # 24mo only
    dates.extend([today - timedelta(days=900 + i * 5) for i in range(40)])  # older
    n = len(dates)
    X = np.column_stack([np.arange(n), np.zeros((n, 4))])
    # Probs hover near 0.6 for class 1; truth is 1 ~ 60% of the time.
    rng = np.random.RandomState(42)
    y = rng.binomial(1, 0.6, size=n)
    prob_lookup = rng.uniform(0.5, 0.7, size=n)
    return X, y, np.array(dates), prob_lookup, today


class TestEvaluatePerSlice:
    """Tests for evaluate_per_slice (3-slice gate harness, D-13)."""

    def test_returns_three_slice_keys(self, synthetic_slices):
        from ufc_prediction.ml.evaluator import evaluate_per_slice

        X, y, dates, prob_lookup, today = synthetic_slices
        model = _SliceModel(prob_lookup)
        result = evaluate_per_slice(model, X, y, dates, today=today)

        assert set(result.keys()) == {
            "most_recent_12mo",
            "most_recent_24mo",
            "random_15pct",
        }

    def test_each_slice_has_evaluate_model_shape(self, synthetic_slices):
        from ufc_prediction.ml.evaluator import evaluate_per_slice

        X, y, dates, prob_lookup, today = synthetic_slices
        model = _SliceModel(prob_lookup)
        result = evaluate_per_slice(model, X, y, dates, today=today)

        for slice_name, metrics in result.items():
            assert "brier_score" in metrics, f"{slice_name} missing brier_score"
            assert "auc_roc" in metrics, f"{slice_name} missing auc_roc"
            assert "accuracy" in metrics, f"{slice_name} missing accuracy"
            assert "calibration_curve" in metrics, f"{slice_name} missing calib"
            assert isinstance(metrics["brier_score"], float)
            assert isinstance(metrics["accuracy"], float)

    def test_12mo_slice_masks_correctly(self, synthetic_slices):
        """Only fights with event_date >= today - 365d enter the 12mo slice."""
        from ufc_prediction.ml.evaluator import evaluate_per_slice

        X, y, dates, prob_lookup, today = synthetic_slices
        # The 12mo slice should include the first 30 rows (within 365 days)
        # and exclude the rest. We verify by counting via known indices.
        cutoff = today - timedelta(days=365)
        expected_n = sum(1 for d in dates if d >= cutoff)
        assert expected_n == 30  # sanity-check the fixture

        model = _SliceModel(prob_lookup)
        # Inject a tracker into the model that records the X length seen.
        seen_lengths = []
        original = model.predict_proba

        def tracking_predict_proba(Xi):
            seen_lengths.append(Xi.shape[0])
            return original(Xi)

        model.predict_proba = tracking_predict_proba  # type: ignore
        evaluate_per_slice(model, X, y, dates, today=today)
        # 3 slices invoke predict_proba 3 times (one per slice for probs +
        # accuracy invokes predict, not predict_proba). Brier+AUC use probs.
        # First call is 12mo. (Order in our impl: 12mo, 24mo, random_15pct.)
        assert seen_lengths[0] == expected_n

    def test_24mo_slice_masks_correctly(self, synthetic_slices):
        """Only fights with event_date >= today - 730d enter the 24mo slice."""
        from ufc_prediction.ml.evaluator import evaluate_per_slice

        X, y, dates, prob_lookup, today = synthetic_slices
        cutoff = today - timedelta(days=730)
        expected_n = sum(1 for d in dates if d >= cutoff)
        assert expected_n == 60  # 30 (12mo) + 30 (24mo-only)

        model = _SliceModel(prob_lookup)
        seen_lengths = []
        original = model.predict_proba

        def tracking_predict_proba(Xi):
            seen_lengths.append(Xi.shape[0])
            return original(Xi)

        model.predict_proba = tracking_predict_proba  # type: ignore
        evaluate_per_slice(model, X, y, dates, today=today)
        # Second predict_proba call is 24mo slice.
        assert seen_lengths[1] == expected_n

    def test_random_15pct_deterministic(self, synthetic_slices):
        """Two calls with same random_seed produce identical sample selection."""
        from ufc_prediction.ml.evaluator import evaluate_per_slice

        X, y, dates, prob_lookup, today = synthetic_slices
        model_a = _SliceModel(prob_lookup)
        model_b = _SliceModel(prob_lookup)
        result_a = evaluate_per_slice(model_a, X, y, dates, today=today, random_seed=42)
        result_b = evaluate_per_slice(model_b, X, y, dates, today=today, random_seed=42)

        # Brier scores on random_15pct slice must match exactly.
        assert (
            result_a["random_15pct"]["brier_score"]
            == result_b["random_15pct"]["brier_score"]
        )

    def test_evaluate_model_unchanged(self):
        """Back-compat: evaluate_model signature and behavior are preserved (Gotcha 8)."""
        from ufc_prediction.ml.evaluator import evaluate_model

        probs = np.array([0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.9, 0.1, 0.5, 0.5])
        model = MockModel(probs)
        y_test = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        X_test = np.zeros((10, 5))

        result = evaluate_model(model, X_test, y_test)
        # Same shape it had before MODEL-01 changes.
        assert "brier_score" in result
        assert "auc_roc" in result
        assert "accuracy" in result
        assert "calibration_curve" in result


@pytest.fixture()
def fixed_v20_compatible_contract(tmp_path):
    """A GateContract with v2.0-compatible per-slice thresholds.

    Each slice has brier_max=0.215, accuracy_min=0.67 — matching v2.0 values
    so the existing fixture metric values (0.21, 0.215, 0.22, 0.66, 0.67, 0.68)
    keep their pass/fail semantics post-rewire (Pitfall A).

    Pitfall C: clears load_gate_contract.cache_clear() in BOTH setup and teardown.
    """
    load_gate_contract.cache_clear()
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
    payload = {
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
        "notes": "v2.0-compatible test fixture; Pitfall A regression suite",
    }
    p = tmp_path / "gate_contract.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    contract = load_gate_contract(p)
    yield contract
    load_gate_contract.cache_clear()


class TestGateVerdict:
    """Tests for gate_verdict (D-13 carry-forward; v2.1 contract-driven).

    Fixture metric values unchanged — they remain a regression suite for
    gate_verdict's pass/fail logic. The injected contract pins v2.0-compatible
    thresholds (brier_max=0.215, accuracy_min=0.67) so the existing pass/fail
    semantics are preserved post-rewire (Pitfall A).
    """

    def test_all_pass(self, fixed_v20_compatible_contract):
        from ufc_prediction.ml.evaluator import gate_verdict

        median = {
            "most_recent_12mo": {"brier_score": 0.21, "accuracy": 0.68},
            "most_recent_24mo": {"brier_score": 0.20, "accuracy": 0.69},
            "random_15pct":     {"brier_score": 0.215, "accuracy": 0.67},
        }
        passed, failed = gate_verdict(median, contract=fixed_v20_compatible_contract)
        assert passed is True
        assert failed == []

    def test_one_slice_fails_brier(self, fixed_v20_compatible_contract):
        from ufc_prediction.ml.evaluator import gate_verdict

        median = {
            "most_recent_12mo": {"brier_score": 0.22, "accuracy": 0.68},
            "most_recent_24mo": {"brier_score": 0.20, "accuracy": 0.69},
            "random_15pct":     {"brier_score": 0.21, "accuracy": 0.68},
        }
        passed, failed = gate_verdict(median, contract=fixed_v20_compatible_contract)
        assert passed is False
        assert len(failed) == 1
        assert "most_recent_12mo" in failed[0]
        assert "brier_score" in failed[0]

    def test_one_slice_fails_accuracy(self, fixed_v20_compatible_contract):
        from ufc_prediction.ml.evaluator import gate_verdict

        median = {
            "most_recent_12mo": {"brier_score": 0.21, "accuracy": 0.68},
            "most_recent_24mo": {"brier_score": 0.20, "accuracy": 0.66},  # below 0.67
            "random_15pct":     {"brier_score": 0.21, "accuracy": 0.68},
        }
        passed, failed = gate_verdict(median, contract=fixed_v20_compatible_contract)
        assert passed is False
        assert len(failed) == 1
        assert "most_recent_24mo" in failed[0]
        assert "accuracy" in failed[0]
