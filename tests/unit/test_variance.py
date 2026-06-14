"""Phase 30 / Plan 30-01 / Task 2 — variance.py unit tests.

Six tests mapped to VALIDATION rows 30-01-01..30-01-06.

TDD discipline (per Plan 30-01 `tdd="true"`): this file is committed in
its filled form BEFORE src/ufc_prediction/ml/variance.py exists. The RED
commit fails on `ModuleNotFoundError` at import; the GREEN commit lands
variance.py and turns these green.

Each test references one VALIDATION row id in its docstring; the row id
is the test-name → contract anchor used by the executor's per-task
verification map.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# Will fail import until Task 2 GREEN commit lands variance.py.
from ufc_prediction.ml.variance import (
    aggregate_variance,
    assert_distinct_seed_brier,
    bootstrap_resample,
    multi_seed_metrics,
)
from ufc_prediction.ml.evaluator import PER_SLICE_KEYS


def _synthetic_xy(
    n: int = 200, d: int = 13, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Local synthetic fixture — deterministic; no DB; sub-millisecond.

    Two probability cols + 11 normal cols + noisy linear-of-probs label.
    Matches the shape of META_V22_FEATURE_COLUMNS (13) but does NOT
    depend on the cached .npz — these unit tests are self-contained per
    30-RESEARCH.md "Recommended test fixtures: synthetic NumPy arrays
    generated inside each test (n=200, d=13)".
    """
    rng = np.random.RandomState(seed)
    X = rng.standard_normal((n, d)).astype(np.float64)
    X[:, 0] = 1.0 / (1.0 + np.exp(-X[:, 0]))  # squash to [0, 1]
    X[:, 1] = 1.0 / (1.0 + np.exp(-X[:, 1]))
    logits = 2.0 * (X[:, 0] - 0.5) + 1.5 * (X[:, 1] - 0.5)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < probs).astype(np.int64)
    return X, y


def _synthetic_eval_set(
    n: int = 200, d: int = 13, seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eval triple (X_eval, y_eval, fight_dates_eval) for multi_seed_metrics."""
    from datetime import date, timedelta
    X, y = _synthetic_xy(n=n, d=d, seed=seed)
    rng = np.random.RandomState(seed + 11)
    today = date(2026, 5, 21)
    days_back = rng.randint(0, 730 + 365, size=n)
    fight_dates = np.array([today - timedelta(days=int(k)) for k in days_back])
    return X, y, fight_dates


def test_bootstrap_resample_shape() -> None:
    """VALIDATION 30-01-01 — preserves shape; replacement observable."""
    X, y = _synthetic_xy(n=200, d=13, seed=0)
    Xp, yp = bootstrap_resample(X, y, seed=42)
    assert Xp.shape == X.shape, (Xp.shape, X.shape)
    assert yp.shape == y.shape, (yp.shape, y.shape)
    # Pitfall: at n=200 with replace=True, expected unique rate ≈ 63.2%
    # → at most ~127 unique rows on average. Observe via index trick:
    # bootstrap_resample uses RandomState(seed).choice(n, n, replace=True);
    # we can re-derive the index draw deterministically and count uniques.
    rng = np.random.RandomState(42)
    idx = rng.choice(200, size=200, replace=True)
    n_unique = len(set(idx.tolist()))
    # Below 200 confirms replacement; well above 0 confirms broad coverage.
    assert 0 < n_unique < 200, n_unique
    # Sanity: arrays returned are actually X[idx] / y[idx].
    assert np.array_equal(Xp, X[idx])
    assert np.array_equal(yp, y[idx])


def test_bootstrap_resample_seed_stable() -> None:
    """VALIDATION 30-01-02 — same seed → bit-identical resample (deterministic)."""
    X, y = _synthetic_xy(n=200, d=13, seed=0)
    Xa, ya = bootstrap_resample(X, y, seed=42)
    Xb, yb = bootstrap_resample(X, y, seed=42)
    assert np.array_equal(Xa, Xb), "X resample drifted across calls at fixed seed"
    assert np.array_equal(ya, yb), "y resample drifted across calls at fixed seed"
    # Different seed → different rows.
    Xc, yc = bootstrap_resample(X, y, seed=43)
    assert not np.array_equal(Xa, Xc), "seeds 42 and 43 produced identical X"
    # Shape mismatch must raise.
    with pytest.raises(ValueError):
        bootstrap_resample(X, y[:50], seed=42)


def test_multi_seed_distinct() -> None:
    """VALIDATION 30-01-03 — 5 seeds → 5 distinct Brier scores per slice."""
    from sklearn.linear_model import LogisticRegression
    X_train, y_train = _synthetic_xy(n=200, d=13, seed=0)
    X_eval, y_eval, fight_dates = _synthetic_eval_set(n=200, d=13, seed=1)

    def fit_fn(Xb: np.ndarray, yb: np.ndarray, seed: int) -> LogisticRegression:
        # Pitfall 4: fresh estimator per seed; LR random_state is a no-op
        # for the closed-form lbfgs solver — variance comes from the
        # bootstrap-resampled (Xb, yb) pair, NOT from random_state.
        return LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs", max_iter=1000,
            random_state=int(seed),
        ).fit(Xb, yb)

    seeds = (42, 43, 44, 45, 46)
    per_seed = multi_seed_metrics(
        X_train, y_train, X_eval, y_eval, fight_dates,
        seeds=seeds, fit_fn=fit_fn,
    )
    assert set(per_seed.keys()) == set(int(s) for s in seeds), per_seed.keys()
    for slc in PER_SLICE_KEYS:
        briers = {per_seed[s][slc]["brier_score"] for s in per_seed}
        # Pitfall 3: exact-equality set (NOT np.isclose) — the failure
        # mode caught is bootstrap-no-op which produces bit-identical
        # floats, not near-misses.
        assert len(briers) == len(seeds), (
            f"slice {slc}: only {len(briers)} distinct Brier across "
            f"{len(seeds)} seeds; bootstrap collapsed"
        )


def test_aggregate_variance_max() -> None:
    """VALIDATION 30-01-04 — std_used = max(seed_std, ci_half) when both finite."""
    # Construct per_seed_results so seed_std_brier ≈ 0.01 across 5 seeds.
    per_seed_results: dict[int, dict] = {}
    brier_values = [0.20, 0.21, 0.22, 0.23, 0.24]  # std (ddof=1) ≈ 0.01581
    acc_values = [0.60, 0.61, 0.62, 0.63, 0.64]
    for i, seed in enumerate((42, 43, 44, 45, 46)):
        per_seed_results[seed] = {
            slc: {
                "brier_score": float(brier_values[i]),
                "accuracy": float(acc_values[i]),
                "auc_roc": 0.65 + i * 0.001,
            }
            for slc in PER_SLICE_KEYS
        }

    # Stub representative_model: aggregate_variance delegates the CI half
    # call to evaluator.bootstrap_per_slice_ci. We monkeypatch that name
    # inside the variance module's namespace.
    from ufc_prediction.ml import variance as variance_mod

    def fake_bootstrap_per_slice_ci(
        model, X_eval, y_eval, fight_dates_eval,
    ) -> dict[str, dict[str, float]]:
        # Return finite CIs > seed_std → std_used = ci_half.
        return {slc: {"brier_ci_half": 0.05, "acc_ci_half": 0.04}
                for slc in PER_SLICE_KEYS}

    saved = variance_mod.bootstrap_per_slice_ci
    variance_mod.bootstrap_per_slice_ci = fake_bootstrap_per_slice_ci
    try:
        aggregated, warnings = aggregate_variance(
            per_seed_results,
            representative_model=object(),  # placeholder; ignored by stub
            X_eval=np.zeros((5, 13)), y_eval=np.zeros(5, dtype=np.int64),
            fight_dates_eval=np.array([]),
        )
    finally:
        variance_mod.bootstrap_per_slice_ci = saved

    assert warnings == [], warnings  # nothing degenerate
    for slc in PER_SLICE_KEYS:
        a = aggregated[slc]
        # seed_std_brier ≈ 0.01581 (ddof=1 over 0.20..0.24).
        assert math.isclose(a["seed_std_brier"], 0.01581, rel_tol=1e-3), a
        assert math.isclose(a["bootstrap_ci_half_brier"], 0.05), a
        # max(0.01581, 0.05) == 0.05
        assert math.isclose(a["std_brier_used"], 0.05), a
        assert math.isclose(a["bootstrap_ci_half_acc"], 0.04), a
        assert math.isclose(a["std_acc_used"], 0.04), a


def test_collapse_detection() -> None:
    """VALIDATION 30-01-05 — assert_distinct_seed_brier raises on collapse."""
    # 5 seeds; slice "most_recent_12mo" gets IDENTICAL Brier across 2 of 5
    # → only 4 distinct values for that slice.
    per_seed_results: dict[int, dict] = {}
    seeds = (42, 43, 44, 45, 46)
    brier_distinct = [0.20, 0.21, 0.22, 0.23, 0.24]
    for i, seed in enumerate(seeds):
        per_seed_results[seed] = {
            "most_recent_12mo": {
                # Force seed 44 to share Brier with seed 43 → collapse.
                "brier_score": float(brier_distinct[1] if i == 2 else brier_distinct[i]),
                "accuracy": 0.6,
            },
            "most_recent_24mo": {
                "brier_score": float(brier_distinct[i]),
                "accuracy": 0.6,
            },
            "random_15pct": {
                "brier_score": float(brier_distinct[i]),
                "accuracy": 0.6,
            },
        }
    with pytest.raises(AssertionError, match="most_recent_12mo"):
        assert_distinct_seed_brier(per_seed_results)

    # All-distinct case → silent (returns None).
    per_seed_ok: dict[int, dict] = {
        seed: {
            slc: {"brier_score": float(brier_distinct[i]), "accuracy": 0.6}
            for slc in PER_SLICE_KEYS
        }
        for i, seed in enumerate(seeds)
    }
    result = assert_distinct_seed_brier(per_seed_ok)
    assert result is None


def test_pitfall_b_nan_fallback() -> None:
    """VALIDATION 30-01-06 — NaN bootstrap_ci_half → fallback + raw NaN preserved."""
    per_seed_results: dict[int, dict] = {}
    brier_values = [0.10, 0.105, 0.11, 0.115, 0.12]  # seed_std ≈ 0.0079
    acc_values = [0.70, 0.71, 0.72, 0.73, 0.74]
    for i, seed in enumerate((42, 43, 44, 45, 46)):
        per_seed_results[seed] = {
            slc: {
                "brier_score": float(brier_values[i]),
                "accuracy": float(acc_values[i]),
                "auc_roc": 0.70,
            }
            for slc in PER_SLICE_KEYS
        }

    from ufc_prediction.ml import variance as variance_mod

    def fake_nan_ci(
        model, X_eval, y_eval, fight_dates_eval,
    ) -> dict[str, dict[str, float]]:
        # Degenerate slice → NaN for both metrics on random_15pct only;
        # other slices have small finite CIs (less than seed_std).
        return {
            "most_recent_12mo": {"brier_ci_half": 0.001, "acc_ci_half": 0.001},
            "most_recent_24mo": {"brier_ci_half": 0.001, "acc_ci_half": 0.001},
            "random_15pct": {
                "brier_ci_half": float("nan"),
                "acc_ci_half": float("nan"),
            },
        }

    saved = variance_mod.bootstrap_per_slice_ci
    variance_mod.bootstrap_per_slice_ci = fake_nan_ci
    try:
        aggregated, warnings = aggregate_variance(
            per_seed_results,
            representative_model=object(),
            X_eval=np.zeros((5, 13)), y_eval=np.zeros(5, dtype=np.int64),
            fight_dates_eval=np.array([]),
        )
    finally:
        variance_mod.bootstrap_per_slice_ci = saved

    # Raw NaN preserved in bootstrap_ci_half_* for the degenerate slice.
    nan_slice = aggregated["random_15pct"]
    assert math.isnan(nan_slice["bootstrap_ci_half_brier"]), nan_slice
    assert math.isnan(nan_slice["bootstrap_ci_half_acc"]), nan_slice
    # Fallback: max() reduces to seed_std (CI treated as 0.0 in the max).
    seed_std_b = nan_slice["seed_std_brier"]
    seed_std_a = nan_slice["seed_std_acc"]
    assert math.isclose(nan_slice["std_brier_used"], seed_std_b, rel_tol=1e-9)
    assert math.isclose(nan_slice["std_acc_used"], seed_std_a, rel_tol=1e-9)
    # Warning list MUST contain at least one Pitfall-B entry naming the slice.
    pitfall_msgs = [w for w in warnings if "random_15pct" in w]
    assert pitfall_msgs, warnings
    # The non-NaN slices' std_used picks seed_std (which exceeds 0.001 CI).
    for slc in ("most_recent_12mo", "most_recent_24mo"):
        a = aggregated[slc]
        assert math.isclose(
            a["std_brier_used"], a["seed_std_brier"], rel_tol=1e-9,
        ), a
