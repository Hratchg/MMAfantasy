"""Gate verifier NaN handling mirrors the canonical META-V22 policy.

Live substrates (scripts/build_*_substrate_v261.py) carry NaN in the
non-baseline Level-1 columns — ``closing_prob_diff`` / ``sharp_money_signal``
(no odds), ``days_since_last_fight_diff`` / ``age_diff`` (debut / missing DOB)
and the TRAVEL cols — about 945 of 2,918 rows on the REF substrate. The
refit-baseline ``LogisticRegression`` rejected them and the verifier crashed.

The canonical meta training path (scripts/train_meta_v22.py, Plan 29-02)
applies ``nan_drop_policy="per_feature_strict_baseline"``: drop rows whose
baseline columns (``xgb_oof_prob`` / ``elo_prob`` — cols 0 and 1) are NaN,
then impute the remaining NaN in non-baseline columns with the fit-set column
medians (computed after the drop; 0.0 when a column has no finite value).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ufc_prediction.ml.gate_verifier import (
    NAN_DROP_POLICY,
    EvalSlice,
    apply_canonical_nan_policy,
    verify_candidate_vs_canonical,
)


def _slice(X: np.ndarray, y: np.ndarray, sha: str) -> EvalSlice:
    return EvalSlice(
        feature_vectors=tuple(tuple(float(v) for v in row) for row in X),
        outcomes=tuple(int(o) for o in y),
        substrate_sha=sha,
    )


def _pipeline(tmp_path: Path, X: np.ndarray, y: np.ndarray, name: str) -> Path:
    import joblib

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=10_000, random_state=0)),
        ]
    )
    pipe.fit(X, y)
    out = tmp_path / f"{name}.joblib"
    joblib.dump(pipe, out)
    return out


def test_policy_name_matches_canonical_meta_training() -> None:
    assert NAN_DROP_POLICY == "per_feature_strict_baseline"


def test_drops_baseline_nan_rows_and_imputes_fit_set_medians() -> None:
    nan = float("nan")
    a = np.array(
        [
            [0.6, 0.5, 1.0, nan],
            [nan, 0.5, 2.0, 5.0],  # baseline NaN -> dropped
            [0.4, 0.5, nan, 7.0],
        ]
    )
    b = np.array(
        [
            [0.7, nan, 3.0, 9.0],  # baseline NaN -> dropped
            [0.3, 0.4, 4.0, nan],
        ]
    )
    slices = {"b": _slice(b, np.array([1, 0]), "sb"), "a": _slice(a, np.array([1, 0, 1]), "sa")}

    clean, medians = apply_canonical_nan_policy(slices)

    # Medians over the surviving fit-set rows (a0, a2, b1): col2 -> {1, 4},
    # col3 -> {7}. Baseline cols are never imputed.
    assert medians == {2: 2.5, 3: 7.0}
    assert clean["a"].feature_vectors == ((0.6, 0.5, 1.0, 7.0), (0.4, 0.5, 2.5, 7.0))
    assert clean["a"].outcomes == (1, 1)
    assert clean["b"].feature_vectors == ((0.3, 0.4, 4.0, 7.0),)
    assert clean["b"].outcomes == (0,)
    # Audit trail: per-slice substrate SHAs pass through unchanged.
    assert clean["a"].substrate_sha == "sa" and clean["b"].substrate_sha == "sb"


def test_all_nan_column_imputes_zero() -> None:
    nan = float("nan")
    X = np.array([[0.6, 0.5, nan], [0.4, 0.5, nan]])
    clean, medians = apply_canonical_nan_policy({"s": _slice(X, np.array([1, 0]), "s")})
    assert medians == {2: 0.0}
    assert clean["s"].feature_vectors == ((0.6, 0.5, 0.0), (0.4, 0.5, 0.0))


def test_nan_free_substrate_passes_through_unchanged() -> None:
    X = np.array([[0.6, 0.5, 1.0], [0.4, 0.5, 2.0]])
    sl = _slice(X, np.array([1, 0]), "s")
    clean, medians = apply_canonical_nan_policy({"s": sl})
    assert clean["s"] == sl
    assert medians == {}


def test_slice_emptied_by_baseline_drop_raises() -> None:
    nan = float("nan")
    X = np.array([[nan, 0.5, 1.0]])
    with pytest.raises(ValueError, match="no rows survive"):
        apply_canonical_nan_policy({"s": _slice(X, np.array([1]), "s")})


def test_verifier_runs_on_substrate_with_non_baseline_nan(tmp_path: Path) -> None:
    """Regression: the refit baseline used to crash with 'Input X contains NaN'."""
    rng = np.random.default_rng(7)
    n = 400
    X = rng.uniform(0.05, 0.95, size=(n, 5))
    y = (X[:, 0] + 0.1 * rng.standard_normal(n) > 0.5).astype(int)
    canon = _pipeline(tmp_path, X, y, "canon")
    cand = _pipeline(tmp_path, X, y, "cand")

    X_nan = X.copy()
    X_nan[rng.random(n) < 0.3, 2] = np.nan  # e.g. closing_prob_diff w/o odds
    X_nan[rng.random(n) < 0.2, 4] = np.nan  # e.g. days_since_last_fight_diff
    X_nan[:5, 1] = np.nan  # a few baseline-NaN rows -> dropped
    slices = {
        "most_recent_12mo": _slice(X_nan[:150], y[:150], "s12"),
        "most_recent_24mo": _slice(X_nan[:300], y[:300], "s24"),
        "random_15pct": _slice(X_nan[300:], y[300:], "sr"),
    }

    verdict = verify_candidate_vs_canonical(cand, canon, slices)

    for per_slice in (
        verdict.aligned_baseline_brier_per_slice,
        verdict.aligned_candidate_brier_per_slice,
        verdict.raw_baseline_brier_per_slice,
    ):
        assert set(per_slice) == set(slices)
        assert all(np.isfinite(v) for v in per_slice.values())


def test_medians_use_the_deduplicated_union_the_refit_trains_on() -> None:
    """Overlapping slices (12mo is a subset of 24mo) count each row once.

    The aligned refit baseline is cross-fit on the de-duplicated slice union
    (S11), so the imputation fit set is that same union: a recent fight
    present in both 12mo and 24mo must not be double-weighted in the median.
    """
    nan = float("nan")
    old = np.array([[0.6, 0.5, 1.0], [0.4, 0.5, 2.0]])
    recent = np.array([[0.7, 0.5, 9.0], [0.3, 0.5, nan]])
    slices = {
        "most_recent_12mo": _slice(recent, np.array([1, 0]), "s12"),
        "most_recent_24mo": _slice(np.vstack([old, recent]), np.array([1, 0, 1, 0]), "s24"),
    }

    clean, medians = apply_canonical_nan_policy(slices)

    # Union col2 = {1, 2, 9} -> 2.0 (the concatenation {9, 1, 2, 9} gives 5.5).
    assert medians == {2: 2.0}
    # The shared NaN row is imputed identically in both slices.
    assert clean["most_recent_12mo"].feature_vectors[1] == (0.3, 0.5, 2.0)
    assert clean["most_recent_24mo"].feature_vectors[3] == (0.3, 0.5, 2.0)
