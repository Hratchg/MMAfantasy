"""S11 finding 1 — the refit (aligned) baseline must be scored out of sample.

Before this fix ``verify_candidate_vs_canonical`` fit the refit baseline on
the concatenation of every eval slice and then scored it on those same rows.
That in-sample Brier is optimistic (measured on the live REF substrate at
+0.0113 summed across the 3 slices, ~3.8x the 0.003 total-margin hurdle), so
every candidate was graded against a baseline that looked better than it
was. The refit also used ``StandardScaler + LR`` instead of the canonical
``MetaLearnerLogistic`` architecture (``PolynomialFeatures`` interactions +
``StandardScaler`` + LR).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

from ufc_prediction.ml.gate_verifier import (
    EvalSlice,
    _refit_baseline_on_substrate,
    verify_candidate_vs_canonical,
)
from ufc_prediction.ml.meta_learner import MetaLearnerLogistic

SLICES = ("most_recent_12mo", "most_recent_24mo", "random_15pct")


def _slice(X: np.ndarray, y: np.ndarray, sha: str) -> EvalSlice:
    return EvalSlice(
        feature_vectors=tuple(tuple(float(v) for v in row) for row in X),
        outcomes=tuple(int(o) for o in y),
        substrate_sha=sha,
    )


def _noise_substrate(seed: int = 0, n: int = 400, d: int = 13) -> dict[str, EvalSlice]:
    """Pure-noise substrate shaped like the real one (12mo is a subset of 24mo)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = rng.integers(0, 2, size=n)
    recent = np.arange(n) >= n // 2  # "24mo" = last half
    most_recent = np.arange(n) >= 3 * n // 4  # "12mo" = last quarter (subset)
    rand = rng.random(n) < 0.3
    return {
        "most_recent_12mo": _slice(X[most_recent], y[most_recent], "sha-12"),
        "most_recent_24mo": _slice(X[recent], y[recent], "sha-24"),
        "random_15pct": _slice(X[rand], y[rand], "sha-r"),
    }


def _persist(tmp_path: Path, model: object, name: str) -> Path:
    out = tmp_path / f"{name}.joblib"
    joblib.dump(model, out)
    return out


def test_aligned_baseline_is_not_in_sample_optimistic(tmp_path: Path) -> None:
    """On pure noise, a no-skill candidate must not trail the aligned baseline.

    The features carry no signal, so no baseline can genuinely beat the
    prior-rate candidate out of sample. An in-sample refit still does,
    because it memorises noise; that shows up as a large negative
    ``hurdle_value`` for a candidate that is exactly as good as anything
    achievable.
    """
    slices = _noise_substrate()
    X_all = np.array([fv for s in SLICES for fv in slices[s].feature_vectors])
    y_all = np.array([o for s in SLICES for o in slices[s].outcomes])
    prior = DummyClassifier(strategy="prior").fit(X_all, y_all)
    cand = _persist(tmp_path, prior, "cand")
    canon = _persist(tmp_path, prior, "canon")

    verdict = verify_candidate_vs_canonical(candidate=cand, canonical=canon, eval_slices=slices)

    # In-sample the refit beat the no-skill candidate by ~0.01+ per slice.
    # Out of sample it cannot beat it by more than sampling noise.
    assert verdict.hurdle_value > -0.003, verdict.aligned_delta_per_slice
    for slc in SLICES:
        assert verdict.aligned_baseline_brier_per_slice[slc] >= 0.245, (
            slc,
            verdict.aligned_baseline_brier_per_slice[slc],
        )


def test_cross_fit_scores_duplicate_rows_identically() -> None:
    """A fight present in two slices (12mo is a subset of 24mo) is one row.

    It must land in exactly one fold, so both slice copies get the SAME
    held-out prediction. Otherwise one copy trains the model that scores
    the other, which leaks the label.
    """
    from ufc_prediction.ml.gate_verifier import _cross_fit_baseline_predictions

    slices = _noise_substrate(seed=3)
    preds = _cross_fit_baseline_predictions(slices)

    by_row: dict[tuple[tuple[float, ...], int], set[float]] = {}
    for slc in SLICES:
        sl = slices[slc]
        assert len(preds[slc]) == len(sl.outcomes)
        for fv, o, p in zip(sl.feature_vectors, sl.outcomes, preds[slc], strict=True):
            by_row.setdefault((fv, o), set()).add(p)
    assert all(len(ps) == 1 for ps in by_row.values())
    # Sanity: the fixture really does share rows across slices.
    shared = set(slices["most_recent_12mo"].feature_vectors) & set(
        slices["most_recent_24mo"].feature_vectors
    )
    assert shared


def test_cross_fit_dedups_rows_containing_nan() -> None:
    """NaN != NaN, so a naive tuple key would never dedup a NaN-bearing row."""
    from ufc_prediction.ml.gate_verifier import _dedup_substrate_rows

    row = (0.5, float("nan"), 1.0)
    slices = {
        "most_recent_12mo": EvalSlice((row,), (1,), "a"),
        "most_recent_24mo": EvalSlice(((0.5, float("nan"), 1.0),), (1,), "b"),
    }
    _X, _y, index = _dedup_substrate_rows(slices)
    assert len(_y) == 1
    assert index["most_recent_12mo"] == index["most_recent_24mo"] == [0]


def test_cross_fit_is_deterministic() -> None:
    from ufc_prediction.ml.gate_verifier import _cross_fit_baseline_predictions

    slices = _noise_substrate(seed=5)
    assert _cross_fit_baseline_predictions(slices) == _cross_fit_baseline_predictions(slices)


def test_refit_baseline_mirrors_canonical_meta_learner_architecture() -> None:
    """The refit must be the canonical MetaLearnerLogistic config, not Scaler+LR."""
    slices = _noise_substrate(seed=7)
    X = tuple(fv for s in SLICES for fv in slices[s].feature_vectors)
    y = tuple(o for s in SLICES for o in slices[s].outcomes)
    refit = _refit_baseline_on_substrate(X, y)

    canonical_steps = [name for name, _ in MetaLearnerLogistic().pipeline.steps]
    assert canonical_steps == ["poly", "scaler", "clf"]
    assert isinstance(refit, MetaLearnerLogistic)
    assert [name for name, _ in refit.pipeline.steps] == canonical_steps
    assert refit.C == 1.0
