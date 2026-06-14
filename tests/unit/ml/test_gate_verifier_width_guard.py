"""Phase 64 FEAT-V261-01 Plan 64-01 — width-mismatch guard unit tests.

Covers the D-01 locked behavior: when the canonical Pipeline's
``n_features_in_`` differs from the per-row width of the substrate's
``feature_vectors``, ``verify_candidate_vs_canonical`` MUST:

  - Skip the raw canonical ``predict_proba`` call entirely (it would
    crash with an opaque sklearn ValueError).
  - Populate ``raw_baseline_brier_per_slice[slice] = None`` and
    ``raw_delta_per_slice[slice] = None`` for every slice.
  - Still run the refit_baseline path end-to-end on the candidate-width
    substrate so ``aligned_*_brier`` carry real numbers (refit pipeline
    is retrained at candidate width, so no width error).
  - Force ``verdict.verdict == "confound_block"`` with ``confound_detected``
    True, ``confound_evidence`` containing the literal substring
    ``"width_mismatch"`` plus both numeric widths, and ``rationale``
    containing ``"width_mismatch"`` plus both widths for operator
    debuggability.

Width-matched cases (existing Phase 55 path) MUST be untouched — the
guard is a surgical early-return for the mismatch case only.

This file intentionally has zero overlap with
``tests/unit/ml/test_gate_verifier_v26.py`` so the Phase 55 suite
stays byte-stable.
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ufc_prediction.ml.gate_verifier import (
    EvalSlice,
    SubstrateDriftSafeGateVerdict,
    verify_candidate_vs_canonical,
)


# ── Synthetic fixtures ────────────────────────────────────────────────────


def _make_pipeline_and_persist(
    tmp_path: Path,
    n_features: int,
    name: str,
    seed: int = 0,
) -> Path:
    """Fit a fresh StandardScaler+LogReg Pipeline at the requested width
    and joblib-dump.

    Mirrors the Phase 55 helper in tests/unit/ml/test_gate_verifier_v26.py
    but parameterizes the column count so we can construct width-mismatched
    pairs.  After ``fit``, the Pipeline exposes ``n_features_in_`` — the
    very attribute the Phase 64 width guard introspects.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((40, n_features))
    # Learnable binary signal off the first column so logreg actually fits.
    y = (X[:, 0] > 0).astype(int)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(C=1.0, max_iter=500, random_state=seed)),
        ]
    )
    pipe.fit(X, y)
    assert pipe.named_steps["logreg"].n_features_in_ == n_features
    out = tmp_path / f"{name}.joblib"
    joblib.dump(pipe, out)
    return out


def _build_eval_slices(
    n_features: int,
    seed: int = 1,
    n_rows_per_slice: int = 20,
) -> dict[str, EvalSlice]:
    """Build three synthetic EvalSlices at the requested width.

    Slice names mirror the locked Phase 63 substrate-slice triple
    (``most_recent_12mo``, ``most_recent_24mo``, ``random_15pct``).
    Each slice gets a distinct ``substrate_sha`` so the per-slice
    SHA aggregator in the verifier sees real variation.
    """
    rng = np.random.default_rng(seed)
    slices: dict[str, EvalSlice] = {}
    for slice_name in ("most_recent_12mo", "most_recent_24mo", "random_15pct"):
        X = rng.standard_normal((n_rows_per_slice, n_features))
        y = (X[:, 0] > 0).astype(int)
        slices[slice_name] = EvalSlice(
            feature_vectors=tuple(tuple(float(v) for v in row) for row in X),
            outcomes=tuple(int(o) for o in y),
            substrate_sha=f"sha_slice_{slice_name}_w{n_features}",
        )
    return slices


# ── Test 1: width-mismatch forces confound_block ──────────────────────────


def test_width_mismatch_forces_confound_block(tmp_path: Path) -> None:
    """13-wide canonical + 15-wide candidate + 15-wide substrate.

    The TRAVEL FEAT-V261-01 production shape. Canonical raw predict would
    crash; the guard must short-circuit to a clean ``confound_block``
    verdict with raw_*=None and aligned_* populated via refit.
    """
    canonical_path = _make_pipeline_and_persist(
        tmp_path, n_features=13, name="canon13", seed=0,
    )
    candidate_path = _make_pipeline_and_persist(
        tmp_path, n_features=15, name="cand15", seed=1,
    )
    slices = _build_eval_slices(n_features=15, seed=2)

    verdict = verify_candidate_vs_canonical(
        candidate=candidate_path,
        canonical=canonical_path,
        eval_slices=slices,
    )

    assert isinstance(verdict, SubstrateDriftSafeGateVerdict)
    assert verdict.verdict == "confound_block"
    assert verdict.confound_detected is True
    # Raw canonical predict was skipped — every slice's raw entry is None.
    assert all(verdict.raw_baseline_brier_per_slice[s] is None for s in slices)
    assert all(verdict.raw_delta_per_slice[s] is None for s in slices)
    # Refit_baseline path still populated the aligned numbers.
    assert all(
        verdict.aligned_baseline_brier_per_slice[s] is not None for s in slices
    )
    assert all(
        verdict.aligned_candidate_brier_per_slice[s] is not None for s in slices
    )
    # Rationale carries the literal width_mismatch marker + both widths.
    assert "width_mismatch" in verdict.rationale
    assert "13" in verdict.rationale
    assert "15" in verdict.rationale


# ── Test 2: width-matched path unaffected (non-regression) ────────────────


def test_width_match_path_unaffected(tmp_path: Path) -> None:
    """13-wide canonical + 13-wide candidate + 13-wide substrate.

    Guard MUST NOT fire — the existing Phase 55 raw-predict loop runs
    and populates ``raw_baseline_brier_per_slice`` with real floats.
    """
    canonical_path = _make_pipeline_and_persist(
        tmp_path, n_features=13, name="canon13b", seed=2,
    )
    candidate_path = _make_pipeline_and_persist(
        tmp_path, n_features=13, name="cand13", seed=3,
    )
    slices = _build_eval_slices(n_features=13, seed=4)

    verdict = verify_candidate_vs_canonical(
        candidate=candidate_path,
        canonical=canonical_path,
        eval_slices=slices,
    )

    assert isinstance(verdict, SubstrateDriftSafeGateVerdict)
    # Guard did NOT fire — raw entries are real floats, not None.
    for s in slices:
        raw_val = verdict.raw_baseline_brier_per_slice[s]
        assert isinstance(raw_val, float), (
            f"raw_baseline_brier_per_slice[{s}] expected float, got {type(raw_val)}"
        )
        assert 0.0 <= raw_val <= 1.0


# ── Test 3: rationale carries both concrete widths ────────────────────────


def test_width_mismatch_rationale_includes_concrete_widths(tmp_path: Path) -> None:
    """Operator debuggability: rationale must name BOTH widths as integers.

    The verdict.rationale string is what surfaces in the JSON sidecar
    and the markdown writeup, so operators must be able to read the
    actual width values without spelunking confound_evidence.
    """
    canonical_path = _make_pipeline_and_persist(
        tmp_path, n_features=13, name="canon13c", seed=5,
    )
    candidate_path = _make_pipeline_and_persist(
        tmp_path, n_features=15, name="cand15c", seed=6,
    )
    slices = _build_eval_slices(n_features=15, seed=7)

    verdict = verify_candidate_vs_canonical(
        candidate=candidate_path,
        canonical=canonical_path,
        eval_slices=slices,
    )

    # Match the locked rationale shape `canonical=<int>, candidate=<int>`
    # (per D-01: rationale must include both widths for debuggability).
    match = re.search(
        r"canonical[=:]?\s*(\d+).*?candidate[=:]?\s*(\d+)",
        verdict.rationale,
    )
    assert match is not None, (
        f"rationale missing canonical=N, candidate=N pattern: {verdict.rationale!r}"
    )
    assert int(match.group(1)) == 13
    assert int(match.group(2)) == 15


# ── Test 4: aligned numbers still populate under refit_baseline ───────────


def test_width_mismatch_aligned_numbers_still_populate_under_refit_baseline(
    tmp_path: Path,
) -> None:
    """Aligned numbers must be real Briers in [0, 1] even on width mismatch.

    The refit_baseline path retrains a fresh Pipeline at candidate width,
    so its predict_proba on candidate-width substrate is well-defined.
    The audit-trail SHA (refit_baseline_sha) must be a real sha256 hex.
    """
    canonical_path = _make_pipeline_and_persist(
        tmp_path, n_features=13, name="canon13d", seed=8,
    )
    candidate_path = _make_pipeline_and_persist(
        tmp_path, n_features=15, name="cand15d", seed=9,
    )
    slices = _build_eval_slices(n_features=15, seed=10)

    verdict = verify_candidate_vs_canonical(
        candidate=candidate_path,
        canonical=canonical_path,
        eval_slices=slices,
    )

    for s in slices:
        cand_brier = verdict.aligned_candidate_brier_per_slice[s]
        assert isinstance(cand_brier, float)
        assert 0.0 <= cand_brier <= 1.0, (
            f"aligned_candidate_brier_per_slice[{s}] out of [0,1]: {cand_brier}"
        )
        base_brier = verdict.aligned_baseline_brier_per_slice[s]
        assert isinstance(base_brier, float)
        assert 0.0 <= base_brier <= 1.0
    assert verdict.methodology == "refit_baseline"
    assert verdict.refit_baseline_sha is not None
    assert len(verdict.refit_baseline_sha) == 64
