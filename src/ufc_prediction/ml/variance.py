"""Phase 30 bootstrap-variance harness — canonical multi-seed entry point.

D-01 / D-02 / D-05 / D-08 binding (per
.planning/phases/30-multi-seed-variance-harness/30-CONTEXT.md):

  D-01: variance is injected by bootstrap resampling the meta-learner
        training rows (`X_meta_train`, `y_meta_train`) — classical
        bootstrap-with-replacement using `np.random.RandomState(seed)`
        per-call (NOT global `np.random.seed`; NOT `default_rng`). The
        meta-learner closed-form solver makes `random_state` a no-op
        for variance; only row resampling produces real variance.

  D-02: this module's public surface is locked at exactly four functions:
          bootstrap_resample
          multi_seed_metrics
          aggregate_variance
          assert_distinct_seed_brier
        Signatures match `30-RESEARCH.md` Pattern 1-4 verbatim.

  D-05: `assert_distinct_seed_brier` is the regression check that catches
        the v2.2 collapsed-variance bug (per `train_meta_v22.py:531-543`)
        — exact-equality set membership over per-seed Brier scores
        (Pitfall 3 in 30-RESEARCH.md; not `np.isclose` — the failure
        mode is bit-identical floats from a bootstrap no-op).

  D-08: `aggregate_variance` applies the operator-approved reduction
          std_used = max(seed_std, bootstrap_ci_half)
        verbatim from `spike_noise_floor_v22.py:_build_per_slice_thresholds`
        (lines 324-389). Pitfall-B fallback: if `bootstrap_ci_half_*` is
        NaN/non-finite (degenerate slice — e.g., `random_15pct` collapse
        on Phase 29 widened slices), the raw NaN is preserved in
        `bootstrap_ci_half_*` for the contract record, but the `max()`
        reduction treats NaN as 0.0 (so `std_used` reduces to
        `seed_std` only) and a Pitfall-B warning is appended.

Formula hash (`std_used = max(seed_std_*, bootstrap_ci_half_*)` per
slice per metric, locked from v2.1 spike):

  7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a

This module is the SINGLE source of truth for that reduction; Phase 30-02
(scripts/spike_noise_floor_v23.py) imports it and Phase 31 (gate spike)
consumes it directly — no parallel implementation.

Implementation notes:

- Pure NumPy + delegations to `evaluator.bootstrap_per_slice_ci` and
  `evaluator.evaluate_per_slice`. No file I/O, no DB, no logging side
  effects.
- `multi_seed_metrics` takes a caller-supplied `fit_fn` closure so the
  harness is fit-fn-agnostic (works for `MetaLearnerLogistic`, any
  sklearn-compatible estimator, or a custom callable).
- xgb_v2's OOF predictions are ALREADY columns of `X_meta_train` (per
  `META_V22_FEATURE_COLUMNS[0:2]`) so bootstrap-resampling rows IS
  bootstrap-resampling the OOF — no xgb_v2 retrain needed, AUDIT-01
  byte-identity preserved by construction (Pitfall 1 in 30-RESEARCH.md).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ufc_prediction.ml.evaluator import (
    PER_SLICE_KEYS,
    bootstrap_per_slice_ci,
    evaluate_per_slice,
)

__all__ = [
    "bootstrap_resample",
    "multi_seed_metrics",
    "aggregate_variance",
    "assert_distinct_seed_brier",
]


def bootstrap_resample(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap-with-replacement row resample of (X, y) per D-01.

    Returns arrays with the same shape; ~63.2% of rows are unique on
    average for large n (1 - (1 - 1/n)^n → 1 - 1/e ≈ 0.632).

    Args:
        X: feature matrix, shape (n, d).
        y: label vector, shape (n,).
        seed: per-call RNG seed. `np.random.RandomState(int(seed))` is
            used (NOT `default_rng`, NOT global `np.random.seed`) to
            match Pitfall 4 + Pattern 1 in 30-RESEARCH.md and the
            `evaluator.py:182` precedent.

    Returns:
        `(X[idx], y[idx])` where `idx = RandomState(seed).choice(n, n,
        replace=True)`.

    Raises:
        ValueError: when `X.shape[0] != y.shape[0]`.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"bootstrap_resample: X.shape[0]={X.shape[0]} != "
            f"y.shape[0]={y.shape[0]}"
        )
    n = int(X.shape[0])
    rng = np.random.RandomState(int(seed))
    idx = rng.choice(n, size=n, replace=True)
    return X[idx], y[idx]


def multi_seed_metrics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    fight_dates_eval: np.ndarray,
    seeds: Sequence[int],
    fit_fn: Callable[[np.ndarray, np.ndarray, int], object],
) -> dict[int, dict[str, dict[str, float]]]:
    """Per-seed bootstrap-fit + 3-slice evaluate per D-01 / D-02.

    For each `seed` in `seeds`:
      1. Bootstrap-resample (X_train, y_train) via `bootstrap_resample`.
      2. Construct a fresh model via `fit_fn(X_boot, y_boot, seed)`
         (Pitfall 4: caller must build a NEW estimator inside the
         closure; do NOT capture an outer mutable instance).
      3. Evaluate the model on the UNCHANGED (X_eval, y_eval,
         fight_dates_eval) via `evaluator.evaluate_per_slice`.

    Args:
        X_train: meta-train feature matrix, shape (n_train, d).
        y_train: meta-train labels, shape (n_train,).
        X_eval: meta-eval feature matrix, shape (n_eval, d).
        y_eval: meta-eval labels, shape (n_eval,).
        fight_dates_eval: per-eval-row event dates (ndarray of
            `datetime.date`-comparable values; length n_eval). Drives
            the 12mo / 24mo / random_15pct slice masks in
            `evaluate_per_slice`.
        seeds: per-seed RNG values; iteration order preserved in the
            returned dict.
        fit_fn: closure `(X_boot, y_boot, seed) -> model`. `model` must
            expose `predict_proba` and `predict` so
            `evaluate_per_slice` can compute Brier / AUC / accuracy /
            calibration.

    Returns:
        `{int(seed): {slice_name: {brier_score, auc_roc, accuracy,
        calibration_curve, ...}}}` — outer key order matches `seeds`
        argument; inner keys match `PER_SLICE_KEYS`.

    Raises:
        ValueError: when `seeds` is empty or row counts mismatch.
    """
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"multi_seed_metrics: X_train.shape[0]={X_train.shape[0]} != "
            f"y_train.shape[0]={y_train.shape[0]}"
        )
    seeds_list = list(seeds)
    if not seeds_list:
        raise ValueError("multi_seed_metrics: seeds must be non-empty")

    per_seed: dict[int, dict[str, dict[str, float]]] = {}
    for seed in seeds_list:
        X_boot, y_boot = bootstrap_resample(X_train, y_train, seed=int(seed))
        model = fit_fn(X_boot, y_boot, int(seed))
        per_seed[int(seed)] = evaluate_per_slice(
            model, X_eval, y_eval, fight_dates_eval,
        )
    return per_seed


def aggregate_variance(
    per_seed_results: dict[int, dict[str, dict[str, float]]],
    *,
    representative_model: object,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    fight_dates_eval: np.ndarray,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Per-slice variance aggregation per D-08 + Pitfall-B fallback.

    For each slice in `PER_SLICE_KEYS`:
      - `seed_std_brier` / `seed_std_acc`: `np.std(..., ddof=1)` across
        the per-seed Brier / accuracy values (sample stddev — chosen
        for unbiased estimation given small n; the v2.2 spike uses
        ddof=0 for formula-hash lineage but Plan 30 is a NEW lineage
        introducing ddof=1 by Claude's discretion per CONTEXT D-08
        Claude's-discretion bullets; the absolute number is consumed
        by `max(seed_std, ci_half)` which is dominated by the larger
        of the two for any practical n).
      - `bootstrap_ci_half_brier` / `bootstrap_ci_half_acc`: BCa 68% CI
        half-widths via `evaluator.bootstrap_per_slice_ci` called once
        on `representative_model` against the unchanged eval set.
        Raw values are PRESERVED in the output (including NaN — for
        Pitfall-B-degenerate slices).
      - `std_brier_used` / `std_acc_used`: `max(seed_std, ci_half_or_0)`
        — Pitfall-B fallback substitutes 0.0 for non-finite ci_half so
        the `max()` reduction degenerates to `seed_std` on degenerate
        slices. A warning string is appended naming that slice.

    Args:
        per_seed_results: output of `multi_seed_metrics`.
        representative_model: a single estimator (typically the
            first-seed model, mirroring `spike_noise_floor_v22.py:506`)
            against which BCa CIs are computed.
        X_eval: meta-eval feature matrix (UNCHANGED across seeds).
        y_eval: meta-eval labels.
        fight_dates_eval: per-eval-row event dates.

    Returns:
        (aggregated, warnings) where
          aggregated[slice] = {
            "seed_std_brier", "seed_std_acc",
            "bootstrap_ci_half_brier", "bootstrap_ci_half_acc",
            "std_brier_used", "std_acc_used",
          }
          warnings: list of Pitfall-B advisory strings (empty if all
                    slices produced finite bootstrap CIs).
    """
    seeds = list(per_seed_results.keys())
    warnings: list[str] = []

    # Per-seed stds for Brier + Accuracy per slice.
    # WR-01 / Phase 30 code-review: guard ddof=1 with n >= 2. On a single-
    # element array, np.std(..., ddof=1) divides by n-ddof=0 → NaN + a
    # RuntimeWarning. The deterministic single-seed --no-bootstrap path
    # (ResearchQ10 lineage anchor) is canonical; reporting NaN for an
    # undefined-variance regime would propagate into max() reductions and
    # bypass the D-06 zero-variance check. Fall back to ddof=0 on n=1
    # (which evaluates to 0.0 — the only coherent "no observed variance"
    # signal for a single sample).
    seed_stds: dict[str, dict[str, float]] = {}
    n = len(seeds)
    ddof_used = 1 if n >= 2 else 0
    for slc in PER_SLICE_KEYS:
        briers = np.array(
            [per_seed_results[s][slc]["brier_score"] for s in seeds],
            dtype=np.float64,
        )
        accs = np.array(
            [per_seed_results[s][slc]["accuracy"] for s in seeds],
            dtype=np.float64,
        )
        seed_stds[slc] = {
            # ddof=1 sample stddev — see docstring on lineage choice.
            # n=1 → ddof=0 fallback so seed_std == 0.0 (degenerate-by-
            # design "no observed variance"); avoids NaN propagation.
            "brier_score": float(np.std(briers, ddof=ddof_used)),
            "accuracy": float(np.std(accs, ddof=ddof_used)),
        }

    # BCa CI half-widths via existing primitive (no duplicate
    # implementation; matches `spike_noise_floor_v22.py:1106-1110`).
    bootstrap_halves = bootstrap_per_slice_ci(
        representative_model, X_eval, y_eval, fight_dates_eval,
    )

    # Apply max() rule + Pitfall-B fallback (verbatim from
    # spike_noise_floor_v22.py:353-388).
    # WR-05 / Phase 30 code-review: consolidate per-slice warnings — when
    # both `bootstrap_ci_half_brier` and `bootstrap_ci_half_acc` are NaN
    # on the same slice, emit ONE warning bullet naming both metrics
    # rather than two near-identical bullets (v2.2 spike lines 353-370
    # used the consolidated form; this fork had drifted to per-metric).
    aggregated: dict[str, dict[str, float]] = {}
    for slc in PER_SLICE_KEYS:
        ss_b = seed_stds[slc]["brier_score"]
        ss_a = seed_stds[slc]["accuracy"]
        bh_b_raw = float(bootstrap_halves[slc]["brier_ci_half"])
        bh_a_raw = float(bootstrap_halves[slc]["acc_ci_half"])

        degenerate_metrics: list[str] = []
        if not np.isfinite(bh_b_raw):
            degenerate_metrics.append("brier")
            bh_b_used = 0.0
        else:
            bh_b_used = bh_b_raw
        if not np.isfinite(bh_a_raw):
            degenerate_metrics.append("acc")
            bh_a_used = 0.0
        else:
            bh_a_used = bh_a_raw

        if degenerate_metrics:
            metric_list = ",".join(degenerate_metrics)
            warnings.append(
                f"{slc}: bootstrap_ci_half_{metric_list} was NaN/non-"
                f"finite (degenerate slice; Pitfall B); falling back to "
                f"0.0 (std_used reduces to seed_std)."
            )

        aggregated[slc] = {
            "seed_std_brier":          ss_b,
            "seed_std_acc":            ss_a,
            # Raw values preserved — Pitfall-B-degenerate slices keep
            # the NaN in the contract record (D-08 + WR-04 lineage).
            "bootstrap_ci_half_brier": bh_b_raw,
            "bootstrap_ci_half_acc":   bh_a_raw,
            "std_brier_used":          max(ss_b, bh_b_used),
            "std_acc_used":            max(ss_a, bh_a_used),
        }
    return aggregated, warnings


def assert_distinct_seed_brier(per_seed_results: dict[int, dict[str, dict[str, float]]]) -> None:
    """D-05 regression check: raise on bootstrap-collapse.

    For every slice in `PER_SLICE_KEYS`, the per-seed Brier scores must
    form a set with cardinality equal to `len(seeds)`. Any collapse
    indicates the bootstrap has silently no-op'd (e.g., `replace=True`
    inadvertently disabled, or `seeds` containing duplicates).

    Args:
        per_seed_results: output of `multi_seed_metrics`.

    Raises:
        AssertionError: with a message naming the offending slice and
            the distinct-vs-total count, on first collapse encountered.

    Returns:
        None on silent pass (all slices have len(seeds) distinct
        Brier scores).

    Pitfall 3 reminder: this uses EXACT-equality `set()` membership,
    NOT `np.isclose` — the failure mode caught is bit-identical floats
    from a bootstrap no-op, not floating-point noise.
    """
    seeds = list(per_seed_results.keys())
    n_seeds = len(seeds)
    for slc in PER_SLICE_KEYS:
        briers = {
            per_seed_results[s][slc]["brier_score"] for s in seeds
        }
        if len(briers) < n_seeds:
            raise AssertionError(
                f"Slice {slc}: only {len(briers)} distinct Brier scores "
                f"across {n_seeds} seeds (expected {n_seeds}). Bootstrap "
                "has collapsed — variance regression detected (D-05; "
                "v2.2 train_meta_v22.py:531-543 lineage)."
            )
    return None
