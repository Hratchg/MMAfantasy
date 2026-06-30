#!/usr/bin/env python
"""Phase 24 / GATE-V22-01..05 — 10-seed × 3-slice noise-floor spike harness (v2.2).

Operator-driven entry point for the v2.2 promotion-gate recalibration on the
90-col `FEATURE_COLUMNS_V22` substrate. Forked from Phase 17's
`scripts/spike_noise_floor.py` (978 lines); v2.1 spike preserved unchanged
for audit lineage.

Reads `models/xgb_v2_meta.json`, asserts AF-1 (best_params verbatim) + AF-2
(`len(FEATURE_COLUMNS_V22) == 90`; NET-* absent), runs Pitfall E REPURPOSED
sanity check on the NO_NET-prefix subspace (first 72 cols of v2.2), runs
10 seeds via `_train_with_fixed_params` (skips Optuna per AF-1) on the FULL
90-col matrix, aggregates per-slice via `median_metrics` + per-seed `np.std`,
computes BCa 68% bootstrap CI half-widths via `evaluator.bootstrap_per_slice_ci`,
applies the operator-approved FORMULA_SOURCE mechanically, applies the
operator empirical floor `accuracy_min = max(formula_output, 0.70)`, and emits:

  - `.planning/gate_contract_v2.2.json` — per CONTEXT D-07 schema (NEW v2.2
    fields: `feature_columns_hash`, `bfo_backfill_committed_at`)
  - `.planning/phases/24-gate-v22-recalibration-spike/24-NOISE-FLOOR-REPORT.md`
  - `.planning/phases/24-gate-v22-recalibration-spike/24-HALT-AND-DECIDE.md`
    (CONDITIONAL — only if any pre-floor slice formula_acc < 0.70)

Locked decisions (CONTEXT D-01 .. D-12):
  - D-01: formula immutability — same formula hash as v2.1
    (7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a).
  - D-02: AF-1 hparam retuning BANNED — xgb_v2 best_params asserted verbatim.
  - D-03: AF-2 feature engineering BANNED — FEATURE_COLUMNS_V22 must be 90
    cols; no NET-*; APPEND-ONLY column order locked.
  - D-04: extend existing harness pattern — this is a fork (keeps v2.1 spike
    untouchable for audit; planner picked fork over in-place edit per
    RESEARCH primary recommendation).
  - D-05: operator empirical floor `accuracy_min = max(formula_output, 0.70)`
    applied AT SPIKE TIME (not at gate-eval time). Single source of truth.
  - D-06: HALT-AND-DECIDE if ANY slice formula_output < 0.70. No auto-
    relaxation. Pre-floor value is the diagnostic; post-floor (clamped)
    value is the contract.
  - D-07: gate_contract_v2.2.json schema-identical to v2.1 PLUS new fields
    `feature_columns_hash` + `bfo_backfill_committed_at`.
  - D-09: 24-SUMMARY.md frontmatter pre-templates 3 outcome paths BEFORE
    spike runs (handled in Plan 24-01 — NOT by this script).
  - D-10: AUDIT-01 MID + END SHA checkpoints (handled by orchestration — NOT
    by this script).

Anti-features (BANNED — same as v2.1 with v2.2-binding column count):
  - AF-1: hyperparameter retuning. Spike asserts xgb_v2_meta["best_params"]
    matches EXPECTED_XGB_V2_BEST_PARAMS verbatim; uses _train_with_fixed_params
    (skips Optuna by reusing xgb_v2's best_params verbatim).
  - AF-2 (v2.2 rewire): feature engineering. Spike asserts
    len(FEATURE_COLUMNS_V22) == 90 + NET-* absent + dispatch parity with
    get_feature_columns(feature_set="v2.2").

Pitfalls mitigated:
  - Pitfall B (degenerate bootstrap): bootstrap_per_slice_ci already returns
    NaN gracefully; spike catches and falls back to seed_std-only with a
    warning.
  - Pitfall C (lru_cache stale): spike does NOT call load_gate_contract()
    before writing the JSON.
  - Pitfall D (auto-commit): spike does NOT call git. Operator owns the
    commit at the Wave-2 final task.
  - Pitfall E REPURPOSED (v2.2 Q1 RECOMMENDATION option b): xgb_v2 was never
    trained on the 90-col substrate, so the v2.1 direct-baseline Pitfall E
    is undefined. REPURPOSED: train seed=42 on the FIRST 72 cols of v2.2
    (the NO_NET prefix subspace — byte-identical to FEATURE_COLUMNS_NO_NET
    by APPEND-ONLY discipline) and assert Brier within PITFALL_E_TOLERANCE
    of EXPECTED_XGB_V2_BRIER. Catches column-reorder bugs without requiring
    a new baseline.
  - Pitfall #2 (legacy trailing-col-subset hardcoding): the v2.1 spike used
    a trailing-3-col negative-index slice to subset away NET-*. v2.2 has
    NO NET-* — that subset slice DELIBERATELY ABSENT from this fork. Full
    90-col matrix is used.
  - Pitfall #7 (formula direction): the formula adds std for accuracy
    (`median_acc + std`) — this is LOCKED, not a typo. Hash 7d221b4a... pins
    the convention. DO NOT "fix" to minus.
  - Pitfall #8 (D-21 row): the D-21(v2.2, GATE) PROJECT.md row is added in
    Plan 24-01 BEFORE this spike runs.

D-09(P15) carry-forward: spike never persists models to disk. Models
live in-memory only; the persistence helper is intentionally not imported.

Usage:
    uv run python scripts/spike_noise_floor_v22.py \\
        --seeds 42 43 44 45 46 47 48 49 50 51 \\
        --meta-v2-path models/xgb_v2_meta.json \\
        --contract-path .planning/gate_contract_v2.2.json \\
        --report-path .planning/phases/24-gate-v22-recalibration-spike/24-NOISE-FLOOR-REPORT.md

After completion, operator reviews 24-NOISE-FLOOR-REPORT.md (+ 24-HALT-AND-
DECIDE.md if emitted), verifies the emitted formula_hash matches the
operator-recorded hash (7d221b4a...), and commits the artifacts manually.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

from ufc_prediction.db.session import SessionLocal
from ufc_prediction.ml.config import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_V22,
    MLConfig,
    get_feature_columns,
)
from ufc_prediction.ml.evaluator import (
    bootstrap_per_slice_ci,
    evaluate_per_slice,
)
from ufc_prediction.ml.feature_matrix import (
    FeatureMatrixAssembler,
    compute_division_medians,
    split_temporal,
)
from ufc_prediction.ml.gate_contract import GateContract, PerSliceThresholds
from ufc_prediction.ml.queries import (
    load_computed_features,
    load_elo_features,
    load_fight_odds,
    load_fight_records,
    load_fighter_physicals,
    load_pre_ufc_records,
    load_round_stats_for_ml,
)
from ufc_prediction.ml.trainer import median_metrics


# ── Locked constants (D-05..D-08(P17)) ────────────────────────────────

K_VALUE: int = 1  # D-05(P17) one-sigma

# D-07(P17): operator-approved formula source. Any change invalidates the
# embedded sha256, halting the spike before contract emission.
FORMULA_SOURCE: str = (
    "gate_brier_max = round(median_brier - 1 * max(seed_std_brier, "
    "bootstrap_ci_half_brier), 4); "
    "gate_accuracy_min = round(median_acc + 1 * max(seed_std_acc, "
    "bootstrap_ci_half_acc), 4)"
)
EXPECTED_FORMULA_HASH: str = "7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a"

# AF-1 enforcement: best_params snapshot from models/xgb_v2_meta.json. The
# 10-key dict is asserted verbatim at startup; any drift halts the spike.
EXPECTED_XGB_V2_BEST_PARAMS: dict = {
    "n_estimators": 253,
    "max_depth": 7,
    "learning_rate": 0.013116743875697326,
    "subsample": 0.6649190225778725,
    "colsample_bytree": 0.7330702307222914,
    "min_child_weight": 6,
    "gamma": 4.585236505363638,
    "reg_alpha": 2.9183206987079522e-06,
    "reg_lambda": 4.437482739059479e-05,
}

# Pitfall E (refined per Option B + 2026-05-04 operator-approved relaxation):
# xgb_v2's seed-42 Brier on the 72-col matrix MUST match
# xgb_v2_meta.json["metrics"]["brier_score"] within PITFALL_E_TOLERANCE.
#
# Columns match xgb_v2 exactly (FEATURE_COLUMNS[:-3] == xgb_v2's training
# column space) — but feature VALUES can drift across module-level changes
# (e.g., Phase 16 HOUSE-04 dedup unification, LIVE-01/02 module refactor).
# xgb_v2.joblib SHA byte-identity on disk is preserved regardless
# (baseline 6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099).
#
# Operator decision (2026-05-04): relax threshold 1e-4 → 0.005 after first
# spike attempt produced diff=0.000826 in the IMPROVEMENT direction (better
# Brier) with row counts EXACT match to xgb_v2 metadata (n_training_fights=
# 13315, n_test_fights=3326). Diagnosis: benign feature-value drift from
# Phase 16 module changes; well within typical 5-seed Brier std (~0.0009
# from Phase 16). 0.005 ≈ 5× typical seed std — catches genuine code drift
# (wrong feature pipeline, broken dedup) but tolerates benign feature
# improvements. The v2.1 gate is calibrated against the spike's own measured
# median Brier (not against xgb_v2's recorded baseline), so absolute baseline
# drift does not affect gate margin.
EXPECTED_XGB_V2_BRIER: float = 0.22061555132914565
PITFALL_E_TOLERANCE: float = 0.005

# Per-slice keys (mirrors evaluator.PER_SLICE_KEYS).
PER_SLICE_KEYS: tuple[str, ...] = (
    "most_recent_12mo",
    "most_recent_24mo",
    "random_15pct",
)


# ── v2.2 additions (CONTEXT D-05 / D-07) ──────────────────────────────

# D-05: operator empirical floor applied AT SPIKE TIME via
# accuracy_min = max(formula_output, OPERATOR_ACCURACY_FLOOR). Single source
# of truth — Phase 26 reads the floored value from the emitted contract; it
# does NOT need to know about the floor logic.
OPERATOR_ACCURACY_FLOOR: float = 0.70

# v2.2 column-space size: 72 (NO_NET baseline) + 3 REF + 6 TRAVEL + 9 META.
# META is 9 cols, NOT 10 — `layoff_days_diff` was DROPPED per RESEARCH Q4
# RESOLVED + D-07 dedup. AF-2 v2.2 assertion checks against this number; any
# drift is a Phase 23 / Phase 24 column-engineering violation.
EXPECTED_FEATURE_COLUMNS_V22_LEN: int = 90


def _train_with_fixed_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_params: dict,
    seed: int,
) -> CalibratedClassifierCV:
    """Single-seed train using xgb_v2 best_params (skips Optuna).

    Lifted verbatim from scripts/retrain_xgb_v3.py:252-280 per AF-1
    enforcement. Mirrors trainer.ModelTrainer.train()'s post-Optuna logic:
      - 80/20 chronological train_proper / calibration_holdout split
      - XGBClassifier fit on train_proper with the supplied params
      - CalibratedClassifierCV (sigmoid) wraps a FrozenEstimator
    """
    n_total = len(X_train)
    split_idx = int(n_total * 0.8)
    X_proper = X_train[:split_idx]
    y_proper = y_train[:split_idx]
    X_calib = X_train[split_idx:]
    y_calib = y_train[split_idx:]

    base = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=int(seed),
        verbosity=0,
    )
    base.fit(X_proper, y_proper)

    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    calibrated.fit(X_calib, y_calib)
    return calibrated


def _expected_calibration_error(
    probs: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Equal-width-bin ECE per Niculescu-Mizil & Caruana 2005.

    ECE = sum_b (|bin_b| / N) * |acc_b - conf_b|
    where acc_b = empirical accuracy in bin b (fraction of positives), and
    conf_b = mean predicted probability in bin b.

    D-03(P17): ECE is reported as secondary metric (un-gated). Inline here
    because evaluator.evaluate_model surfaces calibration_curve but not a
    scalar ECE; we don't add a primitive to evaluator.py for a one-shot
    spike-only readout.

    WR-04: probabilities are clamped to [0, 1] before binning. Sigmoid-
    calibrated outputs can occasionally produce values microscopically
    outside [0, 1] due to floating-point rounding in the calibrator; un-
    clamped, those samples fall outside every bin mask and silently drop
    out of the ECE numerator while still inflating `n` in the denominator.
    """
    # WR-04 clamp — defensive; expected to be a no-op for well-behaved
    # calibrators but protects against FP-rounding artifacts.
    probs = np.clip(probs, 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    if n == 0:
        return float("nan")
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(probs[mask]))
        bin_acc = float(np.mean(y[mask]))
        weight = float(np.sum(mask)) / float(n)
        ece += weight * abs(bin_acc - bin_conf)
    return float(ece)


def _per_seed_secondary_metrics(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fight_dates_test: np.ndarray,
    today: date,
    random_seed_for_slice: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute per-slice (AUC, ECE) for one seed.

    AUC is already in evaluate_per_slice's return as `auc_roc`; ECE is
    computed here. Slice masks reproduce evaluate_per_slice exactly so the
    secondary metrics align with the gated metrics.
    """
    from datetime import timedelta

    cutoff_12mo = today - timedelta(days=365)
    cutoff_24mo = today - timedelta(days=730)
    mask_12mo = np.array([d >= cutoff_12mo for d in fight_dates_test])
    mask_24mo = np.array([d >= cutoff_24mo for d in fight_dates_test])
    rng = np.random.RandomState(random_seed_for_slice)
    mask_random = rng.random(len(fight_dates_test)) < 0.15

    out: dict[str, dict[str, float]] = {}
    for slice_name, mask in (
        ("most_recent_12mo", mask_12mo),
        ("most_recent_24mo", mask_24mo),
        ("random_15pct", mask_random),
    ):
        probs = model.predict_proba(X_test[mask])[:, 1]
        y_slice = y_test[mask]
        ece = _expected_calibration_error(probs, y_slice, n_bins=10)
        out[slice_name] = {"ece": ece}
    return out


def _build_per_slice_thresholds(
    median: dict,
    seed_stds: dict[str, dict[str, float]],
    bootstrap_halves: dict[str, dict[str, float]],
    warnings: list[str],
) -> dict[str, PerSliceThresholds]:
    """Apply FORMULA_SOURCE mechanically per slice per metric (D-05/D-06).

    For each slice:
      median_brier_xgb_v2 = median[slice]["brier_score"]
      median_acc_xgb_v2   = median[slice]["accuracy"]
      seed_std_brier      = seed_stds[slice]["brier_score"]
      seed_std_acc        = seed_stds[slice]["accuracy"]
      bootstrap_ci_half_brier = bootstrap_halves[slice]["brier_ci_half"]
      bootstrap_ci_half_acc   = bootstrap_halves[slice]["acc_ci_half"]

    Pitfall B: if a bootstrap_ci_half is NaN/non-finite, fall back to 0.0
    and append a warning. std_used reduces to seed_std-only on degenerate
    slices.
    """
    per_slice: dict[str, PerSliceThresholds] = {}
    for slice_name in PER_SLICE_KEYS:
        median_brier = float(median[slice_name]["brier_score"])
        median_acc = float(median[slice_name]["accuracy"])
        seed_std_brier = float(seed_stds[slice_name]["brier_score"])
        seed_std_acc = float(seed_stds[slice_name]["accuracy"])
        boot_half_brier_raw = float(bootstrap_halves[slice_name]["brier_ci_half"])
        boot_half_acc_raw = float(bootstrap_halves[slice_name]["acc_ci_half"])

        if not np.isfinite(boot_half_brier_raw):
            warnings.append(
                f"{slice_name}: bootstrap_ci_half_brier was NaN/non-finite "
                "(degenerate slice); falling back to 0.0 (std_used reduces "
                "to seed_std)."
            )
            boot_half_brier = 0.0
        else:
            boot_half_brier = boot_half_brier_raw
        if not np.isfinite(boot_half_acc_raw):
            warnings.append(
                f"{slice_name}: bootstrap_ci_half_acc was NaN/non-finite "
                "(degenerate slice); falling back to 0.0 (std_used reduces "
                "to seed_std)."
            )
            boot_half_acc = 0.0
        else:
            boot_half_acc = boot_half_acc_raw

        std_brier_used = max(seed_std_brier, boot_half_brier)
        std_acc_used = max(seed_std_acc, boot_half_acc)
        gate_brier_max = round(median_brier - K_VALUE * std_brier_used, 4)
        gate_accuracy_min = round(median_acc + K_VALUE * std_acc_used, 4)

        per_slice[slice_name] = PerSliceThresholds(
            brier_max=gate_brier_max,
            accuracy_min=gate_accuracy_min,
            median_brier_xgb_v2=median_brier,
            median_acc_xgb_v2=median_acc,
            seed_std_brier=seed_std_brier,
            seed_std_acc=seed_std_acc,
            bootstrap_ci_half_brier=boot_half_brier_raw,
            bootstrap_ci_half_acc=boot_half_acc_raw,
            std_brier_used=std_brier_used,
            std_acc_used=std_acc_used,
        )
    return per_slice


def _apply_operator_floor_and_detect_breach(
    per_slice: dict[str, PerSliceThresholds],
    *,
    floor: float = OPERATOR_ACCURACY_FLOOR,
) -> tuple[dict[str, PerSliceThresholds], dict[str, dict]]:
    """Apply operator empirical floor (CONTEXT D-05) + detect breach (D-06).

    For each slice:
      pre_floor_accuracy_min = per_slice[slice].accuracy_min  (formula output)
      final_accuracy_min     = max(pre_floor_accuracy_min, floor)
      breach[slice]          = pre_floor_accuracy_min < floor

    Returns:
      (clamped_per_slice, breach_diagnostic)

      clamped_per_slice: PerSliceThresholds dict with accuracy_min clamped
        to >= floor. This is what gets emitted into gate_contract_v2.2.json
        AT SPIKE TIME (pre-operator-decision; provisional).
      breach_diagnostic[slice] = {
        "pre_floor_accuracy_min": float,
        "post_floor_accuracy_min": float,
        "breach": bool,
        "gap": float (positive if breach, else 0.0),
      }
      The diagnostic is reported in 24-HALT-AND-DECIDE.md when any slice
      breached the floor; the contract JSON itself stores the floored
      value at spike time (single source of truth at spike time).

    Note (WR-01 / Phase 24 lineage): when the operator chooses
    ``accept_truth`` (un-floored thresholds carried forward per D-18), the
    clamped contract emitted by this function is REPLACED post-spike by a
    hand-edit to .planning/gate_contract_v2.2.json that:
      (a) restores the un-floored ``accuracy_min`` per slice, and
      (b) injects per-slice ``operator_floor_applied`` /
          ``operator_floor_value`` / ``operator_decision`` plus a top-level
          ``operator_decision`` block.
    The ``accept_truth`` branch is therefore NOT exercised by this function
    — the function unconditionally produces the clamped (floored) contract;
    the post-HALT hand-edit handles the un-floored case. A future spike-v23
    iteration should accept an ``--operator-decision {floor|accept_truth}``
    flag and emit the final contract programmatically rather than via
    hand-edit (deferred; see Phase 24 REVIEW WR-01 + deferred-items.md).
    """
    clamped: dict[str, PerSliceThresholds] = {}
    breach: dict[str, dict] = {}
    for slice_name, thresholds in per_slice.items():
        pre = thresholds.accuracy_min
        post = max(pre, floor)
        is_breach = pre < floor
        gap = max(floor - pre, 0.0)
        clamped[slice_name] = PerSliceThresholds(
            brier_max=thresholds.brier_max,
            accuracy_min=post,  # CLAMPED value goes into contract
            median_brier_xgb_v2=thresholds.median_brier_xgb_v2,
            median_acc_xgb_v2=thresholds.median_acc_xgb_v2,
            seed_std_brier=thresholds.seed_std_brier,
            seed_std_acc=thresholds.seed_std_acc,
            bootstrap_ci_half_brier=thresholds.bootstrap_ci_half_brier,
            bootstrap_ci_half_acc=thresholds.bootstrap_ci_half_acc,
            std_brier_used=thresholds.std_brier_used,
            std_acc_used=thresholds.std_acc_used,
        )
        breach[slice_name] = {
            "pre_floor_accuracy_min": pre,
            "post_floor_accuracy_min": post,
            "breach": is_breach,
            "gap": gap,
        }
    return clamped, breach


def _render_halt_and_decide(
    *,
    breach_diagnostic: dict[str, dict],
    per_slice: dict[str, PerSliceThresholds],
    triggered_at: str,
) -> str:
    """Render 24-HALT-AND-DECIDE.md per CONTEXT D-06 + D-09 pre-templates.

    Args:
        breach_diagnostic: output of _apply_operator_floor_and_detect_breach
            second return value; per-slice pre/post floor values + gap.
        per_slice: PRE-floor PerSliceThresholds dict (so the diagnostic
            tables show the formula output that fell below the floor).
        triggered_at: ISO timestamp for the artifact frontmatter.
    """
    lines = [
        "---",
        "phase: 24-gate-v22-recalibration-spike",
        "artifact: HALT-AND-DECIDE",
        f"triggered_at: {triggered_at}",
        'trigger_condition: "Pre-floor formula_output < OPERATOR_ACCURACY_FLOOR (0.70) on at least one slice"',
        "operator_decision_required: true",
        "outcome_paths:",
        "  - id: v2.2_gate_breaks_floor_accept_truth",
        '    description: "Accept looser empirical truth as v2.2 gate; tighten floor for v2.3+"',
        "  - id: v2.2_gate_breaks_floor_close_partial",
        '    description: "Close v2.2 partial without promotion; META/CALIB/REF/TRAVEL deferred to v2.3"',
        "---",
        "",
        "# Phase 24 HALT-AND-DECIDE",
        "",
        "**The v2.2 gate spike found at least one slice where the mechanically-",
        "derived `accuracy_min` (formula output) fell BELOW the operator empirical",
        f"floor of `{OPERATOR_ACCURACY_FLOOR}`. Per CONTEXT D-06 (D-18 carry-forward),",
        "the spike does NOT silently apply the floor — operator decision is required.**",
        "",
        "## Per-Slice Gap Diagnostic",
        "",
        "| Slice | pre_floor_accuracy_min | post_floor (clamped) | gap | breach? |",
        "|-------|------------------------|---------------------|-----|---------|",
    ]
    for slice_name in PER_SLICE_KEYS:
        d = breach_diagnostic[slice_name]
        lines.append(
            f"| {slice_name} | {d['pre_floor_accuracy_min']:.4f} | "
            f"{d['post_floor_accuracy_min']:.4f} | {d['gap']:.4f} | "
            f"{'YES' if d['breach'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Pre-Templated Outcome Paths (CONTEXT D-09)",
            "",
            "- **v2.2_gate_breaks_floor_accept_truth:** Operator accepts the empirical",
            "  truth — the looser thresholds become the v2.2 contract; the operator floor",
            "  is moved to v2.3+ as a tighter empirical commitment. Phase 26 proceeds.",
            "",
            "- **v2.2_gate_breaks_floor_close_partial:** Operator declines to accept the",
            "  looser thresholds; v2.2 closes partial (v2.0 precedent); META/CALIB/REF/",
            "  TRAVEL deferred to v2.3+. Phase 26 + Phase 27 collapse.",
            "",
            "## Operator Action Required",
            "",
            "Set `actual_outcome` in `24-SUMMARY.md` frontmatter to one of:",
            "- `v2.2_gate_breaks_floor_accept_truth`",
            "- `v2.2_gate_breaks_floor_close_partial`",
            "",
            "Then run `/gsd-execute-phase 24` resumption.",
        ]
    )
    return "\n".join(lines) + "\n"


def _emit_report(
    report_path: Path,
    *,
    spike_started: datetime,
    spike_finished: datetime,
    seeds: list[int],
    n_training_fights: int,
    n_test_fights: int,
    cutoff_str: str,
    formula_hash: str,
    median: dict,
    seed_stds: dict[str, dict[str, float]],
    bootstrap_halves: dict[str, dict[str, float]],
    per_slice: dict[str, PerSliceThresholds],
    secondary_per_slice_median: dict[str, dict[str, float]],
    seed42_repro_brier: float,
    warnings: list[str],
    per_seed_per_slice: list[dict],
) -> None:
    """Emit NOISE-FLOOR-REPORT.md per CONTEXT.md D-09 shape."""
    lines: list[str] = []
    lines.append("# Phase 24 - Noise-Floor Report (v2.2)\n")
    lines.append(
        f"**Spike started:** {spike_started.isoformat()}  \n"
        f"**Spike finished:** {spike_finished.isoformat()}  \n"
        f"**Spike duration:** "
        f"{(spike_finished - spike_started).total_seconds() / 60.0:.1f} min"
    )
    lines.append(f"**Spike seeds:** {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds)  ")
    lines.append(
        "**Base feature set:** `FEATURE_COLUMNS_V22` (90 cols; 72 NO_NET "
        "+ 3 REF + 6 TRAVEL + 9 META appended per Phase 23 D-09; NET-* "
        "absent; CONTEXT D-03 binding)  "
    )
    lines.append(f"**Cutoff date:** `{cutoff_str}` (verbatim from `xgb_v2_meta.json`)  ")
    lines.append(f"**Train fights:** {n_training_fights} / Test fights: {n_test_fights}  ")
    lines.append(
        '**Hparams:** verbatim from `xgb_v2_meta.json["best_params"]` '
        "(AF-1 enforced via EXPECTED_XGB_V2_BEST_PARAMS)  "
    )
    lines.append(
        "**FEATURE_COLUMNS_V22 length at startup:** 90 (AF-2 v2.2 enforced; "
        "full matrix used — no subset)  "
    )
    lines.append(f"**Formula hash (sha256):** `{formula_hash}`  ")
    lines.append(
        "**Formula:** `gate_brier_max = round(median_brier - 1 * "
        "max(seed_std_brier, bootstrap_ci_half_brier), 4)`; "
        "`gate_accuracy_min = round(median_acc + 1 * max(seed_std_acc, "
        "bootstrap_ci_half_acc), 4)`\n"
    )
    lines.append("---\n")

    lines.append("## Verdict\n")
    lines.append(
        "**v2.1 gate per-slice thresholds (mechanically derived; "
        "D-XX(v2.1, GATE) row will land in PROJECT.md):**\n"
    )
    for slice_name in PER_SLICE_KEYS:
        ts = per_slice[slice_name]
        lines.append(f"- `{slice_name}`: brier <= {ts.brier_max:.4f}, acc >= {ts.accuracy_min:.4f}")
    lines.append("\n---\n")

    lines.append("## Per-Slice Tables\n")
    for slice_name in PER_SLICE_KEYS:
        ts = per_slice[slice_name]
        lines.append(f"### `{slice_name}`\n")
        lines.append(
            "| Metric    | Median (xgb_v2 baseline) | seed_std (n=10) | "
            "bootstrap_CI_half (BCa 68%) | std_used = max() | "
            "Threshold derived |"
        )
        lines.append(
            "|-----------|--------------------------|-----------------|"
            "------------------------------|------------------|"
            "---------------------|"
        )
        lines.append(
            f"| Brier     | {ts.median_brier_xgb_v2:.4f}                   | "
            f"{ts.seed_std_brier:.4f}          | "
            f"{ts.bootstrap_ci_half_brier:.4f}                       | "
            f"{ts.std_brier_used:.4f}           | "
            f"<= {ts.brier_max:.4f}            |"
        )
        lines.append(
            f"| Accuracy  | {ts.median_acc_xgb_v2:.4f}                   | "
            f"{ts.seed_std_acc:.4f}          | "
            f"{ts.bootstrap_ci_half_acc:.4f}                       | "
            f"{ts.std_acc_used:.4f}           | "
            f">= {ts.accuracy_min:.4f}            |\n"
        )

    lines.append("---\n")
    lines.append("## Secondary Metrics (Observed; NOT Gated)\n")
    lines.append(
        "| Slice              | AUC (median across 10 seeds) | ECE (median across 10 seeds) |"
    )
    lines.append(
        "|--------------------|------------------------------|-------------------------------|"
    )
    for slice_name in PER_SLICE_KEYS:
        auc_med = secondary_per_slice_median[slice_name]["auc"]
        ece_med = secondary_per_slice_median[slice_name]["ece"]
        lines.append(
            f"| `{slice_name}` | {auc_med:.4f}                       | "
            f"{ece_med:.4f}                        |"
        )
    lines.append(
        "\nPer D-03(P17): AUC and ECE are reported for context but do NOT "
        "gate. Per-division ranking-stability remains a soft flag "
        "(D-14(P16) carry-forward).\n"
    )
    lines.append("---\n")

    lines.append("## Per-Seed Sub-Rows (for audit reproducibility)\n")
    lines.append(
        "| Seed | 12mo Brier | 12mo Acc | 24mo Brier | 24mo Acc | "
        "random_15pct Brier | random_15pct Acc |"
    )
    lines.append(
        "|------|-----------|----------|-----------|----------|"
        "---------------------|--------------------|"
    )
    for seed, per_slice_seed in zip(seeds, per_seed_per_slice, strict=True):
        b12 = per_slice_seed["most_recent_12mo"]["brier_score"]
        a12 = per_slice_seed["most_recent_12mo"]["accuracy"]
        b24 = per_slice_seed["most_recent_24mo"]["brier_score"]
        a24 = per_slice_seed["most_recent_24mo"]["accuracy"]
        br = per_slice_seed["random_15pct"]["brier_score"]
        ar = per_slice_seed["random_15pct"]["accuracy"]
        lines.append(
            f"| {seed}   | {b12:.4f}    | {a12:.4f}   | "
            f"{b24:.4f}    | {a24:.4f}   | "
            f"{br:.4f}              | {ar:.4f}             |"
        )
    lines.append("\n---\n")

    lines.append("## Sanity Checks\n")
    lines.append(
        '- AF-1 (no hparam retuning): `xgb_v2_meta.json["best_params"]` '
        "matched `EXPECTED_XGB_V2_BEST_PARAMS` verbatim (10 keys)."
    )
    lines.append(
        "- AF-2 (no feature engineering): `len(FEATURE_COLUMNS_V22) == 90` "
        "at startup; NET-* absent; dispatch parity with "
        "`get_feature_columns(feature_set='v2.2')` verified."
    )
    lines.append(
        f"- Pitfall E REPURPOSED: seed-42 Brier on the NO_NET prefix "
        f"subspace (`FEATURE_COLUMNS_V22[:72]`) = "
        f"{seed42_repro_brier:.6f} vs. expected "
        f"{EXPECTED_XGB_V2_BRIER:.6f} (within {PITFALL_E_TOLERANCE} "
        "tolerance — column-reorder sanity check)."
    )
    lines.append(
        f"- D-07(P17): `formula_hash == EXPECTED_FORMULA_HASH` "
        f"({formula_hash[:16]}...). Operator-approved formula source "
        "matched verbatim."
    )
    lines.append("")
    lines.append("---\n")

    lines.append("## Warnings\n")
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("(none — all 3 slices produced finite bootstrap CIs)")
    lines.append("\n---\n")

    lines.append("## Reproducibility\n")
    try:
        import scipy

        scipy_version = scipy.__version__
    except ImportError:
        scipy_version = "(import failed)"
    try:
        import xgboost

        xgb_version = xgboost.__version__
    except ImportError:
        xgb_version = "(import failed)"
    try:
        import sklearn

        sk_version = sklearn.__version__
    except ImportError:
        sk_version = "(import failed)"
    lines.append(f"- scipy version: {scipy_version}")
    lines.append(f"- xgboost version: {xgb_version}")
    lines.append(f"- sklearn version: {sk_version}")
    lines.append(
        f"- Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 24 / GATE-V22-01..05 — 10-seed x 3-slice noise-floor "
            "spike (v2.2; 90-col substrate; operator floor + HALT-AND-DECIDE)."
        ),
    )
    parser.add_argument(
        "--meta-v2-path",
        default="models/xgb_v2_meta.json",
        help="Path to xgb_v2 meta JSON (cutoff_date + best_params source).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(42, 52)),
        help="Random seeds for the 10-seed spike (default: 42..51).",
    )
    parser.add_argument(
        "--contract-path",
        default=".planning/gate_contract_v2.2.json",
        help="Output path for the v2.2 gate contract JSON.",
    )
    parser.add_argument(
        "--report-path",
        default=(".planning/phases/24-gate-v22-recalibration-spike/24-NOISE-FLOOR-REPORT.md"),
        help="Output path for NOISE-FLOOR-REPORT.md.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Quick mode: only 1 seed (debugging only). Refuses to write "
            "gate_contract_v2.2.json or NOISE-FLOOR-REPORT.md when "
            "n_seeds_observed != 10."
        ),
    )
    args = parser.parse_args()

    spike_started = datetime.now(timezone.utc)

    # ── AF-2 startup assertion (v2.2 90-col substrate; CONTEXT D-03) ─
    if len(FEATURE_COLUMNS_V22) != EXPECTED_FEATURE_COLUMNS_V22_LEN:
        msg = (
            f"FEATURE_COLUMNS_V22 drift: expected "
            f"{EXPECTED_FEATURE_COLUMNS_V22_LEN} cols (72 NO_NET + 3 REF + "
            f"6 TRAVEL + 9 META per Phase 23 close); got "
            f"{len(FEATURE_COLUMNS_V22)}. AF-2 violation: feature engineering "
            f"during/after Phase 23 close is BANNED (CONTEXT D-03)."
        )
        print(f"[spike-v22] FATAL: {msg}", file=sys.stderr)
        return 2

    # NET-* MUST NOT be in v2.2 (Phase 18 dropped them; pollution would
    # silently break the AUDIT-01 chain in a less obvious way than a count
    # drift).
    v22_set = set(FEATURE_COLUMNS_V22)
    forbidden_net = {"pagerank_diff", "sos_2hop_diff", "is_debutant_in_graph_diff"}
    net_collision = v22_set & forbidden_net
    if net_collision:
        msg = (
            f"FEATURE_COLUMNS_V22 NET-* pollution: {sorted(net_collision)}; "
            f"v2.2 substrate must NOT include NET-* (Phase 18 dropped them)."
        )
        print(f"[spike-v22] FATAL: {msg}", file=sys.stderr)
        return 2

    # Dispatch-parity guard: get_feature_columns must agree with the constant
    dispatched = get_feature_columns(feature_set="v2.2")
    if list(dispatched) != list(FEATURE_COLUMNS_V22):
        msg = "get_feature_columns(feature_set='v2.2') drifted from FEATURE_COLUMNS_V22 constant."
        print(f"[spike-v22] FATAL: {msg}", file=sys.stderr)
        return 2
    print(
        f"[spike-v22] AF-2 OK: len(FEATURE_COLUMNS_V22) == "
        f"{EXPECTED_FEATURE_COLUMNS_V22_LEN}; NET-* absent; "
        f"get_feature_columns dispatch parity OK."
    )

    # ── Read xgb_v2 meta + AF-1 startup assertion ─────────────────────
    print("[spike] Reading xgb_v2 metadata for D-02(P17) cutoff inheritance...")
    meta_v2_path = Path(args.meta_v2_path)
    if not meta_v2_path.exists():
        msg = f"xgb_v2 meta JSON not found at {meta_v2_path}"
        print(f"[spike] FATAL: {msg}", file=sys.stderr)
        return 2
    meta_v2 = json.loads(meta_v2_path.read_text())
    cutoff_str: str = meta_v2["cutoff_date"]
    cutoff_date_obj: date = date.fromisoformat(cutoff_str)
    best_params: dict = meta_v2["best_params"]
    if best_params != EXPECTED_XGB_V2_BEST_PARAMS:
        msg = (
            f"AF-1 violation: xgb_v2 best_params drift detected.\n"
            f"  Got:      {best_params}\n"
            f"  Expected: {EXPECTED_XGB_V2_BEST_PARAMS}"
        )
        print(f"[spike] FATAL: {msg}", file=sys.stderr)
        return 2
    print(
        "[spike] AF-1 OK: xgb_v2 best_params matched "
        "EXPECTED_XGB_V2_BEST_PARAMS verbatim (10 keys)."
    )
    print(f"  cutoff_date: {cutoff_str}")

    # ── D-07(P17) formula hash startup compute + verification ────────
    formula_hash = hashlib.sha256(FORMULA_SOURCE.encode("utf-8")).hexdigest()
    print(f"[spike] FORMULA_HASH = {formula_hash}")
    if formula_hash != EXPECTED_FORMULA_HASH:
        msg = (
            f"D-07(P17) FORMULA drift detected.\n"
            f"  Computed:        {formula_hash}\n"
            f"  Expected:        {EXPECTED_FORMULA_HASH}\n"
            "Operator must re-approve formula before spike can emit "
            "gate_contract.json."
        )
        print(f"[spike] FATAL: {msg}", file=sys.stderr)
        return 2
    print(
        "[spike] D-07(P17) OK: formula_hash matched operator-approved "
        f"sha256 ({EXPECTED_FORMULA_HASH[:16]}...)."
    )

    # ── Data load (mirrors retrain_xgb_v3.py:117-127 verbatim) ────────
    print("[spike] Loading data from DB...")
    t0 = time.time()
    session = SessionLocal()
    try:
        fight_records = load_fight_records(session)
        elo_features = load_elo_features(session)
        computed_features = load_computed_features(session)
        fighter_physicals = load_fighter_physicals(session)
        round_stats = load_round_stats_for_ml(session)
        pre_ufc = load_pre_ufc_records(session)
        fight_odds = load_fight_odds(session)
    finally:
        session.close()
    print(
        f"  Loaded {len(fight_records)} fights / {len(fight_odds)} odds "
        f"rows in {time.time() - t0:.1f}s"
    )

    print("[spike] Computing division medians (training-set only)...")
    division_medians = compute_division_medians(
        fighter_physicals,
        fight_records,
        cutoff_date_obj,
    )

    print(
        "[spike-v22] Assembling feature matrix on v2.2 substrate "
        f"({EXPECTED_FEATURE_COLUMNS_V22_LEN} columns; "
        "REF + TRAVEL + META appended; NET-* absent)..."
    )
    config = MLConfig(cutoff_date=cutoff_str)
    assembler = FeatureMatrixAssembler(config)
    X, y, fight_dates_full = assembler.assemble(
        fight_records,
        elo_features,
        computed_features,
        fighter_physicals,
        division_medians,
        round_stats,
        pre_ufc_records=pre_ufc,
        fight_odds=fight_odds,
        feature_set="v2.2",
    )
    print(f"  X.shape = {X.shape}, y.shape = {y.shape}")

    if X.shape[1] != EXPECTED_FEATURE_COLUMNS_V22_LEN:
        msg = (
            f"FEATURE_COLUMNS_V22 shape drift: assembler with "
            f"feature_set='v2.2' produced {X.shape[1]} cols, expected "
            f"{EXPECTED_FEATURE_COLUMNS_V22_LEN}"
        )
        print(f"[spike-v22] FATAL: {msg}", file=sys.stderr)
        return 2

    print("[spike-v22] Temporal split at cutoff_date...")
    X_train, X_test, y_train, y_test = split_temporal(
        X,
        y,
        fight_dates_full,
        cutoff_date_obj,
    )
    test_mask = np.array([d >= cutoff_date_obj for d in fight_dates_full])
    fight_dates_test = np.array(fight_dates_full)[test_mask]
    print(f"  Train: {X_train.shape[0]} fights, Test: {X_test.shape[0]} fights")

    # Pitfall E corpus-drift shape sanity (v2.2 relaxation):
    # The v2.1 spike halted on n_training_fights drift vs xgb_v2_meta. v2.2
    # cannot do that — adding REF + TRAVEL + META features may introduce
    # different NaN-drop patterns from the 72-col baseline, so row counts
    # may legitimately differ. Column-count check (X.shape[1] == 90 above)
    # already catches structural AF-2 drift. We REPORT the deltas for
    # operator visibility but do NOT halt.
    expected_n_train = meta_v2.get("n_training_fights")
    expected_n_test = meta_v2.get("n_test_fights")
    if expected_n_train is not None:
        train_delta = X_train.shape[0] - expected_n_train
        print(
            f"[spike-v22] n_training_fights: got {X_train.shape[0]}, "
            f"xgb_v2_meta has {expected_n_train} (delta={train_delta:+d}; "
            "expected nonzero on v2.2 substrate)."
        )
    if expected_n_test is not None:
        test_delta = X_test.shape[0] - expected_n_test
        print(
            f"[spike-v22] n_test_fights: got {X_test.shape[0]}, "
            f"xgb_v2_meta has {expected_n_test} (delta={test_delta:+d}; "
            "expected nonzero on v2.2 substrate)."
        )

    # ── v2.2 NO subset — train on full 90-col matrix (CONTEXT D-03) ──
    # NB: v2.1 used X_train[:, :-3] to drop 3 NET-* cols. v2.2 has NO
    # NET-* in FEATURE_COLUMNS_V22 (asserted at startup) — full matrix
    # is the column space being measured. Pitfall #2 prevention: legacy
    # trailing-col-subset hardcoding DELIBERATELY ABSENT in this fork.
    if X_train.shape[1] != EXPECTED_FEATURE_COLUMNS_V22_LEN:
        msg = (
            f"v2.2 column-count drift post-assembly: "
            f"X_train.shape[1]={X_train.shape[1]}, expected "
            f"{EXPECTED_FEATURE_COLUMNS_V22_LEN}. Possible assembler / "
            f"feature_set dispatch drift."
        )
        print(f"[spike-v22] FATAL: {msg}", file=sys.stderr)
        return 2
    X_train_v22 = X_train  # alias for clarity below; NO subset
    X_test_v22 = X_test
    print(
        f"[spike-v22] v2.2 90-col matrix in use: X_train_v22.shape="
        f"{X_train_v22.shape}, X_test_v22.shape={X_test_v22.shape}"
    )

    # ── Pitfall E REPURPOSED (v2.2 Q1 RECOMMENDATION option b) ─────────
    # On v2.2 90-col matrix, xgb_v2 has never trained, so direct-baseline
    # Pitfall E is undefined. REPURPOSE: train seed=42 on FIRST 72 cols
    # of v2.2 (which IS the NO_NET column space per APPEND-ONLY discipline
    # — verified: FEATURE_COLUMNS_V22 = FEATURE_COLUMNS_NO_NET + 18 cols
    # appended at indices 72..89) and assert Brier against
    # EXPECTED_XGB_V2_BRIER within tolerance. Catches accidental
    # column-reorder bugs (Pitfall #2 prevention) without requiring a new
    # baseline.
    print(
        "[spike-v22] Pitfall E REPURPOSED — training seed=42 on "
        "FEATURE_COLUMNS_V22[:72] (NO_NET prefix subspace) for column-"
        "reorder sanity check..."
    )
    t_repro = time.time()
    X_train_no_net_prefix = X_train_v22[:, :72]
    X_test_no_net_prefix = X_test_v22[:, :72]
    repro_model = _train_with_fixed_params(
        X_train_no_net_prefix,
        y_train,
        best_params,
        42,
    )
    repro_probs = repro_model.predict_proba(X_test_no_net_prefix)[:, 1]
    seed42_repro_brier = float(np.mean((repro_probs - y_test) ** 2))
    reproduce_drift = abs(seed42_repro_brier - EXPECTED_XGB_V2_BRIER)
    print(
        f"  reproduce_brier(seed=42, NO_NET prefix)={seed42_repro_brier:.8f} "
        f"vs EXPECTED_XGB_V2_BRIER={EXPECTED_XGB_V2_BRIER:.8f}; "
        f"drift={reproduce_drift:.6f} (took {time.time() - t_repro:.1f}s)"
    )
    if reproduce_drift > PITFALL_E_TOLERANCE:
        msg = (
            f"Pitfall E REPURPOSED check FAILED: NO_NET-prefix Brier "
            f"drift {reproduce_drift:.6f} > tolerance "
            f"{PITFALL_E_TOLERANCE:.4f}. Possible column-reorder bug in "
            f"FEATURE_COLUMNS_V22 — the first 72 cols should be byte-"
            f"identical to FEATURE_COLUMNS_NO_NET."
        )
        print(f"[spike-v22] FATAL: {msg}", file=sys.stderr)
        return 2
    print(
        "[spike-v22] Pitfall E REPURPOSED OK: NO_NET-prefix subspace "
        "reproduces xgb_v2 baseline within tolerance."
    )

    # ── 10-seed loop ──────────────────────────────────────────────────
    seeds = args.seeds if not args.quick else [42]
    if args.quick:
        print(
            "[spike] --quick mode: running ONLY seed 42 (HARD GUARD: refuses "
            "to write contract or report when n_seeds_observed != 10)."
        )
    print(
        f"[spike] Training {len(seeds)} candidates with shared best_params "
        "via _train_with_fixed_params (AF-1 enforced; no Optuna; no "
        "hparam-retuning loop)..."
    )
    candidate_models: list = []
    candidate_per_slice: list[dict] = []
    candidate_secondary: list[dict] = []
    # NB (v2.2 fork): the v2.1 spike reused the Pitfall E repro_model for the
    # seed=42 entry in the 10-seed loop because its repro model was trained on
    # the same column space (72 NO_NET cols). For v2.2 the Pitfall E repro
    # model was trained on the NO_NET *prefix* subspace (X_train_v22[:, :72]),
    # NOT the full 90-col v2.2 matrix. We must train fresh for the 10-seed
    # loop on the full v2.2 column space — do NOT reuse repro_model here.
    for i, seed in enumerate(seeds):
        t_seed = time.time()
        print(
            f"[spike-v22] Seed {seed} ({i + 1}/{len(seeds)}) - training "
            "on full 90-col v2.2 matrix..."
        )
        model = _train_with_fixed_params(
            X_train_v22,
            y_train,
            best_params,
            seed,
        )
        print(f"  seed={seed} trained in {time.time() - t_seed:.1f}s")
        candidate_models.append(model)
        per_slice_seed = evaluate_per_slice(
            model,
            X_test_v22,
            y_test,
            fight_dates_test,
            today=date.today(),
            random_seed=42,
        )
        candidate_per_slice.append(per_slice_seed)
        secondary_seed = _per_seed_secondary_metrics(
            model,
            X_test_v22,
            y_test,
            fight_dates_test,
            date.today(),
        )
        candidate_secondary.append(secondary_seed)
        slice_summary = ", ".join(
            f"{slice_name}: brier={metrics['brier_score']:.4f} acc={metrics['accuracy']:.4f}"
            for slice_name, metrics in per_slice_seed.items()
        )
        print(f"  seed={seed} slices: {slice_summary}")

    # ── Median across seeds (D-16(P16) carry-forward primitive) ──────
    print("[spike] Computing per-slice median across seeds...")
    median = median_metrics(candidate_per_slice)

    # ── Per-seed std (D-06(P17)) ──────────────────────────────────────
    # WR-03: ddof=0 (population stddev — numpy default) is LOCKED IN by
    # D-18 because the operator-approved formula hash
    # 7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a
    # was derived from the v2.1 spike using `np.std(...)` without an
    # explicit ddof kwarg (which defaults to 0). The v2.2 spike re-uses
    # the same formula bit-for-bit per D-18 binding. Switching to
    # `ddof=1` (sample stddev — ~5.4% larger for n=10) would invalidate
    # the v2.1/v2.2 lineage and trigger a fresh formula-hash derivation.
    # DO NOT change to ddof=1 inside a v2.x recalibration.
    seed_stds: dict[str, dict[str, float]] = {}
    for slice_name in PER_SLICE_KEYS:
        brier_arr = np.array([psm[slice_name]["brier_score"] for psm in candidate_per_slice])
        acc_arr = np.array([psm[slice_name]["accuracy"] for psm in candidate_per_slice])
        seed_stds[slice_name] = {
            "brier_score": float(np.std(brier_arr)),  # ddof=0 locked (WR-03)
            "accuracy": float(np.std(acc_arr)),  # ddof=0 locked (WR-03)
        }

    # ── Bootstrap CI half-widths (D-06(P17); Pitfall B NaN-aware) ─────
    print("[spike] Computing BCa 68% bootstrap CI half-widths on the median-Brier seed's model...")
    median_seed_brier_12mo = median["most_recent_12mo"]["brier_score"]
    distances = [
        abs(psm["most_recent_12mo"]["brier_score"] - median_seed_brier_12mo)
        for psm in candidate_per_slice
    ]
    median_seed_idx = int(np.argmin(distances))
    median_seed = seeds[median_seed_idx]
    median_seed_model = candidate_models[median_seed_idx]
    print(f"  median-12mo-Brier seed = {median_seed} (idx {median_seed_idx}); bootstrapping...")
    t_boot = time.time()
    bootstrap_halves = bootstrap_per_slice_ci(
        median_seed_model,
        X_test_v22,
        y_test,
        fight_dates_test,
        today=date.today(),
        confidence_level=0.68,
        n_resamples=9999,
        rng_seed=42,
    )
    print(f"  bootstrap done in {time.time() - t_boot:.1f}s")

    # ── Mechanical formula application + warnings collection ──────────
    warnings_list: list[str] = []
    per_slice_thresholds_raw = _build_per_slice_thresholds(
        median,
        seed_stds,
        bootstrap_halves,
        warnings_list,
    )

    # ── Apply operator empirical floor (CONTEXT D-05) + detect breach (D-06)
    per_slice_thresholds, floor_breach_diagnostic = _apply_operator_floor_and_detect_breach(
        per_slice_thresholds_raw,
        floor=OPERATOR_ACCURACY_FLOOR,
    )
    for slice_name in PER_SLICE_KEYS:
        d = floor_breach_diagnostic[slice_name]
        breach_marker = " (BREACH)" if d["breach"] else ""
        print(
            f"[spike-v22] floor-apply {slice_name}: "
            f"pre={d['pre_floor_accuracy_min']:.4f} -> "
            f"post={d['post_floor_accuracy_min']:.4f} "
            f"(gap={d['gap']:.4f}){breach_marker}"
        )

    # ── Secondary metrics: per-slice median AUC + ECE across seeds ────
    secondary_per_slice_median: dict[str, dict[str, float]] = {}
    for slice_name in PER_SLICE_KEYS:
        auc_seed_vals = [psm[slice_name]["auc_roc"] for psm in candidate_per_slice]
        ece_seed_vals = [sec[slice_name]["ece"] for sec in candidate_secondary]
        secondary_per_slice_median[slice_name] = {
            "auc": float(np.median(auc_seed_vals)),
            "ece": float(np.median(ece_seed_vals)),
        }

    # ── Quick-mode hard guard (D-08(P17) integrity) ───────────────────
    if len(seeds) != 10:
        print(
            f"[spike-v22] --quick mode active (n_seeds_observed={len(seeds)} "
            "!= 10). REFUSING to write contract or report. Re-run with "
            "--seeds 42 43 44 45 46 47 48 49 50 51 to emit canonical "
            "artifacts.",
            file=sys.stderr,
        )
        return 0

    # ── v2.2 feature_columns_hash (CONTEXT D-07 + specifics §1) ──────
    # sort_keys=False is mandatory — the APPEND-ONLY column order is part of
    # the locked identity; sorting alphabetically would silently invalidate
    # the hash for anyone who later compares against the live constant.
    v22_cols = get_feature_columns(feature_set="v2.2")
    feature_columns_hash = hashlib.sha256(
        json.dumps(v22_cols, sort_keys=False).encode("utf-8")
    ).hexdigest()
    print(f"[spike-v22] feature_columns_hash = {feature_columns_hash}")

    # ── v2.2 bfo_backfill_committed_at (CONTEXT D-07 + specifics §2) ──
    # Derive from git log on the Phase 21 BFO coverage JSON's last commit.
    # If git log fails or returns empty, halt — Phase 21 BFO backfill MUST
    # be committed before this spike runs (Pitfall #8 lineage).
    #
    # WR-02: guard `subprocess.run(check=True)` so that running outside a
    # git work-tree (CI runner without `.git`, missing git binary, or
    # permission error) produces a clean fatal message instead of an
    # unhelpful traceback after the ~90-minute training run.
    import subprocess

    try:
        bfo_ts_raw = subprocess.run(
            [
                "git",
                "log",
                "--format=%aI",
                "-1",
                "--",
                ".planning/phases/21-bfo-coverage-backfill/",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        msg = (
            f"git log failed ({type(exc).__name__}: {exc}). "
            "bfo_backfill_committed_at cannot be derived. "
            "Re-run inside a git work-tree with git installed."
        )
        print(f"[spike-v22] FATAL: {msg}", file=sys.stderr)
        return 2
    if not bfo_ts_raw:
        msg = (
            "git log returned empty for .planning/phases/21-bfo-coverage-"
            "backfill/. Phase 21 BFO backfill must be committed before "
            "spike runs."
        )
        print(f"[spike-v22] FATAL: {msg}", file=sys.stderr)
        return 2
    bfo_backfill_committed_at: str = bfo_ts_raw
    print(f"[spike-v22] bfo_backfill_committed_at = {bfo_backfill_committed_at}")

    # ── Build GateContract instance + emit JSON ───────────────────────
    contract = GateContract(
        version="v2.2",
        derived_at=date.today().isoformat(),
        n_seeds_observed=len(seeds),
        base_features_set="FEATURE_COLUMNS_V22",
        n_features=EXPECTED_FEATURE_COLUMNS_V22_LEN,
        k_value=K_VALUE,
        formula_hash=formula_hash,
        cutoff_date=cutoff_str,
        per_slice=per_slice_thresholds,
        secondary_metrics_observed={
            "auc": {
                "per_slice_median": {
                    s: secondary_per_slice_median[s]["auc"] for s in PER_SLICE_KEYS
                },
            },
            "ece": {
                "per_slice_median": {
                    s: secondary_per_slice_median[s]["ece"] for s in PER_SLICE_KEYS
                },
            },
        },
        supersedes=["D-18(P17)", "v2.1-gate"],
        notes=(
            "v2.2 gate contract — mechanically re-derived on the 90-col "
            "FEATURE_COLUMNS_V22 substrate using the same formula hash as "
            "v2.1 (D-18 binding). Operator floor accuracy_min >= 0.70 "
            "applied at spike time via max(formula_output, 0.70). "
            "HALT-AND-DECIDE artifact emitted if any slice formula_output "
            "< 0.70 (D-06 binding). v2.1 contract at "
            ".planning/gate_contract.json PRESERVED for audit lineage. "
            "Formula: " + FORMULA_SOURCE
        ),
        feature_columns_hash=feature_columns_hash,
        bfo_backfill_committed_at=bfo_backfill_committed_at,
    )

    # asdict converts PerSliceThresholds dataclasses to dicts recursively.
    contract_path = Path(args.contract_path)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(
        json.dumps(asdict(contract), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[spike-v22] Wrote gate contract: {contract_path}")

    # ── Emit NOISE-FLOOR-REPORT.md ────────────────────────────────────
    spike_finished = datetime.now(timezone.utc)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _emit_report(
        report_path,
        spike_started=spike_started,
        spike_finished=spike_finished,
        seeds=list(seeds),
        n_training_fights=int(X_train.shape[0]),
        n_test_fights=int(X_test.shape[0]),
        cutoff_str=cutoff_str,
        formula_hash=formula_hash,
        median=median,
        seed_stds=seed_stds,
        bootstrap_halves=bootstrap_halves,
        per_slice=per_slice_thresholds,
        secondary_per_slice_median=secondary_per_slice_median,
        seed42_repro_brier=seed42_repro_brier,
        warnings=warnings_list,
        per_seed_per_slice=candidate_per_slice,
    )
    print(f"[spike-v22] Wrote noise-floor report: {report_path}")

    # ── HALT-AND-DECIDE artifact (CONTEXT D-06; CONDITIONAL emit) ─────
    halt_required = any(d["breach"] for d in floor_breach_diagnostic.values())
    if halt_required:
        halt_path = Path(".planning/phases/24-gate-v22-recalibration-spike/24-HALT-AND-DECIDE.md")
        halt_path.parent.mkdir(parents=True, exist_ok=True)
        halt_path.write_text(
            _render_halt_and_decide(
                breach_diagnostic=floor_breach_diagnostic,
                per_slice=per_slice_thresholds_raw,  # PRE-floor for diagnostic
                triggered_at=datetime.now(timezone.utc).isoformat(),
            ),
            encoding="utf-8",
        )
        print(f"[spike-v22] HALT-AND-DECIDE emitted -> {halt_path}")
        # ── Final stdout (Pitfall D: do NOT call git) ────────────────
        print(
            f"\n[spike-v22] Spike complete WITH BREACH. Wrote:\n"
            f"  {contract_path}\n"
            f"  {report_path}\n"
            f"  {halt_path}\n"
            f"[spike-v22] Operator decision required: review "
            f"24-HALT-AND-DECIDE.md, set actual_outcome in 24-SUMMARY.md,"
            f" then commit artifacts manually.\n"
            f"[spike-v22] FORMULA_HASH = {formula_hash}"
        )
        return 3  # halt-needed exit code (CONTEXT D-06)
    print("[spike-v22] No floor breach detected; no HALT-AND-DECIDE emitted.")

    # ── Final stdout (Pitfall D: do NOT call git) ─────────────────────
    print(
        f"\n[spike-v22] Spike complete. Wrote:\n"
        f"  {contract_path}\n"
        f"  {report_path}\n"
        f"[spike-v22] Review 24-NOISE-FLOOR-REPORT.md, then commit "
        f"gate_contract_v2.2.json + report manually.\n"
        f"[spike-v22] FORMULA_HASH = {formula_hash}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
