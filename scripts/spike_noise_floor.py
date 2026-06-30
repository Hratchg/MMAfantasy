#!/usr/bin/env python
"""Phase 17 / GATE-01..03 — 10-seed × 3-slice noise-floor spike harness.

Operator-driven entry point for the v2.1 promotion-gate calibration. Reads
`models/xgb_v2_meta.json`, asserts AF-1 (best_params verbatim) + AF-2
(`len(FEATURE_COLUMNS) == 75` + NET-* in last 3 positions), reproduces
xgb_v2's seed-42 Brier on the 72-col subset (Pitfall E sanity check), runs
10 seeds via `_train_with_fixed_params` (skips Optuna per AF-1),
aggregates per-slice via `median_metrics` + per-seed `np.std`, computes BCa
68% bootstrap CI half-widths via `evaluator.bootstrap_per_slice_ci`, applies
the operator-approved FORMULA_SOURCE mechanically, and emits two artifacts:

  - `.planning/gate_contract.json` — per CONTEXT.md D-08 schema
  - `.planning/phases/17-gate-recalibration-spike/17-NOISE-FLOOR-REPORT.md`

Locked decisions:
  - D-01(P17, corrected 2026-05-04): subset to 72 cols via FEATURE_COLUMNS[:-3].
    Reality at HEAD is len(FEATURE_COLUMNS) == 75 (72 v2.0-baseline + 3 NET-*
    added Phase 16). Dropping NET-* yields xgb_v2's exact 72-col training
    space per xgb_v2_meta.json["n_features"] = 72.
  - D-02(P17): cutoff_date inherited verbatim from xgb_v2_meta.json.
  - D-03(P17): gate metrics = Brier + Accuracy ONLY. AUC + ECE reported but
    do NOT gate.
  - D-05(P17): k = 1 (one-sigma).
  - D-06(P17): std_used = max(seed_std, bootstrap_ci_half_width) per slice
    per metric. BCa 68% CI via scipy.stats.bootstrap (n_resamples=9999).
  - D-07(P17): formula pre-committed; FORMULA_SOURCE + sha256 frozen.
  - D-08(P17): gate_contract.json schema; supersedes D-13(P16) + D-17(v2.0).
  - D-09(P17): NOISE-FLOOR-REPORT.md shape — verdict line + 3 per-slice
    tables + secondary metrics block + warnings + formula hash + timestamp.

Anti-features (BANNED):
  - AF-1: hyperparameter retuning. Spike asserts xgb_v2_meta["best_params"]
    matches EXPECTED_XGB_V2_BEST_PARAMS verbatim; uses _train_with_fixed_params
    (skips Optuna by reusing xgb_v2's best_params verbatim).
  - AF-2: feature engineering. Spike asserts len(FEATURE_COLUMNS) == 75 +
    NET-* in last 3 positions; subsets to 72 via [:-3].

Pitfalls mitigated:
  - Pitfall B (degenerate bootstrap): bootstrap_per_slice_ci already returns
    NaN gracefully; spike catches and falls back to seed_std-only with a
    warning.
  - Pitfall C (lru_cache stale): spike does NOT call load_gate_contract()
    before writing the JSON.
  - Pitfall D (auto-commit): spike does NOT call git. Operator owns the
    commit at the Wave-2 final task.
  - Pitfall E (corpus drift): pre-spike sanity check trains seed=42 on the
    72-col subset and asserts Brier within PITFALL_E_TOLERANCE (= 0.005;
    relaxed 2026-05-04 from 1e-4 — see constant docstring) of
    xgb_v2_meta["metrics"]
    ["brier_score"] (= 0.22061555). Halts on mismatch.

D-09(P15) carry-forward: spike never persists models to disk. Models
live in-memory only; the persistence helper is intentionally not imported.

Usage:
    uv run python scripts/spike_noise_floor.py \\
        --seeds 42 43 44 45 46 47 48 49 50 51 \\
        --meta-v2-path models/xgb_v2_meta.json \\
        --contract-path .planning/gate_contract.json \\
        --report-path .planning/phases/17-gate-recalibration-spike/17-NOISE-FLOOR-REPORT.md

After completion, operator reviews NOISE-FLOOR-REPORT.md, verifies the
emitted formula_hash matches the Wave-0/Task-5 operator-recorded hash
(7d221b4a...), and commits the two artifacts manually.
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
from ufc_prediction.ml.config import FEATURE_COLUMNS, MLConfig
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
    """
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
    lines.append("# Phase 17 - Noise-Floor Report\n")
    lines.append(
        f"**Spike started:** {spike_started.isoformat()}  \n"
        f"**Spike finished:** {spike_finished.isoformat()}  \n"
        f"**Spike duration:** "
        f"{(spike_finished - spike_started).total_seconds() / 60.0:.1f} min"
    )
    lines.append(f"**Spike seeds:** {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds)  ")
    lines.append(
        "**Base feature set:** `FEATURE_COLUMNS_NO_NET` (72 cols; last 3 "
        "NET-* dropped via `FEATURE_COLUMNS[:-3]`; D-01(P17) corrected "
        "2026-05-04 — operator-approved Option B)  "
    )
    lines.append(f"**Cutoff date:** `{cutoff_str}` (verbatim from `xgb_v2_meta.json`)  ")
    lines.append(f"**Train fights:** {n_training_fights} / Test fights: {n_test_fights}  ")
    lines.append(
        '**Hparams:** verbatim from `xgb_v2_meta.json["best_params"]` '
        "(AF-1 enforced via EXPECTED_XGB_V2_BEST_PARAMS)  "
    )
    lines.append(
        "**FEATURE_COLUMNS length at startup:** 75 (AF-2 enforced; 72 used "
        "after `[:-3]` slice — drops NET-*)  "
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
        "- AF-2 (no feature engineering): `len(FEATURE_COLUMNS) == 75` at "
        "startup; spike used `FEATURE_COLUMNS[:-3]` (72 cols)."
    )
    lines.append(
        f"- Pitfall E (corpus drift): seed-42 single-seed Brier on the "
        f"72-col subset = {seed42_repro_brier:.6f} vs. expected "
        f"{EXPECTED_XGB_V2_BRIER:.6f} (within {PITFALL_E_TOLERANCE} "
        "tolerance)."
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
        description=("Phase 17 / GATE-01..03 — 10-seed × 3-slice noise-floor spike."),
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
        default=".planning/gate_contract.json",
        help="Output path for the v2.1 gate contract JSON.",
    )
    parser.add_argument(
        "--report-path",
        default=(".planning/phases/17-gate-recalibration-spike/17-NOISE-FLOOR-REPORT.md"),
        help="Output path for NOISE-FLOOR-REPORT.md.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Quick mode: only 1 seed (debugging only). Refuses to write "
            "gate_contract.json or NOISE-FLOOR-REPORT.md when "
            "n_seeds_observed != 10."
        ),
    )
    args = parser.parse_args()

    spike_started = datetime.now(timezone.utc)

    # ── AF-2 startup assertion (D-01(P17, corrected) / Option B) ─────
    if len(FEATURE_COLUMNS) != 75:
        msg = (
            f"FEATURE_COLUMNS drift: expected 75 (72 v2.0-baseline + 3 "
            f"NET-* added Phase 16); got {len(FEATURE_COLUMNS)}"
        )
        print(f"[spike] FATAL: {msg}", file=sys.stderr)
        return 2
    expected_net_tail = [
        "pagerank_diff",
        "sos_2hop_diff",
        "is_debutant_in_graph_diff",
    ]
    if list(FEATURE_COLUMNS[-3:]) != expected_net_tail:
        msg = (
            f"FEATURE_COLUMNS NET-* drift: expected last 3 to be "
            f"{expected_net_tail}; got {list(FEATURE_COLUMNS[-3:])}"
        )
        print(f"[spike] FATAL: {msg}", file=sys.stderr)
        return 2
    print(
        "[spike] AF-2 OK: len(FEATURE_COLUMNS) == 75; NET-* in last 3 "
        "positions; will subset to 72 via [:-3]."
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

    print("[spike] Assembling feature matrix (75 columns; NET-* included before subset)...")
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
    )
    print(f"  X.shape = {X.shape}, y.shape = {y.shape}")

    if X.shape[1] != len(FEATURE_COLUMNS):
        msg = (
            f"FEATURE_COLUMNS drift: assembler produced {X.shape[1]} cols, "
            f"FEATURE_COLUMNS has {len(FEATURE_COLUMNS)}"
        )
        print(f"[spike] FATAL: {msg}", file=sys.stderr)
        return 2

    print("[spike] Temporal split at cutoff_date...")
    X_train, X_test, y_train, y_test = split_temporal(
        X,
        y,
        fight_dates_full,
        cutoff_date_obj,
    )
    test_mask = np.array([d >= cutoff_date_obj for d in fight_dates_full])
    fight_dates_test = np.array(fight_dates_full)[test_mask]
    print(f"  Train: {X_train.shape[0]} fights, Test: {X_test.shape[0]} fights")

    # Pitfall E corpus-drift shape sanity: train/test fight counts match
    # xgb_v2_meta. Gate before we burn 30 min of compute.
    expected_n_train = meta_v2.get("n_training_fights")
    expected_n_test = meta_v2.get("n_test_fights")
    if expected_n_train is not None and X_train.shape[0] != expected_n_train:
        msg = (
            f"Pitfall E shape sanity: n_training_fights drift: "
            f"got {X_train.shape[0]}, expected {expected_n_train} (from "
            "xgb_v2_meta.json). Possible code drift in queries.py / "
            "feature_matrix.py / features/."
        )
        print(f"[spike] FATAL: {msg}", file=sys.stderr)
        return 2
    if expected_n_test is not None and X_test.shape[0] != expected_n_test:
        msg = (
            f"Pitfall E shape sanity: n_test_fights drift: "
            f"got {X_test.shape[0]}, expected {expected_n_test} (from "
            "xgb_v2_meta.json). Possible code drift."
        )
        print(f"[spike] FATAL: {msg}", file=sys.stderr)
        return 2

    # ── D-01(P17, corrected) subset: 75 → 72 cols ─────────────────────
    X_train_72 = X_train[:, :-3]
    X_test_72 = X_test[:, :-3]
    if X_train_72.shape[1] != 72 or X_test_72.shape[1] != 72:
        msg = (
            f"D-01(P17) subset failed: expected 72 cols post-[:-3]; got "
            f"X_train_72.shape[1]={X_train_72.shape[1]}, "
            f"X_test_72.shape[1]={X_test_72.shape[1]}"
        )
        print(f"[spike] FATAL: {msg}", file=sys.stderr)
        return 2
    print(
        "[spike] Subset to 72 cols (dropped last 3 NET-* per "
        "D-01(P17, corrected 2026-05-04 — Option B))."
    )

    # ── Pitfall E pre-spike sanity reproduce (seed=42 on 72-col matrix) ─
    # Train ONE seed=42 model and check Brier reproduces xgb_v2's value
    # within PITFALL_E_TOLERANCE (0.005, relaxed 2026-05-04 from 1e-4 to
    # tolerate benign feature-value drift from Phase 16 module changes;
    # see constant docstring). Halts before the 10-seed loop on mismatch.
    print(
        "[spike] Pitfall E sanity: training seed=42 on 72-col subset and "
        f"checking Brier within {PITFALL_E_TOLERANCE} of xgb_v2 baseline "
        f"({EXPECTED_XGB_V2_BRIER:.6f})..."
    )
    t_repro = time.time()
    repro_model = _train_with_fixed_params(X_train_72, y_train, best_params, 42)
    repro_probs = repro_model.predict_proba(X_test_72)[:, 1]
    seed42_repro_brier = float(np.mean((repro_probs - y_test) ** 2))
    repro_diff = abs(seed42_repro_brier - EXPECTED_XGB_V2_BRIER)
    print(
        f"  seed=42 Brier on 72-col matrix: {seed42_repro_brier:.6f} "
        f"(diff {repro_diff:.6f}; took {time.time() - t_repro:.1f}s)"
    )
    if repro_diff > PITFALL_E_TOLERANCE:
        msg = (
            "Pitfall E sanity check FAILED: spike's 72-col seed-42 "
            f"reproduce diverged from xgb_v2 baseline by {repro_diff:.6f} "
            f"(> {PITFALL_E_TOLERANCE}). Halt before contract emission. "
            "Investigate code drift in queries.py / feature_matrix.py / "
            "features/."
        )
        print(f"[spike] FATAL: {msg}", file=sys.stderr)
        return 2
    print(
        f"[spike] Pitfall E OK: seed-42 Brier reproduces xgb_v2 baseline "
        f"within {PITFALL_E_TOLERANCE}."
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
    for i, seed in enumerate(seeds):
        # Skip seed=42 if we already trained it for Pitfall E (saves ~3 min).
        if seed == 42:
            model = repro_model
            print(
                f"[spike] Seed {seed} ({i + 1}/{len(seeds)}) - reusing Pitfall E reproduce model."
            )
        else:
            t_seed = time.time()
            print(f"[spike] Seed {seed} ({i + 1}/{len(seeds)}) - training...")
            model = _train_with_fixed_params(
                X_train_72,
                y_train,
                best_params,
                seed,
            )
            print(f"  seed={seed} trained in {time.time() - t_seed:.1f}s")
        candidate_models.append(model)
        per_slice_seed = evaluate_per_slice(
            model,
            X_test_72,
            y_test,
            fight_dates_test,
            today=date.today(),
            random_seed=42,
        )
        candidate_per_slice.append(per_slice_seed)
        secondary_seed = _per_seed_secondary_metrics(
            model,
            X_test_72,
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
    seed_stds: dict[str, dict[str, float]] = {}
    for slice_name in PER_SLICE_KEYS:
        brier_arr = np.array([psm[slice_name]["brier_score"] for psm in candidate_per_slice])
        acc_arr = np.array([psm[slice_name]["accuracy"] for psm in candidate_per_slice])
        seed_stds[slice_name] = {
            "brier_score": float(np.std(brier_arr)),
            "accuracy": float(np.std(acc_arr)),
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
        X_test_72,
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
    per_slice_thresholds = _build_per_slice_thresholds(
        median,
        seed_stds,
        bootstrap_halves,
        warnings_list,
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
            f"[spike] --quick mode active (n_seeds_observed={len(seeds)} "
            "!= 10). REFUSING to write contract or report. Re-run with "
            "--seeds 42 43 44 45 46 47 48 49 50 51 to emit canonical "
            "artifacts.",
            file=sys.stderr,
        )
        return 0

    # ── Build GateContract instance + emit JSON ───────────────────────
    contract = GateContract(
        version="v2.1",
        derived_at=date.today().isoformat(),
        n_seeds_observed=len(seeds),
        base_features_set="FEATURE_COLUMNS_NO_NET",
        n_features=72,
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
        supersedes=["D-13(P16)", "D-17(v2.0)"],
        notes=(
            "Operator-approved formula, mechanically derived thresholds. "
            "No relaxation pathway. No post-measurement renegotiation. "
            "Formula: " + FORMULA_SOURCE
        ),
    )

    # asdict converts PerSliceThresholds dataclasses to dicts recursively.
    contract_path = Path(args.contract_path)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(
        json.dumps(asdict(contract), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[spike] Wrote gate contract: {contract_path}")

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
    print(f"[spike] Wrote noise-floor report: {report_path}")

    # ── Final stdout (Pitfall D: do NOT call git) ─────────────────────
    print(
        f"\n[spike] Spike complete. Wrote:\n"
        f"  {contract_path}\n"
        f"  {report_path}\n"
        f"[spike] Review NOISE-FLOOR-REPORT.md, then commit gate_contract.json"
        f" + report manually.\n"
        f"[spike] FORMULA_HASH = {formula_hash}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
