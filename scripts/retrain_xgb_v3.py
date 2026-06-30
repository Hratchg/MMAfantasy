#!/usr/bin/env python
"""xgb_v3 retrain driver — Phase 16-04 Tasks 4..8 orchestration.

Per CONTEXT.md decisions:
  - D-04: cutoff_date inherited verbatim from models/xgb_v2_meta.json
          (apples-to-apples comparison; no extra training fights).
  - D-13: 3-slice gate (most-recent-12mo, most-recent-24mo,
          random-15%-temporal). ALL must pass median Brier ≤ 0.215 AND
          Acc ≥ 0.67.
  - D-16: 5 random_seed values (42, 43, 44, 45, 46). Median (not mean)
          across seeds is the gate verdict.
  - D-15 / D-17: gate-fail behavior is HALT at gsd-checkpoint. NO
          relaxation pathway.
  - D-08:  NET-* kill switch. Drop NET-* if Brier improvement vs
          xgb_v2-no-NET baseline < 0.001 OR if NET-03 leakage test
          failed.

Usage:
    uv run python scripts/retrain_xgb_v3.py [--out-dir DIR]

The script writes:
    models/xgb_v3_candidates.pkl  (5 candidate models + per-seed metrics
                                   — for Tasks 5/6/7/8 to consume).
    .planning/phases/.../16-04-XGB-V3-REPORT.md  (median + per-seed
                                                  tables appended).

It does NOT call save_model("xgb_v3") — that runs in Task 8 only on
the gate-PASS / kill-switch-KEEP path (D-07(P15) hard-gate-then-save).

Engineering choices (documented per CONTEXT.md "Important environment notes"):
  - Hyperparameter source: reuses xgb_v2_meta.json["best_params"] for
    each seed instead of re-running Optuna 5×. This is faithful to D-04
    apples-to-apples (same hparams, same cutoff — only features differ).
    Saves 30-90 minutes of compute and removes seed-coupled Optuna noise
    from the comparison.
  - Calibration: same 80/20 chronological split as the single-seed
    train(), driven by ModelTrainer.train_multi_seed.
  - In-memory only: candidate models are pickled to disk for the
    downstream tasks (5..8) to consume but NOT placed at the production
    model path until Task 8.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import replace as dc_replace
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

from ufc_prediction.db.session import SessionLocal
from ufc_prediction.ml.config import FEATURE_COLUMNS, MLConfig
from ufc_prediction.ml.evaluator import evaluate_per_slice
from ufc_prediction.ml.feature_matrix import (
    FeatureMatrixAssembler,
    compute_division_medians,
    split_temporal,
)
from ufc_prediction.ml.queries import (
    load_computed_features,
    load_elo_features,
    load_fight_odds,
    load_fight_records,
    load_fighter_physicals,
    load_pre_ufc_records,
    load_round_stats_for_ml,
)
from ufc_prediction.ml.trainer import ModelTrainer, median_metrics


SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)  # D-16
NO_NET_FEATURE_COUNT: int = 72  # NET-* lives at indices 72/73/74 (Gotcha 9)


def main() -> int:
    parser = argparse.ArgumentParser(description="xgb_v3 retrain driver (Phase 16-04)")
    parser.add_argument("--out-dir", default="models", help="Output dir for candidate pickle")
    parser.add_argument(
        "--report-path",
        default=".planning/phases/16-foundation-opponent-network-live-odds/16-04-XGB-V3-REPORT.md",
        help="Markdown report to append to (Tasks 4/5/6/7/9 sections).",
    )
    parser.add_argument(
        "--meta-v2-path",
        default="models/xgb_v2_meta.json",
        help="Path to xgb_v2 meta JSON (cutoff_date + best_params source).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only 1 seed (debugging only — not for production).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_pickle_path = out_dir / "xgb_v3_candidates.pkl"

    print("[xgb_v3] Reading xgb_v2 metadata for D-04 cutoff inheritance...")
    meta_v2 = json.loads(Path(args.meta_v2_path).read_text())
    cutoff_str: str = meta_v2["cutoff_date"]
    cutoff_date_obj: date = date.fromisoformat(cutoff_str)
    best_params: dict = meta_v2["best_params"]
    print(f"  cutoff_date: {cutoff_str}")
    print(f"  best_params: {best_params}")
    print(f"  FEATURE_COLUMNS length: {len(FEATURE_COLUMNS)} (with NET-*)")

    print("[xgb_v3] Loading data from DB...")
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
        f"  Loaded {len(fight_records)} fights / {len(fight_odds)} odds rows in {time.time() - t0:.1f}s"
    )

    print("[xgb_v3] Computing division medians (training-set only)...")
    division_medians = compute_division_medians(
        fighter_physicals,
        fight_records,
        cutoff_date_obj,
    )

    print("[xgb_v3] Assembling feature matrix (75 columns; NET-* included)...")
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
        print(f"[xgb_v3] FATAL: {msg}", file=sys.stderr)
        return 2

    print("[xgb_v3] Temporal split at cutoff_date...")
    X_train, X_test, y_train, y_test = split_temporal(
        X,
        y,
        fight_dates_full,
        cutoff_date_obj,
    )
    test_mask = np.array([d >= cutoff_date_obj for d in fight_dates_full])
    fight_dates_test = np.array(fight_dates_full)[test_mask]
    print(f"  Train: {X_train.shape[0]} fights, Test: {X_test.shape[0]} fights")

    seeds = SEEDS if not args.quick else (42,)
    print(f"[xgb_v3] Training {len(seeds)} xgb_v3 candidates with shared best_params...")

    candidate_models: list = []
    candidate_per_slice: list[dict] = []

    for i, seed in enumerate(seeds):
        t_seed = time.time()
        print(f"[xgb_v3] Seed {seed} ({i + 1}/{len(seeds)}) — training...")
        model = _train_with_fixed_params(X_train, y_train, best_params, seed)
        candidate_models.append(model)
        per_slice = evaluate_per_slice(
            model,
            X_test,
            y_test,
            fight_dates_test,
            today=date.today(),
            random_seed=42,  # Fixed across seeds for slice consistency.
        )
        candidate_per_slice.append(per_slice)
        elapsed = time.time() - t_seed
        slice_summary = ", ".join(
            f"{slice_name}: brier={metrics['brier_score']:.4f} acc={metrics['accuracy']:.4f}"
            for slice_name, metrics in per_slice.items()
        )
        print(f"  seed={seed} done in {elapsed:.1f}s — {slice_summary}")

    print("[xgb_v3] Computing per-slice median across seeds (D-16)...")
    median = median_metrics(candidate_per_slice)
    for slice_name, metrics in median.items():
        print(
            f"  {slice_name}: brier={metrics['brier_score']:.4f}, "
            f"acc={metrics['accuracy']:.4f}, auc={metrics['auc_roc']:.4f}"
        )

    print("[xgb_v3] Training xgb_v3-no-NET baseline for D-08 (Task 7) attribution...")
    X_train_no_net = X_train[:, :NO_NET_FEATURE_COUNT]
    X_test_no_net = X_test[:, :NO_NET_FEATURE_COUNT]
    candidate_no_net_models: list = []
    candidate_no_net_per_slice: list[dict] = []
    for i, seed in enumerate(seeds):
        t_seed = time.time()
        print(f"[xgb_v3-no-NET] Seed {seed} ({i + 1}/{len(seeds)}) — training (72 cols)...")
        model_no_net = _train_with_fixed_params(
            X_train_no_net,
            y_train,
            best_params,
            seed,
        )
        candidate_no_net_models.append(model_no_net)
        per_slice_no_net = evaluate_per_slice(
            model_no_net,
            X_test_no_net,
            y_test,
            fight_dates_test,
            today=date.today(),
            random_seed=42,
        )
        candidate_no_net_per_slice.append(per_slice_no_net)
        elapsed = time.time() - t_seed
        print(
            f"  seed={seed} (no-NET) done in {elapsed:.1f}s — "
            f"12mo brier={per_slice_no_net['most_recent_12mo']['brier_score']:.4f}"
        )

    median_no_net = median_metrics(candidate_no_net_per_slice)

    bundle = {
        "cutoff_date": cutoff_str,
        "best_params": best_params,
        "feature_columns": list(FEATURE_COLUMNS),
        "n_features": len(FEATURE_COLUMNS),
        "seeds": list(seeds),
        "n_training_fights": int(X_train.shape[0]),
        "n_test_fights": int(X_test.shape[0]),
        "candidate_models": candidate_models,
        "candidate_per_slice": candidate_per_slice,
        "median": median,
        "candidate_no_net_models": candidate_no_net_models,
        "candidate_no_net_per_slice": candidate_no_net_per_slice,
        "median_no_net": median_no_net,
        "fight_dates_test": fight_dates_test.tolist(),
    }
    with candidate_pickle_path.open("wb") as f:
        pickle.dump(bundle, f)
    print(f"[xgb_v3] Wrote candidate bundle to {candidate_pickle_path}")

    print("[xgb_v3] Appending Per-Slice Median + Per-Seed Sub-Rows tables to report...")
    _append_report_tables(
        Path(args.report_path),
        seeds=list(seeds),
        per_seed_per_slice=candidate_per_slice,
        median=median,
        n_training_fights=X_train.shape[0],
        n_test_fights=X_test.shape[0],
    )
    print("[xgb_v3] Task 4 complete. xgb_v3.joblib NOT yet saved (D-07(P15) discipline).")
    return 0


def _train_with_fixed_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_params: dict,
    seed: int,
) -> CalibratedClassifierCV:
    """Single-seed train using the xgb_v2 best_params (skipping Optuna).

    Mirrors trainer.ModelTrainer.train()'s post-Optuna logic:
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


def _append_report_tables(
    report_path: Path,
    *,
    seeds: list[int],
    per_seed_per_slice: list[dict],
    median: dict,
    n_training_fights: int,
    n_test_fights: int,
) -> None:
    """Append the Per-Slice Median + Per-Seed Sub-Rows sections to the report.

    Replaces the placeholder lines '(Filled by Task 5)' under the two
    sections with full markdown tables. Idempotent — safe to re-run.
    """
    text = report_path.read_text()

    median_block = "## Per-Slice Median Metrics (5 seeds × 3 slices)\n\n"
    median_block += f"Train fights: {n_training_fights}; Test fights: {n_test_fights}\n\n"
    median_block += (
        "| Slice | Brier (median) | Acc (median) | AUC (median) |\n"
        "|-------|----------------|--------------|--------------|\n"
    )
    for slice_name in ("most_recent_12mo", "most_recent_24mo", "random_15pct"):
        m = median[slice_name]
        median_block += (
            f"| `{slice_name}` | "
            f"{m['brier_score']:.4f} | "
            f"{m['accuracy']:.4f} | "
            f"{m['auc_roc']:.4f} |\n"
        )

    sub_block = "## Per-Seed Sub-Rows\n\n"
    sub_block += (
        "Seed-to-seed spread is a sanity flag (CONTEXT.md noted ±0.0006 in Phase 15.1).\n\n"
    )
    for slice_name in ("most_recent_12mo", "most_recent_24mo", "random_15pct"):
        sub_block += f"### {slice_name}\n\n"
        sub_block += "| Seed | Brier | Acc | AUC |\n|------|-------|-----|-----|\n"
        briers = []
        accs = []
        for seed, per_slice in zip(seeds, per_seed_per_slice, strict=True):
            metrics = per_slice[slice_name]
            briers.append(metrics["brier_score"])
            accs.append(metrics["accuracy"])
            sub_block += (
                f"| {seed} | "
                f"{metrics['brier_score']:.4f} | "
                f"{metrics['accuracy']:.4f} | "
                f"{metrics['auc_roc']:.4f} |\n"
            )
        sub_block += (
            f"\n**Std dev across seeds:** Brier={float(np.std(briers)):.4f}, "
            f"Acc={float(np.std(accs)):.4f}\n\n"
        )

    text = text.replace(
        "## Per-Slice Median Metrics (5 seeds × 3 slices)\n\n(Filled by Task 5)",
        median_block.rstrip(),
    )
    text = text.replace(
        "## Per-Seed Sub-Rows\n\n(Filled by Task 5)",
        sub_block.rstrip(),
    )
    report_path.write_text(text)


if __name__ == "__main__":
    sys.exit(main())
