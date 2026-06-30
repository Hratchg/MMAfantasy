#!/usr/bin/env python
"""Phase 45 Plan 45-03 — meta_v3 candidate META blender trainer (META3-V25-02).

Fork (do NOT mutate) of `scripts/train_meta_v22.py` orchestration. Trains the
meta_v3 candidate blender with SAME architecture as META-V22:
  - 13-col Level-1 input set per META_V22_FEATURE_COLUMNS (single source of
    truth — `src/ufc_prediction/ml/meta_features_v22.py`)
  - MetaLearnerLogistic (PolynomialFeatures degree=2 interaction_only=True +
    StandardScaler + LogisticRegression C=1.0 penalty=l2 solver=lbfgs)
  - Conservative TRAVEL path: ONLY v2.2 TRAVEL cols (indices 75-80 of
    FEATURE_COLUMNS_V22) flow through naturally as part of the v2.2 substrate.
    NO v2.5 sibling cols (travel_distance_km, tz_shift_hours).

The ONLY architectural difference from META-V22:
  - The `xgb_oof_prob` Level-1 slot is sourced from `xgb_v3` OOF predictions
    (Plan 45-02's `.planning/phases/45-*/45-XGB-V3-OOF-PREDICTIONS.parquet`)
    instead of xgb_v2 OOF.

Apples-to-apples: same architecture, swap base model OOF → if meta_v3 clears
the v2.3 gate, the lift is from BFO disambiguation (Phase 41) + seeded Elo
(Phase 43) substrate work, NOT from blender architecture changes.

Persistence triad (mirrors meta_v2 sibling layout):
  - models/meta/meta_v3.joblib              (NEW candidate)
  - models/meta/meta_v3_meta.json           (NEW metadata sidecar)
  - models/meta/meta_v3-contract.json       (NEW partner contract sibling)

AUDIT-01 discipline:
  - xgb_v2.joblib + meta_v2.joblib BYTE-IDENTICAL throughout (no overwrite).
  - Phase 26/32 protected scripts (train_meta_v22, compose_v23_meta,
    spike_noise_floor_v22/23, retrain_xgb_v3) UNTOUCHED — this is a fork.
  - Pre-flight + post-flight SHA gate inside main() (SystemExit on drift).

Usage:
    uv run python scripts/train_meta_v3_v25.py \
        --oof-parquet .planning/phases/45-meta-v3-candidate-retrain/45-XGB-V3-OOF-PREDICTIONS.parquet \
        --out-dir models/meta
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Phase 45 constants — AUDIT-01 anchor SHAs (PROJECT.md cross-cutting invariants
# #1 and #2). Pre-flight + post-flight checks against these are SystemExit hard
# stops.
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_XGB_V2_SHA256: str = "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
EXPECTED_META_V2_SHA256: str = "77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196"
EXPECTED_CUTOFF_DATE: str = "2023-01-01"

# Phase 26 best_params verbatim — MetaLearnerLogistic uses these defaults; we
# echo them into the metadata for partner-facing reproducibility.
META_V22_BEST_PARAMS: dict[str, Any] = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "PolynomialFeatures": "degree=2 interaction_only=True include_bias=False",
}

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
XGB_V2_PATH: Path = PROJECT_ROOT / "models" / "xgb_v2.joblib"
XGB_V3_PATH: Path = PROJECT_ROOT / "models" / "xgb_v3.joblib"
META_V2_PATH: Path = PROJECT_ROOT / "models" / "meta" / "meta_v2.joblib"
META_DIR: Path = PROJECT_ROOT / "models" / "meta"

PER_SLICE_KEYS: tuple[str, ...] = (
    "most_recent_12mo",
    "most_recent_24mo",
    "random_15pct",
)

# Seeds — single-seed canonical ship plus 5-seed harness for partner metrics
# (matches the META-V22 5-seed pattern, Phase 26).
SEEDS_DEFAULT: tuple[int, ...] = (42, 43, 44, 45, 46)


# ─────────────────────────────────────────────────────────────────────────────
# Public API — defined BEFORE heavy ML imports so unit tests can importlib-load
# the module without instantiating sklearn / pandas DB pipelines. (Matches the
# Plan 45-02 train_xgb_v3_v25 pattern.)
# ─────────────────────────────────────────────────────────────────────────────


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_meta_v3_metadata(
    base_model_sha256: str,
    meta_oof_parquet_sha256: str,
    per_slice_metrics: dict,
    meta_input_distribution_hash: str,
) -> dict:
    """Build the meta_v3_meta.json dict — mirrors meta_v2_meta.json shape.

    Per Plan 45-03 §<read_first>::models/meta/meta_v2_meta.json — meta_v3
    metadata uses the same field set, with:
      - meta_version="v3" (was "v2_candidate" for the pre-promotion META-V22)
      - base_model_version="v3" (was "v2")
      - base_model_sha256=sha256(models/xgb_v3.joblib) (was xgb_v2's SHA)
      - meta_feature_columns == META_V22_FEATURE_COLUMNS verbatim (13 cols;
        D-CONTEXT § "same shape as META-V22" decision — NO v2.5 TRAVEL drift)
    """
    # Local import keeps the public-API surface importable without sklearn at
    # the module top (Plan 45-02 pattern).
    from ufc_prediction.ml.meta_features_v22 import META_V22_FEATURE_COLUMNS

    return {
        "meta_version": "v3",
        "meta_kind": "logistic",
        "meta_feature_columns": list(META_V22_FEATURE_COLUMNS),
        "meta_input_distribution_hash": meta_input_distribution_hash,
        "meta_oof_parquet_sha256": meta_oof_parquet_sha256,
        "base_model_version": "v3",
        "base_model_sha256": base_model_sha256,
        "meta_learner_brier_delta_vs_logistic": 0.0,
        "best_params": dict(META_V22_BEST_PARAMS),
        "metrics": {"per_slice": per_slice_metrics},
        "trained_by_script": "scripts/train_meta_v3_v25.py",
        "phase": "45",
        "requirement": "META3-V25-02",
        "parent_model": "xgb_v3",
        "candidate_or_promoted": "candidate",
    }


def assemble_level1_input(
    oof_df: pd.DataFrame,
    level1_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack the 13-col META-V22 Level-1 matrix with xgb_v3 OOF substitution.

    Per Plan 45-03 <behavior> Test 3: column 0 of returned X_meta is sourced
    from ``oof_df.xgb_v3_oof_prob`` (the xgb_oof_prob slot of
    META_V22_FEATURE_COLUMNS).

    Args:
        oof_df: DataFrame with at minimum ``fight_id`` and ``xgb_v3_oof_prob``
            columns (per Plan 45-02 OOF parquet schema).
        level1_df: DataFrame with ``fight_id``, ``y``, and one column per
            non-xgb-OOF entry in META_V22_FEATURE_COLUMNS (12 of the 13). The
            fight_id sets MUST align — caller is responsible for join semantics.

    Returns:
        (X_meta, y) where X_meta has shape (n, 13) in META_V22_FEATURE_COLUMNS
        order; column 0 (xgb_oof_prob) is xgb_v3_oof_prob from oof_df.
    """
    from ufc_prediction.ml.meta_features_v22 import META_V22_FEATURE_COLUMNS

    # Inner join on fight_id ensures row alignment between the OOF parquet
    # and the Level-1 substrate.
    merged = oof_df[["fight_id", "xgb_v3_oof_prob"]].merge(
        level1_df,
        on="fight_id",
        how="inner",
    )

    # Map META_V22_FEATURE_COLUMNS names → source column in `merged`.
    # The first entry "xgb_oof_prob" is substituted from xgb_v3_oof_prob.
    # The other 12 are pulled verbatim from level1_df by name.
    source_by_meta_col: dict[str, str] = {"xgb_oof_prob": "xgb_v3_oof_prob"}
    for col in META_V22_FEATURE_COLUMNS[1:]:
        source_by_meta_col[col] = col

    missing = [
        meta_col for meta_col, src in source_by_meta_col.items() if src not in merged.columns
    ]
    if missing:
        raise ValueError(
            f"assemble_level1_input: missing Level-1 source columns: {missing}. "
            f"Expected 13 META-V22 cols (xgb_oof_prob ← xgb_v3_oof_prob from "
            f"oof_df; remaining 12 verbatim from level1_df)."
        )

    cols = [
        merged[source_by_meta_col[meta_col]].to_numpy(dtype=float)
        for meta_col in META_V22_FEATURE_COLUMNS
    ]
    X_meta = np.column_stack(cols)

    if "y" not in merged.columns:
        raise ValueError(
            "assemble_level1_input: level1_df must contain a 'y' column "
            "(0/1 win-label) aligned to fight_id."
        )
    y = merged["y"].to_numpy(dtype=int)

    assert X_meta.shape[1] == 13, (
        f"Level-1 shape drift: got {X_meta.shape[1]} cols, expected 13. "
        f"Any v2.5 TRAVEL sibling col leakage would inflate this."
    )
    assert X_meta.shape[0] == y.shape[0]
    return X_meta, y


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers — load DB substrate + build Level-1 source DataFrame
# ─────────────────────────────────────────────────────────────────────────────


def _audit01_preflight() -> tuple[str, str]:
    """Pre-flight: assert xgb_v2 + meta_v2 SHAs match PROJECT.md invariants."""
    xgb_v2_sha = _sha256_file(XGB_V2_PATH)
    meta_v2_sha = _sha256_file(META_V2_PATH)
    if xgb_v2_sha != EXPECTED_XGB_V2_SHA256:
        print(
            f"[train_meta_v3_v25] FATAL AUDIT-01 preflight: xgb_v2 SHA drift. "
            f"got={xgb_v2_sha} expected={EXPECTED_XGB_V2_SHA256}",
            file=sys.stderr,
        )
        sys.exit(2)
    if meta_v2_sha != EXPECTED_META_V2_SHA256:
        print(
            f"[train_meta_v3_v25] FATAL AUDIT-01 preflight: meta_v2 SHA drift. "
            f"got={meta_v2_sha} expected={EXPECTED_META_V2_SHA256}",
            file=sys.stderr,
        )
        sys.exit(2)
    print(
        f"[train_meta_v3_v25] AUDIT-01 preflight OK — "
        f"xgb_v2={xgb_v2_sha[:12]}... meta_v2={meta_v2_sha[:12]}..."
    )
    return xgb_v2_sha, meta_v2_sha


def _audit01_postflight(xgb_v2_sha_pre: str, meta_v2_sha_pre: str) -> None:
    """Post-flight: assert xgb_v2 + meta_v2 SHAs UNCHANGED after our work."""
    xgb_v2_sha_post = _sha256_file(XGB_V2_PATH)
    meta_v2_sha_post = _sha256_file(META_V2_PATH)
    if xgb_v2_sha_post != xgb_v2_sha_pre:
        print(
            f"[train_meta_v3_v25] FATAL AUDIT-01 postflight: xgb_v2 SHA drift!",
            file=sys.stderr,
        )
        sys.exit(2)
    if meta_v2_sha_post != meta_v2_sha_pre:
        print(
            f"[train_meta_v3_v25] FATAL AUDIT-01 postflight: meta_v2 SHA drift!",
            file=sys.stderr,
        )
        sys.exit(2)
    print(f"[train_meta_v3_v25] AUDIT-01 postflight OK — xgb_v2+meta_v2 byte-identical.")


def _load_level1_substrate_from_db() -> pd.DataFrame:
    """Load v2.2 90-col feature matrix + Elo probs + targets from DB.

    Returns a DataFrame with columns:
      [fight_id, event_date, y, elo_prob, <11 non-xgb META-V22 cols>]

    The xgb_oof_prob column is supplied SEPARATELY from the OOF parquet —
    NOT loaded here, since this script's whole purpose is to substitute
    xgb_v3 OOF for xgb_v2 OOF at that slot.
    """
    from ufc_prediction.db.session import SessionLocal
    from ufc_prediction.elo.config import EloConfig
    from ufc_prediction.elo.engine import EloEngine
    from ufc_prediction.ml.config import FEATURE_COLUMNS_V22, MLConfig
    from ufc_prediction.ml.feature_matrix import (
        FeatureMatrixAssembler,
        compute_division_medians,
    )
    from ufc_prediction.ml.meta_features_v22 import META_V22_FEATURE_COLUMNS
    from ufc_prediction.ml.queries import (
        load_computed_features,
        load_elo_features,
        load_fight_odds,
        load_fight_records,
        load_fighter_physicals,
        load_pre_ufc_records,
        load_round_stats_for_ml,
    )

    cutoff_date_obj = date.fromisoformat(EXPECTED_CUTOFF_DATE)
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

    division_medians = compute_division_medians(
        fighter_physicals,
        fight_records,
        cutoff_date_obj,
    )
    config = MLConfig(cutoff_date=EXPECTED_CUTOFF_DATE)
    assembler = FeatureMatrixAssembler(config)
    X_v22, y, fight_dates = assembler.assemble(
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
    assert X_v22.shape[1] == 90, f"v2.2 expected 90 cols, got {X_v22.shape[1]}"

    # Compute as-of-date Elo probability per fight (D-CONTEXT pattern — same
    # EloEngine.expected_win_probability as META-V22 + Plan 45-02).
    engine = EloEngine(EloConfig())
    elo_probs: list[float] = []
    for f in fight_records:
        fa_id = f["fighter_a_id"]
        fb_id = f["fighter_b_id"]
        fid = f["fight_id"]
        ra = float(
            elo_features.get((fa_id, fid), {"elo_overall": 1500.0}).get(
                "elo_overall",
                1500.0,
            )
        )
        rb = float(
            elo_features.get((fb_id, fid), {"elo_overall": 1500.0}).get(
                "elo_overall",
                1500.0,
            )
        )
        elo_probs.append(float(engine.expected_win_probability(ra, rb)))

    # Pull the 11 v2.2-substrate cols by NAME via FEATURE_COLUMNS_V22.index()
    # (Pitfall #1 guard from meta_features_v22 — no hardcoded indices).
    df = pd.DataFrame(
        {
            "fight_id": [f["fight_id"] for f in fight_records],
            "event_date": [f["event_date"] for f in fight_records],
            "y": y.astype(int),
            "elo_prob": np.asarray(elo_probs, dtype=float),
        }
    )
    for col in META_V22_FEATURE_COLUMNS[2:]:  # skip xgb_oof_prob + elo_prob
        idx = FEATURE_COLUMNS_V22.index(col)
        df[col] = X_v22[:, idx]
    return df


def _compute_input_distribution_hash(
    X: np.ndarray,
    y: np.ndarray,
    oof: np.ndarray,
) -> str:
    """sha256 over (X, y, oof) — invariant across parquet writer versions."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(X), allow_pickle=False)
    np.save(buf, np.asarray(y), allow_pickle=False)
    np.save(buf, np.asarray(oof), allow_pickle=False)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _compute_per_slice_metrics(
    model,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    fight_dates_eval: np.ndarray,
    *,
    today: date | None = None,
    random_seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Mirror evaluator.evaluate_per_slice mask construction.

    Returns Brier + accuracy + n per slice; no calibration curves (Wave 3 will
    compute calibration if needed). Output mirrors meta_v2_meta.json::v2_3
    slice_metrics shape so SUMMARY can show side-by-side.
    """
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

    today = today or date.today()
    cutoff_12mo = today - timedelta(days=365)
    cutoff_24mo = today - timedelta(days=730)
    mask_12mo = np.array([d >= cutoff_12mo for d in fight_dates_eval])
    mask_24mo = np.array([d >= cutoff_24mo for d in fight_dates_eval])
    rng = np.random.RandomState(random_seed)
    mask_random = rng.random(len(fight_dates_eval)) < 0.15

    out: dict[str, dict[str, float]] = {}
    masks = {
        "most_recent_12mo": mask_12mo,
        "most_recent_24mo": mask_24mo,
        "random_15pct": mask_random,
    }
    for slc, mask in masks.items():
        y_sl = y_eval[mask]
        X_sl = X_eval[mask]
        n = int(mask.sum())
        if n == 0:
            out[slc] = {
                "brier_score": None,
                "accuracy": None,
                "auc_roc": None,
                "n": 0,
            }
            continue
        proba = model.predict_proba(X_sl)[:, 1]
        brier = float(brier_score_loss(y_sl, proba))
        acc = float(accuracy_score(y_sl, (proba >= 0.5).astype(int)))
        if len(np.unique(y_sl)) > 1:
            auc = float(roc_auc_score(y_sl, proba))
        else:
            auc = None
        out[slc] = {
            "brier_score": brier,
            "accuracy": acc,
            "auc_roc": auc,
            "n": n,
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator — main()
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 45 meta_v3 candidate META blender trainer (META3-V25-02)",
    )
    parser.add_argument(
        "--oof-parquet",
        type=Path,
        required=True,
        help="Path to xgb_v3 OOF parquet (Plan 45-02 output: 45-XGB-V3-OOF-PREDICTIONS.parquet)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=META_DIR,
        help="Output directory for meta_v3 candidate triad",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SEEDS_DEFAULT),
        help="Random seeds for the 5-seed harness (default: 42-46)",
    )
    args = parser.parse_args(argv)

    print(f"[train_meta_v3_v25] Phase 45-03 META3-V25-02 — args: {vars(args)}")

    # ── AUDIT-01 pre-flight ──
    xgb_v2_sha, meta_v2_sha = _audit01_preflight()
    if not XGB_V3_PATH.is_file():
        print(
            f"[train_meta_v3_v25] FATAL: {XGB_V3_PATH} missing — Plan 45-02 dep",
            file=sys.stderr,
        )
        return 2
    xgb_v3_sha = _sha256_file(XGB_V3_PATH)
    print(f"[train_meta_v3_v25] xgb_v3 (Plan 45-02 base): {xgb_v3_sha[:12]}...")

    if not args.oof_parquet.is_file():
        print(
            f"[train_meta_v3_v25] FATAL: OOF parquet missing at {args.oof_parquet}",
            file=sys.stderr,
        )
        return 2
    oof_parquet_sha = _sha256_file(args.oof_parquet)
    print(f"[train_meta_v3_v25] OOF parquet SHA: {oof_parquet_sha[:12]}...")

    # ── Step 1: Load xgb_v3 OOF parquet ──
    oof_df = pd.read_parquet(args.oof_parquet)
    required_oof_cols = {
        "fight_id",
        "xgb_v3_oof_prob",
        "split",
        "train_or_test",
        "event_date",
    }
    missing = required_oof_cols - set(oof_df.columns)
    if missing:
        print(
            f"[train_meta_v3_v25] FATAL: OOF parquet missing cols: {missing}",
            file=sys.stderr,
        )
        return 2
    print(
        f"[train_meta_v3_v25] OOF parquet loaded: {len(oof_df)} rows, "
        f"NaN(xgb_v3_oof_prob)={int(oof_df['xgb_v3_oof_prob'].isna().sum())}, "
        f"train={int((oof_df['train_or_test'] == 'train').sum())}, "
        f"test={int((oof_df['train_or_test'] == 'test').sum())}"
    )

    # ── Step 2: Load Level-1 substrate from DB ──
    print("[train_meta_v3_v25] Loading v2.2 Level-1 substrate from DB...")
    level1_df = _load_level1_substrate_from_db()
    print(f"[train_meta_v3_v25] Level-1 substrate: {len(level1_df)} rows")

    # ── Step 3: Assemble Level-1 — train + eval partitions ──
    # The OOF parquet's train_or_test column already encodes the chronological
    # split at cutoff_date (Plan 45-02 invariant). Use train rows for meta
    # blender fit; use test rows for v2.3 widened-slice eval.
    train_oof = oof_df[oof_df["train_or_test"] == "train"].copy()
    test_oof = oof_df[oof_df["train_or_test"] == "test"].copy()

    X_meta_train, y_meta_train = assemble_level1_input(train_oof, level1_df)
    X_meta_eval, y_meta_eval = assemble_level1_input(test_oof, level1_df)
    print(
        f"[train_meta_v3_v25] Assembled: X_meta_train={X_meta_train.shape}, "
        f"X_meta_eval={X_meta_eval.shape}"
    )

    # ── Step 4: NaN handling — per_feature_strict_baseline (Plan 29-02 carry) ──
    # Baseline cols (xgb_oof_prob, elo_prob) MUST be non-NaN. Non-baseline cols
    # are imputed with train-set medians. Mirrors compose_v23_meta exactly.
    from ufc_prediction.ml.meta_features_v22 import META_V22_FEATURE_COLUMNS

    BASELINE_COLS = ("xgb_oof_prob", "elo_prob")
    baseline_idx = [i for i, c in enumerate(META_V22_FEATURE_COLUMNS) if c in BASELINE_COLS]
    non_baseline_idx = [i for i, c in enumerate(META_V22_FEATURE_COLUMNS) if c not in BASELINE_COLS]

    def _strict_baseline_mask(X: np.ndarray) -> np.ndarray:
        baseline_nan = np.isnan(X[:, baseline_idx]).any(axis=1)
        return ~baseline_nan

    train_keep = _strict_baseline_mask(X_meta_train)
    eval_keep = _strict_baseline_mask(X_meta_eval)
    n_train_dropped = int((~train_keep).sum())
    n_eval_dropped = int((~eval_keep).sum())
    print(
        f"[train_meta_v3_v25] strict-baseline NaN drop: "
        f"train dropped {n_train_dropped}/{len(train_keep)}, "
        f"eval dropped {n_eval_dropped}/{len(eval_keep)}"
    )
    X_meta_train_clean = X_meta_train[train_keep].copy()
    y_meta_train_clean = y_meta_train[train_keep]
    X_meta_eval_clean = X_meta_eval[eval_keep].copy()
    y_meta_eval_clean = y_meta_eval[eval_keep]

    # Align event_date for eval slicing
    merged_eval = test_oof[["fight_id", "event_date"]].merge(
        level1_df[["fight_id"]],
        on="fight_id",
        how="inner",
    )
    eval_dates = pd.to_datetime(
        merged_eval["event_date"].to_numpy(),
    ).date
    eval_dates_arr = np.array(list(eval_dates))[eval_keep]

    # Non-baseline column median imputation (train-medians applied to both)
    nan_imputation_medians: dict[str, float] = {}
    for idx in non_baseline_idx:
        col_train = X_meta_train_clean[:, idx]
        finite = col_train[~np.isnan(col_train)]
        median_val = float(np.median(finite)) if finite.size else 0.0
        nan_imputation_medians[META_V22_FEATURE_COLUMNS[idx]] = median_val
        tr_nan = np.isnan(X_meta_train_clean[:, idx])
        if tr_nan.any():
            X_meta_train_clean[tr_nan, idx] = median_val
        ev_nan = np.isnan(X_meta_eval_clean[:, idx])
        if ev_nan.any():
            X_meta_eval_clean[ev_nan, idx] = median_val

    # ── Step 5: 5-seed harness fit + evaluate ──
    from ufc_prediction.ml.meta_learner import MetaLearnerLogistic

    print(
        f"[train_meta_v3_v25] Fitting MetaLearnerLogistic × {len(args.seeds)} "
        f"seeds + evaluating on 3 v2.3 widened slices..."
    )
    per_seed_slice: dict[int, dict[str, dict[str, float]]] = {}
    per_seed_models: dict[int, MetaLearnerLogistic] = {}
    for seed in args.seeds:
        m = MetaLearnerLogistic(random_state=seed).fit(
            X_meta_train_clean,
            y_meta_train_clean,
        )
        per_seed_models[seed] = m
        per_seed_slice[seed] = _compute_per_slice_metrics(
            m,
            X_meta_eval_clean,
            y_meta_eval_clean,
            eval_dates_arr,
        )

    # ── Step 6: Median per-slice metrics across seeds ──
    median_per_slice: dict[str, dict[str, float]] = {}
    for slc in PER_SLICE_KEYS:
        for metric_key in ("brier_score", "accuracy", "auc_roc", "n"):
            vals = [
                per_seed_slice[seed][slc][metric_key]
                for seed in args.seeds
                if per_seed_slice[seed][slc][metric_key] is not None
            ]
            if not vals:
                median_per_slice.setdefault(slc, {})[metric_key] = None
                continue
            median_per_slice.setdefault(slc, {})[metric_key] = float(np.median(vals))

    # ── Step 7: Persist meta_v3 triad ──
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Ship the canonical model = the first seed (matches Phase 26 META-V22 ship
    # convention: `ship_pipeline = per_seed_meta[args.seeds[0]]`).
    canonical_seed = args.seeds[0]
    ship_model = per_seed_models[canonical_seed]

    meta_v3_joblib_path = args.out_dir / "meta_v3.joblib"
    import joblib

    joblib.dump(ship_model, meta_v3_joblib_path)

    input_dist_hash = _compute_input_distribution_hash(
        X_meta_train_clean,
        y_meta_train_clean,
        X_meta_train_clean[:, 0],
    )

    metadata = build_meta_v3_metadata(
        base_model_sha256=xgb_v3_sha,
        meta_oof_parquet_sha256=oof_parquet_sha,
        per_slice_metrics=median_per_slice,
        meta_input_distribution_hash=input_dist_hash,
    )
    # Enrich metadata with additional fields (mirrors meta_v2_meta.json shape)
    metadata["meta_learner_brier_delta_vs_logistic"] = 0.0
    metadata["per_seed_metrics"] = {str(seed): per_seed_slice[seed] for seed in args.seeds}
    metadata["seeds"] = list(args.seeds)
    metadata["canonical_seed"] = canonical_seed
    metadata["n_meta_train_rows"] = int(len(X_meta_train_clean))
    metadata["n_meta_eval_rows"] = int(len(X_meta_eval_clean))
    metadata["nan_drop_policy"] = "per_feature_strict_baseline"
    metadata["nan_imputation_medians"] = nan_imputation_medians
    metadata["trained_at"] = datetime.now(tz=UTC).isoformat()

    import sklearn

    metadata["sklearn_version"] = sklearn.__version__
    metadata["python_version"] = sys.version

    meta_v3_meta_path = args.out_dir / "meta_v3_meta.json"
    meta_v3_meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Contract sibling — partner-facing schema
    from ufc_prediction.ml.config import FEATURE_COLUMNS_V22
    from ufc_prediction.ml.meta_features_v22 import META_V22_FEATURE_COLUMNS

    meta_v3_artifact_sha = _sha256_file(meta_v3_joblib_path)
    meta_feature_columns_hash = hashlib.sha256(
        json.dumps(list(META_V22_FEATURE_COLUMNS), sort_keys=True).encode("utf-8"),
    ).hexdigest()
    feature_columns_hash = hashlib.sha256(
        "\n".join(FEATURE_COLUMNS_V22).encode("utf-8"),
    ).hexdigest()
    contract = {
        "schema_version": "1.3.0-candidate",
        "gate_contract_ref": ".planning/gate_contract_v2.3.json",
        "feature_columns_hash": feature_columns_hash,
        "meta_feature_columns_hash": meta_feature_columns_hash,
        "base_model_sha256": xgb_v3_sha,
        "model_artifact_sha256": meta_v3_artifact_sha,
        "sha256": meta_v3_artifact_sha,
        "parent_model": "xgb_v3",
        "min_partner_version_supported": "1.0.0",
        "deprecation_policy": "N >= 2 minor versions",
        "candidate_or_promoted": "candidate",
        "created_at": datetime.now(tz=UTC).date().isoformat(),
    }
    contract_path = args.out_dir / "meta_v3-contract.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    print(
        f"[train_meta_v3_v25] Persisted triad:\n"
        f"  - {meta_v3_joblib_path} (SHA {meta_v3_artifact_sha[:12]}...)\n"
        f"  - {meta_v3_meta_path}\n"
        f"  - {contract_path}"
    )

    # ── Step 8: Print summary side-by-side w/ META-V22 baseline ──
    # META-V22 baseline per-slice Brier from gate_contract_v2.3.json /
    # meta_v2_meta.json::metrics.per_slice_median (D-CONTEXT § "baseline 0.2131").
    META_V22_BASELINE_BRIER = {
        "most_recent_12mo": 0.21305929305714250,
        "most_recent_24mo": 0.21305929305714250,
        "random_15pct": 0.18669529738223767,
    }
    print("\n=== meta_v3 candidate vs META-V22 baseline (per-slice median) ===")
    print(
        f"{'slice':<22} | {'v3 Brier':>10} | {'v22 Brier':>10} | {'Δ Brier':>10} "
        f"| {'v3 Acc':>8} | {'n':>6}"
    )
    for slc in PER_SLICE_KEYS:
        v3_b = median_per_slice[slc]["brier_score"]
        v3_a = median_per_slice[slc]["accuracy"]
        v3_n = median_per_slice[slc]["n"]
        v22_b = META_V22_BASELINE_BRIER[slc]
        delta = None if v3_b is None else v22_b - v3_b
        v3_b_s = "n/a" if v3_b is None else f"{v3_b:.4f}"
        v3_a_s = "n/a" if v3_a is None else f"{v3_a:.4f}"
        delta_s = "n/a" if delta is None else f"{delta:+.4f}"
        n_s = "n/a" if v3_n is None else f"{int(v3_n)}"
        print(f"{slc:<22} | {v3_b_s:>10} | {v22_b:>10.4f} | {delta_s:>10} | {v3_a_s:>8} | {n_s:>6}")

    # ── AUDIT-01 post-flight ──
    _audit01_postflight(xgb_v2_sha, meta_v2_sha)
    print("[train_meta_v3_v25] DONE — meta_v3 candidate ready for Wave 3 gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
