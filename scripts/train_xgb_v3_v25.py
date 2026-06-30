#!/usr/bin/env python
"""xgb_v3 training driver — Phase 45 META3-V25-01.

Per 45-CONTEXT.md <decisions>:
  - Training cutoff: match xgb_v2 cutoff verbatim (2023-01-01).
  - Feature columns: FEATURE_COLUMNS_NO_NET (72 cols — same as xgb_v2).
  - Hyperparameters: verbatim from models/xgb_v2_meta.json::best_params
    (no Optuna re-run; xgb_v3 hyperparameter tuning is v2.6+ scope).
  - 5-seed harness: [42, 43, 44, 45, 46]. Median (not mean) wins.
  - AUDIT-01: xgb_v2 + meta_v2 NOT mutated; this is a candidate sibling.
  - Substrate: v2.5 (post-Phase 41 BFO 60.12% + post-Phase 43 800 seeded
    debutant Elos).

Persists:
  - models/xgb_v3.joblib          (5-seed median-by-Brier model)
  - models/xgb_v3_meta.json       (per-seed + median metrics)
  - models/xgb_v3-contract.json   (partner-facing contract)
  - .planning/phases/45-meta-v3-candidate-retrain/45-XGB-V3-OOF-PREDICTIONS.parquet
                                  (per-fight OOF — train rows from TimeSeriesSplit,
                                   test rows from 5-seed median test predictions)

NOTE: A Phase 16-era scripts/retrain_xgb_v3.py exists with stale gate
thresholds (Brier ≤ 0.215, Acc ≥ 0.67 — wrong v2.5 binding). This script is
a NEW Phase 45-specific driver to avoid mutating that legacy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

# ────────────────────────────────────────────────────────────────────────
# Public API — these four functions are unit-tested in
# tests/unit/ml/test_train_xgb_v3_v25.py. They are defined BEFORE any
# heavy DB / ML imports so the test file can importlib-load this module
# without needing a populated SQLite database.
# ────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
XGB_V2_PATH = MODELS_DIR / "xgb_v2.joblib"
XGB_V2_META_PATH = MODELS_DIR / "xgb_v2_meta.json"
XGB_V2_CONTRACT_PATH = MODELS_DIR / "xgb_v2-contract.json"
META_V2_PATH = MODELS_DIR / "meta" / "meta_v2.joblib"
XGB_V3_PATH = MODELS_DIR / "xgb_v3.joblib"
XGB_V3_META_PATH = MODELS_DIR / "xgb_v3_meta.json"
XGB_V3_CONTRACT_PATH = MODELS_DIR / "xgb_v3-contract.json"

PHASE_DIR = PROJECT_ROOT / ".planning/phases/45-meta-v3-candidate-retrain"
OOF_PARQUET_PATH = PHASE_DIR / "45-XGB-V3-OOF-PREDICTIONS.parquet"
SHA_MID_XGB_V2_PATH = PHASE_DIR / "45-XGB-V2-SHA-PHASE-45-MID.txt"
SHA_MID_META_V2_PATH = PHASE_DIR / "45-META-V2-SHA-PHASE-45-MID.txt"

# AUDIT-01 invariants (PROJECT.md cross-cutting invariants #1 + #2).
EXPECTED_XGB_V2_SHA = "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
EXPECTED_META_V2_SHA = "77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196"

# 5-seed harness per D-CONTEXT §Training-Strategy (same as xgb_v2).
SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)


def load_xgb_v2_training_config() -> dict:
    """Load xgb_v2_meta.json + parse cutoff_date, feature_columns, best_params.

    Returns:
        dict with keys:
          - cutoff_date: date (parsed from ISO string)
          - feature_columns: list[str] (== FEATURE_COLUMNS_NO_NET verbatim)
          - best_params: dict (verbatim XGBoost hyperparameter dict from
            xgb_v2_meta.json::best_params)
    """
    # Lazy import (config.py is light; this keeps the importlib-load fast).
    from ufc_prediction.ml.config import FEATURE_COLUMNS_NO_NET

    xgb_v2_meta = json.loads(XGB_V2_META_PATH.read_text(encoding="utf-8"))
    return {
        "cutoff_date": date.fromisoformat(xgb_v2_meta["cutoff_date"]),
        "feature_columns": list(FEATURE_COLUMNS_NO_NET),
        "best_params": xgb_v2_meta["best_params"],
    }


def assert_apples_to_apples(config: dict) -> None:
    """Raise AssertionError if config diverges from xgb_v2's reference.

    Checks:
      - cutoff_date == date(2023, 1, 1)
      - feature_columns == FEATURE_COLUMNS_NO_NET (verbatim, ordered)
      - set(best_params.keys()) == set(xgb_v2_meta.best_params.keys())
    """
    from ufc_prediction.ml.config import FEATURE_COLUMNS_NO_NET

    xgb_v2_meta = json.loads(XGB_V2_META_PATH.read_text(encoding="utf-8"))
    expected_cutoff = date.fromisoformat(xgb_v2_meta["cutoff_date"])

    actual_cutoff = config["cutoff_date"]
    assert actual_cutoff == expected_cutoff, (
        f"cutoff_date drift: {actual_cutoff} vs xgb_v2 {expected_cutoff}"
    )

    expected_cols = list(FEATURE_COLUMNS_NO_NET)
    actual_cols = list(config["feature_columns"])
    assert actual_cols == expected_cols, (
        f"feature_columns drift from FEATURE_COLUMNS_NO_NET "
        f"(len actual={len(actual_cols)} vs expected={len(expected_cols)})"
    )

    expected_bp_keys = set(xgb_v2_meta["best_params"].keys())
    actual_bp_keys = set(config["best_params"].keys())
    assert actual_bp_keys == expected_bp_keys, (
        f"best_params keyset drift: actual={sorted(actual_bp_keys)} "
        f"vs xgb_v2={sorted(expected_bp_keys)}"
    )


def build_seed_list() -> list[int]:
    """Return the 5-seed harness used by xgb_v2 (D-CONTEXT §Training-Strategy)."""
    return [42, 43, 44, 45, 46]


def debutant_elo_is_seeded(fight_record: dict) -> bool:
    """Pre-fight Elo seeded-check for Phase 43 backfill.

    For a debutant (`n_ufc_fights` == 0 at first UFC appearance):
      - Returns True if `elo_overall_pre` != 1500.0 (Phase 43 Sherdog seed).
      - Returns False if `elo_overall_pre` == 1500.0 (un-seeded default).

    For non-debutants (`n_ufc_fights` > 0): returns True unconditionally
    (no-op; debutant seeding only applies to first UFC fight).
    """
    n_ufc = int(fight_record.get("n_ufc_fights", 0))
    elo_pre = float(fight_record.get("elo_overall_pre", 1500.0))
    if n_ufc > 0:
        return True
    # Debutant: seeded iff elo_pre is NOT the default 1500.0.
    return not (abs(elo_pre - 1500.0) < 1e-9)


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit01_preflight() -> None:
    """Refuse to start training if xgb_v2 or meta_v2 SHAs drift from
    PROJECT.md cross-cutting invariants. Belt-and-suspenders against
    accidental retraining."""
    sha_xgb_v2 = _sha256_file(XGB_V2_PATH)
    if sha_xgb_v2 != EXPECTED_XGB_V2_SHA:
        msg = (
            f"AUDIT-01 PRE-FLIGHT FAIL: xgb_v2 SHA = {sha_xgb_v2} "
            f"!= invariant {EXPECTED_XGB_V2_SHA}"
        )
        raise SystemExit(msg)
    if META_V2_PATH.exists():
        sha_meta_v2 = _sha256_file(META_V2_PATH)
        if sha_meta_v2 != EXPECTED_META_V2_SHA:
            msg = (
                f"AUDIT-01 PRE-FLIGHT FAIL: meta_v2 SHA = {sha_meta_v2} "
                f"!= invariant {EXPECTED_META_V2_SHA}"
            )
            raise SystemExit(msg)


def _audit01_postflight() -> None:
    """Confirm xgb_v2 + meta_v2 SHAs unchanged after training. Write MID
    anchor files for Task 3 cross-check."""
    sha_xgb_v2 = _sha256_file(XGB_V2_PATH)
    if sha_xgb_v2 != EXPECTED_XGB_V2_SHA:
        msg = (
            f"AUDIT-01 POST-FLIGHT FAIL: xgb_v2 SHA DRIFTED to {sha_xgb_v2} "
            f"(expected {EXPECTED_XGB_V2_SHA})"
        )
        raise SystemExit(msg)
    sha_meta_v2 = _sha256_file(META_V2_PATH)
    if sha_meta_v2 != EXPECTED_META_V2_SHA:
        msg = (
            f"AUDIT-01 POST-FLIGHT FAIL: meta_v2 SHA DRIFTED to {sha_meta_v2} "
            f"(expected {EXPECTED_META_V2_SHA})"
        )
        raise SystemExit(msg)
    SHA_MID_XGB_V2_PATH.write_text(sha_xgb_v2 + "\n", encoding="utf-8")
    SHA_MID_META_V2_PATH.write_text(sha_meta_v2 + "\n", encoding="utf-8")
    print("[xgb_v3] AUDIT-01 MID anchors written:")
    print(f"  xgb_v2:  {sha_xgb_v2}")
    print(f"  meta_v2: {sha_meta_v2}")


def _train_with_fixed_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_params: dict,
    seed: int,
) -> CalibratedClassifierCV:
    """Single-seed train mirroring scripts/retrain_xgb_v3.py::_train_with_fixed_params.

    80/20 chronological train_proper / calibration_holdout split, XGBClassifier
    fit on train_proper, CalibratedClassifierCV(sigmoid) on FrozenEstimator base.
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


def _brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def _accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(((y_prob >= 0.5).astype(int) == y_true).mean())


# ────────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="xgb_v3 candidate training (Phase 45 META3-V25-01)"
    )
    parser.add_argument(
        "--out-dir",
        default=str(MODELS_DIR),
        help="Output dir for xgb_v3 triad (default: models/)",
    )
    parser.add_argument(
        "--oof-parquet",
        default=str(OOF_PARQUET_PATH),
        help="Path to OOF predictions parquet (default: phase dir)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    oof_path = Path(args.oof_parquet)
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_path.parent.mkdir(parents=True, exist_ok=True)

    # ─── Step 1: AUDIT-01 pre-flight ──────────────────────────────────
    print("[xgb_v3] AUDIT-01 pre-flight check...")
    _audit01_preflight()
    print(f"  xgb_v2 SHA OK: {EXPECTED_XGB_V2_SHA}")
    print(f"  meta_v2 SHA OK: {EXPECTED_META_V2_SHA}")

    # ─── Step 2: Load config + assert apples-to-apples ────────────────
    print("[xgb_v3] Loading xgb_v2 training config (cutoff + features + best_params)...")
    config = load_xgb_v2_training_config()
    assert_apples_to_apples(config)
    cutoff_date_obj: date = config["cutoff_date"]
    cutoff_str: str = cutoff_date_obj.isoformat()
    feature_columns: list[str] = config["feature_columns"]
    best_params: dict = config["best_params"]
    print(f"  cutoff_date: {cutoff_str}")
    print(f"  feature_columns: {len(feature_columns)} cols (FEATURE_COLUMNS_NO_NET)")
    print(f"  best_params: {best_params}")

    # ─── Step 3: Load data + assemble feature matrix ──────────────────
    # Heavy imports deferred until after the public-API tests can pass.
    from ufc_prediction.db.session import SessionLocal
    from ufc_prediction.ml.config import FEATURE_COLUMNS, MLConfig
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

    print("[xgb_v3] Loading data from DB (v2.5 substrate)...")
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
        f"  Loaded {len(fight_records)} fights / {len(fight_odds)} odds rows "
        f"in {time.time() - t0:.1f}s"
    )

    print("[xgb_v3] Computing division medians (training-set only)...")
    division_medians = compute_division_medians(
        fighter_physicals,
        fight_records,
        cutoff_date_obj,
    )

    print("[xgb_v3] Assembling full feature matrix (75 cols; NET-* included)...")
    ml_config = MLConfig(cutoff_date=cutoff_str)
    assembler = FeatureMatrixAssembler(ml_config)
    X_full, y, fight_dates_full = assembler.assemble(
        fight_records,
        elo_features,
        computed_features,
        fighter_physicals,
        division_medians,
        round_stats,
        pre_ufc_records=pre_ufc,
        fight_odds=fight_odds,
    )
    print(f"  X_full.shape = {X_full.shape}, y.shape = {y.shape}")
    if X_full.shape[1] != len(FEATURE_COLUMNS):
        msg = (
            f"FEATURE_COLUMNS drift: assembler produced {X_full.shape[1]} cols, "
            f"FEATURE_COLUMNS has {len(FEATURE_COLUMNS)}"
        )
        print(f"[xgb_v3] FATAL: {msg}", file=sys.stderr)
        return 2

    # Drop NET-* tail to form 72-col FEATURE_COLUMNS_NO_NET view.
    X = X_full[:, : len(feature_columns)]
    print(f"  X (no-NET).shape = {X.shape}")

    # Build per-row fight_id lookup (must match assembler row order).
    fight_ids_full: list[int] = [int(f["fight_id"]) for f in fight_records]
    # Defensive: fight_records → assemble preserves order (CONTEXT D-04 of P15
    # asserts no row drops). Verify length.
    if len(fight_ids_full) != X.shape[0]:
        msg = f"row-count mismatch: fight_records={len(fight_ids_full)} vs X.shape[0]={X.shape[0]}"
        print(f"[xgb_v3] FATAL: {msg}", file=sys.stderr)
        return 2

    # ─── Step 4: Temporal split at cutoff ─────────────────────────────
    print("[xgb_v3] Temporal split at cutoff_date...")
    X_train, X_test, y_train, y_test = split_temporal(
        X,
        y,
        fight_dates_full,
        cutoff_date_obj,
    )
    train_mask = np.array([d < cutoff_date_obj for d in fight_dates_full])
    test_mask = np.array([d >= cutoff_date_obj for d in fight_dates_full])
    fight_ids_train = [fid for fid, m in zip(fight_ids_full, train_mask) if m]
    fight_ids_test = [fid for fid, m in zip(fight_ids_full, test_mask) if m]
    fight_dates_train = np.array(fight_dates_full)[train_mask]
    fight_dates_test = np.array(fight_dates_full)[test_mask]
    print(f"  Train: {X_train.shape[0]} fights, Test: {X_test.shape[0]} fights")

    # ─── Step 5: 5-seed candidate training ────────────────────────────
    seeds = build_seed_list()
    print(f"[xgb_v3] Training {len(seeds)} candidates with shared best_params...")
    per_seed_models: list[CalibratedClassifierCV] = []
    per_seed_metrics: list[dict] = []
    per_seed_test_probs: list[np.ndarray] = []

    for i, seed in enumerate(seeds):
        t_seed = time.time()
        print(f"[xgb_v3] Seed {seed} ({i + 1}/{len(seeds)}) — training...")
        model = _train_with_fixed_params(X_train, y_train, best_params, seed)
        per_seed_models.append(model)

        # Overall test-set metrics (Brier + acc + auc) per seed.
        test_prob = model.predict_proba(X_test)[:, 1]
        per_seed_test_probs.append(test_prob)
        brier_seed = _brier(y_test, test_prob)
        acc_seed = _accuracy(y_test, test_prob)
        per_seed_metrics.append(
            {
                "seed": int(seed),
                "brier_score": brier_seed,
                "accuracy": acc_seed,
            }
        )
        elapsed = time.time() - t_seed
        print(f"  seed={seed} done in {elapsed:.1f}s — brier={brier_seed:.4f} acc={acc_seed:.4f}")

    # ─── Step 6: 5-seed median predictions for OOF parquet + canon model ──
    # Per Plan: 5-seed median is the canonical artifact. For the persisted
    # joblib, pick the seed whose test-set Brier is closest to the median Brier
    # (this avoids needing a model-ensemble wrapper). The OOF parquet stores
    # the per-row 5-seed median probability.
    test_probs_stack = np.stack(per_seed_test_probs, axis=0)  # (5, n_test)
    test_prob_median = np.median(test_probs_stack, axis=0)
    median_brier = _brier(y_test, test_prob_median)
    median_acc = _accuracy(y_test, test_prob_median)

    seed_briers = np.array([m["brier_score"] for m in per_seed_metrics])
    canon_seed_idx = int(np.argmin(np.abs(seed_briers - median_brier)))
    canon_seed = seeds[canon_seed_idx]
    canon_model = per_seed_models[canon_seed_idx]
    print(
        f"[xgb_v3] 5-seed median test Brier = {median_brier:.4f}, "
        f"Acc = {median_acc:.4f} (canonical seed = {canon_seed})"
    )

    # OOF train predictions: TimeSeriesSplit per seed → median across seeds.
    print("[xgb_v3] Generating OOF train predictions (TimeSeriesSplit, 5 splits per seed)...")
    from sklearn.model_selection import TimeSeriesSplit

    t_oof = time.time()
    # Pre-sort train data chronologically once (TimeSeriesSplit requires this).
    sort_idx = np.argsort(fight_dates_train)
    X_train_sorted = X_train[sort_idx]
    y_train_sorted = y_train[sort_idx]
    fight_ids_train_sorted = [fight_ids_train[i] for i in sort_idx]
    fight_dates_train_sorted = fight_dates_train[sort_idx]

    n_train = len(y_train_sorted)
    oof_probs_per_seed: list[np.ndarray] = []
    n_splits = 5
    for seed in seeds:
        oof_seed = np.full(n_train, np.nan)
        cv = TimeSeriesSplit(n_splits=n_splits)
        for tr_idx, te_idx in cv.split(X_train_sorted):
            base = XGBClassifier(
                **best_params,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=int(seed),
                verbosity=0,
            )
            base.fit(X_train_sorted[tr_idx], y_train_sorted[tr_idx])
            oof_seed[te_idx] = base.predict_proba(X_train_sorted[te_idx])[:, 1]
        oof_probs_per_seed.append(oof_seed)
        print(
            f"  OOF seed={seed}: {int(np.isnan(oof_seed).sum())} warm-up NaNs, "
            f"{int((~np.isnan(oof_seed)).sum())} predicted"
        )
    oof_probs_stack = np.stack(oof_probs_per_seed, axis=0)  # (5, n_train)
    # NaN-aware median: median over the 5 seeds per row (all 5 NaN ⇒ NaN).
    oof_probs_train_median = np.nanmedian(oof_probs_stack, axis=0)
    print(f"  OOF generation: {time.time() - t_oof:.1f}s")

    # Build the parquet: train rows (OOF median) + test rows (5-seed test median).
    oof_df = pd.DataFrame(
        {
            "fight_id": fight_ids_train_sorted + fight_ids_test,
            "xgb_v3_oof_prob": np.concatenate(
                [
                    oof_probs_train_median,
                    test_prob_median,
                ]
            ),
            "split": (["train"] * n_train + ["test"] * len(fight_ids_test)),
            "train_or_test": (["train"] * n_train + ["test"] * len(fight_ids_test)),
            "event_date": (
                [str(d) for d in fight_dates_train_sorted] + [str(d) for d in fight_dates_test]
            ),
        }
    )
    print(f"[xgb_v3] Writing OOF parquet ({len(oof_df)} rows) to {oof_path}...")
    oof_df.to_parquet(oof_path, index=False)

    # ─── Step 7: Persist canonical xgb_v3.joblib + meta + contract ────
    xgb_v3_path = out_dir / "xgb_v3.joblib"
    print(f"[xgb_v3] Persisting canonical model to {xgb_v3_path}...")
    joblib.dump(canon_model, xgb_v3_path)
    xgb_v3_sha = _sha256_file(xgb_v3_path)
    print(f"  xgb_v3.joblib SHA-256: {xgb_v3_sha}")

    # Median metrics dict (mirrors xgb_v2_meta shape).
    median_metrics_dict = {
        "brier_score": median_brier,
        "accuracy": median_acc,
    }
    # AUC-ROC of median is a single scalar over median probs.
    try:
        from sklearn.metrics import roc_auc_score

        median_metrics_dict["auc_roc"] = float(roc_auc_score(y_test, test_prob_median))
    except Exception:
        median_metrics_dict["auc_roc"] = float("nan")

    xgb_v3_meta = {
        "model_name": "xgb_v3",
        "version": "v3",
        "trained_at": datetime.now(tz=UTC).isoformat(),
        "xgboost_version": xgboost.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version,
        "cutoff_date": cutoff_str,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "best_params": best_params,
        "seeds": list(seeds),
        "canonical_seed": int(canon_seed),
        "per_seed_metrics": per_seed_metrics,
        "metrics": median_metrics_dict,
        "n_training_fights": int(X_train.shape[0]),
        "n_test_fights": int(X_test.shape[0]),
        "base_model_sha256": xgb_v3_sha,
        "parent_model": "xgb_v2",
        "parent_model_sha256": EXPECTED_XGB_V2_SHA,
        "substrate": "v2.5 (post-Phase 41 BFO 60.12% + post-Phase 43 800 seeded debutants)",
        "phase": "45",
        "requirement": "META3-V25-01",
    }
    meta_path = out_dir / "xgb_v3_meta.json"
    meta_path.write_text(json.dumps(xgb_v3_meta, indent=2), encoding="utf-8")
    print(f"  xgb_v3_meta.json written to {meta_path}")

    # Contract JSON (mirrors xgb_v2-contract.json shape).
    parent_contract = json.loads(XGB_V2_CONTRACT_PATH.read_text(encoding="utf-8"))
    feature_cols_hash = hashlib.sha256(
        json.dumps(feature_columns, separators=(",", ":")).encode("utf-8"),
    ).hexdigest()
    xgb_v3_contract = {
        "schema_version": parent_contract.get("schema_version", "1.0.0"),
        "model_name": "xgb_v3",
        "model_version": "v3",
        "parent_model": "xgb_v2",
        "gate_contract_ref": ".planning/gate_contract_v2.3.json",
        "feature_columns_hash": feature_cols_hash,
        "n_features": len(feature_columns),
        "min_partner_version_supported": parent_contract.get(
            "min_partner_version_supported",
            "1.0.0",
        ),
        "deprecation_policy": parent_contract.get(
            "deprecation_policy",
            "N >= 2 minor versions",
        ),
        "model_artifact_sha256": xgb_v3_sha,
        "sha256": xgb_v3_sha,  # alias for plan verification command
        "candidate_or_promoted": "candidate",
        "created_at": date.today().isoformat(),
        "trained_at": xgb_v3_meta["trained_at"],
    }
    contract_path = out_dir / "xgb_v3-contract.json"
    contract_path.write_text(json.dumps(xgb_v3_contract, indent=2), encoding="utf-8")
    print(f"  xgb_v3-contract.json written to {contract_path}")

    # ─── Step 8: AUDIT-01 post-flight ─────────────────────────────────
    print("[xgb_v3] AUDIT-01 post-flight check + MID anchor write...")
    _audit01_postflight()

    print(
        f"\n[xgb_v3] DONE. 5 seeds trained. Median Brier={median_brier:.4f}, "
        f"Acc={median_acc:.4f}, n_train={X_train.shape[0]}, "
        f"n_test={X_test.shape[0]}, n_oof_rows={len(oof_df)}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
