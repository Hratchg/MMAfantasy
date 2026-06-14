#!/usr/bin/env python
"""FALLBACK-V24-01 — Train xgb_v2_no_odds (67-col ablation of xgb_v2; drops 5 odds-derived columns).

SAME training data + cutoff as xgb_v2 (read from models/xgb_v2_meta.json).
NEVER routes through ModelTrainer.train() — direct XGBClassifier.fit() + CalibratedClassifierCV
(sigmoid) wrap, mirroring scripts/retrain_xgb_v3.py:_train_with_fixed_params. We persist via a
direct joblib.dump() to ``models/xgb_v2_no_odds.joblib`` to prevent any chance of overwriting
``models/xgb_v2.joblib`` (AUDIT-01 byte-identity protected, per RESEARCH Pitfall 6).

Per CONTEXT D-FALLBACK-V24-01 + PATTERNS.md xgb_v2_no_odds training section.

Output artifacts:
    models/xgb_v2_no_odds.joblib          (CalibratedClassifierCV)
    models/xgb_v2_no_odds_meta.json       (metadata + per_slice_metrics)
    models/xgb_v2_no_odds-contract.json   (partner-facing contract — promoted)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
import xgboost
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

from ufc_prediction.db.session import SessionLocal
from ufc_prediction.ml.config import FEATURE_COLUMNS, FEATURE_COLUMNS_NO_NET, MLConfig
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


EXPECTED_XGB_V2_SHA = "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
ODDS_COLS = {
    "opening_prob_diff",
    "closing_prob_diff",
    "line_movement_diff",
    "sharp_money_signal",
    "odds_elo_divergence",
}
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
XGB_V2_PATH = MODELS_DIR / "xgb_v2.joblib"
XGB_V2_META_PATH = MODELS_DIR / "xgb_v2_meta.json"
XGB_V2_CONTRACT_PATH = MODELS_DIR / "xgb_v2-contract.json"
FALLBACK_JOBLIB = MODELS_DIR / "xgb_v2_no_odds.joblib"
FALLBACK_META = MODELS_DIR / "xgb_v2_no_odds_meta.json"
FALLBACK_CONTRACT = MODELS_DIR / "xgb_v2_no_odds-contract.json"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _train_with_fixed_params(
    X_train: np.ndarray, y_train: np.ndarray, best_params: dict, seed: int,
) -> CalibratedClassifierCV:
    """Single-seed train mirroring retrain_xgb_v3.py:_train_with_fixed_params.

    80/20 chronological train_proper/calibration_holdout split, XGBClassifier
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


def main() -> int:
    print("[no-odds] AUDIT-01 SHA pre-flight check...")
    sha_before = _sha256_file(XGB_V2_PATH)
    if sha_before != EXPECTED_XGB_V2_SHA:
        print(f"[no-odds] FATAL: xgb_v2 SHA drift BEFORE train: {sha_before}", file=sys.stderr)
        return 2
    print(f"  xgb_v2 SHA OK: {sha_before}")

    print("[no-odds] Loading xgb_v2 meta for cutoff + best_params inheritance...")
    xgb_v2_meta = json.loads(XGB_V2_META_PATH.read_text())
    best_params = xgb_v2_meta["best_params"]
    cutoff_str: str = xgb_v2_meta["cutoff_date"]
    cutoff_date_obj: date = date.fromisoformat(cutoff_str)
    print(f"  cutoff_date: {cutoff_str}")
    print(f"  best_params: {best_params}")

    print("[no-odds] Loading data from DB...")
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

    print("[no-odds] Computing division medians (training-set only)...")
    division_medians = compute_division_medians(
        fighter_physicals, fight_records, cutoff_date_obj,
    )

    print("[no-odds] Assembling 75-col feature matrix (NET-* included)...")
    config = MLConfig(cutoff_date=cutoff_str)
    assembler = FeatureMatrixAssembler(config)
    X, y, fight_dates_full = assembler.assemble(
        fight_records, elo_features, computed_features,
        fighter_physicals, division_medians, round_stats,
        pre_ufc_records=pre_ufc,
        fight_odds=fight_odds,
    )
    print(f"  X.shape = {X.shape}, y.shape = {y.shape}")

    if X.shape[1] != len(FEATURE_COLUMNS):
        msg = (
            f"FEATURE_COLUMNS drift: assembler produced {X.shape[1]} cols, "
            f"FEATURE_COLUMNS has {len(FEATURE_COLUMNS)}"
        )
        print(f"[no-odds] FATAL: {msg}", file=sys.stderr)
        return 2

    print("[no-odds] Temporal split at cutoff_date...")
    X_train, X_test, y_train, y_test = split_temporal(
        X, y, fight_dates_full, cutoff_date_obj,
    )
    test_mask = np.array([d >= cutoff_date_obj for d in fight_dates_full])
    fight_dates_test = np.array(fight_dates_full)[test_mask]
    print(f"  Train: {X_train.shape[0]} fights, Test: {X_test.shape[0]} fights")

    print("[no-odds] Dropping NET-* tail and 5 odds-derived columns to form 67-col matrix...")
    # First drop NET-* tail to mirror xgb_v2's FEATURE_COLUMNS_NO_NET view.
    X_train_no_net = X_train[:, : len(FEATURE_COLUMNS_NO_NET)]
    X_test_no_net = X_test[:, : len(FEATURE_COLUMNS_NO_NET)]
    full_cols = list(FEATURE_COLUMNS_NO_NET)  # 72
    no_odds_cols = [c for c in full_cols if c not in ODDS_COLS]
    assert len(no_odds_cols) == 67, f"Expected 67 no-odds cols, got {len(no_odds_cols)}"
    keep_idx = [full_cols.index(c) for c in no_odds_cols]
    X_train_no = X_train_no_net[:, keep_idx]
    X_test_no = X_test_no_net[:, keep_idx]
    print(f"  no-odds train shape: {X_train_no.shape}, test shape: {X_test_no.shape}")

    print(f"[no-odds] Training calibrated XGBoost (seed={SEED})...")
    t_train = time.time()
    model = _train_with_fixed_params(X_train_no, y_train, best_params, SEED)
    print(f"  trained in {time.time() - t_train:.1f}s")

    print("[no-odds] Evaluating on 3 v2.3 widened slices...")
    per_slice = evaluate_per_slice(
        model, X_test_no, y_test, fight_dates_test,
        today=date.today(), random_seed=42,
    )
    for slice_name, metrics in per_slice.items():
        print(
            f"  {slice_name}: brier={metrics['brier_score']:.4f} "
            f"acc={metrics['accuracy']:.4f} auc={metrics['auc_roc']:.4f} "
            f"n_obs={metrics.get('n', 'n/a')}"
        )

    print(f"[no-odds] Persisting model to {FALLBACK_JOBLIB}...")
    joblib.dump(model, FALLBACK_JOBLIB)
    fallback_sha = _sha256_file(FALLBACK_JOBLIB)
    print(f"  xgb_v2_no_odds.joblib SHA: {fallback_sha}")

    # Per-slice metrics, serializable form
    per_slice_serializable: dict = {}
    for slice_name, m in per_slice.items():
        per_slice_serializable[slice_name] = {
            "brier_score": float(m["brier_score"]),
            "auc_roc": float(m["auc_roc"]),
            "accuracy": float(m["accuracy"]),
            "calibration_curve": m["calibration_curve"],
        }

    print(f"[no-odds] Writing meta JSON to {FALLBACK_META}...")
    meta_dict = {
        "model_name": "xgb_v2_no_odds",
        "version": "v2_no_odds",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "trained_at": datetime.now(tz=UTC).isoformat(),
        "xgboost_version": xgboost.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version,
        "cutoff_date": cutoff_str,
        "best_params": best_params,
        "feature_columns": no_odds_cols,
        "n_features": 67,
        "n_training_fights": int(X_train_no.shape[0]),
        "n_test_fights": int(X_test_no.shape[0]),
        "train_seed": SEED,
        "parent_model": "xgb_v2",
        "parent_model_sha256": EXPECTED_XGB_V2_SHA,
        "dropped_columns": sorted(ODDS_COLS),
        "per_slice_metrics": per_slice_serializable,
    }
    FALLBACK_META.write_text(json.dumps(meta_dict, indent=2))

    print(f"[no-odds] Writing contract JSON to {FALLBACK_CONTRACT}...")
    parent_contract = json.loads(XGB_V2_CONTRACT_PATH.read_text())
    contract_dict = {
        "schema_version": parent_contract.get("schema_version", "1.0.0"),
        "model_name": "xgb_v2_no_odds",
        "parent_model": "xgb_v2",
        "gate_contract_ref": ".planning/gate_contract_v2.3.json",
        "feature_columns_hash": hashlib.sha256(
            json.dumps(no_odds_cols, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "min_partner_version_supported": parent_contract.get(
            "min_partner_version_supported", "1.0.0"
        ),
        "deprecation_policy": parent_contract.get(
            "deprecation_policy", "N >= 2 minor versions"
        ),
        "model_artifact_sha256": fallback_sha,
        "candidate_or_promoted": "promoted",
        "created_at": date.today().isoformat(),
    }
    FALLBACK_CONTRACT.write_text(json.dumps(contract_dict, indent=2))

    print("[no-odds] AUDIT-01 SHA post-flight check (xgb_v2.joblib byte-identity)...")
    sha_after = _sha256_file(XGB_V2_PATH)
    if sha_after != EXPECTED_XGB_V2_SHA:
        print(f"[no-odds] FATAL: AUDIT-01 BROKEN during train: {sha_after}", file=sys.stderr)
        return 2
    print(f"  xgb_v2 SHA OK (unchanged): {sha_after}")

    # Defensive: confirm meta_v2.joblib also unchanged (script never touches it)
    meta_v2_path = MODELS_DIR / "meta" / "meta_v2.joblib"
    if meta_v2_path.exists():
        meta_sha = _sha256_file(meta_v2_path)
        anchor_path = (
            PROJECT_ROOT
            / ".planning/phases/34-trust-baseline-no-odds-fallback"
            / "34-META-V2-SHA-PHASE-34-START-PLAN-01.txt"
        )
        if anchor_path.exists():
            expected_meta = anchor_path.read_text().strip()
            if meta_sha != expected_meta:
                print(
                    f"[no-odds] FATAL: meta_v2 SHA drift: {meta_sha} vs anchor {expected_meta}",
                    file=sys.stderr,
                )
                return 2
            print(f"  meta_v2 SHA OK (matches START anchor): {meta_sha}")

    print("[no-odds] DONE. All 3 artifacts written; AUDIT-01 byte-identity preserved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
