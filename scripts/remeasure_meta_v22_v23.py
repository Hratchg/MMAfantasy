#!/usr/bin/env python
"""TRUST-V24-02 — Inference-only re-measurement of META-V22 on v2.3 widened slices.

Per CONTEXT D-TRUST-V24-02 (append to meta_v2_meta.json; preserve v2.2-era metrics for
AUDIT-01 lineage) + RESEARCH Open Q2 (re-derive xgb_oof_prob via xgb_v2.predict_proba on
TEST rows — no leakage since these are out-of-fold rows from xgb_v2's perspective).

Side note (revealed during execution): the post-cutoff Event.source='ufcstats' filter +
BFO `closing_implied_prob` join produces 99.5% NaN on test-side `closing_prob_diff` — a
known pre-existing data architecture gap. The re-measurement honestly reports n_after_nan_drop
per slice; the resulting Brier is computed on the small subset of post-cutoff fights with
non-NaN Level-1 features.

This script appends `v2_3_slice_metrics` to `models/meta/meta_v2_meta.json` ONLY — does NOT
modify any other key, does NOT touch `meta_v2.joblib` bytes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from ufc_prediction.db.session import SessionLocal
from ufc_prediction.ml.config import FEATURE_COLUMNS_V22, MLConfig
from ufc_prediction.ml.evaluator import PER_SLICE_KEYS
from ufc_prediction.ml.feature_matrix import (
    FeatureMatrixAssembler,
    compute_division_medians,
    split_temporal,
)
from ufc_prediction.ml.meta_features_v22 import build_meta_features_v22
from ufc_prediction.ml.queries import (
    load_computed_features,
    load_elo_features,
    load_fight_odds,
    load_fight_records,
    load_fighter_physicals,
    load_pre_ufc_records,
    load_round_stats_for_ml,
)


EXPECTED_XGB_V2_SHA256 = "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
SLICES = list(PER_SLICE_KEYS)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
META_META_PATH = MODELS_DIR / "meta" / "meta_v2_meta.json"
META_SHA_START_PATH = (
    PROJECT_ROOT
    / ".planning/phases/34-trust-baseline-no-odds-fallback"
    / "34-META-V2-SHA-PHASE-34-START-PLAN-01.txt"
)
FIXTURE_PATH = PROJECT_ROOT / "tests/fixtures" / "meta_v2_meta_pre_phase_34_t2_snapshot.json"


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _slice_masks(
    fight_dates: np.ndarray, today: date, random_seed: int = 42
) -> dict[str, np.ndarray]:
    """Mirror evaluator.evaluate_per_slice slice construction."""
    from datetime import timedelta

    cutoff_12mo = today - timedelta(days=365)
    cutoff_24mo = today - timedelta(days=730)
    mask_12mo = np.array([d >= cutoff_12mo for d in fight_dates])
    mask_24mo = np.array([d >= cutoff_24mo for d in fight_dates])
    rng = np.random.RandomState(random_seed)
    mask_random = rng.random(len(fight_dates)) < 0.15
    return {
        "most_recent_12mo": mask_12mo,
        "most_recent_24mo": mask_24mo,
        "random_15pct": mask_random,
    }


def main() -> int:
    print("[remeasure] Pre-flight asserts...")
    xgb_sha = _sha256_file(MODELS_DIR / "xgb_v2.joblib")
    if xgb_sha != EXPECTED_XGB_V2_SHA256:
        print(f"[remeasure] FATAL: xgb_v2 SHA drift: {xgb_sha}", file=sys.stderr)
        return 2
    meta_sha_anchor = META_SHA_START_PATH.read_text().strip()
    meta_sha_before = _sha256_file(MODELS_DIR / "meta" / "meta_v2.joblib")
    if meta_sha_before != meta_sha_anchor:
        print(
            f"[remeasure] FATAL: meta_v2 SHA drift: {meta_sha_before} vs anchor {meta_sha_anchor}",
            file=sys.stderr,
        )
        return 2
    assert sklearn.__version__ == "1.8.0"
    print("  SHAs OK")

    # Capture fixture snapshot if missing (idempotent — already done in execution session,
    # but defensive in case of re-run)
    if not FIXTURE_PATH.exists():
        print(f"[remeasure] WARNING: fixture {FIXTURE_PATH} missing — should be pre-captured")
    pre_keys_in_memory = set(json.loads(META_META_PATH.read_text()).keys())
    print(f"  pre-Task-2 keys: {len(pre_keys_in_memory)} (legacy preserved)")

    print("[remeasure] Loading models...")
    xgb_v2 = joblib.load(MODELS_DIR / "xgb_v2.joblib")
    meta_v2 = joblib.load(MODELS_DIR / "meta" / "meta_v2.joblib")

    print("[remeasure] Loading data from DB...")
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

    xgb_v2_meta = json.loads((MODELS_DIR / "xgb_v2_meta.json").read_text())
    cutoff_str: str = xgb_v2_meta["cutoff_date"]
    cutoff_date_obj: date = date.fromisoformat(cutoff_str)

    print("[remeasure] Assembling feature matrices (72-col xgb baseline + 90-col v2.2)...")
    division_medians = compute_division_medians(
        fighter_physicals,
        fight_records,
        cutoff_date_obj,
    )
    config = MLConfig(cutoff_date=cutoff_str)
    assembler = FeatureMatrixAssembler(config)
    X72, y, fight_dates_full = assembler.assemble(
        fight_records,
        elo_features,
        computed_features,
        fighter_physicals,
        division_medians,
        round_stats,
        pre_ufc_records=pre_ufc,
        fight_odds=fight_odds,
        feature_set="v2.1-no-net",
    )
    assembler_v22 = FeatureMatrixAssembler(config)
    X_v22, y_v22, _ = assembler_v22.assemble(
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

    print("[remeasure] Temporal split at cutoff_date...")
    X72_train, X72_test, _, y_test = split_temporal(X72, y, fight_dates_full, cutoff_date_obj)
    _, Xv22_test, _, _ = split_temporal(X_v22, y_v22, fight_dates_full, cutoff_date_obj)
    test_mask = np.array([d >= cutoff_date_obj for d in fight_dates_full])
    fight_dates_test = np.array(fight_dates_full)[test_mask]
    print(f"  test rows: {X72_test.shape[0]}")

    print("[remeasure] Computing meta_v22 probabilities on full test set...")
    xgb_oof_test = xgb_v2.predict_proba(X72_test)[:, 1]
    elo_overall_diff_idx_v22 = FEATURE_COLUMNS_V22.index("elo_overall_diff")
    elo_diff_test = Xv22_test[:, elo_overall_diff_idx_v22]
    elo_prob_test = 1.0 / (1.0 + 10.0 ** (-elo_diff_test / 400.0))
    level1_test = build_meta_features_v22(
        xgb_oof_prob=xgb_oof_test, elo_prob=elo_prob_test, X_v22=Xv22_test
    )
    nan_mask_full = np.isnan(level1_test).any(axis=1)
    meta_proba_full = np.full(len(y_test), np.nan)
    if (~nan_mask_full).sum() > 0:
        meta_proba_full[~nan_mask_full] = meta_v2.pipeline.predict_proba(
            level1_test[~nan_mask_full]
        )[:, 1]

    print("[remeasure] Computing per-slice metrics...")
    today = date.today()
    masks = _slice_masks(fight_dates_test, today, random_seed=42)
    new_metrics: dict = {}
    for sl in SLICES:
        mask = masks[sl]
        y_sl = y_test[mask]
        proba_sl = meta_proba_full[mask]
        valid = ~np.isnan(proba_sl)
        n_total = int(mask.sum())
        n_valid = int(valid.sum())
        print(f"  {sl}: n_total={n_total}, n_valid_after_nan_drop={n_valid}")

        if n_valid == 0:
            new_metrics[sl] = {
                "brier_score": None,
                "auc_roc": None,
                "accuracy": None,
                "calibration_curve": None,
                "n": 0,
                "n_total_slice": n_total,
                "n_after_nan_drop": 0,
                "measured_at": datetime.now(tz=UTC).isoformat(),
                "nan_drop_caveat": (
                    "Level-1 features include closing_prob_diff which is 99.5% NaN on "
                    "post-cutoff Event.source='ufcstats' rows (pre-existing data gap — "
                    "BFO Phase 21 backfill primarily covered training-side fights). "
                    "Per RESEARCH Open Q1 + Pitfall 3."
                ),
            }
            continue

        y_valid = y_sl[valid]
        proba_valid = proba_sl[valid]
        brier = float(brier_score_loss(y_valid, proba_valid))
        # AUC needs ≥1 positive AND ≥1 negative
        if len(np.unique(y_valid)) > 1:
            auc = float(roc_auc_score(y_valid, proba_valid))
        else:
            auc = None
        acc = float(accuracy_score(y_valid, (proba_valid >= 0.5).astype(int)))

        # Calibration curve uniform 10-bin
        try:
            n_bins = min(10, max(2, n_valid // 2))
            fop, mpv = calibration_curve(y_valid, proba_valid, n_bins=n_bins, strategy="uniform")
            cc = {
                "fraction_of_positives": fop.tolist(),
                "mean_predicted_value": mpv.tolist(),
                "n_bins": n_bins,
                "strategy": "uniform",
            }
        except ValueError as e:
            cc = {"error": str(e)}

        new_metrics[sl] = {
            "brier_score": brier,
            "auc_roc": auc,
            "accuracy": acc,
            "calibration_curve": cc,
            "n": n_valid,
            "n_total_slice": n_total,
            "n_after_nan_drop": n_valid,
            "measured_at": datetime.now(tz=UTC).isoformat(),
        }

    print("[remeasure] Appending v2_3_slice_metrics to meta_v2_meta.json (APPEND-ONLY)...")
    existing = json.loads(META_META_PATH.read_text())
    if "v2_3_slice_metrics" in existing:
        print("  WARNING: v2_3_slice_metrics already present — overwriting (idempotent re-run)")
    existing["v2_3_slice_metrics"] = new_metrics
    META_META_PATH.write_text(json.dumps(existing, indent=2))

    # Post-write assertions
    print("[remeasure] Post-write assertions...")
    cur = json.loads(META_META_PATH.read_text())
    pre_keys = set(pre_keys_in_memory)
    cur_keys = set(cur.keys())
    missing = pre_keys - cur_keys
    if missing:
        print(f"[remeasure] FATAL: legacy keys dropped: {missing}", file=sys.stderr)
        return 2
    if "v2_3_slice_metrics" not in cur:
        print("[remeasure] FATAL: append failed", file=sys.stderr)
        return 2

    # Fixture comparison (revision n2)
    if FIXTURE_PATH.exists():
        fx = json.loads(FIXTURE_PATH.read_text())
        for k in fx:
            if cur.get(k) != fx[k]:
                print(
                    f"[remeasure] FATAL: legacy key {k!r} VALUE corrupted; "
                    f"snapshot vs current diverged",
                    file=sys.stderr,
                )
                return 2
        print(f"  legacy-key value parity verified against git-fixture snapshot")

    # meta_v2.joblib byte identity (script must never touch the .joblib)
    meta_sha_after = _sha256_file(MODELS_DIR / "meta" / "meta_v2.joblib")
    assert meta_sha_after == meta_sha_anchor, (
        f"meta_v2.joblib SHA drift! {meta_sha_after} vs {meta_sha_anchor}"
    )
    print(f"  meta_v2.joblib SHA OK (unchanged): {meta_sha_after}")

    print("[remeasure] DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
