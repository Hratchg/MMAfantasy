#!/usr/bin/env python
"""Corrected-corpus refit of xgb_v2 (RETRAIN-PLAN step 3) + head-to-head eval (step 4).

Apples-to-apples: LOCKED best_params, cutoff 2023-01-01, 72-col FEATURE_COLUMNS_NO_NET
(feature_set="v2.1-no-net"), on the CORRECTED DB. Replicates ModelTrainer.train()'s
final-fit path EXACTLY (80/20 chronological proper/calib split; XGBClassifier fit on
proper; CalibratedClassifierCV(FrozenEstimator, method=_pick_calibration_method(n_calib))
on calib). Skips Optuna (params fixed).

Writes candidate to models/xgb_v2_corrected.joblib (+ _meta.json). Does NOT touch frozen.
Then evaluates frozen (v2) vs candidate on the SAME X_test: overall + per-slice.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

from ufc_prediction.cli.predict import (
    compute_division_medians,
    load_computed_features,
    load_elo_features,
    load_fight_odds,
    load_fight_records,
    load_fighter_physicals,
    load_pre_ufc_records,
    load_round_stats_for_ml,
)
from ufc_prediction.db.session import SessionLocal
from ufc_prediction.ml.config import FEATURE_COLUMNS_NO_NET, MLConfig
from ufc_prediction.ml.evaluator import evaluate_model, evaluate_per_slice
from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler, split_temporal
from ufc_prediction.ml.persistence import load_model, save_model
from ufc_prediction.ml.trainer import _pick_calibration_method

LOCKED_BEST_PARAMS = {
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
OUT = "/private/tmp/claude-501/-Users-hratchghanime/a4fa50d2-8de3-4664-8bb1-545563627650/scratchpad"


def main() -> int:
    config = MLConfig()
    cutoff = date.fromisoformat(config.cutoff_date)

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

    division_medians = compute_division_medians(fighter_physicals, fight_records, cutoff)
    assembler = FeatureMatrixAssembler(config)
    X90, y, fight_dates = assembler.assemble(
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
    # Row count is the DEDUPLICATED ufcstats-canonical corpus (Plan 28-04:
    # load_fight_records filters Event.source=='ufcstats', ~8,473 baseline +
    # new post-cutoff fights). The frozen xgb_v2's 16641 rows came from the
    # PRE-dedup cross-source corpus (inflated ~1.88x); this is the correct
    # smaller substrate, NOT a regression.
    print(
        f"v2.2 matrix shape={X90.shape} (expect 90 cols; ~8.5k dedup rows, not frozen's inflated 16641)"
    )
    assert X90.shape[1] == 90, "v2.2 column count drift"
    # Pitfall-E discipline (spike_noise_floor_v22): first 72 cols of the v2.2
    # matrix are byte-identical to FEATURE_COLUMNS_NO_NET (APPEND-ONLY order).
    X = X90[:, :72]
    assert len(FEATURE_COLUMNS_NO_NET) == 72

    X_train, X_test, y_train, y_test = split_temporal(X, y, fight_dates, cutoff)
    test_mask = np.array([d >= cutoff for d in fight_dates])
    fight_dates_test = fight_dates[test_mask]
    print(
        f"n_train={len(X_train)} n_test={len(X_test)} (frozen inflated: 13315 / 3326; dedup ~6792 / ~1.7k+)"
    )

    # Step 3: candidate refit (replicate trainer.train final-fit)
    n_total = len(X_train)
    split_idx = int(n_total * 0.8)
    X_proper, y_proper = X_train[:split_idx], y_train[:split_idx]
    X_calib, y_calib = X_train[split_idx:], y_train[split_idx:]
    calib_method = _pick_calibration_method(len(X_calib))
    print(f"n_proper={len(X_proper)} n_calib={len(X_calib)} calib_method={calib_method}")

    base = XGBClassifier(
        **LOCKED_BEST_PARAMS,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=config.random_seed,
        verbosity=0,
    )
    base.fit(X_proper, y_proper)
    candidate = CalibratedClassifierCV(FrozenEstimator(base), method=calib_method)
    candidate.fit(X_calib, y_calib)

    cand_overall = evaluate_model(candidate, X_test, y_test)
    save_model(
        model=candidate,
        metrics=cand_overall,
        feature_columns=list(FEATURE_COLUMNS_NO_NET),
        best_params=LOCKED_BEST_PARAMS,
        model_dir="models",
        version="v2_corrected",
        cutoff_date=config.cutoff_date,
        n_training_fights=len(X_train),
        n_test_fights=len(X_test),
    )
    print("saved models/xgb_v2_corrected.joblib")

    # Step 4: head-to-head frozen vs candidate on identical X_test
    frozen = load_model("models", "v2")
    frozen_overall = evaluate_model(frozen, X_test, y_test)

    def strip(d):
        return {k: v for k, v in d.items() if k != "calibration_curve"}

    frozen_slices = {
        k: strip(v)
        for k, v in evaluate_per_slice(
            frozen, X_test, y_test, fight_dates_test, today=date.today()
        ).items()
    }
    cand_slices = {
        k: strip(v)
        for k, v in evaluate_per_slice(
            candidate, X_test, y_test, fight_dates_test, today=date.today()
        ).items()
    }

    c12 = date.today() - timedelta(days=365)
    c24 = date.today() - timedelta(days=730)
    rng = np.random.RandomState(42)
    counts = {
        "most_recent_12mo": int(np.sum([d >= c12 for d in fight_dates_test])),
        "most_recent_24mo": int(np.sum([d >= c24 for d in fight_dates_test])),
        "random_15pct": int(np.sum(rng.random(len(fight_dates_test)) < 0.15)),
    }

    result = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_proper": len(X_proper),
        "n_calib": len(X_calib),
        "calib_method": calib_method,
        "today": date.today().isoformat(),
        "slice_counts": counts,
        "overall": {"frozen": strip(frozen_overall), "candidate": strip(cand_overall)},
        "per_slice": {"frozen": frozen_slices, "candidate": cand_slices},
    }
    with open(f"{OUT}/headtohead.json", "w") as f:
        json.dump(result, f, indent=2, default=float)

    print("\n=== OVERALL (full test set) ===")
    print(
        f"  frozen    brier={frozen_overall['brier_score']:.6f} "
        f"auc={frozen_overall['auc_roc']:.4f} acc={frozen_overall['accuracy']:.4f}"
    )
    print(
        f"  candidate brier={cand_overall['brier_score']:.6f} "
        f"auc={cand_overall['auc_roc']:.4f} acc={cand_overall['accuracy']:.4f}"
    )
    print("\n=== PER-SLICE (frozen -> candidate) ===")
    for s in ("most_recent_12mo", "most_recent_24mo", "random_15pct"):
        fb, cb = frozen_slices[s]["brier_score"], cand_slices[s]["brier_score"]
        fa, ca = frozen_slices[s]["auc_roc"], cand_slices[s]["auc_roc"]
        fc, cc = frozen_slices[s]["accuracy"], cand_slices[s]["accuracy"]
        print(f"  [{s}] n={counts[s]}")
        print(f"      brier {fb:.6f} -> {cb:.6f}  (delta={cb - fb:+.6f})")
        print(f"      auc   {fa:.4f} -> {ca:.4f}")
        print(f"      acc   {fc:.4f} -> {cc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
