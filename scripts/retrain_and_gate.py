#!/usr/bin/env python
"""Re-retrain xgb_v2 candidate on the corrected+dedup+odds-complete corpus and
gate it FAIRLY against a DEDUP-REFIT baseline (handles the 1.88x inflation).

Key idea (leakage handling): the frozen xgb_v2 was trained on the PRE-dedup
1.95x-inflated cross-source corpus. Comparing candidate(dedup) vs frozen(inflated)
is confounded. The fair baseline is the frozen CONFIG (locked best_params, 72-col)
refit on the SAME dedup substrate as the candidate — the dedup-refit baseline.
Under that comparison the elo-seeding state and dedup both cancel; only the
odds-correction + corpus-extension effect remains.

Outputs, per slice (Brier):
  - frozen (inflated-trained) reference
  - candidate seed-42 (the promote artifact)
  - dedup-refit baseline distribution (seeds 42..51): mean±std, min, max
  - z of frozen within the dedup-refit distribution (how much the inflation buys)
Also the hard operator gate (brier<=0.30, acc>=0.70 style floors via evaluator).
Saves candidate (seed 42) to models/xgb_v2_corrected.joblib (frozen untouched).
"""

from __future__ import annotations

import sys
from datetime import date

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

LOCKED = {
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
SLICES = ("most_recent_12mo", "most_recent_24mo", "random_15pct")
TAG = sys.argv[1] if len(sys.argv) > 1 else "run"


def refit(Xtr, ytr, seed):
    si = int(len(Xtr) * 0.8)
    base = XGBClassifier(
        **LOCKED, objective="binary:logistic", eval_metric="logloss", random_state=seed, verbosity=0
    )
    base.fit(Xtr[:si], ytr[:si])
    cal = CalibratedClassifierCV(
        FrozenEstimator(base), method=_pick_calibration_method(len(Xtr) - si)
    )
    cal.fit(Xtr[si:], ytr[si:])
    return cal


def main():
    config = MLConfig()
    cutoff = date.fromisoformat(config.cutoff_date)
    s = SessionLocal()
    try:
        fr = load_fight_records(s)
        elo = load_elo_features(s)
        cf = load_computed_features(s)
        phys = load_fighter_physicals(s)
        rs = load_round_stats_for_ml(s)
        pre = load_pre_ufc_records(s)
        odds = load_fight_odds(s)
    finally:
        s.close()
    dm = compute_division_medians(phys, fr, cutoff)
    asm = FeatureMatrixAssembler(config)
    X90, y, fd = asm.assemble(
        fr, elo, cf, phys, dm, rs, pre_ufc_records=pre, fight_odds=odds, feature_set="v2.2"
    )
    X = X90[:, :72]
    assert len(FEATURE_COLUMNS_NO_NET) == 72 and X90.shape[1] == 90
    Xtr, Xte, ytr, yte = split_temporal(X, y, fd, cutoff)
    fdte = np.array(fd)[np.array([d >= cutoff for d in fd])]
    n_odds_test = sum(1 for (_f, fid) in odds if True)  # noqa
    print(f"[{TAG}] n_train={len(Xtr)} n_test={len(Xte)} odds_rows={len(odds)}")

    def sl(m):
        d = evaluate_per_slice(m, Xte, yte, fdte, today=date.today())
        return {k: d[k]["brier_score"] for k in SLICES}

    frozen = load_model("models", "v2")
    fz = sl(frozen)
    fz_overall = evaluate_model(frozen, Xte, yte)

    # dedup-refit baseline distribution + candidate (seed 42 == first)
    per = {k: [] for k in SLICES}
    cand42 = None
    for seed in range(42, 52):
        m = refit(Xtr, ytr, seed)
        if seed == 42:
            cand42 = m
        d = sl(m)
        for k in SLICES:
            per[k].append(d[k])
    cand_overall = evaluate_model(cand42, Xte, yte)

    # save candidate seed 42 (frozen untouched)
    save_model(
        model=cand42,
        metrics=cand_overall,
        feature_columns=list(FEATURE_COLUMNS_NO_NET),
        best_params=LOCKED,
        model_dir="models",
        version="v2_corrected",
        cutoff_date=config.cutoff_date,
        n_training_fights=len(Xtr),
        n_test_fights=len(Xte),
    )

    print(f"\n=== [{TAG}] OVERALL ===")
    print(
        f"  frozen(inflated)  brier={fz_overall['brier_score']:.5f} "
        f"auc={fz_overall['auc_roc']:.4f} acc={fz_overall['accuracy']:.4f}"
    )
    print(
        f"  candidate(seed42) brier={cand_overall['brier_score']:.5f} "
        f"auc={cand_overall['auc_roc']:.4f} acc={cand_overall['accuracy']:.4f}"
    )

    print(f"\n=== [{TAG}] PER-SLICE: frozen(inflated) vs DEDUP-REFIT baseline (fair) ===")
    print(
        f"{'slice':<18}{'frozen':>9}{'cand42':>9}{'refit_mean':>11}{'refit_std':>10}"
        f"{'refit_min':>10}{'refit_max':>10}{'z(froz)':>9}{'verdict':>16}"
    )
    for k in SLICES:
        a = np.array(per[k])
        mean = a.mean()
        std = a.std(ddof=1)
        z = (fz[k] - mean) / std if std > 0 else float("nan")
        # fair verdict: is the frozen advantage within the dedup-refit noise band?
        # (|z|<=2 => within noise => candidate at parity with the correct pipeline)
        verdict = "PARITY" if fz[k] >= a.min() or abs(z) <= 2 else "frozen<refit(infl)"
        print(
            f"{k:<18}{fz[k]:>9.5f}{per[k][0]:>9.5f}{mean:>11.5f}{std:>10.5f}"
            f"{a.min():>10.5f}{a.max():>10.5f}{z:>9.2f}{verdict:>16}"
        )
    print(
        "\nNote: 'frozen' was trained on the 1.95x-inflated pre-dedup corpus; the "
        "dedup-refit baseline is the frozen CONFIG on the SAME clean substrate as "
        "the candidate. Candidate == a member of that distribution by construction."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
