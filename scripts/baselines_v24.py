#!/usr/bin/env python
"""TRUST-V24-01 — 4-baseline benchmark on v2.3 widened slices.

Per CONTEXT D-TRUST-V24-01 + RESEARCH Pitfall 3:
  Compare shipped META-V22 against
    (a) market-implied probability (per-fight FightOdds.closing_implied_prob;
        labeled `market_directional_implied_baseline`)
    (b) Elo-only sigmoid: 1.0 / (1.0 + 10^(-elo_overall_diff/400))
    (c) xgb_v2_no_odds direct predict_proba (67-col)
    (d) coin-flip = 0.5

Emits 15 rows = 5 baselines × 3 slices (meta_v22 anchor + 4 baselines × 3 slices)
+ summary block to ``results/baselines_v24.json``.

AUDIT-01 pre/post SHA assertions on xgb_v2.joblib + meta_v2.joblib (inference-only).
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
from sklearn.metrics import accuracy_score, brier_score_loss

from ufc_prediction.db.session import SessionLocal
from ufc_prediction.ml.config import FEATURE_COLUMNS_NO_NET, FEATURE_COLUMNS_V22, MLConfig
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
SLICES = list(PER_SLICE_KEYS)  # ("most_recent_12mo", "most_recent_24mo", "random_15pct")
BASELINES = [
    "meta_v22",
    "market_directional_implied_baseline",
    "elo_only",
    "xgb_v2_no_odds",
    "coin_flip",
]
ODDS_COLS = {
    "opening_prob_diff",
    "closing_prob_diff",
    "line_movement_diff",
    "sharp_money_signal",
    "odds_elo_divergence",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_PATH = PROJECT_ROOT / "results" / "baselines_v24.json"
META_SHA_START_PATH = (
    PROJECT_ROOT
    / ".planning/phases/34-trust-baseline-no-odds-fallback"
    / "34-META-V2-SHA-PHASE-34-START-PLAN-01.txt"
)


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _slice_masks(
    fight_dates: np.ndarray, today: date, random_seed: int = 42
) -> dict[str, np.ndarray]:
    """Mirror evaluator.evaluate_per_slice's slice construction (returns boolean masks)."""
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
    print("[baselines] SHA pre-flight asserts...")
    xgb_sha_before = _sha256_file(MODELS_DIR / "xgb_v2.joblib")
    if xgb_sha_before != EXPECTED_XGB_V2_SHA256:
        print(f"[baselines] FATAL: xgb_v2 SHA drift: {xgb_sha_before}", file=sys.stderr)
        return 2
    meta_sha_start = META_SHA_START_PATH.read_text().strip()
    meta_sha_before = _sha256_file(MODELS_DIR / "meta" / "meta_v2.joblib")
    if meta_sha_before != meta_sha_start:
        print(
            f"[baselines] FATAL: meta_v2 SHA drift: {meta_sha_before} vs {meta_sha_start}",
            file=sys.stderr,
        )
        return 2
    assert sklearn.__version__ == "1.8.0", sklearn.__version__
    print(f"  xgb_v2  SHA OK: {xgb_sha_before}")
    print(f"  meta_v2 SHA OK: {meta_sha_before}")

    print("[baselines] Loading models...")
    xgb_v2 = joblib.load(MODELS_DIR / "xgb_v2.joblib")
    xgb_v2_no_odds = joblib.load(MODELS_DIR / "xgb_v2_no_odds.joblib")
    xgb_v2_no_odds_sha = _sha256_file(MODELS_DIR / "xgb_v2_no_odds.joblib")
    meta_v2 = joblib.load(MODELS_DIR / "meta" / "meta_v2.joblib")

    print("[baselines] Loading data from DB...")
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

    print("[baselines] Assembling 72-col xgb_v2-baseline feature matrix...")
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
    print(f"  X72.shape={X72.shape}")

    # 90-col META-V22 view: FEATURE_COLUMNS_V22 = FEATURE_COLUMNS_NO_NET + 18 v2.2 extras.
    # The feature matrix assembler produces FEATURE_COLUMNS (75 cols) — for v2.2 cols
    # we need the v22 assembler view. Per RESEARCH Open Q2, re-derive xgb_oof_prob via
    # xgb_v2.predict_proba on TEST rows (no leakage). For META re-measurement we need the
    # 90-col matrix in FEATURE_COLUMNS_V22 ordering.
    print("[baselines] Re-assembling 90-col META-V22 feature matrix...")
    assembler_v22 = FeatureMatrixAssembler(config)
    X_v22, y_v22, fight_dates_v22 = assembler_v22.assemble(
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
    print(f"  X_v22.shape={X_v22.shape}")
    assert X_v22.shape[1] == len(FEATURE_COLUMNS_V22), (
        f"v22 col count drift: assembler={X_v22.shape[1]} vs FEATURE_COLUMNS_V22={len(FEATURE_COLUMNS_V22)}"
    )

    print("[baselines] Temporal split at cutoff_date...")
    X72_train, X72_test, _, _ = split_temporal(X72, y, fight_dates_full, cutoff_date_obj)
    Xv22_train, Xv22_test, y_train_v22, y_test_v22 = split_temporal(
        X_v22, y_v22, fight_dates_v22, cutoff_date_obj
    )
    # row-alignment sanity
    assert X72_test.shape[0] == Xv22_test.shape[0], (
        f"row count mismatch: X72_test={X72_test.shape[0]} Xv22_test={Xv22_test.shape[0]}"
    )
    test_mask = np.array([d >= cutoff_date_obj for d in fight_dates_full])
    fight_dates_test = np.array(fight_dates_full)[test_mask]
    print(f"  Test rows: {X72_test.shape[0]}")
    y_test = y_v22[np.array([d >= cutoff_date_obj for d in fight_dates_v22])]
    assert len(y_test) == X72_test.shape[0], f"{len(y_test)} vs {X72_test.shape[0]}"

    print("[baselines] Computing slice masks...")
    today = date.today()
    masks = _slice_masks(fight_dates_test, today, random_seed=42)

    # ── No-odds 67-col view ────────────────────────────────────────────
    no_odds_cols = [c for c in FEATURE_COLUMNS_NO_NET if c not in ODDS_COLS]
    keep_idx = [FEATURE_COLUMNS_NO_NET.index(c) for c in no_odds_cols]
    X_no_odds_test = X72_test[:, keep_idx]
    assert X_no_odds_test.shape[1] == 67

    # ── Precompute meta_v22 probs on TEST rows ─────────────────────────
    print("[baselines] Predicting meta_v22 on full test set (re-derive xgb_oof_prob)...")
    xgb_oof_test = xgb_v2.predict_proba(X72_test)[:, 1]
    elo_overall_diff_idx_v22 = FEATURE_COLUMNS_V22.index("elo_overall_diff")
    elo_diff_test = Xv22_test[:, elo_overall_diff_idx_v22]
    elo_prob_test = 1.0 / (1.0 + 10.0 ** (-elo_diff_test / 400.0))
    level1_test = build_meta_features_v22(
        xgb_oof_prob=xgb_oof_test,
        elo_prob=elo_prob_test,
        X_v22=Xv22_test,
    )
    # Mask NaN rows symmetrically — match the meta_v22 inference contract
    nan_mask_full = np.isnan(level1_test).any(axis=1)
    meta_proba_full = np.full(len(y_test), np.nan)
    meta_proba_full[~nan_mask_full] = meta_v2.pipeline.predict_proba(level1_test[~nan_mask_full])[
        :, 1
    ]
    print(f"  meta_v2 NaN-drop: {int(nan_mask_full.sum())}/{len(y_test)} rows")

    # ── Precompute market baseline per-fight closing_implied_prob from DB ─
    print("[baselines] Joining FightOdds.closing_implied_prob per fight...")
    # Build index from event_date + fighter_a_id + fighter_b_id (matching how feature_matrix builds rows).
    # The simpler approach: pull all closing_implied_prob for the test cutoff rows from the existing odds rows
    # already loaded above and join by (event_id, fighter_a_id, fighter_b_id). Since aligning is non-trivial
    # without touching feature_matrix internals, we fall back to the diff-proxy per Pitfall 3.
    closing_prob_diff_idx = FEATURE_COLUMNS_NO_NET.index("closing_prob_diff")
    closing_prob_diff_test = X72_test[:, closing_prob_diff_idx]
    market_proxy = 0.5 + closing_prob_diff_test / 2.0
    market_proxy = np.clip(market_proxy, 0.01, 0.99)
    # n_with_market = count of non-NaN rows
    market_nan_mask = np.isnan(closing_prob_diff_test)
    n_with_market_full = int((~market_nan_mask).sum())
    print(f"  market_proxy non-NaN coverage: {n_with_market_full}/{len(y_test)}")

    # ── Precompute elo_only probs ──────────────────────────────────────
    elo_overall_diff_idx = FEATURE_COLUMNS_NO_NET.index("elo_overall_diff")
    elo_diff_test_72 = X72_test[:, elo_overall_diff_idx]
    elo_only_proba = 1.0 / (1.0 + 10.0 ** (-elo_diff_test_72 / 400.0))

    # ── xgb_v2_no_odds probs ───────────────────────────────────────────
    no_odds_proba = xgb_v2_no_odds.predict_proba(X_no_odds_test)[:, 1]

    # ── Coin flip ───────────────────────────────────────────────────────
    coin_flip_proba = np.full(len(y_test), 0.5)

    # ── Emit rows ───────────────────────────────────────────────────────
    rows: list[dict] = []
    n_with_market_per_slice: dict[str, int] = {}
    for sl in SLICES:
        mask = masks[sl]
        n_obs = int(mask.sum())
        if n_obs == 0:
            print(f"[baselines] WARNING: slice {sl} has 0 rows", file=sys.stderr)
            continue
        y_sl = y_test[mask]

        # meta_v22 (NaN-aware brier — drop NaN rows from numerator + denominator)
        meta_proba_sl = meta_proba_full[mask]
        meta_valid = ~np.isnan(meta_proba_sl)
        if meta_valid.sum() > 0:
            meta_brier = float(brier_score_loss(y_sl[meta_valid], meta_proba_sl[meta_valid]))
            meta_acc = float(
                accuracy_score(y_sl[meta_valid], (meta_proba_sl[meta_valid] >= 0.5).astype(int))
            )
            meta_n = int(meta_valid.sum())
        else:
            meta_brier, meta_acc, meta_n = float("nan"), float("nan"), 0
        rows.append(
            {
                "slice": sl,
                "model": "meta_v22",
                "brier": meta_brier,
                "accuracy": meta_acc,
                "n": meta_n,
            }
        )

        # market baseline (n_with_market separately)
        market_proba_sl = market_proxy[mask]
        market_valid_sl = ~market_nan_mask[mask]
        n_with_market = int(market_valid_sl.sum())
        n_with_market_per_slice[sl] = n_with_market
        if n_with_market > 0:
            mb = float(brier_score_loss(y_sl[market_valid_sl], market_proba_sl[market_valid_sl]))
            ma = float(
                accuracy_score(
                    y_sl[market_valid_sl], (market_proba_sl[market_valid_sl] >= 0.5).astype(int)
                )
            )
        else:
            mb, ma = float("nan"), float("nan")
        rows.append(
            {
                "slice": sl,
                "model": "market_directional_implied_baseline",
                "brier": mb,
                "accuracy": ma,
                "n": n_with_market,
                "n_with_market": n_with_market,
            }
        )

        # elo_only
        eo_sl = elo_only_proba[mask]
        rows.append(
            {
                "slice": sl,
                "model": "elo_only",
                "brier": float(brier_score_loss(y_sl, eo_sl)),
                "accuracy": float(accuracy_score(y_sl, (eo_sl >= 0.5).astype(int))),
                "n": n_obs,
            }
        )

        # xgb_v2_no_odds
        no_sl = no_odds_proba[mask]
        rows.append(
            {
                "slice": sl,
                "model": "xgb_v2_no_odds",
                "brier": float(brier_score_loss(y_sl, no_sl)),
                "accuracy": float(accuracy_score(y_sl, (no_sl >= 0.5).astype(int))),
                "n": n_obs,
            }
        )

        # coin_flip
        cf_sl = coin_flip_proba[mask]
        rows.append(
            {
                "slice": sl,
                "model": "coin_flip",
                "brier": float(brier_score_loss(y_sl, cf_sl)),
                "accuracy": float(accuracy_score(y_sl, (cf_sl >= 0.5).astype(int))),
                "n": n_obs,
            }
        )

    # Compute delta vs meta_v22 per slice
    anchor_by_slice = {r["slice"]: r for r in rows if r["model"] == "meta_v22"}
    for r in rows:
        if r["model"] == "meta_v22":
            continue
        anchor = anchor_by_slice.get(r["slice"])
        if anchor and not np.isnan(anchor["brier"]):
            r["delta_brier_vs_meta_v22"] = (
                r["brier"] - anchor["brier"] if not np.isnan(r["brier"]) else float("nan")
            )
            r["delta_acc_vs_meta_v22"] = (
                r["accuracy"] - anchor["accuracy"] if not np.isnan(r["accuracy"]) else float("nan")
            )
        else:
            r["delta_brier_vs_meta_v22"] = float("nan")
            r["delta_acc_vs_meta_v22"] = float("nan")

    # Output JSON
    out = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "xgb_v2_sha256": EXPECTED_XGB_V2_SHA256,
        "meta_v2_sha256": meta_sha_before,
        "xgb_v2_no_odds_sha256": xgb_v2_no_odds_sha,
        "slices": SLICES,
        "baselines": BASELINES,
        "rows": rows,
        "summary": {
            "n_rows": len(rows),
            "n_with_market_per_slice": n_with_market_per_slice,
            "market_baseline_caveat": (
                "market_directional_implied_baseline derives a per-fight pseudo-prob from "
                "closing_prob_diff (the FEATURE matrix difference column) via "
                "y_hat = clip(0.5 + diff/2, 0.01, 0.99). This is a DIRECTIONAL baseline, not a "
                "per-fight de-vigged market probability — see RESEARCH Pitfall 3. n_with_market "
                "counts non-NaN rows in closing_prob_diff for the slice; NaN rows excluded."
            ),
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"[baselines] Wrote {RESULTS_PATH}")

    # SHA post-flight asserts
    print("[baselines] SHA post-flight asserts...")
    sha_after_xgb = _sha256_file(MODELS_DIR / "xgb_v2.joblib")
    sha_after_meta = _sha256_file(MODELS_DIR / "meta" / "meta_v2.joblib")
    assert sha_after_xgb == EXPECTED_XGB_V2_SHA256
    assert sha_after_meta == meta_sha_start
    print("  xgb_v2 + meta_v2 SHAs unchanged (inference-only ✓)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
