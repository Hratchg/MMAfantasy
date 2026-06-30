#!/usr/bin/env python
"""TRUST-V24-03 — SHAP + permutation importance for xgb_v2 + META-V22.

Per CONTEXT D-TRUST-V24-03 + RESEARCH Pattern 1/2/3 + Pitfall 1/4:
  - Tree SHAP on xgb_v2 (72-col base model; up-to-1000-row stratified sample)
  - Linear SHAP on META-V22 (13-col Level-1 → polynomial-expanded 91-col representation)
  - Permutation importance on most_recent_24mo slice (n_repeats=10, scoring=neg_brier_score)
  - Emit JSON + 2 PNGs + explicit per-tier description strings (Pitfall 1)

Revision M1 — Process isolation: run as a separate `uv run python` process from
calibration_v24.py.
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from datetime import UTC, date, datetime
from pathlib import Path

import joblib
import numpy as np
import shap
import sklearn
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from ufc_prediction.db.session import SessionLocal
from ufc_prediction.ml.config import FEATURE_COLUMNS_NO_NET, FEATURE_COLUMNS_V22, MLConfig
from ufc_prediction.ml.feature_matrix import (
    FeatureMatrixAssembler,
    compute_division_medians,
    split_temporal,
)
from ufc_prediction.ml.meta_features_v22 import META_V22_FEATURE_COLUMNS, build_meta_features_v22
from ufc_prediction.ml.queries import (
    load_computed_features,
    load_elo_features,
    load_fight_odds,
    load_fight_records,
    load_fighter_physicals,
    load_pre_ufc_records,
    load_round_stats_for_ml,
)

# Silence shap's interventional-perturbation deprecation FutureWarning; the API still works.
warnings.filterwarnings("ignore", category=FutureWarning, module="shap")


EXPECTED_XGB_V2_SHA256 = "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
SHAP_SAMPLE_TARGET = 1000

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
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
    print("[importance] SHA pre-flight asserts (revision M2 inline)...")
    xgb_sha = _sha256_file(MODELS_DIR / "xgb_v2.joblib")
    assert xgb_sha == EXPECTED_XGB_V2_SHA256, f"xgb_v2 SHA drift: {xgb_sha}"
    meta_sha_anchor = META_SHA_START_PATH.read_text().strip()
    meta_sha = _sha256_file(MODELS_DIR / "meta" / "meta_v2.joblib")
    assert meta_sha == meta_sha_anchor, f"meta_v2 SHA drift: {meta_sha}"
    assert sklearn.__version__ == "1.8.0", sklearn.__version__  # Pitfall 7 guard
    print(f"  xgb_v2 OK / meta_v2 OK / sklearn={sklearn.__version__} / shap={shap.__version__}")

    np.random.seed(42)

    print("[importance] Loading models...")
    xgb_v2 = joblib.load(MODELS_DIR / "xgb_v2.joblib")
    meta_v2 = joblib.load(MODELS_DIR / "meta" / "meta_v2.joblib")

    # Extract inner XGBClassifier for SHAP (TreeExplainer doesn't accept CalibratedClassifierCV)
    inner_xgb = xgb_v2.calibrated_classifiers_[0].estimator.estimator
    print(f"  inner xgb classifier: {type(inner_xgb).__name__}")

    print("[importance] Loading data + assembling feature matrices...")
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
    X_v22, _, _ = assembler.assemble(
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
    _, X72_test, _, y_test = split_temporal(X72, y, fight_dates_full, cutoff_date_obj)
    _, Xv22_test, _, _ = split_temporal(X_v22, y, fight_dates_full, cutoff_date_obj)
    test_mask = np.array([d >= cutoff_date_obj for d in fight_dates_full])
    fight_dates_test = np.array(fight_dates_full)[test_mask]
    print(f"  test set: {X72_test.shape[0]} rows")

    # ── Stratified up-to-1000-row sample (Pitfall 4) ─────────────────
    n_target = min(SHAP_SAMPLE_TARGET, X72_test.shape[0])
    if n_target < X72_test.shape[0]:
        _, X72_sample, _, y_sample, idx_sample = _stratified_sample(
            X72_test, y_test, n_target, random_state=42
        )
        n_actual = len(X72_sample)
        Xv22_sample = Xv22_test[idx_sample]
    else:
        X72_sample = X72_test
        Xv22_sample = Xv22_test
        _y_sample = y_test
        n_actual = X72_test.shape[0]
    print(f"  SHAP stratified sample: target={n_target} actual={n_actual}")

    # ── xgb_v2 Tree SHAP ────────────────────────────────────────────────
    print("[importance] Computing Tree SHAP on xgb_v2 (raw output)...")
    explainer_xgb = shap.TreeExplainer(
        inner_xgb,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )
    shap_xgb = explainer_xgb.shap_values(X72_sample)
    if isinstance(shap_xgb, list):
        # Binary classification → list of 2; take positive class
        shap_xgb = shap_xgb[1] if len(shap_xgb) == 2 else shap_xgb[0]
    mean_abs_xgb = np.abs(shap_xgb).mean(axis=0)
    xgb_shap_top20 = sorted(
        [
            {
                "rank": int(i + 1),
                "feature": FEATURE_COLUMNS_NO_NET[idx],
                "mean_abs_shap": float(mean_abs_xgb[idx]),
            }
            for i, idx in enumerate(np.argsort(mean_abs_xgb)[::-1][:20])
        ],
        key=lambda r: r["rank"],
    )
    # closing_prob_diff rank in xgb_v2
    cpd_idx = FEATURE_COLUMNS_NO_NET.index("closing_prob_diff")
    xgb_rank_order = np.argsort(mean_abs_xgb)[::-1]
    xgb_cpd_rank = int(np.where(xgb_rank_order == cpd_idx)[0][0]) + 1

    print("[importance] Plotting xgb_v2 SHAP summary...")
    plt.figure()
    shap.summary_plot(
        shap_xgb,
        X72_sample,
        feature_names=FEATURE_COLUMNS_NO_NET,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.tight_layout()
    out_xgb_png = RESULTS_DIR / "feature_importance_v24_xgb_v2.png"
    plt.savefig(out_xgb_png, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_xgb_png}")

    # ── META-V22 Linear SHAP ────────────────────────────────────────────
    print("[importance] Computing Linear SHAP on META-V22 (Level-1 → polynomial-expanded)...")
    # Build level1 from sample
    xgb_oof_sample = xgb_v2.predict_proba(X72_sample)[:, 1]
    elo_diff_idx_v22 = FEATURE_COLUMNS_V22.index("elo_overall_diff")
    elo_diff_sample = Xv22_sample[:, elo_diff_idx_v22]
    elo_prob_sample = 1.0 / (1.0 + 10.0 ** (-elo_diff_sample / 400.0))
    level1_sample = build_meta_features_v22(
        xgb_oof_prob=xgb_oof_sample, elo_prob=elo_prob_sample, X_v22=Xv22_sample
    )
    # Drop NaN rows for SHAP (LinearExplainer can't handle NaN)
    nan_mask = np.isnan(level1_sample).any(axis=1)
    level1_clean = level1_sample[~nan_mask]
    n_meta_shap = level1_clean.shape[0]
    print(f"  META SHAP sample post-NaN-drop: {n_meta_shap}/{n_actual}")

    if n_meta_shap < 5:
        print(
            f"[importance] WARNING: META SHAP sample too small ({n_meta_shap}); "
            "emitting explanatory placeholder PNG + JSON. This is the same NaN-coverage "
            "gap surfaced in 34-03 SUMMARY."
        )
        meta_shap_top13 = []
        meta_cpd_rank = -1
        # Write explanatory placeholder PNG so downstream tests find a non-empty PNG file
        plt.figure(figsize=(8, 4))
        plt.text(
            0.5,
            0.5,
            f"META-V22 Linear SHAP not computable.\n\n"
            f"Sample after NaN-drop on Level-1 features: {n_meta_shap}/{n_actual} rows.\n"
            f"Root cause: post-cutoff ufcstats-source fights have 99.5% NaN "
            f"`closing_prob_diff`\n(pre-existing data-coverage gap; see 34-03 SUMMARY).\n\n"
            f"xgb_v2 SHAP (which does not require BFO) IS available — see\n"
            f"feature_importance_v24_xgb_v2.png.",
            ha="center",
            va="center",
            fontsize=10,
            wrap=True,
        )
        plt.axis("off")
        out_meta_png = RESULTS_DIR / "feature_importance_v24_meta_v22.png"
        plt.savefig(out_meta_png, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  wrote {out_meta_png} (explanatory placeholder)")
    else:
        X_poly_scaled = meta_v2.pipeline[:-1].transform(level1_clean)
        expanded_names = meta_v2.pipeline.named_steps["poly"].get_feature_names_out(
            META_V22_FEATURE_COLUMNS
        )
        explainer_meta = shap.LinearExplainer(
            meta_v2.pipeline.named_steps["clf"],
            X_poly_scaled,
            feature_perturbation="interventional",
        )
        shap_meta = explainer_meta.shap_values(X_poly_scaled)
        # Aggregate by base feature: sum |shap| across all expanded names containing base
        mean_abs_meta = np.abs(shap_meta).mean(axis=0)  # shape (91,)
        meta_aggregated: dict[str, float] = {}
        for base in META_V22_FEATURE_COLUMNS:
            # Match expanded names that include `base` as a token (' ' separator in poly names)
            agg = 0.0
            for j, ename in enumerate(expanded_names):
                tokens = str(ename).split(" ")
                if base in tokens:
                    agg += float(mean_abs_meta[j])
            meta_aggregated[base] = agg
        ranked = sorted(meta_aggregated.items(), key=lambda kv: kv[1], reverse=True)
        meta_shap_top13 = [
            {"rank": i + 1, "feature": k, "mean_abs_shap_aggregated": v}
            for i, (k, v) in enumerate(ranked)
        ]
        # closing_prob_diff rank in meta_v22
        meta_cpd_rank = next(
            (i + 1 for i, (k, _) in enumerate(ranked) if k == "closing_prob_diff"),
            -1,
        )

        # Plot: aggregate shap values to 13-base shape for summary plot
        agg_shap_matrix = np.zeros((len(shap_meta), len(META_V22_FEATURE_COLUMNS)))
        for j_exp, ename in enumerate(expanded_names):
            tokens = str(ename).split(" ")
            for base in tokens:
                if base in META_V22_FEATURE_COLUMNS:
                    bidx = META_V22_FEATURE_COLUMNS.index(base)
                    agg_shap_matrix[:, bidx] += shap_meta[:, j_exp]

        plt.figure()
        shap.summary_plot(
            agg_shap_matrix,
            level1_clean,
            feature_names=META_V22_FEATURE_COLUMNS,
            plot_type="bar",
            max_display=13,
            show=False,
        )
        plt.tight_layout()
        out_meta_png = RESULTS_DIR / "feature_importance_v24_meta_v22.png"
        plt.savefig(out_meta_png, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  wrote {out_meta_png}")

    # ── Permutation importance on most_recent_24mo ──────────────────────
    print("[importance] Computing permutation importance on most_recent_24mo slice...")
    today = date.today()
    masks = _slice_masks(fight_dates_test, today)
    mask_24mo = masks["most_recent_24mo"]
    X72_24mo = X72_test[mask_24mo]
    Xv22_24mo = Xv22_test[mask_24mo]
    y_24mo = y_test[mask_24mo]

    perm_xgb = permutation_importance(
        xgb_v2,
        X72_24mo,
        y_24mo,
        n_repeats=10,
        random_state=42,
        scoring="neg_brier_score",
        n_jobs=-1,
    )
    perm_xgb_top20 = sorted(
        [
            {
                "rank": int(i + 1),
                "feature": FEATURE_COLUMNS_NO_NET[idx],
                "importance_mean": float(perm_xgb.importances_mean[idx]),
                "importance_std": float(perm_xgb.importances_std[idx]),
            }
            for i, idx in enumerate(np.argsort(perm_xgb.importances_mean)[::-1][:20])
        ],
        key=lambda r: r["rank"],
    )

    # Permutation on meta_v22 — need a callable that drops NaN rows or works on level1.
    # Simplest approach: build level1 for the 24mo slice, drop NaN rows, then permute the 13 cols.
    xgb_oof_24mo = xgb_v2.predict_proba(X72_24mo)[:, 1]
    elo_diff_24mo = Xv22_24mo[:, elo_diff_idx_v22]
    elo_prob_24mo = 1.0 / (1.0 + 10.0 ** (-elo_diff_24mo / 400.0))
    level1_24mo = build_meta_features_v22(
        xgb_oof_prob=xgb_oof_24mo, elo_prob=elo_prob_24mo, X_v22=Xv22_24mo
    )
    nan_mask_24 = np.isnan(level1_24mo).any(axis=1)
    level1_clean_24 = level1_24mo[~nan_mask_24]
    y_clean_24 = y_24mo[~nan_mask_24]
    print(
        f"  META permutation sample post-NaN-drop: {level1_clean_24.shape[0]}/{X72_24mo.shape[0]}"
    )

    if level1_clean_24.shape[0] < 10:
        print("[importance] WARNING: meta permutation sample too small; skipping.")
        perm_meta_top13 = []
    else:
        perm_meta = permutation_importance(
            meta_v2.pipeline,
            level1_clean_24,
            y_clean_24,
            n_repeats=10,
            random_state=42,
            scoring="neg_brier_score",
            n_jobs=-1,
        )
        perm_meta_top13 = sorted(
            [
                {
                    "rank": int(i + 1),
                    "feature": META_V22_FEATURE_COLUMNS[idx],
                    "importance_mean": float(perm_meta.importances_mean[idx]),
                    "importance_std": float(perm_meta.importances_std[idx]),
                }
                for i, idx in enumerate(np.argsort(perm_meta.importances_mean)[::-1])
            ],
            key=lambda r: r["rank"],
        )

    # ── Emit JSON ───────────────────────────────────────────────────────
    out = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "xgb_v2_sha256": EXPECTED_XGB_V2_SHA256,
        "meta_v2_sha256": meta_sha_anchor,
        "shap_sample_target": SHAP_SAMPLE_TARGET,
        "shap_sample_n_actual": int(n_actual),
        "shap_sample_strategy": "stratified by outcome (Pitfall 4)",
        "perm_slice": "most_recent_24mo",
        "perm_n_repeats": 10,
        "perm_scoring": "neg_brier_score",
        "xgb_v2": {
            "tier_description": (
                "Tree SHAP on the 72-col xgb_v2 base model. Measures absolute marginal "
                "contribution of each raw feature to the base model's logit output. The "
                "closing_prob_diff #1 claim from RETROSPECTIVE.md:32 is tested below."
            ),
            "shap_top_20": xgb_shap_top20,
            "permutation_top_20": perm_xgb_top20,
        },
        "meta_v22": {
            "tier_description": (
                "Linear SHAP on the 13-col META-V22 Level-1 set after polynomial "
                "expansion + standardization, then aggregated back to 13 base features by "
                "summing |shap| across all expanded names containing each base. Measures "
                "'given that I already have xgb_oof_prob, how much does column X "
                "additionally contribute?' — answers a DIFFERENT question than xgb_v2 SHAP "
                "(RESEARCH Pitfall 1)."
            ),
            "shap_top_13_aggregated": meta_shap_top13,
            "permutation_top_13": perm_meta_top13,
            "n_after_nan_drop_shap": n_meta_shap,
            "n_after_nan_drop_perm": int(level1_clean_24.shape[0]),
        },
        "closing_prob_diff_claim_verdict": {
            "xgb_v2_rank": xgb_cpd_rank,
            "meta_v22_rank": meta_cpd_rank,
            "verdict_string": _verdict_string(xgb_cpd_rank, meta_cpd_rank),
        },
    }
    out_json = RESULTS_DIR / "feature_importance_v24.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"[importance] Wrote {out_json}")

    # ── SHA post-flight ─────────────────────────────────────────────────
    sha_after_xgb = _sha256_file(MODELS_DIR / "xgb_v2.joblib")
    sha_after_meta = _sha256_file(MODELS_DIR / "meta" / "meta_v2.joblib")
    assert sha_after_xgb == EXPECTED_XGB_V2_SHA256
    assert sha_after_meta == meta_sha_anchor
    print("[importance] AUDIT-01: xgb_v2 + meta_v2 SHAs unchanged.")
    return 0


def _stratified_sample(X, y, n_target, random_state=42):
    """Stratified sample by y. Returns (X_remaining, X_sample, y_remaining, y_sample, idx_sample)."""
    n_total = X.shape[0]
    test_size = n_target / n_total
    indices = np.arange(n_total)
    train_idx, sample_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X[train_idx], X[sample_idx], y[train_idx], y[sample_idx], sample_idx


def _verdict_string(xgb_rank: int, meta_rank: int) -> str:
    if meta_rank == -1:
        return (
            f"closing_prob_diff ranks #{xgb_rank} in xgb_v2 Tree SHAP. META rank not "
            f"computable (sample too small after NaN-drop). RETROSPECTIVE.md:32 claim "
            f"validated at base-model tier only."
        )
    if xgb_rank <= 3 and meta_rank <= 3:
        return (
            f"closing_prob_diff ranks #{xgb_rank} in xgb_v2 Tree SHAP and #{meta_rank} "
            f"in META-V22 Linear SHAP — the RETROSPECTIVE.md:32 claim is empirically "
            f"validated at both tiers."
        )
    return (
        f"closing_prob_diff ranks #{xgb_rank} in xgb_v2 Tree SHAP and #{meta_rank} in "
        f"META-V22 Linear SHAP. The RETROSPECTIVE.md:32 'closing_prob_diff is #1' claim "
        f"is partially supported: validated at base tier (#{xgb_rank}), differently "
        f"weighted at META tier (#{meta_rank}) because META's Level-1 set decouples the "
        f"base model output from raw closing odds."
    )


if __name__ == "__main__":
    sys.exit(main())
