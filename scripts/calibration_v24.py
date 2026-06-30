#!/usr/bin/env python
"""TRUST-V24-04 — Per-slice calibration: uniform 10-bin ECE + reliability plots +
Platt-vs-isotonic A/B (isotonic reused from Phase 32; Platt fit fresh).

Per CONTEXT D-TRUST-V24-04 + RESEARCH Pattern 4/5/6 + Pitfall 5 + revision M4 Option B.

Revision M1 — Process isolation: run as a separate `uv run python` process from
feature_importance_v24.py.
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
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
META_SHA_START_PATH = (
    PROJECT_ROOT
    / ".planning/phases/34-trust-baseline-no-odds-fallback"
    / "34-META-V2-SHA-PHASE-34-START-PLAN-01.txt"
)
PHASE_32_REPORT = (
    PROJECT_ROOT
    / ".planning/milestones/v2.3-phases/32-forward-stepwise-recomposition-partner-v110"
    / "COMPOSITION_V23_REPORT.json"
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


def _bootstrap_ci_per_bin(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    n_bins: int = 10,
    min_n_for_ci: int = 10,
    seed: int = 42,
):
    """Returns (ci_lo[10], ci_hi[10], bin_n[10]). NaN where bin has < min_n_for_ci."""
    rng = np.random.default_rng(seed)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ci_lo = np.full(n_bins, np.nan)
    ci_hi = np.full(n_bins, np.nan)
    bin_n = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            bin_mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            bin_mask = (y_prob >= lo) & (y_prob < hi)
        bin_n[b] = int(bin_mask.sum())
        if bin_n[b] < min_n_for_ci:
            continue
        y_in = y_true[bin_mask]
        n = len(y_in)
        samples = np.empty(n_bootstrap, dtype=float)
        for k in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            samples[k] = float(y_in[idx].mean())
        ci_lo[b], ci_hi[b] = float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))
    return ci_lo, ci_hi, bin_n


def _plot_reliability(
    slice_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ece: float,
    ci_lo: np.ndarray,
    ci_hi: np.ndarray,
    bin_n: np.ndarray,
    out_path: Path,
    min_n_for_ci: int = 10,
) -> None:
    n_bins = len(bin_n)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fop, mpv = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    # reliability points
    # calibration_curve returns only bins with at least 1 sample — map back to bin index by mpv ≈ bin_center
    ax.plot(mpv, fop, "o-", color="C0", label=f"META-V22 (ECE={ece:.3f})")

    # Bootstrap CI for bins with n ≥ threshold
    for b in range(n_bins):
        if bin_n[b] >= min_n_for_ci and not np.isnan(ci_lo[b]):
            ax.vlines(
                bin_centers[b],
                ci_lo[b],
                ci_hi[b],
                color="C0",
                alpha=0.4,
                linewidth=2,
            )
        elif bin_n[b] > 0:
            ax.scatter(
                [bin_centers[b]],
                [0.05],
                marker="x",
                color="red",
                s=50,
            )
            ax.annotate(
                f"n={bin_n[b]}",
                (bin_centers[b], 0.05),
                fontsize=7,
                xytext=(2, 6),
                textcoords="offset points",
            )

    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Reliability — {slice_name}")
    ax.legend(loc="upper left")

    ax2.bar(bin_centers, bin_n, width=0.09, color="steelblue", alpha=0.7)
    ax2.set_xlabel("Predicted probability bin")
    ax2.set_ylabel("Count")
    ax2.set_xlim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def _ece_uniform_10bin(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n_bins = 10
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    bin_counts, _ = np.histogram(y_prob, bins=np.linspace(0, 1, n_bins + 1))
    # calibration_curve returns only non-empty bins — align by index using bin_counts > 0 mask
    non_empty_mask = bin_counts > 0
    aligned_counts = bin_counts[non_empty_mask].astype(float)
    if aligned_counts.sum() == 0:
        return float("nan")
    weights = aligned_counts / aligned_counts.sum()
    return float(np.sum(weights * np.abs(frac_pos - mean_pred)))


def main() -> int:
    print("[calibration] SHA pre-flight asserts (revision M2 inline)...")
    xgb_sha = _sha256_file(MODELS_DIR / "xgb_v2.joblib")
    assert xgb_sha == EXPECTED_XGB_V2_SHA256, f"xgb_v2 SHA drift: {xgb_sha}"
    meta_sha_anchor = META_SHA_START_PATH.read_text().strip()
    meta_sha = _sha256_file(MODELS_DIR / "meta" / "meta_v2.joblib")
    assert meta_sha == meta_sha_anchor, f"meta_v2 SHA drift: {meta_sha}"
    assert sklearn.__version__ == "1.8.0", sklearn.__version__  # Pitfall 7 guard
    print("  SHAs OK")

    # ── Revision M4 Option B: hardcoded Phase 32 CALIB path with stop semantics ────
    print(f"[calibration] Reading isotonic CALIB outputs from {PHASE_32_REPORT}...")
    p32 = json.loads(PHASE_32_REPORT.read_text())
    calib_step = next((s for s in p32.get("steps", []) if s.get("step") == "CALIB"), None)
    if calib_step is None or calib_step.get("calibration_method") != "isotonic":
        raise SystemExit(
            "Phase 32 CALIB step not found OR calibration_method != 'isotonic'. "
            "Expected schema: data['steps'][i]['step']=='CALIB' with "
            "['calibration_method']=='isotonic' and ['candidate_brier_per_slice'] dict. "
            "Surface to operator with the observed schema."
        )
    isotonic_brier_per_slice = calib_step["candidate_brier_per_slice"]
    expected_slices = {"most_recent_12mo", "most_recent_24mo", "random_15pct"}
    assert set(isotonic_brier_per_slice.keys()) == expected_slices, (
        f"isotonic slices mismatch: {list(isotonic_brier_per_slice.keys())}"
    )
    print(f"  isotonic Brier per slice (Phase 32): {isotonic_brier_per_slice}")

    # ── Load models + data ─────────────────────────────────────────────
    print("[calibration] Loading models + data...")
    xgb_v2 = joblib.load(MODELS_DIR / "xgb_v2.joblib")
    meta_v2 = joblib.load(MODELS_DIR / "meta" / "meta_v2.joblib")
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
    X72_train, X72_test, y_train, y_test = split_temporal(X72, y, fight_dates_full, cutoff_date_obj)
    Xv22_train, Xv22_test, _, _ = split_temporal(X_v22, y, fight_dates_full, cutoff_date_obj)
    test_mask = np.array([d >= cutoff_date_obj for d in fight_dates_full])
    fight_dates_test = np.array(fight_dates_full)[test_mask]
    print(f"  test rows: {X72_test.shape[0]}")

    # ── Compute meta_v22 probs on full TEST + TRAIN (for Platt fit) ────
    print("[calibration] Computing meta_v22 probs on test + train...")
    elo_idx_v22 = FEATURE_COLUMNS_V22.index("elo_overall_diff")

    def _compute_meta_probs(
        X72_in: np.ndarray, Xv22_in: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (proba_full_len_NaN_aware, valid_mask)."""
        xgb_oof = xgb_v2.predict_proba(X72_in)[:, 1]
        elo_diff = Xv22_in[:, elo_idx_v22]
        elo_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        level1 = build_meta_features_v22(xgb_oof_prob=xgb_oof, elo_prob=elo_prob, X_v22=Xv22_in)
        nan_mask = np.isnan(level1).any(axis=1)
        proba = np.full(len(X72_in), np.nan)
        if (~nan_mask).sum() > 0:
            proba[~nan_mask] = meta_v2.pipeline.predict_proba(level1[~nan_mask])[:, 1]
        return proba, ~nan_mask

    proba_test, valid_test = _compute_meta_probs(X72_test, Xv22_test)
    proba_train, valid_train = _compute_meta_probs(X72_train, Xv22_train)

    # ── Fit Platt fresh on the same training set ──────────────────────
    print("[calibration] Fitting Platt (sigmoid 2-param) on training meta_v22 probs...")
    from sklearn.linear_model import LogisticRegression

    proba_train_valid = proba_train[valid_train]
    y_train_valid = y_train[valid_train]
    print(f"  Platt fit input: {len(y_train_valid)} rows after NaN-drop")
    if len(y_train_valid) >= 20 and len(np.unique(y_train_valid)) > 1:
        platt = LogisticRegression(C=1e6, max_iter=1000)
        platt.fit(proba_train_valid.reshape(-1, 1), y_train_valid)
        print(f"  Platt coef: {platt.coef_[0]}, intercept: {platt.intercept_}")
    else:
        print(
            f"[calibration] WARNING: insufficient train data for Platt fit "
            f"({len(y_train_valid)} rows; need ≥20 with both classes). "
            "Skipping Platt baseline — same NaN-coverage gap as 34-03/04.",
            file=sys.stderr,
        )
        platt = None

    # ── Per-slice calibration analysis ─────────────────────────────────
    today = date.today()
    slice_masks = _slice_masks(fight_dates_test, today)
    per_slice: dict = {}
    overall_winner_votes = {"isotonic": 0, "platt": 0, "tie": 0}
    sparse_bin_slices: list[str] = []

    for sl in PER_SLICE_KEYS:
        mask = slice_masks[sl]
        # Combine with valid_test
        slice_valid = mask & valid_test
        y_sl = y_test[slice_valid]
        p_sl = proba_test[slice_valid]
        n = len(y_sl)
        print(f"[calibration] slice {sl}: n_after_nan_drop={n}")
        if n < 5:
            per_slice[sl] = {
                "ece": None,
                "n": n,
                "calibration_curve": None,
                "bin_counts": None,
                "ci_lo": None,
                "ci_hi": None,
                "uncalibrated_brier": None,
                "platt_brier": None,
                "isotonic_brier_reused_from_phase_32": isotonic_brier_per_slice[sl],
                "platt_vs_isotonic_brier_delta": None,
                "winner": None,
                "note": f"n_after_nan_drop={n} too small for calibration analysis",
            }
            # Still emit a placeholder PNG
            short = {
                "most_recent_12mo": "12mo",
                "most_recent_24mo": "24mo",
                "random_15pct": "random_15pct",
            }[sl]
            out_path = RESULTS_DIR / f"calibration_v24_{short}.png"
            plt.figure(figsize=(8, 4))
            plt.text(
                0.5,
                0.5,
                f"Calibration not computable for slice {sl}.\n\n"
                f"Sample after NaN-drop on Level-1 features: {n} rows.\n"
                f"Root cause: post-cutoff ufcstats-source fights have 99.5% NaN "
                f"closing_prob_diff\n(see 34-03 SUMMARY).",
                ha="center",
                va="center",
                fontsize=10,
                wrap=True,
            )
            plt.axis("off")
            plt.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"  wrote {out_path} (explanatory placeholder)")
            continue

        # ECE uniform 10-bin
        ece = _ece_uniform_10bin(y_sl, p_sl)
        # bootstrap CI per bin
        ci_lo, ci_hi, bin_n = _bootstrap_ci_per_bin(
            y_sl,
            p_sl,
            n_bootstrap=1000,
            n_bins=10,
            min_n_for_ci=10,
        )
        # calibration curve arrays
        try:
            fop, mpv = calibration_curve(y_sl, p_sl, n_bins=10, strategy="uniform")
            cc_dict = {
                "fraction_of_positives": fop.tolist(),
                "mean_predicted_value": mpv.tolist(),
                "n_bins": 10,
                "strategy": "uniform",
            }
        except ValueError as e:
            cc_dict = {"error": str(e)}

        # Uncalibrated Brier
        uncal_brier = float(brier_score_loss(y_sl, p_sl))
        # Platt-applied
        if platt is not None:
            p_platt = platt.predict_proba(p_sl.reshape(-1, 1))[:, 1]
            platt_brier = float(brier_score_loss(y_sl, p_platt))
        else:
            platt_brier = float("nan")
        # Isotonic reused
        iso_brier = float(isotonic_brier_per_slice[sl])
        # Delta = platt - isotonic. Positive means isotonic wins.
        delta = platt_brier - iso_brier
        winner = "isotonic" if delta > 1e-4 else ("platt" if delta < -1e-4 else "tie")
        overall_winner_votes[winner] += 1

        # Plot
        short = {
            "most_recent_12mo": "12mo",
            "most_recent_24mo": "24mo",
            "random_15pct": "random_15pct",
        }[sl]
        out_path = RESULTS_DIR / f"calibration_v24_{short}.png"
        _plot_reliability(sl, y_sl, p_sl, ece, ci_lo, ci_hi, bin_n, out_path)
        print(f"  wrote {out_path}")

        # Track sparse-bin slices
        if (bin_n < 10).sum() > 0 and (bin_n > 0).sum() > (bin_n >= 10).sum():
            sparse_bin_slices.append(sl)

        per_slice[sl] = {
            "ece": ece,
            "n": n,
            "calibration_curve": cc_dict,
            "bin_counts": bin_n.tolist(),
            "ci_lo": [None if np.isnan(v) else v for v in ci_lo.tolist()],
            "ci_hi": [None if np.isnan(v) else v for v in ci_hi.tolist()],
            "uncalibrated_brier": uncal_brier,
            "platt_brier": platt_brier,
            "isotonic_brier_reused_from_phase_32": iso_brier,
            "platt_vs_isotonic_brier_delta": delta,
            "winner": winner,
        }

    # Overall winner
    if overall_winner_votes["isotonic"] > overall_winner_votes["platt"]:
        overall_winner = "isotonic"
    elif overall_winner_votes["platt"] > overall_winner_votes["isotonic"]:
        overall_winner = "platt"
    else:
        overall_winner = "mixed"

    out_doc = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "xgb_v2_sha256": EXPECTED_XGB_V2_SHA256,
        "meta_v2_sha256": meta_sha_anchor,
        "ece_bin_strategy": "uniform_10_bin",
        "bootstrap_n_per_bin": 1000,
        "sparse_bin_threshold": 10,
        "per_slice": per_slice,
        "platt_vs_isotonic_summary": {
            "isotonic_source": str(PHASE_32_REPORT.relative_to(PROJECT_ROOT)),
            "isotonic_source_path_in_json": (
                "data['steps'][i]['candidate_brier_per_slice'] where step=='CALIB'"
            ),
            "platt_fit_freshly_on_phase_32_train": True,
            "overall_winner": overall_winner,
        },
        "sparse_bin_slices": sparse_bin_slices,
    }
    out_path = RESULTS_DIR / "calibration_v24.json"
    out_path.write_text(json.dumps(out_doc, indent=2))
    print(f"[calibration] Wrote {out_path}")

    # ── SHA post-flight ───────────────────────────────────────────────
    sha_after = _sha256_file(MODELS_DIR / "meta" / "meta_v2.joblib")
    assert sha_after == meta_sha_anchor
    print("[calibration] AUDIT-01: meta_v2 SHA unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
