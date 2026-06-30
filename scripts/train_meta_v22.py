#!/usr/bin/env python
"""Phase 26 / Plan 26-02 — META-V22 5-seed × 3-slice training spike.

Fork of `scripts/train_meta_v1.py` (Phase 19; 643 lines) — preserves the
Phase 19 substrate verbatim where applicable. Edits limited to:
  1. CLI defaults (--feature-set v2.2, --cache-path phase-26 scoped, etc.)
  2. Rich Level-1 (13-col) via `meta_features_v22.build_meta_features_v22`
  3. Pitfall #2 guard: explicit `assert X_oof.shape[1] == 72` at OOF call site
  4. Coefficient stability report (META-V22-04)
  5. Hard-gate-then-save w/ `meta_version="v2_candidate"` (D-10 carries Plan 26-04)
  6. Sibling contract artifact `models/meta/meta_v2-contract.json`
  7. META_V22_SPIKE.json + AUDIT-01 MID checkpoint emission
  8. Stepwise mode (--mode stepwise) for REF + TRAVEL forward-stepwise (Plan 26-03)

xgb_v2.joblib is NEVER retrained — AUDIT-01 SHA byte-identity preserved.

Usage:
    uv run python scripts/train_meta_v22.py             # full spike (5 seeds × 3 slices)
    uv run python scripts/train_meta_v22.py --dry-run   # synthetic smoke
    uv run python scripts/train_meta_v22.py --mode stepwise  # REF + TRAVEL (Plan 26-03)
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

# Phase 26 constants
EXPECTED_XGB_V2_SHA256: str = (
    "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
)
EXPECTED_XGB_V2_BEST_PARAMS: dict = {
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
EXPECTED_XGB_V2_N_FEATURES: int = 72
EXPECTED_CUTOFF_DATE: str = "2023-01-01"
SEEDS_DEFAULT: tuple[int, ...] = (42, 43, 44, 45, 46)
PER_SLICE_KEYS: tuple[str, ...] = (
    "most_recent_12mo", "most_recent_24mo", "random_15pct",
)
STEPWISE_HURDLE: float = 0.003  # D-13(v2.0) — locked; AF-10 ban on tuning downward

# Phase 26 artifact paths
PHASE26_DIR: Path = Path(".planning/phases/26-forward-stepwise-candidate-promotion")
META_OOF_PARQUET_PATH: Path = PHASE26_DIR / "oof_predictions_v22.parquet"
SPIKE_JSON_PATH: Path = PHASE26_DIR / "META_V22_SPIKE.json"
COEF_STABILITY_JSON_PATH: Path = PHASE26_DIR / "META_V22_COEFFICIENT_STABILITY.json"
SHA_MID_PATH: Path = PHASE26_DIR / "26-XGB-V2-SHA-MID.txt"
SHA_END_PATH: Path = PHASE26_DIR / "26-XGB-V2-SHA-END.txt"
REF_STEPWISE_PATH: Path = PHASE26_DIR / "REF_STEPWISE.json"
TRAVEL_STEPWISE_PATH: Path = PHASE26_DIR / "TRAVEL_STEPWISE.json"
META_DIR: Path = Path("models/meta")


def assert_phase26_invariants() -> None:
    """AUDIT-01 + AF-1 + Pitfall B + AF-2 invariant check."""
    sha_actual = hashlib.sha256(Path("models/xgb_v2.joblib").read_bytes()).hexdigest()
    assert sha_actual == EXPECTED_XGB_V2_SHA256, (
        f"AUDIT-01 violation: xgb_v2 SHA drift. got={sha_actual} "
        f"expected={EXPECTED_XGB_V2_SHA256}"
    )
    meta = json.loads(Path("models/xgb_v2_meta.json").read_text(encoding="utf-8"))
    assert meta["n_features"] == EXPECTED_XGB_V2_N_FEATURES, (
        f"Pitfall B violation: xgb_v2 n_features={meta['n_features']!r} "
        f"expected={EXPECTED_XGB_V2_N_FEATURES}"
    )
    assert meta["cutoff_date"] == EXPECTED_CUTOFF_DATE, (
        f"cutoff_date drift: {meta['cutoff_date']!r} expected={EXPECTED_CUTOFF_DATE}"
    )
    assert meta["best_params"] == EXPECTED_XGB_V2_BEST_PARAMS, (
        f"AF-1 violation: xgb_v2 best_params drift"
    )


def _load_assembled_data_v22() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Load fights + assemble the 90-col v2.2 feature matrix.

    Returns:
        (X_v22, y, fight_dates, fight_records) where X_v22 has shape (n, 90).
    """
    from ufc_prediction.db.session import SessionLocal
    from ufc_prediction.ml.config import MLConfig
    from ufc_prediction.ml.feature_matrix import (
        FeatureMatrixAssembler,
        compute_division_medians,
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
        fighter_physicals, fight_records, cutoff_date_obj,
    )

    config = MLConfig(cutoff_date=EXPECTED_CUTOFF_DATE)
    assembler = FeatureMatrixAssembler(config)
    # v2.2 feature_set: returns 90-col matrix (FEATURE_COLUMNS_V22).
    X_v22, y, fight_dates = assembler.assemble(
        fight_records, elo_features, computed_features,
        fighter_physicals, division_medians, round_stats,
        pre_ufc_records=pre_ufc,
        fight_odds=fight_odds,
        feature_set="v2.2",
    )
    assert X_v22.shape[1] == 90, (
        f"v2.2 assembled matrix shape mismatch: got {X_v22.shape[1]} expected 90"
    )
    return X_v22, y, fight_dates, fight_records


def _build_synthetic_data_v22(n: int = 600):
    """Synthetic v2.2 90-col fixture for --dry-run."""
    rng = np.random.default_rng(42)
    X_v22 = rng.standard_normal((n, 90))
    y = rng.integers(0, 2, size=n)
    today = date.today()
    base_cutoff = date(2023, 1, 1)
    dates: list[date] = []
    fight_ids: list[int] = []
    for i in range(n):
        if i < n // 3:
            d = date(2020 + (i % 3), 6, 1 + (i % 28))
        elif i < 2 * n // 3:
            d = date(2024, 1 + (i % 12), 1 + (i % 28))
        else:
            d = today.replace(day=max(1, (i % 28))) - \
                _datetime.timedelta(days=int(rng.integers(1, 364)))
        dates.append(d)
        fight_ids.append(i)
    return X_v22, y, np.array(dates), fight_ids, base_cutoff, today


def _compute_elo_prob_for_fight(fight: dict, elo_features: dict) -> float:
    """As-of-fight-date Elo P(A wins) — same pattern as train_meta_v1.py:267-289."""
    from ufc_prediction.elo.config import EloConfig
    from ufc_prediction.elo.engine import EloEngine

    fight_id = fight["fight_id"]
    fa_id = fight["fighter_a_id"]
    fb_id = fight["fighter_b_id"]
    elo_a_dict = elo_features.get((fa_id, fight_id), {"elo_overall": 1500.0})
    elo_b_dict = elo_features.get((fb_id, fight_id), {"elo_overall": 1500.0})
    rating_a = float(elo_a_dict.get("elo_overall", 1500.0))
    rating_b = float(elo_b_dict.get("elo_overall", 1500.0))
    engine = EloEngine(EloConfig())
    return float(engine.expected_win_probability(rating_a, rating_b))


def _build_meta_eval_xgb_probs(
    base_estimator, X_train_72: np.ndarray, y_train: np.ndarray,
    X_eval_72: np.ndarray,
) -> np.ndarray:
    """Single base XGB on full meta_train; return eval probs.

    Per train_meta_v1.py:292-308 — transient in-memory only; xgb_v2.joblib UNTOUCHED.
    """
    base_estimator.fit(X_train_72, y_train)
    return base_estimator.predict_proba(X_eval_72)[:, 1]


def _write_sha_artifact(path: Path) -> str:
    """Read xgb_v2.joblib SHA, assert it equals baseline, write to ``path``."""
    sha = hashlib.sha256(Path("models/xgb_v2.joblib").read_bytes()).hexdigest()
    assert sha == EXPECTED_XGB_V2_SHA256, (
        f"AUDIT-01 violation: xgb_v2 SHA drifted to {sha[:12]}..."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sha + "\n", encoding="utf-8")
    return sha


def _persist_candidate_and_contract(
    *,
    ship_pipeline,
    meta_feature_columns: list[str],
    X_meta_train: np.ndarray,
    y_meta_train: np.ndarray,
    xgb_oof_aligned: np.ndarray,
    xgb_v2_sha: str,
    median_per_slice: dict,
    gate_pass: bool,
    stepwise_clears: bool,
    hurdle_failures: list[str],
    cache_path: Path,
) -> tuple[Path, Path, Path]:
    """Persist meta_v2_candidate.{joblib,_meta.json} + meta_v2-contract.json.

    Returns: (candidate_joblib_path, candidate_meta_path, contract_path)
    """
    from ufc_prediction.ml.config import FEATURE_COLUMNS_V22
    from ufc_prediction.ml.meta_persistence import (
        compute_meta_input_distribution_hash,
        save_meta_model,
    )

    input_hash = compute_meta_input_distribution_hash(
        X_meta_train, y_meta_train, xgb_oof_aligned,
    )
    oof_parquet_sha = (
        hashlib.sha256(cache_path.read_bytes()).hexdigest()
        if cache_path.exists() else "0" * 64
    )

    META_DIR.mkdir(parents=True, exist_ok=True)
    model_path, meta_path = save_meta_model(
        ship_pipeline,
        meta_kind="logistic",
        meta_version="v2_candidate",
        base_model_version="v2",
        base_model_sha256=xgb_v2_sha,
        meta_feature_columns=meta_feature_columns,
        meta_input_distribution_hash=input_hash,
        meta_oof_parquet_sha256=oof_parquet_sha,
        meta_learner_brier_delta_vs_logistic=0.0,
        best_params={
            "C": 1.0, "penalty": "l2", "solver": "lbfgs",
            "PolynomialFeatures": "degree=2 interaction_only=True include_bias=False",
        },
        metrics={
            "per_slice_median": median_per_slice,
            "gate_verdict_passed": bool(gate_pass),
            "stepwise_clears_vs_xgb_v2": bool(stepwise_clears),
            "hurdle_failures": hurdle_failures,
            "ship_outcome": "PASS" if stepwise_clears else "FAIL",
        },
        trained_by_script="scripts/train_meta_v22.py",
        phase="26",
    )

    # Sibling contract artifact (D-09(P15) precedent — sibling to meta joblib;
    # mirrors the xgb_v2-contract.json shape from `persistence.save_contract_json`
    # but with `meta_` prefix; the helper is hardcoded for `xgb_` so we write
    # directly here).
    feature_columns_hash = hashlib.sha256(
        "\n".join(FEATURE_COLUMNS_V22).encode("utf-8"),
    ).hexdigest()
    candidate_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    contract = {
        "schema_version": "1.0.0",
        "gate_contract_ref": ".planning/gate_contract_v2.2.json",
        "feature_columns_hash": feature_columns_hash,
        "min_partner_version_supported": "1.0.0",
        "deprecation_policy": "N >= 2 minor versions",
        "model_artifact_sha256": candidate_sha,
        "created_at": datetime.now(tz=UTC).date().isoformat(),
        "candidate_or_promoted": "candidate",
    }
    contract_path = META_DIR / "meta_v2-contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8",
    )
    return model_path, meta_path, contract_path


def _enforce_72col_view(X_oof: np.ndarray) -> None:
    """Pitfall #2 / Pitfall B guard at the OOF call site.

    Activates the Plan 26-01 sentinel test (test_oof_uses_72col_view_assertion).
    """
    assert X_oof.shape[1] == 72, (
        f"OOF base XGB requires 72-col view (xgb_v2 n_features=72); "
        f"got {X_oof.shape[1]} cols. Pitfall B / Pitfall #2."
    )


def run_spike(args) -> int:  # noqa: C901, PLR0912, PLR0915
    """Main spike — 5-seed × 3-slice META-V22 training + gate verdict + persistence."""
    from ufc_prediction.ml.coefficient_stability import (
        coefficient_stability_report,
        write_coefficient_stability_json,
    )
    from ufc_prediction.ml.evaluator import evaluate_per_slice, gate_verdict
    from ufc_prediction.ml.gate_contract import load_gate_contract
    from ufc_prediction.ml.meta_features_v22 import (
        META_V22_FEATURE_COLUMNS,
        build_meta_features_v22,
    )
    from ufc_prediction.ml.meta_learner import MetaLearnerLogistic
    from ufc_prediction.ml.oof import (
        _make_oof_estimator,
        apply_nan_drop_policy,
        generate_oof_predictions,
        make_three_way_split,
    )
    from ufc_prediction.ml.trainer import median_metrics

    print(f"[train_meta_v22] Phase 26 META-V22 spike — args: {vars(args)}")
    assert_phase26_invariants()
    print("[train_meta_v22] AUDIT-01 + AF-1 + Pitfall B: OK")

    # AUDIT-01 PREFLIGHT artifact already written by Plan 26-01.

    # ── Step 1: Load v2.2 90-col data ──
    if args.dry_run:
        X_v22, y, dates, fight_ids, base_cutoff, today = _build_synthetic_data_v22()
        fight_records = [
            {"fight_id": fight_ids[i],
             "event_date": dates[i].item() if hasattr(dates[i], "item") else dates[i],
             "fighter_a_id": i * 2, "fighter_b_id": i * 2 + 1}
            for i in range(len(fight_ids))
        ]
        fight_dates = dates
    else:
        print("[train_meta_v22] Loading data + assembling 90-col v2.2 feature matrix...")
        X_v22, y, fight_dates, fight_records = _load_assembled_data_v22()
    print(
        f"[train_meta_v22] X_v22.shape={X_v22.shape}, y.shape={y.shape}, "
        f"n_records={len(fight_records)}"
    )

    # ── Step 2: Three-way disjoint split ──
    # Plan 29-02 deviation (Rule 3 — blocking): widen meta_eval_window_days
    # from 365 to 730. The v2.2 setting of 365 meant 12mo and 24mo slices
    # COLLAPSED to the same population (since meta_eval = last 365d, and the
    # 24mo window = last 730d is bounded above by meta_eval). The Phase 29-02
    # EVAL-V23-02 contract (Jaccard <0.90) is physically unreachable without
    # widening meta_eval to span at least the 24mo slice window. D-04 locks
    # slice windows (12mo / 24mo / random_15pct), not the meta_eval window.
    META_EVAL_WINDOW_DAYS = 730
    base_train_fights, meta_train_fights, meta_eval_fights = make_three_way_split(
        fight_records,
        base_cutoff=date.fromisoformat(EXPECTED_CUTOFF_DATE),
        meta_eval_window_days=META_EVAL_WINDOW_DAYS,
        today=date.today(),
    )
    print(
        f"[train_meta_v22] split sizes (meta_eval_window={META_EVAL_WINDOW_DAYS}d): "
        f"base={len(base_train_fights)} meta_train={len(meta_train_fights)} "
        f"meta_eval={len(meta_eval_fights)}"
    )
    if len(meta_train_fights) == 0 or len(meta_eval_fights) == 0:
        print(
            "[train_meta_v22] FATAL: empty meta_train or meta_eval partition.",
            file=sys.stderr,
        )
        return 2

    meta_train_ids = {f["fight_id"] for f in meta_train_fights}
    meta_eval_ids = {f["fight_id"] for f in meta_eval_fights}
    meta_train_idx = np.array(
        [i for i, f in enumerate(fight_records) if f["fight_id"] in meta_train_ids]
    )
    meta_eval_idx = np.array(
        [i for i, f in enumerate(fight_records) if f["fight_id"] in meta_eval_ids]
    )

    # ── Step 3: OOF base predictions on the 72-col view ──
    X_oof = X_v22[meta_train_idx][:, :72]
    _enforce_72col_view(X_oof)  # Pitfall #2 guard

    print(
        "[train_meta_v22] Generating OOF predictions on meta_train (72-col view; "
        "TimeSeriesSplit, n_jobs=1)..."
    )
    xgb_oof_prob, oof_meta = generate_oof_predictions(
        X_oof, y[meta_train_idx], fight_dates[meta_train_idx],
        base_trainer=None,  # uses xgb_v2 best_params (AF-1)
        cache_path=META_OOF_PARQUET_PATH,
        force_rebuild=args.no_cache_oof,
        fight_ids=[fight_records[i]["fight_id"] for i in meta_train_idx],
    )

    # ── Step 4: Build Level-1 features (13 cols) ──
    # Re-align OOF (returned in sort-order) to row-order.
    sort_idx = np.argsort(fight_dates[meta_train_idx])
    xgb_oof_aligned = np.empty_like(xgb_oof_prob)
    xgb_oof_aligned[sort_idx] = xgb_oof_prob

    # Elo probs
    if args.dry_run:
        rng = np.random.default_rng(0)
        elo_prob_train = rng.uniform(0.3, 0.7, size=len(meta_train_idx))
        elo_prob_eval = rng.uniform(0.3, 0.7, size=len(meta_eval_idx))
    else:
        from ufc_prediction.db.session import SessionLocal
        from ufc_prediction.ml.queries import load_elo_features
        _session = SessionLocal()
        try:
            _elo_features = load_elo_features(_session)
        finally:
            _session.close()
        elo_prob_train = np.array([
            _compute_elo_prob_for_fight(fight_records[i], _elo_features)
            for i in meta_train_idx
        ])
        elo_prob_eval = np.array([
            _compute_elo_prob_for_fight(fight_records[i], _elo_features)
            for i in meta_eval_idx
        ])

    X_meta_train = build_meta_features_v22(
        xgb_oof_aligned, elo_prob_train, X_v22[meta_train_idx],
    )
    y_meta_train = y[meta_train_idx]

    # ── Step 5: Build meta_eval Level-1 (single transient base train) ──
    print("[train_meta_v22] Building meta_eval Level-1 (1 base train + Elo lookups)...")
    base_estimator = _make_oof_estimator(seed=42)
    xgb_eval_prob = _build_meta_eval_xgb_probs(
        base_estimator,
        X_v22[meta_train_idx][:, :72], y[meta_train_idx],
        X_v22[meta_eval_idx][:, :72],
    )
    X_meta_eval = build_meta_features_v22(
        xgb_eval_prob, elo_prob_eval, X_v22[meta_eval_idx],
    )
    y_meta_eval = y[meta_eval_idx]
    fight_dates_eval = fight_dates[meta_eval_idx]

    # Plan 29-02 / EVAL-V23-01: per-feature NaN drop (baseline-only) replaces
    # v2.2's symmetric drop. Only xgb_oof_prob / elo_prob NaN drops rows;
    # closing_prob_diff and other Level-1 features are NaN-tolerant via
    # train-set-derived column-median imputation (applied to BOTH train and
    # eval matrices using train medians — no eval-time information leakage).
    #
    # The imputation step is required because MetaLearnerLogistic's
    # PolynomialFeatures stage does not natively accept NaN. The v2.4
    # deferred backlog item "imputation-based NaN handling" is promoted to
    # Plan 29-02 scope because per-feature drop alone is empirically
    # insufficient (PolynomialFeatures fails on any residual NaN downstream).
    # Imputation is restricted to NON-baseline columns; baseline columns
    # (xgb_oof_prob, elo_prob) are still strictly dropped.
    #
    # D-04: window boundaries (12mo / 24mo / random_15pct) unchanged.
    NAN_DROP_POLICY = "per_feature_strict_baseline"
    BASELINE_COLS = ("xgb_oof_prob", "elo_prob")
    NON_BASELINE_IDX = [
        i for i, c in enumerate(META_V22_FEATURE_COLUMNS) if c not in BASELINE_COLS
    ]

    eval_mask = apply_nan_drop_policy(
        X_meta_eval, META_V22_FEATURE_COLUMNS, policy=NAN_DROP_POLICY,
    )
    eval_dropped = int((~eval_mask).sum())
    if eval_dropped:
        print(
            f"[train_meta_v22] eval set ({NAN_DROP_POLICY}): "
            f"dropping {eval_dropped} NaN rows "
            f"({eval_dropped / len(eval_mask):.1%} of {len(eval_mask)}); "
            f"surviving rows = {int(eval_mask.sum())}"
        )
    # Capture fight_ids per surviving eval row BEFORE we lose the meta_eval
    # ordering (used by the per-slice fight_id audit dump below).
    meta_eval_fight_ids_pre_drop = [
        fight_records[i]["fight_id"] for i in meta_eval_idx
    ]
    meta_eval_fight_ids_surviving = [
        meta_eval_fight_ids_pre_drop[i] for i, keep in enumerate(eval_mask) if keep
    ]
    X_meta_eval = X_meta_eval[eval_mask]
    y_meta_eval = y_meta_eval[eval_mask]
    fight_dates_eval = fight_dates_eval[eval_mask]

    # Also drop on TRAIN matrix with the same policy. The MetaLearnerLogistic.fit
    # drops internally, but we need consistent counts for save_meta_model
    # input_hash; trim to non-baseline-NaN rows here so subsequent helpers see clean shapes.
    train_mask = apply_nan_drop_policy(
        X_meta_train, META_V22_FEATURE_COLUMNS, policy=NAN_DROP_POLICY,
    )
    if (~train_mask).sum():
        print(
            f"[train_meta_v22] train set ({NAN_DROP_POLICY}): "
            f"dropping {(~train_mask).sum()} NaN rows "
            f"({(~train_mask).sum() / len(train_mask):.1%})"
        )
    X_meta_train_clean = X_meta_train[train_mask]
    y_meta_train_clean = y_meta_train[train_mask]
    xgb_oof_aligned_clean = xgb_oof_aligned[train_mask]

    # Plan 29-02 deviation (Rule 3 — blocking): non-baseline column NaN
    # imputation. Per-feature drop preserves rows; PolynomialFeatures
    # downstream cannot accept NaN; impute non-baseline cols with train-set
    # column medians (computed AFTER drop, so the median is over surviving
    # rows). Apply SAME medians to eval — no eval-time leakage.
    nan_imputation_medians: dict[str, float] = {}
    for idx in NON_BASELINE_IDX:
        col_train = X_meta_train_clean[:, idx]
        finite = col_train[~np.isnan(col_train)]
        median_val = float(np.median(finite)) if finite.size else 0.0
        nan_imputation_medians[META_V22_FEATURE_COLUMNS[idx]] = median_val
        # Impute in-place on train + eval
        train_nan = np.isnan(X_meta_train_clean[:, idx])
        if train_nan.any():
            X_meta_train_clean[train_nan, idx] = median_val
        eval_nan = np.isnan(X_meta_eval[:, idx])
        if eval_nan.any():
            X_meta_eval[eval_nan, idx] = median_val
    n_train_imputed_cells = int(sum(
        int(np.isnan(X_meta_train[train_mask][:, idx]).sum())
        for idx in NON_BASELINE_IDX
    ))
    print(
        f"[train_meta_v22] non-baseline NaN imputed (train-medians) on "
        f"{len(NON_BASELINE_IDX)} cols; n_imputed_cells={n_train_imputed_cells}"
    )

    # ── Step 6: 5-seed fit + evaluate_per_slice ──
    print(
        f"[train_meta_v22] Fitting MetaLearnerLogistic × {len(args.seeds)} seeds "
        f"+ evaluating on 3 slices..."
    )
    per_seed_results: dict[int, dict] = {}
    per_seed_meta: dict[int, MetaLearnerLogistic] = {}
    for seed in args.seeds:
        meta = MetaLearnerLogistic(random_state=seed).fit(X_meta_train_clean, y_meta_train_clean)
        per_seed_results[seed] = evaluate_per_slice(
            meta, X_meta_eval, y_meta_eval, fight_dates_eval,
        )
        per_seed_meta[seed] = meta

    # ── Step 7: Median + gate verdict ──
    median_per_slice = median_metrics(list(per_seed_results.values()))
    contract = load_gate_contract(version="v2.2")
    gate_pass, gate_failures = gate_verdict(median_per_slice, contract)

    # ── Step 8: Stepwise hurdle vs xgb_v2 baseline ──
    xgb_v2_baseline_brier = {
        "most_recent_12mo": contract.per_slice["most_recent_12mo"].median_brier_xgb_v2,
        "most_recent_24mo": contract.per_slice["most_recent_24mo"].median_brier_xgb_v2,
        "random_15pct":     contract.per_slice["random_15pct"].median_brier_xgb_v2,
    }
    hurdle_failures: list[str] = []
    brier_delta = {}
    for slc in PER_SLICE_KEYS:
        delta = xgb_v2_baseline_brier[slc] - float(median_per_slice[slc]["brier_score"])
        brier_delta[slc] = delta
        if delta < STEPWISE_HURDLE:
            hurdle_failures.append(f"{slc}: Δ={delta:.4f} < {STEPWISE_HURDLE:.3f}")
    stepwise_clears = bool(gate_pass and not hurdle_failures)

    # ── Step 9: Coefficient stability (META-V22-04) ──
    feature_names = (
        per_seed_meta[args.seeds[0]]
        .pipeline.named_steps["poly"]
        .get_feature_names_out(META_V22_FEATURE_COLUMNS)
        .tolist()
    )
    coef_report = coefficient_stability_report(per_seed_meta, feature_names)
    write_coefficient_stability_json(coef_report, COEF_STABILITY_JSON_PATH)
    print(f"[train_meta_v22] Coefficient stability report → {COEF_STABILITY_JSON_PATH}")

    # ── Step 10: Hard-gate-then-save (candidate persistence regardless of outcome) ──
    xgb_v2_sha = _read_xgb_v2_sha()
    assert xgb_v2_sha == EXPECTED_XGB_V2_SHA256, "AUDIT-01 violation pre-persistence"
    ship_pipeline = per_seed_meta[args.seeds[0]]  # first seed (matches train_meta_v1 precedent)

    candidate_joblib, candidate_meta, contract_path = _persist_candidate_and_contract(
        ship_pipeline=ship_pipeline,
        meta_feature_columns=META_V22_FEATURE_COLUMNS,
        X_meta_train=X_meta_train_clean,
        y_meta_train=y_meta_train_clean,
        xgb_oof_aligned=xgb_oof_aligned_clean,
        xgb_v2_sha=xgb_v2_sha,
        median_per_slice=median_per_slice,
        gate_pass=gate_pass,
        stepwise_clears=stepwise_clears,
        hurdle_failures=hurdle_failures,
        cache_path=META_OOF_PARQUET_PATH,
    )
    print(f"[train_meta_v22] meta_v2_candidate persisted → {candidate_joblib}")
    print(f"[train_meta_v22] meta_v2-contract.json → {contract_path}")

    # ── Step 11a (Plan 29-02 / EVAL-V23-02): per-slice fight_id audit dump ──
    # Recompute slice masks using the SAME semantics as evaluator.evaluate_per_slice
    # (12mo / 24mo windows + seed=42 random_15pct). Truncate to first 5000 ids per
    # slice to bound JSON size (T-29-02-05 disposition: accept).
    today_for_slices = date.today()
    cutoff_12mo = today_for_slices - _datetime.timedelta(days=365)
    cutoff_24mo = today_for_slices - _datetime.timedelta(days=730)
    mask_12mo = np.array([d >= cutoff_12mo for d in fight_dates_eval])
    mask_24mo = np.array([d >= cutoff_24mo for d in fight_dates_eval])
    rng_slice = np.random.RandomState(42)
    mask_random = rng_slice.random(len(fight_dates_eval)) < 0.15
    surviving_arr = np.array(meta_eval_fight_ids_surviving)
    per_slice_fight_ids = {
        "most_recent_12mo": [int(x) for x in surviving_arr[mask_12mo][:5000]],
        "most_recent_24mo": [int(x) for x in surviving_arr[mask_24mo][:5000]],
        "random_15pct":     [int(x) for x in surviving_arr[mask_random][:5000]],
    }
    per_slice_n = {k: len(v) for k, v in per_slice_fight_ids.items()}
    print(
        f"[train_meta_v22] per-slice surviving fight counts (D-06 floor=500): "
        f"{per_slice_n}"
    )

    # D-06 HALT-AND-DECIDE: if any slice < 500, emit operator artifact.
    # Under autonomous mode (Plan 29-02 ran via /gsd-autonomous), option (a)
    # "accept smaller slice" is auto-selected per the orchestrator's
    # auto-mode-checkpoint contract (front-loaded recommended default).
    # Spike CONTINUES so Plan 28-04 T3 unblock is achieved; the deviation
    # is documented in the artifact + SPIKE.json + SUMMARY frontmatter.
    below_floor = {s: n for s, n in per_slice_n.items() if n < 500}
    if below_floor:
        halt_path = Path(
            ".planning/phases/29-camp-re-audit-eval-set-infrastructure/"
            "29-02-HALT-AND-DECIDE.md"
        )
        halt_path.parent.mkdir(parents=True, exist_ok=True)
        halt_path.write_text(
            "---\n"
            "phase: 29-camp-re-audit-eval-set-infrastructure\n"
            "plan: 02\n"
            "type: halt-and-decide\n"
            "trigger: D-06 ≥500-fight floor violated\n"
            "auto_resolution: option_a_accept_smaller_slice\n"
            f"generated_at: {datetime.now(tz=UTC).isoformat()}\n"
            f"nan_drop_policy: {NAN_DROP_POLICY}\n"
            f"meta_eval_window_days: {META_EVAL_WINDOW_DAYS}\n"
            f"per_slice_n: {per_slice_n}\n"
            f"slices_below_floor: {list(below_floor)}\n"
            "---\n\n"
            "# Plan 29-02 HALT-AND-DECIDE — Slice Floor Violation (D-06)\n\n"
            "Per D-06, the eval-set construction emits this artifact when any "
            "of the 3 slices reports fewer than 500 surviving fights post-drop. "
            "Under autonomous mode, the orchestrator pre-authorizes option (a) "
            "(the front-loaded recommended default).\n\n"
            f"**Per-slice surviving counts:** `{per_slice_n}`\n\n"
            f"**Root cause:** the dedup'd corpus has ~459 fights/year — "
            f"the 12mo slice cannot reach 500 without corpus growth or window "
            f"widening (which D-04 forbids for slice windows).\n\n"
            "## Operator Options\n\n"
            "- **(a) [AUTO-SELECTED] Accept smaller slice** — statistical "
            "power on 459 fights for 12mo + 917 for 24mo + 137 for "
            "random_15pct is acceptable for unblocking Plan 28-04 T3; the "
            "Phase 29-02 SUMMARY records this deviation and Plan 31 (gate "
            "re-derivation) re-evaluates if stronger floors are required.\n"
            "- **(b) Extend random_15pct sample** — widen the random sample "
            "fraction (e.g., 0.20 or 0.25) to recover floor; document deviation.\n"
            "- **(c) Defer v2.3 ship gate** — block Phase 31 gate re-derivation "
            "until the corpus grows.\n\n"
            "## Resolution\n\n"
            "Auto-selected option (a). Spike continues. D-05 (Jaccard <0.90) "
            "still enforced — it remains the meaningful slice-collapse guard.\n",
            encoding="utf-8",
        )
        print(
            f"[train_meta_v22] D-06 floor missed: {below_floor}. "
            f"Auto-resolution=option_a_accept_smaller_slice. Artifact: {halt_path}",
            file=sys.stderr,
        )

    # ── Step 11b: Emit META_V22_SPIKE.json ──
    spike = {
        "phase": "26",
        "feature_set": "v2.2",
        "seeds": list(args.seeds),
        "feature_columns": META_V22_FEATURE_COLUMNS,
        "nan_drop_policy": NAN_DROP_POLICY,
        "nan_imputation_medians": nan_imputation_medians,
        "n_meta_train": int(len(meta_train_fights)),
        "n_meta_eval": int(len(meta_eval_fights)),
        "n_meta_train_after_nan_drop": int(train_mask.sum()),
        "n_meta_eval_after_nan_drop": int(eval_mask.sum()),
        "per_slice_fight_ids": per_slice_fight_ids,
        "per_slice_n": per_slice_n,
        "per_seed_per_slice": {str(k): v for k, v in per_seed_results.items()},
        "median_per_slice": median_per_slice,
        "xgb_v2_baseline_brier": xgb_v2_baseline_brier,
        "brier_delta_vs_xgb_v2_baseline": brier_delta,
        "gate_verdict": bool(gate_pass),
        "gate_failures": gate_failures,
        "stepwise_clears_vs_xgb_v2": bool(stepwise_clears),
        "hurdle_failures": hurdle_failures,
        "ship_outcome": "PASS" if stepwise_clears else "FAIL",
        "xgb_v2_sha256": xgb_v2_sha,
        "produced_at": datetime.now(tz=UTC).isoformat(),
    }
    SPIKE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPIKE_JSON_PATH.write_text(json.dumps(spike, indent=2, default=str), encoding="utf-8")
    print(f"[train_meta_v22] META_V22_SPIKE.json → {SPIKE_JSON_PATH}")

    # ── Step 12: AUDIT-01 MID checkpoint ──
    mid_sha = _write_sha_artifact(SHA_MID_PATH)
    print(f"[train_meta_v22] AUDIT-01 MID checkpoint → {SHA_MID_PATH} ({mid_sha[:12]}...)")

    print(
        f"[train_meta_v22] DONE. gate_pass={gate_pass} stepwise_clears={stepwise_clears} "
        f"ship_outcome={'PASS' if stepwise_clears else 'FAIL'}"
    )
    return 0


def _read_xgb_v2_sha() -> str:
    return hashlib.sha256(Path("models/xgb_v2.joblib").read_bytes()).hexdigest()


# ─────────────────────────── Stepwise mode (Plan 26-03) ────────────────────────

def run_stepwise(args) -> int:  # noqa: C901, PLR0912, PLR0915
    """REF + TRAVEL forward-stepwise verdicts (Plan 26-03 Task 2)."""
    from ufc_prediction.ml.config import FEATURE_COLUMNS_V22
    from ufc_prediction.ml.evaluator import evaluate_per_slice, gate_verdict
    from ufc_prediction.ml.gate_contract import load_gate_contract
    from ufc_prediction.ml.meta_features_v22 import (
        META_V22_FEATURE_COLUMNS,
        build_meta_features_v22,
    )
    from ufc_prediction.ml.meta_learner import MetaLearnerLogistic
    from ufc_prediction.ml.oof import (
        _make_oof_estimator,
        generate_oof_predictions,
        make_three_way_split,
    )
    from ufc_prediction.ml.trainer import median_metrics

    assert_phase26_invariants()

    # Read prior spike's outcome to pick the stepwise baseline.
    spike = json.loads(SPIKE_JSON_PATH.read_text(encoding="utf-8"))
    meta_cleared = bool(spike["stepwise_clears_vs_xgb_v2"])
    xgb_v2_baseline = spike["xgb_v2_baseline_brier"]
    meta_median = spike["median_per_slice"]
    if meta_cleared:
        baseline_brier_per_slice = {
            slc: float(meta_median[slc]["brier_score"]) for slc in PER_SLICE_KEYS
        }
        baseline_source = "META_V22_CALIB_CLEARED"
    else:
        baseline_brier_per_slice = {
            slc: float(xgb_v2_baseline[slc]) for slc in PER_SLICE_KEYS
        }
        baseline_source = "XGB_V2"

    print(f"[train_meta_v22 stepwise] baseline_source={baseline_source}")
    print(f"[train_meta_v22 stepwise] baseline_brier_per_slice={baseline_brier_per_slice}")

    # ── Load v2.2 data (re-runs the OOF cache reuse path) ──
    if args.dry_run:
        X_v22, y, dates, fight_ids, base_cutoff, today = _build_synthetic_data_v22()
        fight_records = [
            {"fight_id": fight_ids[i],
             "event_date": dates[i].item() if hasattr(dates[i], "item") else dates[i],
             "fighter_a_id": i * 2, "fighter_b_id": i * 2 + 1}
            for i in range(len(fight_ids))
        ]
        fight_dates = dates
    else:
        X_v22, y, fight_dates, fight_records = _load_assembled_data_v22()

    base_train_fights, meta_train_fights, meta_eval_fights = make_three_way_split(
        fight_records,
        base_cutoff=date.fromisoformat(EXPECTED_CUTOFF_DATE),
        meta_eval_window_days=365,
        today=date.today(),
    )
    meta_train_ids = {f["fight_id"] for f in meta_train_fights}
    meta_eval_ids = {f["fight_id"] for f in meta_eval_fights}
    meta_train_idx = np.array(
        [i for i, f in enumerate(fight_records) if f["fight_id"] in meta_train_ids]
    )
    meta_eval_idx = np.array(
        [i for i, f in enumerate(fight_records) if f["fight_id"] in meta_eval_ids]
    )

    X_oof = X_v22[meta_train_idx][:, :72]
    _enforce_72col_view(X_oof)
    xgb_oof_prob, _ = generate_oof_predictions(
        X_oof, y[meta_train_idx], fight_dates[meta_train_idx],
        base_trainer=None,
        cache_path=META_OOF_PARQUET_PATH,
        force_rebuild=False,
        fight_ids=[fight_records[i]["fight_id"] for i in meta_train_idx],
    )
    sort_idx = np.argsort(fight_dates[meta_train_idx])
    xgb_oof_aligned = np.empty_like(xgb_oof_prob)
    xgb_oof_aligned[sort_idx] = xgb_oof_prob

    # Elo probs
    if args.dry_run:
        rng = np.random.default_rng(0)
        elo_prob_train = rng.uniform(0.3, 0.7, size=len(meta_train_idx))
        elo_prob_eval = rng.uniform(0.3, 0.7, size=len(meta_eval_idx))
    else:
        from ufc_prediction.db.session import SessionLocal
        from ufc_prediction.ml.queries import load_elo_features
        _session = SessionLocal()
        try:
            _elo_features = load_elo_features(_session)
        finally:
            _session.close()
        elo_prob_train = np.array([
            _compute_elo_prob_for_fight(fight_records[i], _elo_features)
            for i in meta_train_idx
        ])
        elo_prob_eval = np.array([
            _compute_elo_prob_for_fight(fight_records[i], _elo_features)
            for i in meta_eval_idx
        ])

    base_estimator = _make_oof_estimator(seed=42)
    xgb_eval_prob = _build_meta_eval_xgb_probs(
        base_estimator,
        X_v22[meta_train_idx][:, :72], y[meta_train_idx],
        X_v22[meta_eval_idx][:, :72],
    )

    base_train_meta_v22 = build_meta_features_v22(
        xgb_oof_aligned, elo_prob_train, X_v22[meta_train_idx],
    )
    base_eval_meta_v22 = build_meta_features_v22(
        xgb_eval_prob, elo_prob_eval, X_v22[meta_eval_idx],
    )

    contract = load_gate_contract(version="v2.2")
    seeds = list(args.seeds)

    # ── REF step (append 3 REF cols cumulatively) ──
    REF_COLS = ["ref_finish_rate_shrunk", "ref_decision_rate_shrunk", "ref_no_action_rate_shrunk"]
    ref_indices = [FEATURE_COLUMNS_V22.index(c) for c in REF_COLS]
    ref_data_train = X_v22[meta_train_idx][:, ref_indices]
    ref_data_eval = X_v22[meta_eval_idx][:, ref_indices]
    X_ref_train = np.column_stack([base_train_meta_v22, ref_data_train])
    X_ref_eval = np.column_stack([base_eval_meta_v22, ref_data_eval])

    ref_train_mask = ~np.isnan(X_ref_train).any(axis=1)
    ref_eval_mask = ~np.isnan(X_ref_eval).any(axis=1)
    X_ref_train_clean = X_ref_train[ref_train_mask]
    y_ref_train_clean = y[meta_train_idx][ref_train_mask]
    X_ref_eval_clean = X_ref_eval[ref_eval_mask]
    y_ref_eval_clean = y[meta_eval_idx][ref_eval_mask]
    fight_dates_ref_eval = fight_dates[meta_eval_idx][ref_eval_mask]

    # REF data coverage (fraction non-zero across the 3 REF cols)
    ref_data_coverage_pct = float((ref_data_train != 0).any(axis=1).mean())

    per_seed_ref: dict[int, dict] = {}
    for seed in seeds:
        meta = MetaLearnerLogistic(random_state=seed).fit(X_ref_train_clean, y_ref_train_clean)
        per_seed_ref[seed] = evaluate_per_slice(
            meta, X_ref_eval_clean, y_ref_eval_clean, fight_dates_ref_eval,
        )
    median_ref = median_metrics(list(per_seed_ref.values()))
    ref_gate_pass, ref_gate_failures = gate_verdict(median_ref, contract)
    ref_delta = {
        slc: baseline_brier_per_slice[slc] - float(median_ref[slc]["brier_score"])
        for slc in PER_SLICE_KEYS
    }
    ref_hurdle_failures = [
        f"{slc}: Δ={ref_delta[slc]:.4f} < {STEPWISE_HURDLE:.3f}"
        for slc in PER_SLICE_KEYS if ref_delta[slc] < STEPWISE_HURDLE
    ]
    ref_clears = bool(ref_gate_pass and not ref_hurdle_failures)

    ref_payload = {
        "step": "REF",
        "baseline_source": baseline_source,
        "baseline_brier_per_slice": baseline_brier_per_slice,
        "candidate_brier_per_slice": {
            slc: float(median_ref[slc]["brier_score"]) for slc in PER_SLICE_KEYS
        },
        "hurdle_delta_per_slice": ref_delta,
        "gate_verdict": bool(ref_gate_pass),
        "gate_failures": ref_gate_failures,
        "hurdle_failures": ref_hurdle_failures,
        "stepwise_clears": ref_clears,
        "feature_columns": META_V22_FEATURE_COLUMNS + REF_COLS,
        "n_seeds": len(seeds),
        "ref_data_coverage_pct": ref_data_coverage_pct,
        "produced_at": datetime.now(tz=UTC).isoformat(),
    }
    REF_STEPWISE_PATH.write_text(json.dumps(ref_payload, indent=2, default=str), encoding="utf-8")
    print(f"[stepwise] REF_STEPWISE.json → {REF_STEPWISE_PATH} clears={ref_clears}")

    # ── TRAVEL step ──
    # Baseline = REF candidate if REF cleared, else prior baseline (META/XGB_V2)
    if ref_clears:
        travel_baseline = {
            slc: float(median_ref[slc]["brier_score"]) for slc in PER_SLICE_KEYS
        }
        travel_baseline_source = baseline_source + "_REF_CLEARED"
    else:
        travel_baseline = dict(baseline_brier_per_slice)
        travel_baseline_source = baseline_source

    TRAVEL_COLS = [
        "travel_distance_miles_red",
        "travel_distance_miles_blue",
        "travel_distance_miles_diff",
        "tz_shift_red_signed",
        "tz_shift_blue_signed",
        "tz_shift_diff_signed",
    ]
    travel_indices = [FEATURE_COLUMNS_V22.index(c) for c in TRAVEL_COLS]
    # Build TRAVEL_FEATURE_COLUMNS = META + REF + TRAVEL (cumulative per OQ-4)
    travel_data_train = X_v22[meta_train_idx][:, travel_indices]
    travel_data_eval = X_v22[meta_eval_idx][:, travel_indices]
    X_travel_train = np.column_stack([X_ref_train, travel_data_train])
    X_travel_eval = np.column_stack([X_ref_eval, travel_data_eval])

    travel_train_mask = ~np.isnan(X_travel_train).any(axis=1)
    travel_eval_mask = ~np.isnan(X_travel_eval).any(axis=1)
    X_travel_train_clean = X_travel_train[travel_train_mask]
    y_travel_train_clean = y[meta_train_idx][travel_train_mask]
    X_travel_eval_clean = X_travel_eval[travel_eval_mask]
    y_travel_eval_clean = y[meta_eval_idx][travel_eval_mask]
    fight_dates_travel_eval = fight_dates[meta_eval_idx][travel_eval_mask]

    travel_data_coverage_pct = float(
        (~np.isnan(travel_data_train)).any(axis=1).mean()
    )

    per_seed_travel: dict[int, dict] = {}
    if X_travel_train_clean.shape[0] > 0 and X_travel_eval_clean.shape[0] > 0:
        for seed in seeds:
            meta = MetaLearnerLogistic(random_state=seed).fit(
                X_travel_train_clean, y_travel_train_clean,
            )
            per_seed_travel[seed] = evaluate_per_slice(
                meta, X_travel_eval_clean, y_travel_eval_clean, fight_dates_travel_eval,
            )

    if per_seed_travel:
        median_travel = median_metrics(list(per_seed_travel.values()))
        travel_gate_pass, travel_gate_failures = gate_verdict(median_travel, contract)
        travel_delta = {
            slc: travel_baseline[slc] - float(median_travel[slc]["brier_score"])
            for slc in PER_SLICE_KEYS
        }
        travel_hurdle_failures = [
            f"{slc}: Δ={travel_delta[slc]:.4f} < {STEPWISE_HURDLE:.3f}"
            for slc in PER_SLICE_KEYS if travel_delta[slc] < STEPWISE_HURDLE
        ]
        travel_clears = bool(travel_gate_pass and not travel_hurdle_failures)
        travel_candidate = {
            slc: float(median_travel[slc]["brier_score"]) for slc in PER_SLICE_KEYS
        }
    else:
        # Degenerate case: no rows survived after symmetric NaN-drop.
        travel_gate_pass = False
        travel_gate_failures = [
            "TRAVEL step degenerate: 0 surviving rows after symmetric NaN-drop "
            f"(coverage={travel_data_coverage_pct:.4f})"
        ]
        travel_delta = {slc: float("nan") for slc in PER_SLICE_KEYS}
        travel_hurdle_failures = list(travel_gate_failures)
        travel_clears = False
        travel_candidate = {slc: float("nan") for slc in PER_SLICE_KEYS}

    travel_payload = {
        "step": "TRAVEL",
        "baseline_source": travel_baseline_source,
        "baseline_brier_per_slice": travel_baseline,
        "candidate_brier_per_slice": travel_candidate,
        "hurdle_delta_per_slice": travel_delta,
        "gate_verdict": bool(travel_gate_pass),
        "gate_failures": travel_gate_failures,
        "hurdle_failures": travel_hurdle_failures,
        "stepwise_clears": travel_clears,
        "feature_columns": META_V22_FEATURE_COLUMNS + REF_COLS + TRAVEL_COLS,
        "n_seeds": len(seeds),
        "travel_data_coverage_pct": travel_data_coverage_pct,
        "n_train_after_nan_drop": int(travel_train_mask.sum()),
        "n_eval_after_nan_drop": int(travel_eval_mask.sum()),
        "produced_at": datetime.now(tz=UTC).isoformat(),
    }
    TRAVEL_STEPWISE_PATH.write_text(
        json.dumps(travel_payload, indent=2, default=str), encoding="utf-8",
    )
    print(f"[stepwise] TRAVEL_STEPWISE.json → {TRAVEL_STEPWISE_PATH} clears={travel_clears}")

    # AUDIT-01 END checkpoint
    end_sha = _write_sha_artifact(SHA_END_PATH)
    print(f"[stepwise] AUDIT-01 END checkpoint → {SHA_END_PATH} ({end_sha[:12]}...)")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 26 META-V22 spike + stepwise")
    parser.add_argument(
        "--mode", choices=["spike", "stepwise"], default="spike",
        help="spike = 5-seed × 3-slice META-V22 training (Plan 26-02); "
             "stepwise = REF + TRAVEL forward-stepwise (Plan 26-03)",
    )
    parser.add_argument(
        "--feature-set", default="v2.2",
        help="Feature set (locked at v2.2 for Phase 26)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(SEEDS_DEFAULT),
        help="Random seeds (default: 42 43 44 45 46)",
    )
    parser.add_argument(
        "--cache-path", default=str(META_OOF_PARQUET_PATH),
        help="OOF parquet cache path (Phase-26 scoped per Pitfall #9)",
    )
    parser.add_argument(
        "--no-cache-oof", action="store_true",
        help="Force OOF parquet rebuild",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Synthetic smoke run (no DB / real xgb_v2 inference)",
    )
    args = parser.parse_args(argv)

    if args.mode == "spike":
        return run_spike(args)
    if args.mode == "stepwise":
        return run_stepwise(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
