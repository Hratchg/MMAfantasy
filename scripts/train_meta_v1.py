#!/usr/bin/env python
"""Phase 19 / META-01 — operator-driven 5-seed × 3-slice META-01 training spike.

Reads `models/xgb_v2.joblib` (sha256 6e7641...0a99 — IMMUTABLE per AUDIT-01)
and `models/xgb_v2_meta.json`, asserts AF-1 (best_params verbatim) + AUDIT-01
SHA byte-identity + Pitfall B n_features=72, applies the three-way split per
D-01(P19), generates OOF predictions via `oof.generate_oof_predictions`
(cross_val_predict + TimeSeriesSplit + n_jobs=1), fits META-01 per D-02(P19)
across 5 seeds, evaluates against the v2.1 gate contract on the 3-slice
harness (ALL-pass discipline per D-13(P16)), and emits:

  - `models/meta/meta_v1.joblib` + `meta_v1_meta.json` (ONLY if gate clears)
  - `.planning/phases/19-meta-learner/oof_predictions.parquet` (cached per D-06(P19))
  - `.planning/phases/19-meta-learner/archive/meta_v1_<sha8>.{joblib,_meta.json,sha256}` (AUDIT-03)
  - `.planning/phases/19-meta-learner/19-META-LEARNER-REPORT.md`

Locked decisions (Phase 19):
  - D-01(P19): three-way split with disjoint-id assertion (meta_train ⊂ post-cutoff)
  - D-02(P19): minimal 3-feature set [xgb_oof_prob, elo_prob, closing_prob_diff]
  - D-06(P19): OOF cache to .planning/phases/19-meta-learner/oof_predictions.parquet
  - D-15(P16): n_jobs=1 (Py 3.14 spawn pickling)
  - D-16(P16): seeds (42, 43, 44, 45, 46), median wins
  - D-13(P16): 3-slice ALL-pass discipline
  - D-07(P15): hard-gate-then-save discipline

Anti-features (BANNED):
  - AF-1 (no hparam retuning): xgb_v2 best_params verbatim hold
  - AF-10 (no tuning hurdle downward): META_02_DELTA_BRIER_MIN locked at 0.003

Pitfalls mitigated:
  - Pitfall #11 (stacking leakage): cross_val_predict cv=TimeSeriesSplit, n_jobs=1
  - Pitfall #14 (distribution mismatch): three-way split disjoint assertion
  - Pitfall #15 (schema collision): meta_persistence.py SIBLING module
  - Pitfall D (auto-commit): script does NOT call git. Operator owns commits.
  - Pitfall E (corpus drift): Wave-2 dry-run reproduces xgb_v2 baseline first.

Usage:
    uv run python scripts/train_meta_v1.py --dry-run         # synthetic smoke
    uv run python scripts/train_meta_v1.py --seeds 42 43 44 45 46  # full spike
    uv run python scripts/train_meta_v1.py --no-cache-oof    # forced OOF rebuild
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import shutil
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from ufc_prediction.ml.meta_persistence import (
    compute_meta_input_distribution_hash,
    save_meta_model,
)

# Module-level locked constants
EXPECTED_XGB_V2_SHA256: str = "0b0b40afc8ec41d87508745a9b5f40a46f7d86c054b1ab2acece03d319f6fecd"
EXPECTED_FORMULA_HASH: str = "7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a"
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
    "most_recent_12mo",
    "most_recent_24mo",
    "random_15pct",
)
META_OOF_PARQUET_PATH: Path = Path(".planning/phases/19-meta-learner/oof_predictions.parquet")
META_ARCHIVE_DIR: Path = Path(".planning/phases/19-meta-learner/archive")
META_DIR: Path = Path("models/meta")
META_LEARNER_REPORT_PATH: Path = Path(".planning/phases/19-meta-learner/19-META-LEARNER-REPORT.md")

# META-01 logistic baseline hyperparams (mirrors meta_learner.MetaLearnerLogistic
# defaults). Promoted to module-level so save_meta_model receives a stable dict
# and tests can introspect it without instantiating the meta model.
META01_BEST_PARAMS: dict = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 42,
}

# Module-global capture of the most recent 5-seed run results, populated at the
# end of main() so the Plan 19-02 Task 1a integration tests can inspect the
# per-seed/per-slice metrics dict without parsing stdout. Keys are seed ints;
# values are the dict returned by ``evaluator.evaluate_per_slice``.
LAST_PER_SEED_RESULTS: dict[int, dict] = {}


# Re-exported here so monkeypatching ``train_meta_v1.gate_verdict`` in Plan
# 19-02 Task 1a tests redirects the gate decision without touching the
# evaluator module's binding.
from ufc_prediction.ml.evaluator import gate_verdict


def assert_phase19_invariants() -> None:
    """Halt if any precondition is violated (AF-1 + AUDIT-01 + Pitfall B + Phase 17 gate identity)."""
    # AUDIT-01 — xgb_v2 byte-identity
    sha_actual = hashlib.sha256(Path("models/xgb_v2.joblib").read_bytes()).hexdigest()
    assert sha_actual == EXPECTED_XGB_V2_SHA256, (
        f"AUDIT-01 violation: xgb_v2 SHA drift. got={sha_actual} expected={EXPECTED_XGB_V2_SHA256}"
    )
    meta = json.loads(Path("models/xgb_v2_meta.json").read_text(encoding="utf-8"))
    # Pitfall B — n_features dispatch
    assert meta["n_features"] == EXPECTED_XGB_V2_N_FEATURES, (
        f"Pitfall B violation: xgb_v2 n_features={meta['n_features']!r} "
        f"expected={EXPECTED_XGB_V2_N_FEATURES}"
    )
    # D-04(P16) cutoff parity
    assert meta["cutoff_date"] == EXPECTED_CUTOFF_DATE, (
        f"D-04(P16) violation: cutoff_date={meta['cutoff_date']!r} expected={EXPECTED_CUTOFF_DATE}"
    )
    # AF-1 — no hparam retuning
    assert meta["best_params"] == EXPECTED_XGB_V2_BEST_PARAMS, (
        f"AF-1 violation: xgb_v2 best_params drift. Got: {meta['best_params']}"
    )
    # Phase 17 gate contract identity
    contract = json.loads(Path(".planning/gate_contract.json").read_text(encoding="utf-8"))
    assert contract["formula_hash"] == EXPECTED_FORMULA_HASH, (
        f"Phase 17 gate_contract.json formula_hash drift. "
        f"got={contract['formula_hash']} expected={EXPECTED_FORMULA_HASH}"
    )


def archive_meta_candidate(
    name: str,
    joblib_path: Path,
    meta_path: Path,
    archive_dir: Path = META_ARCHIVE_DIR,
) -> str:
    """AUDIT-03 (D-10(P19) carry-forward of D-09(P18)): archive candidate bundle.

    Mirrors scripts/spike_net_v2_design.py::_archive_loser shape.
    Returns the sha256 of the archived joblib for traceability.
    """
    archive_dir.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(joblib_path.read_bytes()).hexdigest()
    suffix = f"_{sha[:8]}"
    target_joblib = archive_dir / f"{name}{suffix}.joblib"
    target_meta = archive_dir / f"{name}{suffix}_meta.json"
    target_sha_anchor = archive_dir / f"{name}{suffix}.sha256"
    shutil.copy2(joblib_path, target_joblib)
    shutil.copy2(meta_path, target_meta)
    target_sha_anchor.write_text(sha + "\n", encoding="utf-8")
    return sha


def _build_synthetic_demo_data(n: int = 600, n_features: int = 72, seed: int = 42):
    """Synthetic --dry-run data: 3 partitions of 200 rows each, dates spanning
    pre-cutoff / mid-window / last-365-days."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 2, size=n)
    dates: list[date] = []
    fight_ids: list[int] = []
    today = date.today()
    base_cutoff = date(2023, 1, 1)
    for i in range(n):
        if i < n // 3:
            d = date(2020 + (i % 3), 6, 1 + (i % 28))  # pre-cutoff
        elif i < 2 * n // 3:
            d = date(2024, 1 + (i % 12), 1 + (i % 28))  # post-cutoff, > 365d ago
        else:
            d = today.replace(day=max(1, (i % 28))) - _datetime.timedelta(
                days=int(rng.integers(1, 364))
            )
        dates.append(d)
        fight_ids.append(i)
    return X, y, np.array(dates), fight_ids, base_cutoff, today


def _load_assembled_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Load fights from the DB and assemble the canonical 72-col feature matrix.

    Mirrors the loader pattern established by
    ``scripts/spike_net_v2_design.py::_load_data + assemble`` (lines 640-680).
    Returns ``(X, y, fight_dates, fight_records)`` where:

      - X is ``(n, 72)`` float64 — the 72-col view (NET-* dropped, per D-19(v2.1, NET))
      - y is ``(n,)`` int — A wins (1) / B wins (0)
      - fight_dates is ``(n,)`` object ndarray of ``datetime.date`` (event_date)
      - fight_records is the original ``load_fight_records()`` list-of-dicts,
        kept aligned by row-index with X/y/fight_dates so the wiring can pull
        ``fight_id`` / ``fighter_a_id`` / ``fighter_b_id`` for the three-way
        split + Elo lookups without rebuilding from X.

    Encapsulated as a separate helper so the Plan 19-02 Task 1a integration
    tests can mock this single function and run the full main() loop against
    synthetic data without a live DB or BFO HTTP path.
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
        fighter_physicals,
        fight_records,
        cutoff_date_obj,
    )

    # include_net=False per D-19(v2.1, NET) — Phase 18 outcome neither_clears
    # ⇒ NET-* dropped; META-01 stacks on the 72-col xgb_v2 surface.
    config = MLConfig(cutoff_date=EXPECTED_CUTOFF_DATE)
    assembler = FeatureMatrixAssembler(config)
    X, y, fight_dates = assembler.assemble(
        fight_records,
        elo_features,
        computed_features,
        fighter_physicals,
        division_medians,
        round_stats,
        pre_ufc_records=pre_ufc,
        fight_odds=fight_odds,
        include_net=False,
    )
    if X.shape[1] != EXPECTED_XGB_V2_N_FEATURES:
        msg = (
            f"feature_matrix shape mismatch: got {X.shape[1]} cols, expected "
            f"{EXPECTED_XGB_V2_N_FEATURES} (FEATURE_COLUMNS_NO_NET — Phase 18 dispatch)"
        )
        raise RuntimeError(msg)
    return X, y, fight_dates, fight_records


def _compute_elo_prob_for_fight(fight: dict, elo_features: dict) -> float:
    """Return as-of-fight-date Elo expected_win_probability for fighter A.

    Per RESEARCH.md F3: elo_prob MUST be as-of fight-date (NOT post-fight).
    ``elo_features`` is the dict returned by ``queries.load_elo_features``,
    keyed by ``(fighter_id, fight_id)`` and storing ``elo_before`` (pre-fight
    snapshot — leakage-safe by construction).

    Encapsulated as a separate helper so the Plan 19-02 Task 1a integration
    tests can mock this single function with a deterministic stub.
    """
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
    base_estimator,
    X_meta_train: np.ndarray,
    y_meta_train: np.ndarray,
    X_meta_eval: np.ndarray,
) -> np.ndarray:
    """Train a single base XGB on the meta_train partition and return its
    inference probs on the meta_eval partition.

    Plan 19-02 Task 1a — for the meta_eval xgb_eval_prob input, we train ONE
    base estimator on ALL of meta_train (no CV needed; we're not generating
    OOF for meta_eval — only for meta_train). The resulting estimator does NOT
    overwrite ``models/xgb_v2.joblib`` (that joblib is byte-locked end-to-end
    per AUDIT-01); this is a transient in-memory model used only to produce
    eval-set probs for META-01 evaluation.
    """
    base_estimator.fit(X_meta_train, y_meta_train)
    return base_estimator.predict_proba(X_meta_eval)[:, 1]


def _write_meta_learner_report(
    *,
    median: dict,
    per_seed: dict,
    passed: bool,
    failures: list[str],
    n_meta_train: int,
    n_meta_eval: int,
    out_path: Path = META_LEARNER_REPORT_PATH,
) -> None:
    """Append `19-META-LEARNER-REPORT.md` per Task 1b artifact contract."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Phase 19 / Plan 19-02 — META-01 Spike Report\n")
    lines.append(f"_Generated: {datetime.now(tz=UTC).isoformat()}_\n")
    lines.append(f"_Gate contract formula_hash: `{EXPECTED_FORMULA_HASH}`_\n")
    lines.append("")
    lines.append(
        f"**Partition sizes (D-01(P19) three-way split):** "
        f"meta_train={n_meta_train}, meta_eval={n_meta_eval}"
    )
    lines.append("")
    lines.append(
        f"**Verdict (D-13 ALL-pass discipline):** "
        f"{'PASS — META-01 ships' if passed else 'FAIL — META-01 does NOT clear gate'}"
    )
    if failures:
        lines.append("")
        lines.append("**Failed criteria:**")
        for f in failures:
            lines.append(f"  - {f}")
    lines.append("")
    lines.append("## Per-slice median (5 seeds)")
    lines.append("")
    lines.append("| Slice | Brier (median) | Accuracy (median) |")
    lines.append("|---|---|---|")
    for slice_name in PER_SLICE_KEYS:
        m = median.get(slice_name, {})
        b = m.get("brier_score", float("nan"))
        a = m.get("accuracy", float("nan"))
        lines.append(f"| {slice_name} | {b:.4f} | {a:.4f} |")
    lines.append("")
    lines.append("## Per-seed sub-rows")
    lines.append("")
    lines.append("| Seed | Slice | Brier | Accuracy |")
    lines.append("|---|---|---|---|")
    for seed in sorted(per_seed.keys()):
        for slice_name in PER_SLICE_KEYS:
            m = per_seed[seed].get(slice_name, {})
            b = m.get("brier_score", float("nan"))
            a = m.get("accuracy", float("nan"))
            lines.append(f"| {seed} | {slice_name} | {b:.4f} | {a:.4f} |")
    lines.append("")
    lines.append("## MLP A/B (Plan 19-02b)")
    lines.append("")
    lines.append(
        "_Placeholder — populated by Plan 19-02b Task 1 if operator chooses option-escalate._"
    )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 19 META-01 spike")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(SEEDS_DEFAULT),
        help="Random seeds (default: 42 43 44 45 46)",
    )
    parser.add_argument(
        "--no-cache-oof",
        action="store_true",
        help="Force OOF parquet rebuild (ignore cache invariants)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Synthetic smoke run (no DB, no real xgb_v2 inference)",
    )
    parser.add_argument(
        "--meta-train-strategy",
        choices=["temporal_50_50", "ts_split_post_cutoff", "holdout_only"],
        default="ts_split_post_cutoff",
        help="Three-way split strategy (per D-01(P19) sub-decision OQ-1)",
    )
    args = parser.parse_args(argv)

    print(f"[train_meta_v1] Phase 19 META-01 spike — args: {vars(args)}")

    # Step 1: invariants (skipped in dry-run for synthetic-only smoke)
    if not args.dry_run:
        try:
            assert_phase19_invariants()
            print("[train_meta_v1] AUDIT-01 + AF-1 + Pitfall B + Phase 17 gate: OK")
        except AssertionError as exc:
            print(f"[train_meta_v1] FATAL: invariant violated — {exc}", file=sys.stderr)
            return 2

    # Step 2: data load + three-way split
    if args.dry_run:
        from ufc_prediction.ml.oof import make_three_way_split

        X, y, dates, fight_ids, base_cutoff, today = _build_synthetic_demo_data()
        # Build a "fights" list-of-dicts shape for make_three_way_split
        fights = [
            {
                "fight_id": fight_ids[i],
                "event_date": dates[i].item() if hasattr(dates[i], "item") else dates[i],
            }
            for i in range(len(fight_ids))
        ]
        base_train, meta_train, meta_eval = make_three_way_split(
            fights,
            base_cutoff=base_cutoff,
            meta_eval_window_days=365,
            today=today,
        )
        print(
            f"[train_meta_v1] split sizes: base={len(base_train)} "
            f"meta_train={len(meta_train)} meta_eval={len(meta_eval)}"
        )
        print("[train_meta_v1] DRY-RUN complete (no real OOF / META fit / gate eval).")
        return 0

    # ── Step 3: Real-data wiring (Plan 19-02 Task 1a deliverable) ──
    from ufc_prediction.ml.config import FEATURE_COLUMNS_NO_NET
    from ufc_prediction.ml.evaluator import evaluate_per_slice
    from ufc_prediction.ml.gate_contract import load_gate_contract
    from ufc_prediction.ml.meta_learner import (
        META_FEATURE_COLUMNS,
        MetaLearnerLogistic,
        build_meta_features,
    )
    from ufc_prediction.ml.oof import (
        _make_oof_estimator,
        generate_oof_predictions,
        make_three_way_split,
    )
    from ufc_prediction.ml.trainer import median_metrics

    # (a) Load fights + assemble feature matrix (mockable via _load_assembled_data).
    print("[train_meta_v1] Loading data + assembling 72-col feature matrix...")
    X, y, fight_dates, fight_records = _load_assembled_data()
    print(f"[train_meta_v1] X.shape={X.shape}, y.shape={y.shape}, n_records={len(fight_records)}")

    # (b) Three-way disjoint split per D-01(P19).
    base_train_fights, meta_train_fights, meta_eval_fights = make_three_way_split(
        fight_records,
        base_cutoff=date.fromisoformat(EXPECTED_CUTOFF_DATE),
        meta_eval_window_days=365,
        today=date.today(),
    )
    print(
        f"[train_meta_v1] split sizes: base={len(base_train_fights)} "
        f"meta_train={len(meta_train_fights)} meta_eval={len(meta_eval_fights)}"
    )

    if len(meta_train_fights) == 0 or len(meta_eval_fights) == 0:
        print(
            "[train_meta_v1] FATAL: empty meta_train or meta_eval partition; "
            "cannot run META-01 spike.",
            file=sys.stderr,
        )
        return 2

    # Index lookups by fight_id (avoid O(n^2) row-by-row searches).
    meta_train_ids = {f["fight_id"] for f in meta_train_fights}
    meta_eval_ids = {f["fight_id"] for f in meta_eval_fights}
    meta_train_idx = np.array(
        [i for i, f in enumerate(fight_records) if f["fight_id"] in meta_train_ids]
    )
    meta_eval_idx = np.array(
        [i for i, f in enumerate(fight_records) if f["fight_id"] in meta_eval_ids]
    )

    # (c) Compute the three Level-1 features for meta_train per D-02(P19).
    closing_idx = FEATURE_COLUMNS_NO_NET.index("closing_prob_diff")

    print("[train_meta_v1] Generating OOF predictions on meta_train (TimeSeriesSplit, n_jobs=1)...")
    xgb_oof_prob, oof_meta = generate_oof_predictions(
        X[meta_train_idx],
        y[meta_train_idx],
        fight_dates[meta_train_idx],
        base_trainer=None,  # uses xgb_v2 best_params from xgb_v2_meta.json (OQ-2)
        cache_path=META_OOF_PARQUET_PATH,
        force_rebuild=args.no_cache_oof,
        fight_ids=[fight_records[i]["fight_id"] for i in meta_train_idx],
    )
    # Pre-load elo_features once for the elo_prob lookups (real path); the
    # mocked test path bypasses this entirely via patch.object.
    try:
        from ufc_prediction.db.session import SessionLocal
        from ufc_prediction.ml.queries import load_elo_features

        _session = SessionLocal()
        try:
            _elo_features = load_elo_features(_session)
        finally:
            _session.close()
    except Exception:
        _elo_features = {}

    elo_prob_train = np.array(
        [_compute_elo_prob_for_fight(fight_records[i], _elo_features) for i in meta_train_idx]
    )
    closing_prob_diff_train = X[meta_train_idx, closing_idx]

    # NOTE: generate_oof_predictions sorts internally by fight_dates, so
    # xgb_oof_prob is sorted-order. Re-align to the meta_train_idx row order
    # by mirroring the sort permutation.
    sort_idx = np.argsort(fight_dates[meta_train_idx])
    xgb_oof_aligned = np.empty_like(xgb_oof_prob)
    xgb_oof_aligned[sort_idx] = xgb_oof_prob

    X_meta_train = build_meta_features(
        xgb_oof_aligned,
        elo_prob_train,
        closing_prob_diff_train,
    )
    y_meta_train = y[meta_train_idx]

    # (d) Build meta_eval Level-1 features. xgb_eval_prob via single base train.
    print("[train_meta_v1] Building meta_eval Level-1 features (1 base train + Elo lookups)...")
    base_estimator = _make_oof_estimator(seed=42)
    xgb_eval_prob = _build_meta_eval_xgb_probs(
        base_estimator,
        X[meta_train_idx],
        y[meta_train_idx],
        X[meta_eval_idx],
    )
    elo_prob_eval = np.array(
        [_compute_elo_prob_for_fight(fight_records[i], _elo_features) for i in meta_eval_idx]
    )
    closing_prob_diff_eval = X[meta_eval_idx, closing_idx]
    X_meta_eval = build_meta_features(
        xgb_eval_prob,
        elo_prob_eval,
        closing_prob_diff_eval,
    )
    y_meta_eval = y[meta_eval_idx]
    fight_dates_eval = fight_dates[meta_eval_idx]

    # Symmetric NaN-drop on eval (mirrors MetaLearnerLogistic.fit's training-time drop).
    # Per D-04(P19), production predictor falls back to base xgb_v2 prob on NaN Level-1
    # features (meta_skipped=true). For gate evaluation we measure META-01 on rows where
    # it actually fires — the predictor's skip-fallback path doesn't go through META so
    # it should not contribute to the META gate verdict.
    eval_mask = ~np.isnan(X_meta_eval).any(axis=1)
    eval_dropped = int((~eval_mask).sum())
    if eval_dropped:
        print(
            f"[train_meta_v1] eval set: dropping {eval_dropped} NaN rows "
            f"({eval_dropped / len(eval_mask):.1%} of {len(eval_mask)}); "
            f"surviving rows = {int(eval_mask.sum())}",
        )
    X_meta_eval = X_meta_eval[eval_mask]
    y_meta_eval = y_meta_eval[eval_mask]
    fight_dates_eval = fight_dates_eval[eval_mask]

    # (e) Fit MetaLearnerLogistic once per seed; evaluate on the 3-slice harness.
    print(f"[train_meta_v1] Fitting MetaLearnerLogistic × {len(args.seeds)} seeds + evaluating...")
    per_seed_results: dict[int, dict] = {}
    last_meta_for_save = None
    for seed in args.seeds:
        meta = MetaLearnerLogistic(random_state=seed).fit(X_meta_train, y_meta_train)
        # evaluate_per_slice expects a model with predict_proba + predict;
        # MetaLearnerLogistic provides both via the underlying Pipeline.
        per_seed_results[seed] = evaluate_per_slice(
            meta,
            X_meta_eval,
            y_meta_eval,
            fight_dates_eval,
        )
        last_meta_for_save = meta

    # Capture for the integration tests' introspection contract.
    global LAST_PER_SEED_RESULTS
    LAST_PER_SEED_RESULTS = per_seed_results

    # (f) Compute median + gate verdict + hard-gate-then-save.
    median = median_metrics(list(per_seed_results.values()))
    contract = load_gate_contract()
    passed, failures = gate_verdict(median, contract)

    _METRIC_KEYS_FOR_OVERALL = ("brier_score",)
    median_brier_overall = float(np.median([median[s]["brier_score"] for s in PER_SLICE_KEYS]))

    # (g) Persist + archive.
    META_DIR.mkdir(parents=True, exist_ok=True)
    if passed:
        # Hard-gate-then-save (D-07(P15)) — only persist on PASS.
        oof_parquet_sha256 = (
            hashlib.sha256(META_OOF_PARQUET_PATH.read_bytes()).hexdigest()
            if META_OOF_PARQUET_PATH.exists()
            else "0" * 64
        )
        meta_input_hash = compute_meta_input_distribution_hash(
            X_meta_train,
            y_meta_train,
            xgb_oof_aligned,
        )
        model_path, meta_path = save_meta_model(
            last_meta_for_save,
            meta_kind="logistic",
            meta_version="v1",
            base_model_version="v2",
            base_model_sha256=EXPECTED_XGB_V2_SHA256,
            meta_feature_columns=list(META_FEATURE_COLUMNS),
            meta_input_distribution_hash=meta_input_hash,
            meta_oof_parquet_sha256=oof_parquet_sha256,
            meta_learner_brier_delta_vs_logistic=0.0,
            best_params=dict(META01_BEST_PARAMS),
            metrics={
                "per_slice": median,
                "median_brier_overall": median_brier_overall,
            },
            meta_dir=str(META_DIR),
        )
        print(f"[train_meta_v1] PASS — meta_v1 saved: {model_path}")
        candidate_joblib = Path(model_path) if not isinstance(model_path, Path) else model_path
        candidate_meta = Path(meta_path) if not isinstance(meta_path, Path) else meta_path
    else:
        # FAIL path — write a candidate bundle to disk so AUDIT-03 can archive
        # it (Task 3 requirement: candidate exists REGARDLESS of ship outcome).
        import joblib as _joblib

        candidate_joblib = META_DIR / "meta_v1_candidate.joblib"
        candidate_meta = META_DIR / "meta_v1_candidate_meta.json"
        _joblib.dump(last_meta_for_save, candidate_joblib)
        candidate_meta.write_text(
            json.dumps(
                {
                    "meta_version": "v1_candidate",
                    "meta_kind": "logistic",
                    "meta_feature_columns": list(META_FEATURE_COLUMNS),
                    "base_model_version": "v2",
                    "base_model_sha256": EXPECTED_XGB_V2_SHA256,
                    "ship_outcome": "FAIL",
                    "failures": failures,
                    "metrics": {
                        "per_slice": median,
                        "median_brier_overall": median_brier_overall,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[train_meta_v1] FAIL — candidate written (not promoted): {candidate_joblib}")
        for f in failures:
            print(f"  - {f}")

    # AUDIT-03 archive — fires regardless of ship outcome.
    # Reference module-level constants at call time so monkeypatch in
    # tests (which rewrites the module attributes) is honored.
    try:
        sha = archive_meta_candidate(
            "meta_v1",
            candidate_joblib,
            candidate_meta,
            archive_dir=META_ARCHIVE_DIR,
        )
        print(f"[train_meta_v1] AUDIT-03 archived: meta_v1_<{sha[:8]}>.joblib")
    except Exception as exc:
        print(f"[train_meta_v1] AUDIT-03 archive failed (Task 3 fallback expected): {exc}")

    # (h) Write the markdown spike report. Pass META_LEARNER_REPORT_PATH
    # explicitly so monkeypatched test paths win over the function default.
    _write_meta_learner_report(
        median=median,
        per_seed=per_seed_results,
        passed=passed,
        failures=failures,
        n_meta_train=len(meta_train_idx),
        n_meta_eval=len(meta_eval_idx),
        out_path=META_LEARNER_REPORT_PATH,
    )
    print(f"[train_meta_v1] Report written: {META_LEARNER_REPORT_PATH}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
