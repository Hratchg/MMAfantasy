#!/usr/bin/env python
"""Phase 30 / VARIANCE-V23-01..03 — META-V22 multi-seed bootstrap-variance spike (v2.3).

Fork of `scripts/spike_noise_floor_v22.py` (978 lines; v2.2 base-XGB spike on
the 90-col FEATURE_COLUMNS_V22 substrate) per CONTEXT D-03 fork-not-mutate
discipline. The v2.2 spike is preserved untouched at the AUDIT-01 byte-
identity level; this v2.3 fork is a CLEAN REWIRE that:

  - Operates on the META learner (`MetaLearnerLogistic`) over 13-col Level-1
    features (`META_V22_FEATURE_COLUMNS`), NOT the base XGB.
  - Replaces the v2.2 spike's inline `_train_with_fixed_params × N seeds`
    loop with a single call to `variance.multi_seed_metrics(...)` from
    `src/ufc_prediction/ml/variance.py` (Plan 30-01 deliverable; D-02 4-fn
    locked public surface).
  - Variance is injected via bootstrap-with-replacement on
    `(X_meta_train, y_meta_train)` rows per seed (D-01). The closed-form
    LogisticRegression solver means `random_state` is a no-op for
    variance; only data resampling produces real per-seed variance.
  - xgb_v2 OOF predictions are upstream of `X_meta_train` (already columns
    0..1 of META_V22_FEATURE_COLUMNS — `xgb_oof_prob` / `elo_prob`); the
    spike NEVER retrains xgb_v2. AUDIT-01 SC4 byte-identity preserved by
    construction (Pitfall 1 in 30-RESEARCH.md).

What's PRESERVED verbatim from v2.2 (audit-lineage parity):
  - AUDIT-01 SHA assertion pattern (read `models/xgb_v2.joblib` bytes,
    hashlib.sha256, assert == EXPECTED_XGB_V2_SHA — mirrors
    `train_meta_v22.py:577-579`).
  - Formula source + hash (`std_used = max(seed_std, bootstrap_ci_half)`
    locked at `7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a`
    per D-08; reduction lifted into `variance.aggregate_variance`).
  - Pitfall-B NaN fallback (verbatim from v2.2 spike lines 353-370; lives
    in `variance.aggregate_variance` now, mirrored here in report-render
    for warning-line emission).
  - HALT artifact rendering pattern (mirrors `_render_halt_and_decide` at
    v2.2 spike lines 463-531; this fork renames to `_render_variance_halt`
    per D-06 and emits `30-VARIANCE-HALT.md` on `seed_std == 0` collapse).
  - Exit-code convention: 0 = Path A (advance to Phase 31); 3 = Path B
    (operator HALT on collapse); 2 = fatal pre-spike invariant violation.

What's NEW vs. v2.2 (Plan 30 scope):
  - SEEDS_DEFAULT = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51) per D-04
    (10-seed extension of v2.2's 5-seed set 42..46).
  - `--no-bootstrap` flag: deterministic path that skips
    `variance.bootstrap_resample` and fits on the raw `(X_meta_train,
    y_meta_train)` rows. Used by the canonical-lineage test
    (`test_seed42_deterministic_matches_plan_29_03`) to reproduce Plan
    29-03 META-V22 random_15pct Brier 0.1409 +/- 0.0001.
  - Output artifact paths under `.planning/phases/30-multi-seed-variance-
    harness/`:
      - `30-VARIANCE-REPORT.md` (always; final report with Path A or B
        outcome materialized from runtime).
      - `30-VARIANCE-HALT.md` (CONDITIONAL — only if any slice produced
        `seed_std_brier == 0` or `seed_std_acc == 0`).
      - `30-XGB-V2-SHA-PHASE-30-END.txt` (always; bare 64-char hex + \\n).
  - No `gate_contract_v2.3.json` emission — that's Phase 31 scope per D-07.
    Phase 30 only ships the variance numbers Phase 31 reads.

Usage:
    # Canonical 10-seed bootstrap run (Path A or Path B outcome):
    uv run python scripts/spike_noise_floor_v23.py

    # Deterministic single-seed lineage check (no bootstrap):
    uv run python scripts/spike_noise_floor_v23.py --seeds 42 --no-bootstrap

After completion, operator reviews `30-VARIANCE-REPORT.md` (+ `30-VARIANCE-
HALT.md` if emitted), verifies `30-XGB-V2-SHA-PHASE-30-END.txt` matches the
baseline SHA, and commits the artifacts manually (Pitfall D — spike does
NOT call git).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# ── Locked constants (D-04 / D-08 / AUDIT-01) ─────────────────────────

# AUDIT-01 binding: models/xgb_v2.joblib SHA-256. Preserved end-to-end via
# MID + END SHA assertions (Phase 30 MID artifact was emitted by Plan 30-01;
# this script emits the END artifact at completion).
EXPECTED_XGB_V2_SHA: str = "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"

# D-08 formula hash binding — preserved verbatim from v2.2 spike. The reduction
# `std_used = max(seed_std, bootstrap_ci_half)` lives in
# `variance.aggregate_variance` (single source of truth per Plan 30-01).
FORMULA_SOURCE: str = (
    "gate_brier_max = round(median_brier - 1 * max(seed_std_brier, "
    "bootstrap_ci_half_brier), 4); "
    "gate_accuracy_min = round(median_acc + 1 * max(seed_std_acc, "
    "bootstrap_ci_half_acc), 4)"
)
EXPECTED_FORMULA_HASH: str = "7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a"

# D-04: 10-seed default (extends v2.2's 5-seed set 42..46 by appending 47..51).
SEEDS_DEFAULT: tuple[int, ...] = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)

# Canonical-lineage anchor (ResearchQ10 per 30-RESEARCH.md). The deterministic
# path (seed=42, --no-bootstrap) MUST reproduce Plan 29-03's META-V22
# random_15pct Brier within PLAN_29_03_TOLERANCE for AUDIT-01 lineage.
PLAN_29_03_RANDOM_15PCT_BRIER: float = 0.1409
PLAN_29_03_TOLERANCE: float = 0.0001

# Per-slice keys (mirrors evaluator.PER_SLICE_KEYS).
PER_SLICE_KEYS: tuple[str, str, str] = (
    "most_recent_12mo",
    "most_recent_24mo",
    "random_15pct",
)

# Artifact output paths.
PHASE30_DIR: Path = Path(".planning/phases/30-multi-seed-variance-harness")
VARIANCE_REPORT_PATH: Path = PHASE30_DIR / "30-VARIANCE-REPORT.md"
VARIANCE_HALT_PATH: Path = PHASE30_DIR / "30-VARIANCE-HALT.md"
SHA_END_PATH: Path = PHASE30_DIR / "30-XGB-V2-SHA-PHASE-30-END.txt"


def _assert_xgb_v2_sha(label: str) -> str:
    """AUDIT-01 byte-identity check (MID-equivalent at startup; END at exit).

    Reads `models/xgb_v2.joblib` bytes, computes hashlib.sha256, asserts
    == EXPECTED_XGB_V2_SHA. Pitfall 5 (30-RESEARCH.md) — even though the
    spike does NOT load xgb_v2 (it consumes the OOF columns already baked
    into X_meta_train), the AUDIT-01 binding requires byte-identity at the
    file level, so we read the file's bytes explicitly.
    """
    joblib_bytes = Path("models/xgb_v2.joblib").read_bytes()
    sha = hashlib.sha256(joblib_bytes).hexdigest()
    if sha != EXPECTED_XGB_V2_SHA:
        raise SystemExit(
            f"[spike-v23] AUDIT-01 violation at {label}: got {sha}, expected {EXPECTED_XGB_V2_SHA}"
        )
    print(f"[spike-v23] AUDIT-01 {label} SHA OK: {sha[:16]}...")
    return sha


def _assert_formula_hash() -> str:
    """D-08 formula-hash drift guard (verbatim from v2.2 spike pattern).

    Hash any drift in the FORMULA_SOURCE constant fails the spike before any
    artifact is emitted.
    """
    computed = hashlib.sha256(FORMULA_SOURCE.encode("utf-8")).hexdigest()
    if computed != EXPECTED_FORMULA_HASH:
        raise SystemExit(
            f"[spike-v23] FORMULA drift: got {computed}, expected "
            f"{EXPECTED_FORMULA_HASH}. Operator must re-approve formula "
            "before spike can emit variance report."
        )
    print(f"[spike-v23] FORMULA_HASH OK: {computed[:16]}... (D-08 / v2.2 lineage preserved)")
    return computed


def _meta_fit_fn(X_train: np.ndarray, y_train: np.ndarray, seed: int):
    """Closure: construct a FRESH MetaLearnerLogistic per seed (Pitfall 4).

    Per 30-RESEARCH.md Pitfall 4: sklearn pipeline objects carry fitted
    state (`coef_`, `intercept_`, etc.) after `.fit(...)`. We MUST build a
    new estimator inside the closure to avoid silent state-sharing across
    seeds. Mirrors `train_meta_v22.py:539` precedent.
    """
    # Import inside the closure so module-level import time stays cheap and
    # the dry-path tests don't pay the sklearn cost twice.
    from ufc_prediction.ml.meta_learner import MetaLearnerLogistic

    return MetaLearnerLogistic(random_state=int(seed)).fit(X_train, y_train)


def _load_meta_train_eval_matrices(*, dry_run: bool):
    """Load (X_meta_train, y_meta_train, X_meta_eval, y_meta_eval,
    fight_dates_eval) via the same path `train_meta_v22.py` uses.

    Returns the cleaned matrices (post NaN-drop + non-baseline imputation
    per Plan 29-02 EVAL-V23-01 policy) so downstream `multi_seed_metrics`
    consumers receive matrices indistinguishable from what
    `train_meta_v22.py` feeds `MetaLearnerLogistic(random_state=seed)`.

    The path is intentionally a thin delegation to `train_meta_v22` helpers
    — Plan 29-03 canonical T3 byte-identity REQUIRES bit-identical data
    construction (any divergence breaks the seed=42 lineage anchor).
    """
    from datetime import date as _date

    from ufc_prediction.ml.meta_features_v22 import (
        META_V22_FEATURE_COLUMNS,
        build_meta_features_v22,
    )
    from ufc_prediction.ml.oof import (
        _make_oof_estimator,
        apply_nan_drop_policy,
        generate_oof_predictions,
        make_three_way_split,
    )

    # Re-use train_meta_v22 helpers verbatim so the data path is
    # bit-identical (Plan 29-03 canonical lineage binding).
    # NOTE: scripts/ is not on sys.path by default; add it lazily.
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from train_meta_v22 import (  # noqa: PLC0415
        EXPECTED_CUTOFF_DATE,
        META_OOF_PARQUET_PATH,
        _build_meta_eval_xgb_probs,
        _build_synthetic_data_v22,
        _compute_elo_prob_for_fight,
        _enforce_72col_view,
        _load_assembled_data_v22,
    )

    if dry_run:
        X_v22, y, fight_dates, fight_ids, _base_cutoff, _today = _build_synthetic_data_v22()
        fight_records = [
            {
                "fight_id": fight_ids[i],
                "event_date": (
                    fight_dates[i].item() if hasattr(fight_dates[i], "item") else fight_dates[i]
                ),
                "fighter_a_id": i * 2,
                "fighter_b_id": i * 2 + 1,
            }
            for i in range(len(fight_ids))
        ]
    else:
        X_v22, y, fight_dates, fight_records = _load_assembled_data_v22()

    META_EVAL_WINDOW_DAYS = 730
    base_train_fights, meta_train_fights, meta_eval_fights = make_three_way_split(
        fight_records,
        base_cutoff=_date.fromisoformat(EXPECTED_CUTOFF_DATE),
        meta_eval_window_days=META_EVAL_WINDOW_DAYS,
        today=_date.today(),
    )
    if len(meta_train_fights) == 0 or len(meta_eval_fights) == 0:
        raise SystemExit("[spike-v23] FATAL: empty meta_train or meta_eval partition.")

    meta_train_ids = {f["fight_id"] for f in meta_train_fights}
    meta_eval_ids = {f["fight_id"] for f in meta_eval_fights}
    meta_train_idx = np.array(
        [i for i, f in enumerate(fight_records) if f["fight_id"] in meta_train_ids]
    )
    meta_eval_idx = np.array(
        [i for i, f in enumerate(fight_records) if f["fight_id"] in meta_eval_ids]
    )

    # OOF base predictions on the 72-col view (Pitfall #2 guard active).
    X_oof = X_v22[meta_train_idx][:, :72]
    _enforce_72col_view(X_oof)

    xgb_oof_prob, _oof_meta = generate_oof_predictions(
        X_oof,
        y[meta_train_idx],
        fight_dates[meta_train_idx],
        base_trainer=None,
        cache_path=META_OOF_PARQUET_PATH,
        force_rebuild=False,
        fight_ids=[fight_records[i]["fight_id"] for i in meta_train_idx],
    )

    # Re-align OOF (returned in sort-order) to row-order.
    sort_idx = np.argsort(fight_dates[meta_train_idx])
    xgb_oof_aligned = np.empty_like(xgb_oof_prob)
    xgb_oof_aligned[sort_idx] = xgb_oof_prob

    if dry_run:
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
        elo_prob_train = np.array(
            [_compute_elo_prob_for_fight(fight_records[i], _elo_features) for i in meta_train_idx]
        )
        elo_prob_eval = np.array(
            [_compute_elo_prob_for_fight(fight_records[i], _elo_features) for i in meta_eval_idx]
        )

    X_meta_train = build_meta_features_v22(
        xgb_oof_aligned,
        elo_prob_train,
        X_v22[meta_train_idx],
    )
    y_meta_train = y[meta_train_idx]

    # Single transient base train for the eval OOF (mirrors train_meta_v22.py
    # Step 5; xgb_v2.joblib NOT loaded).
    base_estimator = _make_oof_estimator(seed=42)
    xgb_eval_prob = _build_meta_eval_xgb_probs(
        base_estimator,
        X_v22[meta_train_idx][:, :72],
        y[meta_train_idx],
        X_v22[meta_eval_idx][:, :72],
    )
    X_meta_eval = build_meta_features_v22(
        xgb_eval_prob,
        elo_prob_eval,
        X_v22[meta_eval_idx],
    )
    y_meta_eval = y[meta_eval_idx]
    fight_dates_eval = fight_dates[meta_eval_idx]

    # Plan 29-02 EVAL-V23-01 NaN-drop + non-baseline imputation (verbatim
    # from train_meta_v22.py:444-529).
    NAN_DROP_POLICY = "per_feature_strict_baseline"
    BASELINE_COLS = ("xgb_oof_prob", "elo_prob")
    NON_BASELINE_IDX = [i for i, c in enumerate(META_V22_FEATURE_COLUMNS) if c not in BASELINE_COLS]

    eval_mask = apply_nan_drop_policy(
        X_meta_eval,
        META_V22_FEATURE_COLUMNS,
        policy=NAN_DROP_POLICY,
    )
    X_meta_eval = X_meta_eval[eval_mask]
    y_meta_eval = y_meta_eval[eval_mask]
    fight_dates_eval = fight_dates_eval[eval_mask]

    train_mask = apply_nan_drop_policy(
        X_meta_train,
        META_V22_FEATURE_COLUMNS,
        policy=NAN_DROP_POLICY,
    )
    X_meta_train_clean = X_meta_train[train_mask]
    y_meta_train_clean = y_meta_train[train_mask]

    # Non-baseline column NaN imputation with train-set medians (applied to
    # train and eval) — no eval-time leakage (Plan 29-02).
    for idx in NON_BASELINE_IDX:
        col_train = X_meta_train_clean[:, idx]
        finite = col_train[~np.isnan(col_train)]
        median_val = float(np.median(finite)) if finite.size else 0.0
        train_nan = np.isnan(X_meta_train_clean[:, idx])
        if train_nan.any():
            X_meta_train_clean[train_nan, idx] = median_val
        eval_nan = np.isnan(X_meta_eval[:, idx])
        if eval_nan.any():
            X_meta_eval[eval_nan, idx] = median_val

    return (
        X_meta_train_clean,
        y_meta_train_clean,
        X_meta_eval,
        y_meta_eval,
        fight_dates_eval,
    )


def _no_bootstrap_metrics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    fight_dates_eval: np.ndarray,
    seeds: list[int],
) -> dict[int, dict]:
    """Deterministic path (skips bootstrap_resample per --no-bootstrap).

    Mirrors `multi_seed_metrics` shape but feeds the RAW (X_train, y_train)
    pair to fit_fn for each seed. The deterministic LR solver means N seeds
    produce N IDENTICAL metric tuples on this path — that's the POINT:
    seed=42 deterministic reproduces Plan 29-03 META-V22 random_15pct
    Brier = 0.1409 to 4-decimal precision (ResearchQ10 canonical-lineage
    anchor).

    DO NOT call `assert_distinct_seed_brier` on the output of this function
    — it would (correctly) raise because the deterministic path doesn't
    inject variance; this function is for the lineage anchor only.
    """
    # WR-03 / Phase 30 code-review: parity guard with multi_seed_metrics
    # (variance.py:157-161). A row-count mismatch in (X_train, y_train)
    # would otherwise surface as a deep-in-sklearn broadcasting error
    # rather than a clean ValueError at the harness boundary.
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"_no_bootstrap_metrics: X_train.shape[0]={X_train.shape[0]} != "
            f"y_train.shape[0]={y_train.shape[0]}"
        )
    if not seeds:
        raise ValueError("_no_bootstrap_metrics: seeds must be non-empty")
    from ufc_prediction.ml.evaluator import evaluate_per_slice

    per_seed: dict[int, dict] = {}
    for seed in seeds:
        model = _meta_fit_fn(X_train, y_train, int(seed))
        per_seed[int(seed)] = evaluate_per_slice(
            model,
            X_eval,
            y_eval,
            fight_dates_eval,
        )
    return per_seed


def _render_variance_halt(
    zero_variance_slices: list[str],
    aggregated: dict[str, dict[str, float]],
    per_seed: dict[int, dict],
    *,
    triggered_at: str,
) -> str:
    """Emit 30-VARIANCE-HALT.md body per D-06 + 30-RESEARCH.md template.

    Frontmatter shape mirrors `24-HALT-AND-DECIDE.md` (Open Question 2 in
    30-RESEARCH.md — operator markdown; ~50 LOC inline). Lists each
    affected slice with seed_std + per-seed distinct Brier count, plus
    operator-decision instructions.
    """
    lines = [
        "---",
        "phase: 30-multi-seed-variance-harness",
        "artifact: VARIANCE-HALT",
        f"triggered_at: {triggered_at}",
        ('trigger_condition: "seed_std_brier == 0 OR seed_std_acc == 0 on at least one slice"'),
        "operator_decision_required: true",
        "outcome_paths:",
        "  - id: bootstrap_no_op_investigate",
        (
            '    description: "Diagnose why bootstrap collapsed; fix '
            'variance.py or fit_fn closure; re-run spike"'
        ),
        "  - id: accept_collapse_and_proceed",
        (
            '    description: "(Discouraged) accept zero-variance for this '
            "slice; Phase 31 will gate on seed_std=0 -- gate formula "
            'effectively degenerates"'
        ),
        "---",
        "",
        "# Phase 30 VARIANCE-HALT",
        "",
        ("**One or more slices produced `seed_std_brier == 0` or `seed_std_acc == 0`.**"),
        "",
        (
            "**This is the v2.2 collapsed-variance regression D-06 was added "
            "to catch.** The v2.2 5-seed META spike at `train_meta_v22.py:"
            "531-543` shipped 5 bit-identical metric tuples because "
            "`MetaLearnerLogistic(random_state=seed)` is a no-op for the "
            "closed-form LR solver. Phase 30 injects bootstrap resamples "
            "on the meta-train rows to produce real variance -- if THIS "
            "spike still collapses, the bootstrap itself has no-op'd."
        ),
        "",
        "## Affected Slices",
        "",
        (f"| Slice | seed_std_brier | seed_std_acc | distinct Brier count (n={len(per_seed)}) |"),
        ("|-------|---------------|--------------|--------------------------------|"),
    ]
    for slc in zero_variance_slices:
        n_distinct = len({per_seed[s][slc]["brier_score"] for s in per_seed})
        lines.append(
            f"| {slc} | "
            f"{aggregated[slc]['seed_std_brier']:.6f} | "
            f"{aggregated[slc]['seed_std_acc']:.6f} | "
            f"{n_distinct} |"
        )
    lines.extend(
        [
            "",
            "## Per-Seed Brier (all slices, all seeds)",
            "",
            ("| Seed | 12mo Brier | 24mo Brier | random_15pct Brier |"),
            ("|------|-----------|-----------|---------------------|"),
        ]
    )
    for seed in per_seed:
        ps = per_seed[seed]
        b12 = ps["most_recent_12mo"]["brier_score"]
        b24 = ps["most_recent_24mo"]["brier_score"]
        br = ps["random_15pct"]["brier_score"]
        lines.append(f"| {seed} | {b12:.6f} | {b24:.6f} | {br:.6f} |")
    lines.extend(
        [
            "",
            "## Operator Action Required",
            "",
            "Set `actual_outcome` in `30-SUMMARY.md` frontmatter to one of:",
            "- `bootstrap_no_op_investigate`",
            "- `accept_collapse_and_proceed`",
            "",
            "Then run `/gsd-execute-phase 30` resumption.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_variance_report(
    *,
    spike_started: datetime,
    spike_finished: datetime,
    seeds: list[int],
    bootstrap_enabled: bool,
    n_meta_train: int,
    n_meta_eval: int,
    aggregated: dict[str, dict[str, float]],
    per_seed: dict[int, dict],
    warnings: list[str],
    outcome_path: str,
    canonical_lineage_brier: float | None,
    xgb_v2_sha: str,
) -> str:
    """Render 30-VARIANCE-REPORT.md per D-09 carry-forward (2 outcome paths
    pre-templated; actual outcome populated from runtime).

    Body MUST include the per-slice variance table (6 columns:
    seed_std_brier, seed_std_acc, bootstrap_ci_half_brier,
    bootstrap_ci_half_acc, std_brier_used, std_acc_used) plus the Pitfall-B
    warnings list from `aggregate_variance`.
    """
    duration_min = (spike_finished - spike_started).total_seconds() / 60.0
    lines: list[str] = [
        "---",
        "phase: 30-multi-seed-variance-harness",
        "artifact: VARIANCE-REPORT",
        f"spike_started: {spike_started.isoformat()}",
        f"spike_finished: {spike_finished.isoformat()}",
        f"duration_minutes: {duration_min:.2f}",
        f"seeds: {list(seeds)}",
        f"n_seeds: {len(seeds)}",
        f"bootstrap_enabled: {bootstrap_enabled}",
        f"n_meta_train: {n_meta_train}",
        f"n_meta_eval: {n_meta_eval}",
        f"formula_hash: {EXPECTED_FORMULA_HASH}",
        f"xgb_v2_sha256: {xgb_v2_sha}",
        "outcome_paths:",
        "  - id: path_a_non_zero_variance",
        (
            '    description: "All 3 slices produced non-zero seed_std -- '
            'advance to Phase 31 gate spike with empirically-honest variance"'
        ),
        "  - id: path_b_collapse_halted",
        (
            '    description: "One or more slices collapsed to seed_std == 0 '
            '-- 30-VARIANCE-HALT.md emitted; operator decision required"'
        ),
        f"actual_outcome: {outcome_path}",
    ]
    if canonical_lineage_brier is not None:
        lines.extend(
            [
                f"canonical_lineage_brier_random_15pct: {canonical_lineage_brier:.6f}",
                f"canonical_lineage_target: {PLAN_29_03_RANDOM_15PCT_BRIER:.4f}",
                f"canonical_lineage_tolerance: {PLAN_29_03_TOLERANCE:.4f}",
                (
                    "canonical_lineage_match: "
                    + str(
                        abs(canonical_lineage_brier - PLAN_29_03_RANDOM_15PCT_BRIER)
                        < PLAN_29_03_TOLERANCE
                    )
                ),
            ]
        )
    lines.extend(
        [
            "---",
            "",
            "# Phase 30 -- Multi-Seed Variance Report (v2.3)",
            "",
            f"**Spike started:** {spike_started.isoformat()}  ",
            f"**Spike finished:** {spike_finished.isoformat()}  ",
            f"**Spike duration:** {duration_min:.2f} min",
            "",
            f"**Seeds:** {list(seeds)} (n={len(seeds)})  ",
            f"**Bootstrap enabled:** {bootstrap_enabled}  ",
            f"**Meta-train rows (clean):** {n_meta_train}  ",
            f"**Meta-eval rows (clean):** {n_meta_eval}  ",
            f"**Formula hash:** `{EXPECTED_FORMULA_HASH}`  ",
            f"**xgb_v2.joblib SHA-256:** `{xgb_v2_sha}` (AUDIT-01 byte-identity)  ",
            "",
            "**Formula:** "
            "`gate_brier_max = round(median_brier - 1 * max(seed_std_brier, "
            "bootstrap_ci_half_brier), 4)`; "
            "`gate_accuracy_min = round(median_acc + 1 * max(seed_std_acc, "
            "bootstrap_ci_half_acc), 4)` "
            "(D-08; preserved verbatim from v2.2 spike via "
            "`variance.aggregate_variance`).",
            "",
            "---",
            "",
            "## Outcome",
            "",
            f"**Actual outcome:** `{outcome_path}`",
            "",
            "**Pre-templated outcome paths (D-09 carry-forward):**",
            "- `path_a_non_zero_variance`: All 3 slices produced non-zero "
            "seed_std -- advance to Phase 31 gate spike with empirically-honest "
            "variance.",
            "- `path_b_collapse_halted`: One or more slices collapsed to "
            "seed_std == 0 -- `30-VARIANCE-HALT.md` emitted; operator decision "
            "required before Phase 31 can run.",
            "",
            "---",
            "",
            "## Per-Slice Variance (D-08 + Pitfall-B carry-forward)",
            "",
            (
                "| Slice | seed_std_brier | seed_std_acc | bootstrap_ci_half_brier"
                " | bootstrap_ci_half_acc | std_brier_used | std_acc_used |"
            ),
            (
                "|-------|---------------|--------------|------------------------"
                "-|----------------------|----------------|---------------|"
            ),
        ]
    )
    for slc in PER_SLICE_KEYS:
        a = aggregated[slc]
        # Raw NaN preserved in the bootstrap_ci_half_* columns even when
        # std_used uses the 0.0 fallback (Pitfall-B carry-forward).
        bh_b = (
            f"{a['bootstrap_ci_half_brier']:.6f}"
            if np.isfinite(a["bootstrap_ci_half_brier"])
            else "NaN"
        )
        bh_a = (
            f"{a['bootstrap_ci_half_acc']:.6f}"
            if np.isfinite(a["bootstrap_ci_half_acc"])
            else "NaN"
        )
        lines.append(
            f"| `{slc}` | "
            f"{a['seed_std_brier']:.6f} | {a['seed_std_acc']:.6f} | "
            f"{bh_b} | {bh_a} | "
            f"{a['std_brier_used']:.6f} | {a['std_acc_used']:.6f} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Per-Seed Sub-Rows (audit reproducibility)",
            "",
            (
                "| Seed | 12mo Brier | 12mo Acc | 24mo Brier | 24mo Acc | "
                "random_15pct Brier | random_15pct Acc |"
            ),
            (
                "|------|-----------|----------|-----------|----------|"
                "---------------------|--------------------|"
            ),
        ]
    )
    for seed in per_seed:
        ps = per_seed[seed]
        b12 = ps["most_recent_12mo"]["brier_score"]
        a12 = ps["most_recent_12mo"]["accuracy"]
        b24 = ps["most_recent_24mo"]["brier_score"]
        a24 = ps["most_recent_24mo"]["accuracy"]
        br = ps["random_15pct"]["brier_score"]
        ar = ps["random_15pct"]["accuracy"]
        lines.append(
            f"| {seed} | {b12:.6f} | {a12:.6f} | {b24:.6f} | {a24:.6f} | {br:.6f} | {ar:.6f} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Warnings (Pitfall-B carry-forward)",
            "",
        ]
    )
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
        lines.append(
            "**Pitfall-B note:** When `bootstrap_ci_half_*` is NaN/non-"
            "finite (degenerate slice -- typical on small `random_15pct`), "
            "the raw NaN is PRESERVED in the table above for the contract "
            "record, but the `max()` reduction substitutes `0.0` so "
            "`std_used` degenerates to `seed_std`. Verbatim from "
            "`scripts/spike_noise_floor_v22.py` lines 353-370 -- lifted "
            "into `variance.aggregate_variance` (single source of truth)."
        )
    else:
        lines.append("(none -- all 3 slices produced finite bootstrap CIs)")

    if canonical_lineage_brier is not None:
        match = abs(canonical_lineage_brier - PLAN_29_03_RANDOM_15PCT_BRIER) < PLAN_29_03_TOLERANCE
        lines.extend(
            [
                "",
                "---",
                "",
                "## Canonical-Lineage Anchor (ResearchQ10)",
                "",
                f"**seed=42, --no-bootstrap path:** "
                f"random_15pct Brier = {canonical_lineage_brier:.6f}",
                f"**Plan 29-03 META-V22 target:** "
                f"{PLAN_29_03_RANDOM_15PCT_BRIER:.4f} (+/- {PLAN_29_03_TOLERANCE})",
                f"**Match within tolerance:** {match}",
                "",
                (
                    "Per ResearchQ10 binding, the deterministic seed=42 path on "
                    "the canonical META-V22 data path MUST reproduce Plan 29-03's "
                    "random_15pct Brier to 4-decimal precision. This is the byte-"
                    "identity anchor against Phase 29 closure."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Reproducibility",
            "",
        ]
    )
    try:
        import scipy

        scipy_version = scipy.__version__
    except ImportError:
        scipy_version = "(import failed)"
    try:
        import sklearn

        sk_version = sklearn.__version__
    except ImportError:
        sk_version = "(import failed)"
    lines.extend(
        [
            f"- scipy version: {scipy_version}",
            f"- sklearn version: {sk_version}",
            f"- Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            f"- AUDIT-01 baseline SHA: `{EXPECTED_XGB_V2_SHA}`",
            f"- Formula hash: `{EXPECTED_FORMULA_HASH}`",
            f"- Seeds default (D-04): `{SEEDS_DEFAULT}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 30 / VARIANCE-V23-01..03 -- META-V22 multi-seed "
            "bootstrap-variance spike (v2.3; bootstrap-on-meta-train-rows; "
            "fork of v2.2 spike per D-03)."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(SEEDS_DEFAULT),
        help=(
            "Random seeds for the bootstrap variance harness "
            f"(default: SEEDS_DEFAULT={SEEDS_DEFAULT}; D-04)."
        ),
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help=(
            "Deterministic path: skip bootstrap_resample and fit on the raw "
            "(X_meta_train, y_meta_train) rows. Used by the canonical-"
            "lineage test to reproduce Plan 29-03 META-V22 random_15pct "
            "Brier 0.1409 +/- 0.0001 at seed=42."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Use synthetic v2.2 90-col data instead of live DB load "
            "(integration-test path; mirrors train_meta_v22.py --dry-run)."
        ),
    )
    parser.add_argument(
        "--report-path",
        default=str(VARIANCE_REPORT_PATH),
        help="Output path for 30-VARIANCE-REPORT.md.",
    )
    parser.add_argument(
        "--halt-path",
        default=str(VARIANCE_HALT_PATH),
        help="Output path for 30-VARIANCE-HALT.md (conditional).",
    )
    parser.add_argument(
        "--sha-end-path",
        default=str(SHA_END_PATH),
        help="Output path for 30-XGB-V2-SHA-PHASE-30-END.txt.",
    )
    args = parser.parse_args(argv)

    spike_started = datetime.now(timezone.utc)
    print(f"[spike-v23] Phase 30 META variance spike -- args: {vars(args)}")

    # ── AUDIT-01 startup (MID-equivalent runtime check) ──────────────
    xgb_v2_sha_start = _assert_xgb_v2_sha("STARTUP")

    # ── D-08 formula-hash drift guard ────────────────────────────────
    _assert_formula_hash()

    # ── Load META train + eval matrices (verbatim train_meta_v22 path) ─
    print("[spike-v23] Loading meta-train + meta-eval matrices...")
    t0 = time.time()
    (
        X_meta_train,
        y_meta_train,
        X_meta_eval,
        y_meta_eval,
        fight_dates_eval,
    ) = _load_meta_train_eval_matrices(dry_run=args.dry_run)
    print(
        f"[spike-v23]   X_meta_train.shape={X_meta_train.shape}, "
        f"X_meta_eval.shape={X_meta_eval.shape} "
        f"(load time {time.time() - t0:.1f}s)"
    )

    # ── Multi-seed harness call (D-01 / D-02 / D-05) ─────────────────
    seeds_list = [int(s) for s in args.seeds]
    if args.no_bootstrap:
        print(
            f"[spike-v23] --no-bootstrap mode: running deterministic path on "
            f"{len(seeds_list)} seed(s). DO NOT call assert_distinct_seed_brier "
            "-- deterministic LR + no row resample means N seeds -> N "
            "IDENTICAL tuples (canonical-lineage anchor)."
        )
        per_seed = _no_bootstrap_metrics(
            X_meta_train,
            y_meta_train,
            X_meta_eval,
            y_meta_eval,
            fight_dates_eval,
            seeds=seeds_list,
        )
    else:
        print(
            f"[spike-v23] Bootstrap mode: running variance harness with "
            f"{len(seeds_list)} seeds (D-04 default: {SEEDS_DEFAULT})."
        )
        from ufc_prediction.ml.variance import (
            assert_distinct_seed_brier,
            multi_seed_metrics,
        )

        per_seed = multi_seed_metrics(
            X_meta_train,
            y_meta_train,
            X_meta_eval,
            y_meta_eval,
            fight_dates_eval,
            seeds=seeds_list,
            fit_fn=_meta_fit_fn,
        )
        # D-05 runtime check -- complements the unit-test guard.
        assert_distinct_seed_brier(per_seed)
        print(
            f"[spike-v23] assert_distinct_seed_brier OK -- "
            f"{len(seeds_list)} distinct Brier on every slice."
        )

    # ── Aggregate variance (D-08 + Pitfall-B fallback) ───────────────
    # Representative model: first-seed bootstrap fit (mirrors v2.2 spike
    # line 506 -- `per_seed_meta[args.seeds[0]]`). On --no-bootstrap path,
    # representative is fit on raw (X_meta_train, y_meta_train).
    #
    # WR-02 / Phase 30 code-review: this path-asymmetry IS observable in
    # the BCa CI half-widths (bootstrap_per_slice_ci computes per-fight
    # residuals from the representative's predictions, and a model fit on
    # bootstrap-resampled rows vs raw rows produces different residuals).
    # The asymmetry is internally CONSISTENT on each path:
    #   - bootstrap path  : per_seed Brier values come from bootstrap-fit
    #                       models; representative is ALSO bootstrap-fit
    #                       (seed=seeds_list[0]) → CI half from same regime.
    #   - --no-bootstrap  : per_seed Brier comes from raw-fit models;
    #                       representative is ALSO raw-fit → CI half from
    #                       same regime.
    # Documented asymmetry preserves bit-identity with v2.2 spike line 506
    # (lineage anchor) on the bootstrap path; the --no-bootstrap path's CI
    # half-width is therefore NOT directly comparable to the bootstrap
    # path's. Downstream consumers must read `bootstrap_enabled` in the
    # report frontmatter before comparing CI half-widths across runs.
    from ufc_prediction.ml.variance import (
        aggregate_variance,
        bootstrap_resample,
    )

    if args.no_bootstrap:
        representative_model = _meta_fit_fn(
            X_meta_train,
            y_meta_train,
            seeds_list[0],
        )
    else:
        Xb, yb = bootstrap_resample(
            X_meta_train,
            y_meta_train,
            seed=seeds_list[0],
        )
        representative_model = _meta_fit_fn(Xb, yb, seeds_list[0])
    aggregated, warnings = aggregate_variance(
        per_seed,
        representative_model=representative_model,
        X_eval=X_meta_eval,
        y_eval=y_meta_eval,
        fight_dates_eval=fight_dates_eval,
    )

    # Log Pitfall-B warnings to stdout for operator visibility.
    if warnings:
        for w in warnings:
            print(f"[spike-v23] WARN (Pitfall-B): {w}")

    # ── D-06 HARD HALT check ─────────────────────────────────────────
    # CR-01 / Phase 30 code-review: gate the zero-variance HALT to the
    # bootstrap branch only. The `--no-bootstrap` deterministic path
    # produces N identical metric tuples by construction (closed-form LR
    # + no row resample), so `seed_std == 0.0` is EXPECTED on that path,
    # NOT a collapse signal. Firing HALT there would mislead the
    # operator (HALT body claims "bootstrap collapsed" — but the user
    # explicitly disabled bootstrap). Single-seed --no-bootstrap → N=1
    # → `np.std([x], ddof=1) == NaN` (silently bypasses the `== 0.0`
    # check); multi-seed --no-bootstrap → 5 identical tuples → seed_std
    # == 0.0 → spurious HALT before the fix.
    if args.no_bootstrap:
        zero_variance_slices: list[str] = []
    else:
        zero_variance_slices = [
            slc
            for slc, v in aggregated.items()
            if v["seed_std_brier"] == 0.0 or v["seed_std_acc"] == 0.0
        ]

    # Capture canonical-lineage Brier for report frontmatter.
    canonical_lineage_brier: float | None = None
    if args.no_bootstrap and 42 in per_seed:
        canonical_lineage_brier = float(per_seed[42]["random_15pct"]["brier_score"])
        delta = abs(canonical_lineage_brier - PLAN_29_03_RANDOM_15PCT_BRIER)
        match = delta < PLAN_29_03_TOLERANCE
        print(
            f"[spike-v23] Canonical-lineage check (seed=42 no-bootstrap): "
            f"brier_random_15pct={canonical_lineage_brier:.6f}, "
            f"target={PLAN_29_03_RANDOM_15PCT_BRIER:.4f}, delta={delta:.6f}, "
            f"match (+/- {PLAN_29_03_TOLERANCE})={match}"
        )

    spike_finished = datetime.now(timezone.utc)

    if zero_variance_slices:
        # Path B: emit HALT artifact + exit 3.
        outcome_path = "path_b_collapse_halted"
        halt_body = _render_variance_halt(
            zero_variance_slices,
            aggregated,
            per_seed,
            triggered_at=spike_finished.isoformat(),
        )
        halt_path = Path(args.halt_path)
        halt_path.parent.mkdir(parents=True, exist_ok=True)
        halt_path.write_text(halt_body, encoding="utf-8")
        print(
            f"[spike-v23] D-06 HALT: {zero_variance_slices} produced "
            f"seed_std == 0. Wrote {halt_path}."
        )
    else:
        outcome_path = "path_a_non_zero_variance"
        print("[spike-v23] No zero-variance slices detected -- Path A (advance to Phase 31).")

    # ── Emit VARIANCE-REPORT.md (always -- Path A or B materialized) ──
    report_body = _render_variance_report(
        spike_started=spike_started,
        spike_finished=spike_finished,
        seeds=seeds_list,
        bootstrap_enabled=not args.no_bootstrap,
        n_meta_train=int(X_meta_train.shape[0]),
        n_meta_eval=int(X_meta_eval.shape[0]),
        aggregated=aggregated,
        per_seed=per_seed,
        warnings=warnings,
        outcome_path=outcome_path,
        canonical_lineage_brier=canonical_lineage_brier,
        xgb_v2_sha=xgb_v2_sha_start,
    )
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_body, encoding="utf-8")
    print(f"[spike-v23] Wrote variance report: {report_path}")

    # ── AUDIT-01 END SHA artifact ────────────────────────────────────
    # WR-04 / Phase 30 code-review: the END SHA artifact is gated to Path A
    # only. Its presence is the downstream "spike succeeded → advance to
    # Phase 31" signal; emitting it on Path B HALT would mislead
    # consumers into treating a HALTed run as success-advance-ready. The
    # AUDIT-01 byte-identity SHA check itself still runs on every path
    # (via _assert_xgb_v2_sha at STARTUP); the END artifact is the
    # operator-facing "Path A complete" marker.
    sha_end_path = Path(args.sha_end_path)

    # Pitfall D -- do NOT call git; operator owns the commit.
    if zero_variance_slices:
        # Run the AUDIT-01 END SHA check for byte-identity assurance, but
        # do NOT emit the END artifact file on Path B.
        _assert_xgb_v2_sha("END")
        print(
            "[spike-v23] Spike complete WITH HALT. Operator decision "
            "required: review 30-VARIANCE-HALT.md, set actual_outcome in "
            "30-SUMMARY.md, then commit artifacts manually."
        )
        print(
            "[spike-v23] END SHA artifact NOT emitted on Path B (its "
            "presence is reserved for Path A 'advance to Phase 31' "
            "signaling per WR-04 fix)."
        )
        return 3  # Path B exit code (D-06)

    xgb_v2_sha_end = _assert_xgb_v2_sha("END")
    sha_end_path.parent.mkdir(parents=True, exist_ok=True)
    sha_end_path.write_text(xgb_v2_sha_end + "\n", encoding="utf-8")
    print(f"[spike-v23] Wrote END SHA artifact: {sha_end_path}")

    print(
        "[spike-v23] Spike complete (Path A). Wrote:\n"
        f"  {report_path}\n"
        f"  {sha_end_path}\n"
        "[spike-v23] Review 30-VARIANCE-REPORT.md, then commit "
        "artifacts manually."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
