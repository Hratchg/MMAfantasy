#!/usr/bin/env python
"""Phase 32 Plan 32-01 — Forward-stepwise composition (META → CALIB → REF → TRAVEL).

FORK SOURCE: ``scripts/train_meta_v22.py::run_stepwise`` (lines 723-999).
The source is READ-ONLY per AUDIT-01 + Phase 32 fork-not-mutate binding.
This module PREPENDS META baseline + CALIB steps in front of the existing
REF + TRAVEL composition + adds a triple-gate decision tree + conditional
``models/meta/meta_v3.joblib`` promotion per D-07(P15) hard-gate-then-save.

Per CONTEXT.md D-02 (operator-locked): META → CALIB → REF → TRAVEL ordering.
NO CAMP step (Phase 29 audit MISSED top-30 at 45.92% < 60%).

Per CONTEXT.md D-03 (operator-locked): triple-gate hard-AND for promotion:
  1. Gate clearance: candidate Brier <= per_slice.brier_max AND
     candidate accuracy >= per_slice.accuracy_min on all 3 slices
     (gate_contract_v2.3.json).
  2. Total-margin hurdle: candidate Brier <= META-V22 Brier - 0.003
     on all 3 slices.
  3. Per-step hurdle: each composition step (META→CALIB, CALIB→REF,
     REF→TRAVEL) clears >= 0.003 Brier improvement over prior step.

  If ALL THREE clear → promote to ``models/meta/meta_v3.joblib`` per
  D-07(P15) hard-gate-then-save. If ANY fails → Path C materializes:
  META-V22 stays canonical; STEPWISE-V23-03 "tried and didn't help"
  documented in SUMMARY frontmatter.

Per Pitfall 1: META + CALIB are DISCRETE STEPS, not implicit. Step 1
fits ``MetaLearnerLogistic`` on ``META_V22_FEATURE_COLUMNS`` (13 cols).
Step 2 wraps Step 1's surviving model in
``CalibratedClassifierCV(method='isotonic', cv=5)`` and re-evaluates on
the SAME META Level-1 substrate (no new columns; CALIB does not change
the feature set, only the decision-function calibration).

Per Pitfall 9: if Step 1 META Brier exceeds gate brier_max on ALL 3
slices, the driver short-circuits to ``path_c`` and skips Steps 2-4
(still emits MID + END SHA + report).

Usage:
    uv run python scripts/compose_v23_meta.py                       # live DB
    uv run python scripts/compose_v23_meta.py --cached-fixtures-only --seeds 5
    uv run python scripts/compose_v23_meta.py --dry-run             # synthetic smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

# ─────────────────────────── Phase 32 constants ──────────────────────────────

EXPECTED_XGB_V2_SHA256: str = "0b0b40afc8ec41d87508745a9b5f40a46f7d86c054b1ab2acece03d319f6fecd"
EXPECTED_CUTOFF_DATE: str = "2023-01-01"
SEEDS_DEFAULT: tuple[int, ...] = (42, 43, 44, 45, 46)

# D-13(v2.0) AF-10 LOCKED — never tune downward.
STEPWISE_HURDLE: float = 0.003

# D-03 leg 2 — META-V22 baseline gap.
TOTAL_MARGIN_HURDLE: float = 0.003

PER_SLICE_KEYS: tuple[str, ...] = (
    "most_recent_12mo",
    "most_recent_24mo",
    "random_15pct",
)

# META-V22 baseline Brier per slice — single source of truth is
# .planning/phases/26-forward-stepwise-candidate-promotion/META_V22_SPIKE.json
# (median_per_slice). Embedded here as the D-03 leg-2 anchor; matches
# the values in the file at module-import time (we also validate against
# the file at module-import for drift detection).
META_V22_SPIKE_PATH: Path = (
    Path(".planning/phases/26-forward-stepwise-candidate-promotion") / "META_V22_SPIKE.json"
)
META_V22_BASELINE_BRIER: dict[str, float] = {
    # Phase 73 (DEBT-V261-01): reconciled with .planning/phases/26-…/META_V22_SPIKE.json::median_per_slice.
    # Drift from Phase 26 → v2.6 close was a stale constant; SPIKE.json is the canonical source of truth.
    "most_recent_12mo": 0.15465991222123496,
    "most_recent_24mo": 0.15935635076986582,
    "random_15pct": 0.14167626892373050,
}


def _validate_meta_v22_baseline_anchor() -> None:
    """Cross-check embedded META_V22_BASELINE_BRIER against META_V22_SPIKE.json.

    Drift here = embedded constants disagree with the file. Halts loudly.
    """
    if not META_V22_SPIKE_PATH.exists():
        # Module is importable even if the file is missing (tests + dev mode).
        return
    spike = json.loads(META_V22_SPIKE_PATH.read_text(encoding="utf-8"))
    median = spike.get("median_per_slice", {})
    for slc in PER_SLICE_KEYS:
        if slc not in median:
            return  # incomplete file; skip silently
        file_brier = float(median[slc]["brier_score"])
        const_brier = META_V22_BASELINE_BRIER[slc]
        if abs(file_brier - const_brier) > 1e-9:
            raise AssertionError(
                f"META_V22_BASELINE_BRIER drift on {slc}: const={const_brier!r} "
                f"file={file_brier!r} (META_V22_SPIKE.json)"
            )


_validate_meta_v22_baseline_anchor()


# Phase 32 artifact paths.
PHASE32_DIR: Path = Path(
    ".planning/phases/32-forward-stepwise-recomposition-partner-v110",
)
COMPOSITION_REPORT_PATH: Path = PHASE32_DIR / "COMPOSITION_V23_REPORT.json"
SHA_MID_PATH: Path = PHASE32_DIR / "32-XGB-V2-SHA-PHASE-32-MID-PLAN-01.txt"
SHA_END_PATH: Path = PHASE32_DIR / "32-XGB-V2-SHA-PHASE-32-END-PLAN-01.txt"
META_DIR: Path = Path("models/meta")

# Cached fixtures path (fork source uses this for OOF reuse).
PHASE26_DIR: Path = Path(".planning/phases/26-forward-stepwise-candidate-promotion")
META_OOF_PARQUET_PATH: Path = PHASE26_DIR / "oof_predictions_v22.parquet"


# ─────────────────────── Pure functions (importable by tests) ───────────────


def per_step_brier_clears(
    candidate: dict[str, float],
    baseline: dict[str, float],
    hurdle: float = STEPWISE_HURDLE,
) -> tuple[bool, list[str]]:
    """Forward-stepwise per-step Brier hurdle check.

    Returns (all_slices_clear, failures) where failures is a list of
    human-readable strings shaped like ``"{slice}: Δ={delta:.4f} < {hurdle:.3f}"``
    (verbatim format from train_meta_v22.py:872-875).

    Per D-13(v2.0) AF-10 binding: ALL 3 slices must clear by Δ >= hurdle.
    Per Pitfall 6: NaN candidate Brier on ANY slice → explicit reject
    (not silent treat-as-zero-improvement).
    """
    failures: list[str] = []
    for slc in PER_SLICE_KEYS:
        cand = float(candidate[slc])
        base = float(baseline[slc])
        if math.isnan(cand) or math.isnan(base):
            failures.append(f"{slc}: NaN delta (cand={cand!r}, base={base!r})")
            continue
        delta = base - cand
        if math.isnan(delta):
            failures.append(f"{slc}: NaN delta")
            continue
        if delta < hurdle:
            failures.append(f"{slc}: Δ={delta:.4f} < {hurdle:.3f}")
    return (len(failures) == 0, failures)


def _check_gate_clearance(
    median_per_slice: dict[str, dict[str, float]],
    contract: Any,
) -> tuple[bool, list[str]]:
    """Leg 1 of triple-gate: per-slice Brier + accuracy thresholds.

    Mirrors ``ufc_prediction.ml.evaluator.gate_verdict`` but accepts a
    duck-typed contract (dataclass-or-mock) so unit tests can supply
    lightweight fixtures.
    """
    failed: list[str] = []
    for slc in PER_SLICE_KEYS:
        metrics = median_per_slice[slc]
        brier = float(metrics["brier_score"])
        acc = float(metrics["accuracy"])
        if math.isnan(brier) or math.isnan(acc):
            failed.append(f"{slc}: NaN gate metric (brier={brier!r}, acc={acc!r})")
            continue
        thresholds = contract.per_slice[slc]
        if brier > thresholds.brier_max:
            failed.append(f"{slc}: brier_score {brier:.4f} > {thresholds.brier_max}")
        if acc < thresholds.accuracy_min:
            failed.append(f"{slc}: accuracy {acc:.4f} < {thresholds.accuracy_min}")
    return (len(failed) == 0, failed)


def _check_total_margin(
    final_candidate_brier: dict[str, float],
    hurdle: float = TOTAL_MARGIN_HURDLE,
) -> tuple[bool, list[str]]:
    """Leg 2 of triple-gate: total-margin >= 0.003 over META-V22 on all 3 slices."""
    failed: list[str] = []
    for slc in PER_SLICE_KEYS:
        cand = float(final_candidate_brier[slc])
        baseline = META_V22_BASELINE_BRIER[slc]
        if math.isnan(cand):
            failed.append(f"{slc}: NaN total-margin (cand brier is NaN)")
            continue
        margin = baseline - cand
        if margin < hurdle:
            failed.append(
                f"{slc}: total margin Δ={margin:.4f} < {hurdle:.3f} "
                f"vs META-V22 baseline {baseline:.5f}"
            )
    return (len(failed) == 0, failed)


def _check_per_step_clears(prior_step_clears: dict[str, bool]) -> tuple[bool, list[str]]:
    """Leg 3 of triple-gate: every transition cleared its >=0.003 hurdle."""
    failed: list[str] = []
    for transition, clears in prior_step_clears.items():
        if not clears:
            failed.append(f"per-step transition '{transition}' did not clear hurdle")
    return (len(failed) == 0, failed)


def triple_gate_decision(
    *,
    final_candidate_brier: dict[str, float],
    prior_step_clears: dict[str, bool],
    contract: Any,
    median_final_per_slice: dict[str, dict[str, float]],
) -> tuple[str, str, list[str]]:
    """Compose the 3 legs into a final verdict + outcome path.

    Per D-03 hard-AND: ALL three legs must pass for promotion.

    Args:
        final_candidate_brier: per-slice Brier for the final surviving
            candidate (TRAVEL step if it cleared, else the highest step
            that cleared).
        prior_step_clears: per-transition bool dict (keys =
            "meta_to_calib", "calib_to_ref", "ref_to_travel"). True if
            transition cleared >= 0.003 per-step hurdle on ALL 3 slices.
        contract: GateContract-shaped object with .per_slice[slc].brier_max
            and .per_slice[slc].accuracy_min. Supports both production
            ``GateContract`` and lightweight test mocks.
        median_final_per_slice: median Brier + accuracy per slice for
            the final candidate (used for gate-clearance leg 1).

    Returns:
        (verdict, path, failures) where
        - verdict ∈ {"promote", "reject"}
        - path ∈ {"path_a", "path_b", "path_c"}
        - failures: list of human-readable reasons (empty on path_a)

    Path semantics:
        - Path A: all 3 legs clear → promote.
        - Path B: some-but-not-all steps cleared (per-step leg failure
          where SOME transitions cleared) AND surviving candidate misses
          gate → partial composition documented but not promoted.
        - Path C: every other rejection (gate-miss, margin-miss,
          all-step-miss, NaN).
    """
    gate_pass, gate_failures = _check_gate_clearance(median_final_per_slice, contract)
    margin_pass, margin_failures = _check_total_margin(final_candidate_brier)
    per_step_pass, per_step_failures = _check_per_step_clears(prior_step_clears)

    failures = gate_failures + margin_failures + per_step_failures

    if gate_pass and margin_pass and per_step_pass:
        return ("promote", "path_a", [])

    # Determine Path B vs Path C.
    # Path B = some prior steps cleared AND gate-leg failed
    # (partial composition with surviving candidate missing gate).
    # Path C = everything else (total failure or NaN).
    any_step_cleared = any(prior_step_clears.values())
    all_steps_cleared = all(prior_step_clears.values())
    nan_in_brier = any(math.isnan(float(final_candidate_brier[slc])) for slc in PER_SLICE_KEYS)

    if nan_in_brier:
        return ("reject", "path_c", failures)

    # Path B: partial composition. Per CONTEXT.md D-07 / spec narrative:
    # "some steps clear, some don't; surviving META + CALIB candidate
    #  clears gate but REF/TRAVEL don't add value; documented but not
    #  promoted (META-V22 stays canonical)".
    # Two valid Path B signatures:
    #   (a) surviving candidate misses gate (gate-leg fails) AND some
    #       prior step cleared — partial composition with non-promotable
    #       Brier;
    #   (b) surviving candidate CLEARS gate + total-margin, but the
    #       per-step ALL-must-clear discipline fails because REF/TRAVEL
    #       didn't add value (later steps rejected). This is the "tried,
    #       META+CALIB worked, later steps didn't help" outcome — Path B.
    if any_step_cleared and not all_steps_cleared:
        only_per_step_failed = gate_pass and margin_pass and not per_step_pass
        if (not gate_pass) or only_per_step_failed:
            return ("reject", "path_b", failures)

    # Path C: everything else (gate-miss with no prior steps cleared, or
    # margin-miss when steps did clear but the surviving candidate has
    # too thin a margin over META-V22, etc.).
    return ("reject", "path_c", failures)


# ─────────────────────────── SHA helpers (AUDIT-01) ──────────────────────────


def _read_xgb_v2_sha() -> str:
    """Compute hashlib.sha256 of models/xgb_v2.joblib (current bytes)."""
    return hashlib.sha256(Path("models/xgb_v2.joblib").read_bytes()).hexdigest()


def _write_sha_artifact(path: Path) -> str:
    """Read xgb_v2.joblib SHA, assert it equals baseline, write to path.

    Verbatim port of train_meta_v22.py:195-203 (renamed for clarity).
    """
    sha = _read_xgb_v2_sha()
    assert sha == EXPECTED_XGB_V2_SHA256, (
        f"AUDIT-01 violation: xgb_v2 SHA drifted to {sha[:12]}... "
        f"expected {EXPECTED_XGB_V2_SHA256[:12]}..."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sha + "\n", encoding="utf-8")
    return sha


# ─────────────────────────── Data loading helpers ────────────────────────────


def _load_assembled_data_v22(
    cutoff_date_iso: str = EXPECTED_CUTOFF_DATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Load fights + assemble the 90-col v2.2 feature matrix from live DB.

    Mirror of train_meta_v22._load_assembled_data_v22 (verbatim port; the
    function is needed at runtime but we keep a fresh copy here per
    fork-not-mutate binding).
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

    cutoff_date_obj = date.fromisoformat(cutoff_date_iso)
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

    config = MLConfig(cutoff_date=cutoff_date_iso)
    assembler = FeatureMatrixAssembler(config)
    X_v22, y, fight_dates = assembler.assemble(
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
    assert X_v22.shape[1] == 90, (
        f"v2.2 assembled matrix shape mismatch: got {X_v22.shape[1]} expected 90"
    )
    return X_v22, y, fight_dates, fight_records


def _build_synthetic_data_v22(n: int = 600):
    """Synthetic v2.2 90-col fixture for --dry-run."""
    import datetime as _datetime

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
            d = today.replace(day=max(1, (i % 28))) - _datetime.timedelta(
                days=int(rng.integers(1, 364))
            )
        dates.append(d)
        fight_ids.append(i)
    return X_v22, y, np.array(dates), fight_ids, base_cutoff, today


def _compute_elo_prob_for_fight(fight: dict, elo_features: dict) -> float:
    """As-of-fight-date Elo P(A wins). Port of train_meta_v22._compute_elo_prob_for_fight."""
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


# ─────────────────────────── Step driver ─────────────────────────────────────


def _eval_meta_step(
    *,
    meta_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    fight_dates_eval: np.ndarray,
    seeds: list[int],
    fit_factory,
) -> dict[str, dict[str, float]]:
    """Fit per-seed + evaluate per slice; return median dict.

    fit_factory(seed) returns a NEW estimator already fitted on (X_train, y_train).
    """
    from ufc_prediction.ml.evaluator import evaluate_per_slice
    from ufc_prediction.ml.trainer import median_metrics

    per_seed_results: dict[int, dict] = {}
    for seed in seeds:
        model = fit_factory(seed, X_train, y_train)
        per_seed_results[seed] = evaluate_per_slice(
            model,
            X_eval,
            y_eval,
            fight_dates_eval,
        )
    return median_metrics(list(per_seed_results.values()))


def _step_payload(
    *,
    step: str,
    baseline_brier: dict[str, float],
    median_step: dict[str, dict[str, float]],
    gate_pass: bool,
    gate_failures: list[str],
    feature_columns: list[str],
    n_seeds: int,
    n_train_after_nan_drop: int | None = None,
    n_eval_after_nan_drop: int | None = None,
    extra: dict | None = None,
) -> dict:
    """Build per-step payload dict matching train_meta_v22.py:878-894 shape."""
    candidate_brier = {slc: float(median_step[slc]["brier_score"]) for slc in PER_SLICE_KEYS}
    delta = {slc: baseline_brier[slc] - candidate_brier[slc] for slc in PER_SLICE_KEYS}
    step_clears_bool, hurdle_failures = per_step_brier_clears(
        candidate_brier,
        baseline_brier,
    )
    stepwise_clears = bool(gate_pass and step_clears_bool)
    payload = {
        "step": step,
        "baseline_brier_per_slice": baseline_brier,
        "candidate_brier_per_slice": candidate_brier,
        "candidate_accuracy_per_slice": {
            slc: float(median_step[slc]["accuracy"]) for slc in PER_SLICE_KEYS
        },
        "hurdle_delta_per_slice": delta,
        "gate_verdict": bool(gate_pass),
        "gate_failures": list(gate_failures),
        "hurdle_failures": hurdle_failures,
        "step_clears": step_clears_bool,
        "stepwise_clears": stepwise_clears,
        "feature_columns": list(feature_columns),
        "n_seeds": int(n_seeds),
        "produced_at": datetime.now(tz=UTC).isoformat(),
    }
    if n_train_after_nan_drop is not None:
        payload["n_train_after_nan_drop"] = int(n_train_after_nan_drop)
    if n_eval_after_nan_drop is not None:
        payload["n_eval_after_nan_drop"] = int(n_eval_after_nan_drop)
    if extra:
        payload.update(extra)
    return payload


def _short_circuit_path_c(reason: str, steps: list[dict]) -> dict:
    """Build a Path C report skeleton when Step 1 META fails gate decisively."""
    return {
        "outcome_path": "path_c",
        "triple_gate_pass": False,
        "failures": [reason],
        "short_circuited": True,
        "steps": steps,
    }


def run_composition(args) -> dict:
    """Main orchestration — runs 4-step composition + triple-gate decision.

    Returns the full report dict (also written to COMPOSITION_V23_REPORT.json).
    On Path A, also saves models/meta/meta_v3.joblib via save_meta_model.
    """
    from sklearn.calibration import CalibratedClassifierCV

    from ufc_prediction.ml.config import FEATURE_COLUMNS_V22
    from ufc_prediction.ml.evaluator import evaluate_per_slice, gate_verdict
    from ufc_prediction.ml.gate_contract import load_gate_contract
    from ufc_prediction.ml.meta_features_v22 import (
        META_V22_FEATURE_COLUMNS,
        build_meta_features_v22,
    )
    from ufc_prediction.ml.meta_learner import MetaLearnerLogistic
    from ufc_prediction.ml.meta_persistence import (
        compute_meta_input_distribution_hash,
        save_meta_model,
    )
    from ufc_prediction.ml.oof import (
        _make_oof_estimator,
        apply_nan_drop_policy,
        generate_oof_predictions,
        make_three_way_split,
    )
    from ufc_prediction.ml.trainer import median_metrics

    # Plan 29-02 carry-forward (EVAL-V23-01): per-feature strict-baseline
    # NaN-drop policy. Baseline cols (xgb_oof_prob, elo_prob) MUST be non-NaN
    # (gate-critical inputs); non-baseline cols (closing_prob_diff,
    # sharp_money_signal, days_since_last_fight_diff, ...) are imputed with
    # train-set column medians (no eval-time leakage). This is the canonical
    # v2.3 substrate handling — symmetric drop on the 13-col META Level-1
    # leaves only ~1 surviving eval row (BFO cols are 99% NaN on the most
    # recent 365-day window).
    NAN_DROP_POLICY = "per_feature_strict_baseline"
    BASELINE_COLS = ("xgb_oof_prob", "elo_prob")

    print(f"[compose_v23] args: {vars(args)}")
    sha_actual = _read_xgb_v2_sha()
    assert sha_actual == EXPECTED_XGB_V2_SHA256, (
        f"AUDIT-01 violation at composition start: xgb_v2 SHA drift. "
        f"got={sha_actual} expected={EXPECTED_XGB_V2_SHA256}"
    )
    print(f"[compose_v23] AUDIT-01 preflight: OK ({sha_actual[:12]}...)")

    # ── Step 0: Load data ──
    if args.dry_run:
        print("[compose_v23] DRY-RUN: synthetic 90-col v2.2 fixture")
        X_v22, y, fight_dates, fight_ids, _, _ = _build_synthetic_data_v22()
        fight_records = [
            {
                "fight_id": fight_ids[i],
                "event_date": fight_dates[i].item()
                if hasattr(fight_dates[i], "item")
                else fight_dates[i],
                "fighter_a_id": i * 2,
                "fighter_b_id": i * 2 + 1,
            }
            for i in range(len(fight_ids))
        ]
    else:
        print("[compose_v23] Loading data + assembling 90-col v2.2 feature matrix from DB...")
        X_v22, y, fight_dates, fight_records = _load_assembled_data_v22()
    print(f"[compose_v23] X_v22.shape={X_v22.shape}, n_records={len(fight_records)}")

    base_train_fights, meta_train_fights, meta_eval_fights = make_three_way_split(
        fight_records,
        base_cutoff=date.fromisoformat(EXPECTED_CUTOFF_DATE),
        meta_eval_window_days=365,
        today=date.today(),
    )
    print(
        f"[compose_v23] split: base={len(base_train_fights)} "
        f"meta_train={len(meta_train_fights)} meta_eval={len(meta_eval_fights)}"
    )

    meta_train_ids = {f["fight_id"] for f in meta_train_fights}
    meta_eval_ids = {f["fight_id"] for f in meta_eval_fights}
    meta_train_idx = np.array(
        [i for i, f in enumerate(fight_records) if f["fight_id"] in meta_train_ids]
    )
    meta_eval_idx = np.array(
        [i for i, f in enumerate(fight_records) if f["fight_id"] in meta_eval_ids]
    )

    # OOF on 72-col view (xgb_v2 sub-matrix).
    X_oof = X_v22[meta_train_idx][:, :72]
    assert X_oof.shape[1] == 72, f"OOF base XGB requires 72-col view; got {X_oof.shape[1]}"
    # Dry-run uses synthetic event dates that don't match the cached parquet's
    # date range — force a fresh OOF compute on synthetic fixtures.
    use_cache = not args.no_cache_oof and not args.dry_run
    cache_path = META_OOF_PARQUET_PATH if use_cache else None
    xgb_oof_prob, _oof_meta = generate_oof_predictions(
        X_oof,
        y[meta_train_idx],
        fight_dates[meta_train_idx],
        base_trainer=None,
        cache_path=cache_path,
        force_rebuild=not use_cache,
        fight_ids=[fight_records[i]["fight_id"] for i in meta_train_idx],
    )
    sort_idx = np.argsort(fight_dates[meta_train_idx])
    xgb_oof_aligned = np.empty_like(xgb_oof_prob)
    xgb_oof_aligned[sort_idx] = xgb_oof_prob

    # Elo probs.
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
        elo_prob_train = np.array(
            [_compute_elo_prob_for_fight(fight_records[i], _elo_features) for i in meta_train_idx]
        )
        elo_prob_eval = np.array(
            [_compute_elo_prob_for_fight(fight_records[i], _elo_features) for i in meta_eval_idx]
        )

    # Build meta_eval Level-1 (single transient base train, per Phase 26 pattern).
    print("[compose_v23] Building meta_eval Level-1 (1 base train + Elo lookups)...")
    base_estimator = _make_oof_estimator(seed=42)
    base_estimator.fit(X_v22[meta_train_idx][:, :72], y[meta_train_idx])
    xgb_eval_prob = base_estimator.predict_proba(X_v22[meta_eval_idx][:, :72])[:, 1]

    base_train_meta_v22 = build_meta_features_v22(
        xgb_oof_aligned,
        elo_prob_train,
        X_v22[meta_train_idx],
    )
    base_eval_meta_v22 = build_meta_features_v22(
        xgb_eval_prob,
        elo_prob_eval,
        X_v22[meta_eval_idx],
    )

    contract = load_gate_contract(version="v2.3")
    seeds = list(args.seeds)
    print(f"[compose_v23] gate contract v2.3 loaded (formula_hash={contract.formula_hash[:12]}...)")

    # ────────────────────── Step 1: META baseline ─────────────────────────
    # Per Plan 29-02 carry-forward: per_feature_strict_baseline drop
    # (baseline cols MUST be non-NaN) + non-baseline column train-median
    # imputation (no eval-time leakage).
    print("[compose_v23] STEP 1 (META): fitting MetaLearnerLogistic on 13 cols...")
    meta_feature_cols = list(META_V22_FEATURE_COLUMNS)
    meta_train_mask = apply_nan_drop_policy(
        base_train_meta_v22,
        meta_feature_cols,
        policy=NAN_DROP_POLICY,
        baseline_columns=BASELINE_COLS,
    )
    meta_eval_mask = apply_nan_drop_policy(
        base_eval_meta_v22,
        meta_feature_cols,
        policy=NAN_DROP_POLICY,
        baseline_columns=BASELINE_COLS,
    )
    X_meta_train_clean = base_train_meta_v22[meta_train_mask].copy()
    y_meta_train_clean = y[meta_train_idx][meta_train_mask]
    X_meta_eval_clean = base_eval_meta_v22[meta_eval_mask].copy()
    y_meta_eval_clean = y[meta_eval_idx][meta_eval_mask]
    fight_dates_meta_eval = fight_dates[meta_eval_idx][meta_eval_mask]

    # Plan 29-02 imputation step — non-baseline column NaN values get
    # train-set column medians (computed AFTER drop, then applied to BOTH
    # train + eval matrices using the same medians). This is what makes
    # PolynomialFeatures downstream not crash on residual NaNs after the
    # per-feature drop.
    non_baseline_idx_meta = [i for i, c in enumerate(meta_feature_cols) if c not in BASELINE_COLS]
    nan_imputation_medians_meta: dict[str, float] = {}
    for idx in non_baseline_idx_meta:
        col_train = X_meta_train_clean[:, idx]
        finite = col_train[~np.isnan(col_train)]
        median_val = float(np.median(finite)) if finite.size else 0.0
        nan_imputation_medians_meta[meta_feature_cols[idx]] = median_val
        train_nan_mask = np.isnan(X_meta_train_clean[:, idx])
        if train_nan_mask.any():
            X_meta_train_clean[train_nan_mask, idx] = median_val
        eval_nan_mask = np.isnan(X_meta_eval_clean[:, idx])
        if eval_nan_mask.any():
            X_meta_eval_clean[eval_nan_mask, idx] = median_val
    print(
        f"[compose_v23] STEP 1 NaN policy={NAN_DROP_POLICY}: "
        f"train={meta_train_mask.sum()}/{len(meta_train_mask)} kept "
        f"eval={meta_eval_mask.sum()}/{len(meta_eval_mask)} kept "
        f"(imputed {len(non_baseline_idx_meta)} non-baseline cols)"
    )

    per_seed_meta: dict[int, dict] = {}
    per_seed_meta_models: dict[int, Any] = {}
    for seed in seeds:
        m = MetaLearnerLogistic(random_state=seed).fit(
            X_meta_train_clean,
            y_meta_train_clean,
        )
        per_seed_meta[seed] = evaluate_per_slice(
            m,
            X_meta_eval_clean,
            y_meta_eval_clean,
            fight_dates_meta_eval,
        )
        per_seed_meta_models[seed] = m
    median_meta = median_metrics(list(per_seed_meta.values()))
    meta_gate_pass, meta_gate_failures = gate_verdict(median_meta, contract)

    # META baseline is the META-V22 spike baseline (constant).
    meta_payload = _step_payload(
        step="META",
        baseline_brier=dict(META_V22_BASELINE_BRIER),
        median_step=median_meta,
        gate_pass=meta_gate_pass,
        gate_failures=meta_gate_failures,
        feature_columns=list(META_V22_FEATURE_COLUMNS),
        n_seeds=len(seeds),
        n_train_after_nan_drop=int(meta_train_mask.sum()),
        n_eval_after_nan_drop=int(meta_eval_mask.sum()),
        extra={"baseline_source": "META_V22_BASELINE_BRIER (Phase 26 spike median)"},
    )
    meta_to_calib_baseline = {slc: float(median_meta[slc]["brier_score"]) for slc in PER_SLICE_KEYS}
    print(
        f"[compose_v23] STEP 1 (META): gate_pass={meta_gate_pass} "
        f"brier_per_slice={meta_to_calib_baseline}"
    )

    # AUDIT-01 MID checkpoint — after META baseline confirm.
    mid_sha = _write_sha_artifact(SHA_MID_PATH)
    print(f"[compose_v23] AUDIT-01 MID → {SHA_MID_PATH} ({mid_sha[:12]}...)")

    # Pitfall 9 short-circuit: if META Brier exceeds ALL 3 gate brier_max
    # thresholds, skip Steps 2-4 and materialize Path C immediately.
    meta_brier_above_all_gates = all(
        float(median_meta[slc]["brier_score"]) > contract.per_slice[slc].brier_max
        for slc in PER_SLICE_KEYS
    )
    if meta_brier_above_all_gates:
        print(
            "[compose_v23] PITFALL 9 SHORT-CIRCUIT: META Brier exceeds gate "
            "brier_max on ALL 3 slices — materializing Path C immediately."
        )
        # Mark subsequent steps as skipped.
        skipped_steps = [
            {"step": s, "skipped": True, "reason": "META Step 1 short-circuited Path C"}
            for s in ("CALIB", "REF", "TRAVEL")
        ]
        report = {
            "phase": "32",
            "plan": "01",
            "gate_contract_ref": ".planning/gate_contract_v2.3.json",
            "gate_contract_version": "v2.3",
            "feature_columns_hash": contract.feature_columns_hash,
            "meta_v22_baseline_brier": dict(META_V22_BASELINE_BRIER),
            "produced_at": datetime.now(tz=UTC).isoformat(),
            "xgb_v2_sha256": EXPECTED_XGB_V2_SHA256,
            "steps": [meta_payload, *skipped_steps],
            "triple_gate_pass": False,
            "outcome_path": "path_c",
            "failures": [
                f"META Step 1 short-circuit: Brier per slice "
                f"{meta_to_calib_baseline} exceeds gate_max thresholds "
                f"on ALL 3 slices (Pitfall 9)."
            ],
            "short_circuited": True,
            "meta_v3_promoted": False,
            "n_seeds": len(seeds),
        }
        _emit_report_and_end_sha(report)
        return report

    # ────────────────────── Step 2: CALIB (isotonic wrap) ─────────────────
    # Wrap a fresh sklearn Pipeline equivalent to MetaLearnerLogistic in
    # CalibratedClassifierCV(method='isotonic', cv=5) on the SAME META
    # Level-1 substrate. Pitfall 1 binding: CALIB is a DISCRETE step.
    #
    # Implementation detail (deviation Rule 1 - bug fix): MetaLearnerLogistic
    # is NOT a clonable sklearn estimator (no get_params/set_params), so
    # CalibratedClassifierCV cannot use it via sklearn.base.clone. We
    # construct the equivalent sklearn Pipeline directly inside the wrap.
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline as _SkPipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    def _make_calib_pipeline(seed: int) -> _SkPipeline:
        """Equivalent to MetaLearnerLogistic.pipeline (clone-friendly)."""
        return _SkPipeline(
            [
                (
                    "poly",
                    PolynomialFeatures(
                        degree=2,
                        interaction_only=True,
                        include_bias=False,
                    ),
                ),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=seed,
                    ),
                ),
            ]
        )

    print("[compose_v23] STEP 2 (CALIB): isotonic CalibratedClassifierCV wrap...")
    per_seed_calib: dict[int, dict] = {}
    per_seed_calib_models: dict[int, Any] = {}
    calib_n_train_clean = int(meta_train_mask.sum())
    cv_folds = min(5, max(2, calib_n_train_clean // 30))
    for seed in seeds:
        base_pipeline = _make_calib_pipeline(seed)
        try:
            calibrated = CalibratedClassifierCV(
                estimator=base_pipeline,
                method="isotonic",
                cv=cv_folds,
            ).fit(X_meta_train_clean, y_meta_train_clean)
            per_seed_calib[seed] = evaluate_per_slice(
                calibrated,
                X_meta_eval_clean,
                y_meta_eval_clean,
                fight_dates_meta_eval,
            )
            per_seed_calib_models[seed] = calibrated
        except Exception as e:
            print(f"[compose_v23] CALIB seed={seed} failed: {e!r}", file=sys.stderr)
            per_seed_calib[seed] = {
                slc: {"brier_score": float("nan"), "accuracy": float("nan")}
                for slc in PER_SLICE_KEYS
            }

    median_calib = median_metrics(list(per_seed_calib.values()))
    calib_gate_pass, calib_gate_failures = gate_verdict(median_calib, contract)
    calib_payload = _step_payload(
        step="CALIB",
        baseline_brier=meta_to_calib_baseline,
        median_step=median_calib,
        gate_pass=calib_gate_pass,
        gate_failures=calib_gate_failures,
        feature_columns=list(META_V22_FEATURE_COLUMNS),
        n_seeds=len(seeds),
        n_train_after_nan_drop=int(meta_train_mask.sum()),
        n_eval_after_nan_drop=int(meta_eval_mask.sum()),
        extra={
            "baseline_source": "META (Step 1 median brier)",
            "calibration_method": "isotonic",
            "calibration_cv_folds": cv_folds,
        },
    )
    meta_to_calib_clears = bool(calib_payload["step_clears"])
    # Forward-stepwise pruning: if CALIB rejected, REF baseline = META baseline.
    if meta_to_calib_clears:
        calib_to_ref_baseline = {
            slc: float(median_calib[slc]["brier_score"]) for slc in PER_SLICE_KEYS
        }
        baseline_source_for_ref = "CALIB (cleared)"
        _surviving_after_calib_brier = calib_to_ref_baseline
    else:
        calib_to_ref_baseline = dict(meta_to_calib_baseline)
        baseline_source_for_ref = "META (CALIB rejected; pruned)"
        _surviving_after_calib_brier = meta_to_calib_baseline
    print(
        f"[compose_v23] STEP 2 (CALIB): step_clears={meta_to_calib_clears} "
        f"gate_pass={calib_gate_pass}"
    )

    # ────────────────────── Step 3: REF (3 cols cumulatively) ─────────────
    print("[compose_v23] STEP 3 (REF): + 3 referee cols...")
    REF_COLS = [
        "ref_finish_rate_shrunk",
        "ref_decision_rate_shrunk",
        "ref_no_action_rate_shrunk",
    ]
    ref_indices = [FEATURE_COLUMNS_V22.index(c) for c in REF_COLS]
    ref_data_train = X_v22[meta_train_idx][:, ref_indices]
    ref_data_eval = X_v22[meta_eval_idx][:, ref_indices]
    X_ref_train = np.column_stack([base_train_meta_v22, ref_data_train])
    X_ref_eval = np.column_stack([base_eval_meta_v22, ref_data_eval])
    ref_feature_cols = meta_feature_cols + REF_COLS

    # Per-feature strict-baseline drop on the cumulative META + REF matrix.
    ref_train_mask = apply_nan_drop_policy(
        X_ref_train,
        ref_feature_cols,
        policy=NAN_DROP_POLICY,
        baseline_columns=BASELINE_COLS,
    )
    ref_eval_mask = apply_nan_drop_policy(
        X_ref_eval,
        ref_feature_cols,
        policy=NAN_DROP_POLICY,
        baseline_columns=BASELINE_COLS,
    )
    X_ref_train_clean = X_ref_train[ref_train_mask].copy()
    y_ref_train_clean = y[meta_train_idx][ref_train_mask]
    X_ref_eval_clean = X_ref_eval[ref_eval_mask].copy()
    y_ref_eval_clean = y[meta_eval_idx][ref_eval_mask]
    fight_dates_ref_eval = fight_dates[meta_eval_idx][ref_eval_mask]
    ref_data_coverage_pct = float((ref_data_train != 0).any(axis=1).mean())

    # Impute non-baseline NaNs with train-medians (Plan 29-02 carry-forward).
    non_baseline_idx_ref = [i for i, c in enumerate(ref_feature_cols) if c not in BASELINE_COLS]
    for idx in non_baseline_idx_ref:
        col_train = X_ref_train_clean[:, idx]
        finite = col_train[~np.isnan(col_train)]
        median_val = float(np.median(finite)) if finite.size else 0.0
        train_nan_mask = np.isnan(X_ref_train_clean[:, idx])
        if train_nan_mask.any():
            X_ref_train_clean[train_nan_mask, idx] = median_val
        eval_nan_mask = np.isnan(X_ref_eval_clean[:, idx])
        if eval_nan_mask.any():
            X_ref_eval_clean[eval_nan_mask, idx] = median_val

    per_seed_ref: dict[int, dict] = {}
    per_seed_ref_models: dict[int, Any] = {}
    for seed in seeds:
        m = MetaLearnerLogistic(random_state=seed).fit(
            X_ref_train_clean,
            y_ref_train_clean,
        )
        per_seed_ref[seed] = evaluate_per_slice(
            m,
            X_ref_eval_clean,
            y_ref_eval_clean,
            fight_dates_ref_eval,
        )
        per_seed_ref_models[seed] = m
    median_ref = median_metrics(list(per_seed_ref.values()))
    ref_gate_pass, ref_gate_failures = gate_verdict(median_ref, contract)
    ref_payload = _step_payload(
        step="REF",
        baseline_brier=calib_to_ref_baseline,
        median_step=median_ref,
        gate_pass=ref_gate_pass,
        gate_failures=ref_gate_failures,
        feature_columns=list(META_V22_FEATURE_COLUMNS) + REF_COLS,
        n_seeds=len(seeds),
        n_train_after_nan_drop=int(ref_train_mask.sum()),
        n_eval_after_nan_drop=int(ref_eval_mask.sum()),
        extra={
            "baseline_source": baseline_source_for_ref,
            "ref_data_coverage_pct": ref_data_coverage_pct,
        },
    )
    calib_to_ref_clears = bool(ref_payload["step_clears"])
    if calib_to_ref_clears:
        ref_to_travel_baseline = {
            slc: float(median_ref[slc]["brier_score"]) for slc in PER_SLICE_KEYS
        }
        baseline_source_for_travel = "REF (cleared)"
    else:
        ref_to_travel_baseline = dict(calib_to_ref_baseline)
        baseline_source_for_travel = f"{baseline_source_for_ref} (REF rejected; pruned)"
    print(
        f"[compose_v23] STEP 3 (REF): step_clears={calib_to_ref_clears} "
        f"gate_pass={ref_gate_pass} coverage={ref_data_coverage_pct:.4f}"
    )

    # ────────────────────── Step 4: TRAVEL (6 cols cumulatively) ──────────
    print("[compose_v23] STEP 4 (TRAVEL): + 6 travel cols (cumulative on REF)...")
    TRAVEL_COLS = [
        "travel_distance_miles_red",
        "travel_distance_miles_blue",
        "travel_distance_miles_diff",
        "tz_shift_red_signed",
        "tz_shift_blue_signed",
        "tz_shift_diff_signed",
    ]
    travel_indices = [FEATURE_COLUMNS_V22.index(c) for c in TRAVEL_COLS]
    travel_data_train = X_v22[meta_train_idx][:, travel_indices]
    travel_data_eval = X_v22[meta_eval_idx][:, travel_indices]
    X_travel_train = np.column_stack([X_ref_train, travel_data_train])
    X_travel_eval = np.column_stack([X_ref_eval, travel_data_eval])
    travel_feature_cols = ref_feature_cols + TRAVEL_COLS

    travel_train_mask = apply_nan_drop_policy(
        X_travel_train,
        travel_feature_cols,
        policy=NAN_DROP_POLICY,
        baseline_columns=BASELINE_COLS,
    )
    travel_eval_mask = apply_nan_drop_policy(
        X_travel_eval,
        travel_feature_cols,
        policy=NAN_DROP_POLICY,
        baseline_columns=BASELINE_COLS,
    )
    X_travel_train_clean = X_travel_train[travel_train_mask].copy()
    y_travel_train_clean = y[meta_train_idx][travel_train_mask]
    X_travel_eval_clean = X_travel_eval[travel_eval_mask].copy()
    y_travel_eval_clean = y[meta_eval_idx][travel_eval_mask]
    fight_dates_travel_eval = fight_dates[meta_eval_idx][travel_eval_mask]
    travel_data_coverage_pct = float(
        (~np.isnan(travel_data_train)).any(axis=1).mean(),
    )

    # Impute non-baseline NaNs.
    non_baseline_idx_travel = [
        i for i, c in enumerate(travel_feature_cols) if c not in BASELINE_COLS
    ]
    for idx in non_baseline_idx_travel:
        col_train = X_travel_train_clean[:, idx] if X_travel_train_clean.size else np.array([])
        finite = col_train[~np.isnan(col_train)] if col_train.size else np.array([])
        median_val = float(np.median(finite)) if finite.size else 0.0
        if X_travel_train_clean.size:
            train_nan_mask = np.isnan(X_travel_train_clean[:, idx])
            if train_nan_mask.any():
                X_travel_train_clean[train_nan_mask, idx] = median_val
        if X_travel_eval_clean.size:
            eval_nan_mask = np.isnan(X_travel_eval_clean[:, idx])
            if eval_nan_mask.any():
                X_travel_eval_clean[eval_nan_mask, idx] = median_val

    per_seed_travel: dict[int, dict] = {}
    per_seed_travel_models: dict[int, Any] = {}
    if X_travel_train_clean.shape[0] > 0 and X_travel_eval_clean.shape[0] > 0:
        for seed in seeds:
            m = MetaLearnerLogistic(random_state=seed).fit(
                X_travel_train_clean,
                y_travel_train_clean,
            )
            per_seed_travel[seed] = evaluate_per_slice(
                m,
                X_travel_eval_clean,
                y_travel_eval_clean,
                fight_dates_travel_eval,
            )
            per_seed_travel_models[seed] = m

    if per_seed_travel:
        median_travel = median_metrics(list(per_seed_travel.values()))
        travel_gate_pass, travel_gate_failures = gate_verdict(median_travel, contract)
        travel_payload = _step_payload(
            step="TRAVEL",
            baseline_brier=ref_to_travel_baseline,
            median_step=median_travel,
            gate_pass=travel_gate_pass,
            gate_failures=travel_gate_failures,
            feature_columns=list(META_V22_FEATURE_COLUMNS) + REF_COLS + TRAVEL_COLS,
            n_seeds=len(seeds),
            n_train_after_nan_drop=int(travel_train_mask.sum()),
            n_eval_after_nan_drop=int(travel_eval_mask.sum()),
            extra={
                "baseline_source": baseline_source_for_travel,
                "travel_data_coverage_pct": travel_data_coverage_pct,
            },
        )
        ref_to_travel_clears = bool(travel_payload["step_clears"])
        median_travel_for_decision = median_travel
        travel_degenerate = False
    else:
        # Degenerate case: 0 surviving rows after symmetric NaN-drop.
        travel_payload = {
            "step": "TRAVEL",
            "baseline_brier_per_slice": ref_to_travel_baseline,
            "candidate_brier_per_slice": {slc: float("nan") for slc in PER_SLICE_KEYS},
            "candidate_accuracy_per_slice": {slc: float("nan") for slc in PER_SLICE_KEYS},
            "hurdle_delta_per_slice": {slc: float("nan") for slc in PER_SLICE_KEYS},
            "gate_verdict": False,
            "gate_failures": [
                "TRAVEL step degenerate: 0 surviving rows after symmetric "
                f"NaN-drop (coverage={travel_data_coverage_pct:.4f})"
            ],
            "hurdle_failures": ["NaN delta on degenerate TRAVEL"],
            "step_clears": False,
            "stepwise_clears": False,
            "feature_columns": list(META_V22_FEATURE_COLUMNS) + REF_COLS + TRAVEL_COLS,
            "n_seeds": len(seeds),
            "n_train_after_nan_drop": int(travel_train_mask.sum()),
            "n_eval_after_nan_drop": int(travel_eval_mask.sum()),
            "baseline_source": baseline_source_for_travel,
            "travel_data_coverage_pct": travel_data_coverage_pct,
            "degenerate": True,
            "produced_at": datetime.now(tz=UTC).isoformat(),
        }
        ref_to_travel_clears = False
        median_travel_for_decision = {
            slc: {"brier_score": float("nan"), "accuracy": float("nan")} for slc in PER_SLICE_KEYS
        }
        travel_degenerate = True
    print(
        f"[compose_v23] STEP 4 (TRAVEL): step_clears={ref_to_travel_clears} "
        f"degenerate={travel_degenerate} coverage={travel_data_coverage_pct:.4f}"
    )

    # ────────────────────── Triple-gate decision ──────────────────────────
    # Pick the final candidate: the highest step that cleared.
    # If TRAVEL cleared: final = TRAVEL
    # elif REF cleared: final = REF
    # elif CALIB cleared: final = CALIB
    # else: final = META baseline
    prior_step_clears = {
        "meta_to_calib": meta_to_calib_clears,
        "calib_to_ref": calib_to_ref_clears,
        "ref_to_travel": ref_to_travel_clears,
    }
    if ref_to_travel_clears:
        final_label = "TRAVEL"
        final_brier = {
            slc: float(median_travel_for_decision[slc]["brier_score"]) for slc in PER_SLICE_KEYS
        }
        median_final = median_travel_for_decision
        per_seed_final_models = per_seed_travel_models
        final_feature_columns = list(META_V22_FEATURE_COLUMNS) + REF_COLS + TRAVEL_COLS
        final_train_clean = X_travel_train_clean
        final_y_train_clean = y_travel_train_clean
        final_xgb_oof_aligned = (
            xgb_oof_aligned[
                np.isin(np.arange(len(xgb_oof_aligned)), np.where(travel_train_mask)[0])
            ]
            if travel_train_mask.sum() > 0
            else xgb_oof_aligned
        )
    elif calib_to_ref_clears:
        final_label = "REF"
        final_brier = {slc: float(median_ref[slc]["brier_score"]) for slc in PER_SLICE_KEYS}
        median_final = median_ref
        per_seed_final_models = per_seed_ref_models
        final_feature_columns = list(META_V22_FEATURE_COLUMNS) + REF_COLS
        final_train_clean = X_ref_train_clean
        final_y_train_clean = y_ref_train_clean
        final_xgb_oof_aligned = xgb_oof_aligned[
            np.isin(np.arange(len(xgb_oof_aligned)), np.where(ref_train_mask)[0])
        ]
    elif meta_to_calib_clears:
        final_label = "CALIB"
        final_brier = {slc: float(median_calib[slc]["brier_score"]) for slc in PER_SLICE_KEYS}
        median_final = median_calib
        per_seed_final_models = per_seed_calib_models
        final_feature_columns = list(META_V22_FEATURE_COLUMNS)
        final_train_clean = X_meta_train_clean
        final_y_train_clean = y_meta_train_clean
        final_xgb_oof_aligned = xgb_oof_aligned[meta_train_mask]
    else:
        final_label = "META"
        final_brier = {slc: float(median_meta[slc]["brier_score"]) for slc in PER_SLICE_KEYS}
        median_final = median_meta
        per_seed_final_models = per_seed_meta_models
        final_feature_columns = list(META_V22_FEATURE_COLUMNS)
        final_train_clean = X_meta_train_clean
        final_y_train_clean = y_meta_train_clean
        final_xgb_oof_aligned = xgb_oof_aligned[meta_train_mask]

    verdict, outcome_path, failures = triple_gate_decision(
        final_candidate_brier=final_brier,
        prior_step_clears=prior_step_clears,
        contract=contract,
        median_final_per_slice=median_final,
    )

    print(
        f"[compose_v23] Triple-gate decision: verdict={verdict} "
        f"outcome_path={outcome_path} final_step={final_label} "
        f"failures={failures}"
    )

    triple_gate_pass = verdict == "promote"
    meta_v3_promoted = False
    meta_v3_sha256: str | None = None

    # ────────────────────── Conditional meta_v3 promotion ─────────────────
    # Path A only: save_meta_model per D-07(P15) hard-gate-then-save.
    if outcome_path == "path_a":
        # Pick the seed=42 model as canonical (per train_meta_v22 convention).
        chosen_seed = seeds[0]
        final_model = per_seed_final_models.get(chosen_seed)
        if final_model is None:
            raise RuntimeError(
                f"Path A triple-gate cleared but no model available for "
                f"seed={chosen_seed} (final_label={final_label})"
            )
        input_hash = compute_meta_input_distribution_hash(
            final_train_clean,
            final_y_train_clean,
            final_xgb_oof_aligned[: len(final_train_clean)]
            if len(final_xgb_oof_aligned) >= len(final_train_clean)
            else np.full(len(final_train_clean), np.nan),
        )
        oof_parquet_sha = (
            hashlib.sha256(META_OOF_PARQUET_PATH.read_bytes()).hexdigest()
            if META_OOF_PARQUET_PATH.exists()
            else "0" * 64
        )

        META_DIR.mkdir(parents=True, exist_ok=True)
        model_path, meta_path = save_meta_model(
            final_model,
            meta_kind="logistic",
            meta_version="v3",
            base_model_version="v2",
            base_model_sha256=EXPECTED_XGB_V2_SHA256,
            meta_feature_columns=final_feature_columns,
            meta_input_distribution_hash=input_hash,
            meta_oof_parquet_sha256=oof_parquet_sha,
            meta_learner_brier_delta_vs_logistic=0.0,
            best_params={
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "PolynomialFeatures": ("degree=2 interaction_only=True include_bias=False"),
                "final_step": final_label,
                "composition_chain": ["META", "CALIB", "REF", "TRAVEL"],
            },
            metrics={
                "per_slice_median": _convert_for_json(median_final),
                "gate_verdict_passed": True,
                "stepwise_clears_vs_meta_v22": True,
                "triple_gate_pass": True,
                "outcome_path": "path_a",
                "final_brier_per_slice": final_brier,
                "meta_v22_baseline_brier": dict(META_V22_BASELINE_BRIER),
                "prior_step_clears": prior_step_clears,
            },
            trained_by_script="scripts/compose_v23_meta.py",
            phase="32",
        )
        meta_v3_promoted = True
        meta_v3_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        print(
            f"[compose_v23] PATH A: meta_v3 promoted → {model_path} "
            f"(sha256={meta_v3_sha256[:12]}...)"
        )

    # ────────────────────── Build report ──────────────────────────────────
    report = {
        "phase": "32",
        "plan": "01",
        "gate_contract_ref": ".planning/gate_contract_v2.3.json",
        "gate_contract_version": "v2.3",
        "feature_columns_hash": contract.feature_columns_hash,
        "meta_v22_baseline_brier": dict(META_V22_BASELINE_BRIER),
        "produced_at": datetime.now(tz=UTC).isoformat(),
        "xgb_v2_sha256": EXPECTED_XGB_V2_SHA256,
        "steps": [
            meta_payload,
            calib_payload,
            ref_payload,
            travel_payload,
        ],
        "prior_step_clears": prior_step_clears,
        "final_step_chosen": final_label,
        "final_candidate_brier_per_slice": final_brier,
        "triple_gate_pass": bool(triple_gate_pass),
        "outcome_path": outcome_path,
        "failures": failures,
        "meta_v3_promoted": meta_v3_promoted,
        "meta_v3_sha256": meta_v3_sha256,
        "n_seeds": len(seeds),
        "short_circuited": False,
    }
    _emit_report_and_end_sha(report)
    return report


def _convert_for_json(d: Any) -> Any:
    """Recursively convert numpy types to Python scalars for JSON dump."""
    if isinstance(d, dict):
        return {k: _convert_for_json(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_convert_for_json(x) for x in d]
    if isinstance(d, np.integer):
        return int(d)
    if isinstance(d, np.floating):
        return float(d)
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d


def _emit_report_and_end_sha(report: dict) -> None:
    """Write COMPOSITION_V23_REPORT.json + AUDIT-01 END SHA artifact."""
    COMPOSITION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPOSITION_REPORT_PATH.write_text(
        json.dumps(_convert_for_json(report), indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[compose_v23] COMPOSITION_V23_REPORT → {COMPOSITION_REPORT_PATH}")
    end_sha = _write_sha_artifact(SHA_END_PATH)
    print(f"[compose_v23] AUDIT-01 END → {SHA_END_PATH} ({end_sha[:12]}...)")


# ─────────────────────────── CLI entry ────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 32 Plan 32-01 — Forward-stepwise composition + triple-gate",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(SEEDS_DEFAULT),
        help="Random seeds for MetaLearnerLogistic (default: 42 43 44 45 46)",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Run with bootstrap variance (slower; ≥10 seeds recommended)",
    )
    parser.add_argument(
        "--cached-fixtures-only",
        action="store_true",
        help="Force using cached OOF parquet (Phase 26 fixture); no live DB OOF rebuild",
    )
    parser.add_argument(
        "--no-cache-oof",
        action="store_true",
        help="Force OOF parquet rebuild (ignores cache_path)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Synthetic 90-col v2.2 smoke (no DB / real xgb_v2 inference)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = run_composition(args)
    return 0 if report["outcome_path"] in {"path_a", "path_b", "path_c"} else 2


if __name__ == "__main__":
    sys.exit(main())
