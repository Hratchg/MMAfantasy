#!/usr/bin/env python
"""Phase 42 Plan 42-02 — TRAVEL recomposition spike (META-V22 + CALIB + TRAVEL-V25).

FORK SOURCE: ``scripts/compose_v23_meta.py`` (READ-ONLY per AUDIT-01
fork-not-mutate binding; Plan 42-02 verifies via ``git diff --stat``).

This module takes the v2.3 canonical Path A stack META-V22 + CALIB
(already materialised in ``models/meta/meta_v2.joblib``) as a FIXED
baseline and adds TRAVEL-V25 (2 new Level-1 cols ``travel_distance_km`` +
``tz_shift_hours`` from Plan 42-01) as additional inputs. A fresh OOF
blender candidate is trained in memory (multi-seed median per Phase 32
pattern), evaluated per slice on the v2.3 widened slices (12mo / 24mo /
random_15pct), and the candidate is HELD IN MEMORY ONLY — persistence is
Plan 42-03's job under Path A only.

D-18 binding gate (CONTEXT-locked, formula hash ``7d221b4a...0a``):
  - **Floor**: candidate Brier ≤ baseline Brier on ALL 3 slices AND
    candidate accuracy ≥ 0.70 on ALL 3 slices.
  - **Hurdle**: candidate Brier improvement ≥ 0.003 over baseline on
    MAJORITY (≥2/3) of slices.

NB: the 0.70 accuracy floor is the COARSE CONTEXT D-18 lock and is
DELIBERATELY looser than the per-slice ``gate_contract_v2.3.json``
accuracy_min values (0.7814 / 0.7627 / 0.7812). Per CONTEXT
§Recomposition the locked gate is 0.70 verbatim — Plan 42-02 implements
THIS threshold; the report ALSO carries the gate_contract per-slice
numbers for operator transparency (consumed by Plan 42-03).

Per AUDIT-01: ``models/xgb_v2.joblib`` + ``models/meta/meta_v2.joblib``
SHAs byte-identical end-to-end. The candidate blender is NOT persisted in
this plan — Plan 42-03 conditionally persists it as
``models/meta/meta_v22_travel.joblib`` (sibling, NOT canonical) only on
Path A.

Per Pitfall 6 (Phase 32): NaN Brier on ANY slice → that slice does NOT
count toward majority hurdle (silent treat-as-fail rejected — explicit
NaN handling at the slice level).

Per Pitfall 9 (Phase 32): if TRAVEL-V25 cols are NaN-heavy enough that
post-NaN-drop candidate eval set has <50% of baseline eval rows on ANY
slice, set ``report.candidate_substrate_warning = "small_effective_n"``
so Plan 42-03 can flag the operator.

Usage:
    uv run python scripts/compose_v25_travel.py --seeds 5            # live DB
    uv run python scripts/compose_v25_travel.py --cached-fixtures-only --seeds 3
    uv run python scripts/compose_v25_travel.py --dry-run            # synthetic
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

# ───────────────────────── Phase 42 LOCKED constants ─────────────────────────

# D-18 LOCKED — formula hash NO post-measurement renegotiation
# (PROJECT.md + CONTEXT §AUDIT-01 invariant).
FORMULA_HASH: str = (
    "7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a"
)

# CONTEXT D-18 floor — 0.70 verbatim. DELIBERATELY looser than per-slice
# gate_contract_v2.3.json accuracy_min (0.7814 / 0.7627 / 0.7812). The
# coarse 0.70 floor is the locked operator surface for Phase 42; the
# per-slice contract numbers are recorded in the report for transparency
# but Plan 42-02 + 42-03 gate against 0.70 verbatim.
FLOOR_ACCURACY_MIN: float = 0.70

# CONTEXT D-18 + D-13(v2.0) AF-10 — Brier hurdle LOCKED at 0.003; never
# tune downward.
HURDLE_BRIER_MIN: float = 0.003

# CONTEXT D-18 — majority threshold ≥2/3 slices.
HURDLE_MAJORITY_THRESHOLD: int = 2

# Phase 16 / Plan 16-04 / MODEL-01 — 3-slice canonical convention.
PER_SLICE_KEYS: tuple[str, ...] = (
    "most_recent_12mo",
    "most_recent_24mo",
    "random_15pct",
)

# PROJECT.md byte-identity invariants (AUDIT-01).
EXPECTED_XGB_V2_SHA256: str = (
    "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
)
EXPECTED_META_V2_SHA256: str = (
    "77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196"
)
EXPECTED_CUTOFF_DATE: str = "2023-01-01"
SEEDS_DEFAULT: tuple[int, ...] = (42, 43, 44, 45, 46)
SUBSTRATE_VERSION: str = "v2.5"

# Effective-n warning threshold per Pitfall 9.
SMALL_EFFECTIVE_N_FRACTION: float = 0.50

# Phase 42 artifact paths.
PHASE42_DIR: Path = Path(
    ".planning/phases/42-travel-feature-engineering-closeout",
)
TRAVEL_REPORT_PATH: Path = PHASE42_DIR / "TRAVEL_COMPOSITION_V25_REPORT.json"
PHASE26_DIR: Path = Path(".planning/phases/26-forward-stepwise-candidate-promotion")
META_OOF_PARQUET_PATH: Path = PHASE26_DIR / "oof_predictions_v22.parquet"
META_V2_PATH: Path = Path("models/meta/meta_v2.joblib")
XGB_V2_PATH: Path = Path("models/xgb_v2.joblib")
GATE_CONTRACT_V23_PATH: Path = Path(".planning/gate_contract_v2.3.json")


# ────────────────── Pure decision functions (importable by tests) ──────────────

def floor_clears(
    baseline: dict[str, float] | dict[str, dict[str, float]],
    candidate_brier: dict[str, float],
    candidate_accuracy: dict[str, float],
    *,
    floor_accuracy_min: float = FLOOR_ACCURACY_MIN,
) -> tuple[bool, list[str]]:
    """D-18 Floor: candidate Brier ≤ baseline Brier on ALL slices AND
    candidate accuracy ≥ ``floor_accuracy_min`` (default 0.70) on ALL slices.

    Args:
        baseline: per-slice baseline. Either a flat ``{slc: brier_float}`` dict
            OR a ``{slc: {"brier_score": float, "accuracy": float, ...}}`` dict.
            The function reads ``baseline[slc]`` and unwraps ``brier_score`` if
            the value is a dict; otherwise treats it as the flat baseline Brier.
        candidate_brier: per-slice candidate Brier (flat).
        candidate_accuracy: per-slice candidate accuracy (flat).
        floor_accuracy_min: floor (default 0.70 per CONTEXT D-18).

    Returns:
        ``(all_pass, failures)`` where ``failures`` is a list of human-readable
        strings shaped as
        ``"{slc}: brier {cand:.4f} > baseline {base:.4f}"`` and/or
        ``"{slc}: accuracy {acc:.4f} < {min:.2f}"``.

    NaN policy: NaN candidate Brier OR NaN candidate accuracy → that slice
    FAILS (the comparison ``nan > x`` is False but we still emit a failure
    message; same for accuracy).
    """
    failures: list[str] = []
    for slc in PER_SLICE_KEYS:
        base_raw = baseline[slc]
        if isinstance(base_raw, dict):
            base = float(base_raw["brier_score"])
        else:
            base = float(base_raw)
        cand = float(candidate_brier[slc])
        acc = float(candidate_accuracy[slc])

        # Brier-floor check (≤ baseline).
        if math.isnan(cand) or math.isnan(base):
            failures.append(
                f"{slc}: brier NaN (cand={cand!r}, base={base!r})"
            )
        elif cand > base:
            failures.append(
                f"{slc}: brier {cand:.4f} > baseline {base:.4f}"
            )

        # Accuracy-floor check (≥ floor_accuracy_min).
        if math.isnan(acc):
            failures.append(f"{slc}: accuracy NaN (acc={acc!r})")
        elif acc < floor_accuracy_min:
            failures.append(
                f"{slc}: accuracy {acc:.4f} < {floor_accuracy_min:.2f}"
            )
    return (len(failures) == 0, failures)


def hurdle_clears(
    baseline_brier: dict[str, float] | dict[str, dict[str, float]],
    candidate_brier: dict[str, float] | dict[str, dict[str, float]],
    *,
    hurdle_min: float = HURDLE_BRIER_MIN,
    majority_threshold: int = HURDLE_MAJORITY_THRESHOLD,
) -> tuple[bool, list[str]]:
    """D-18 Hurdle: ≥ ``hurdle_min`` Brier improvement on ≥ ``majority_threshold``
    slices (default 0.003 on ≥2/3).

    Args:
        baseline_brier: per-slice baseline. Either flat
            ``{slc: brier_float}`` OR nested ``{slc: {"brier_score": float}}``.
        candidate_brier: per-slice candidate (same shape).
        hurdle_min: per-slice Brier improvement floor (default 0.003 per
            CONTEXT D-18 + D-13(v2.0) AF-10 LOCKED).
        majority_threshold: minimum count of clearing slices required
            (default 2 of 3 per CONTEXT D-18).

    Returns:
        ``(majority_pass, per_slice_msgs)`` where ``per_slice_msgs`` is a list
        of human-readable strings; one per slice.

    NaN policy (Pitfall 6): NaN delta (from NaN cand OR NaN baseline) → that
    slice DOES NOT count as cleared (silent treat-as-zero-improvement
    rejected).
    """
    n_cleared = 0
    per_slice_msgs: list[str] = []
    for slc in PER_SLICE_KEYS:
        base_raw = baseline_brier[slc]
        cand_raw = candidate_brier[slc]
        if isinstance(base_raw, dict):
            base = float(base_raw["brier_score"])
        else:
            base = float(base_raw)
        if isinstance(cand_raw, dict):
            cand = float(cand_raw["brier_score"])
        else:
            cand = float(cand_raw)

        if math.isnan(base) or math.isnan(cand):
            per_slice_msgs.append(
                f"{slc}: NaN delta (cand={cand!r}, base={base!r}) — not cleared"
            )
            continue
        delta = base - cand
        if math.isnan(delta):
            per_slice_msgs.append(
                f"{slc}: NaN delta — not cleared"
            )
            continue
        if delta >= hurdle_min:
            n_cleared += 1
            per_slice_msgs.append(
                f"{slc}: Δ={delta:.4f} >= {hurdle_min:.3f} ✓"
            )
        else:
            per_slice_msgs.append(
                f"{slc}: Δ={delta:.4f} < {hurdle_min:.3f} ✗"
            )
    return (n_cleared >= majority_threshold, per_slice_msgs)


def determine_path(
    floor_result: tuple[bool, list[str]],
    hurdle_result: tuple[bool, list[str]],
) -> dict[str, bool]:
    """Mutually-exclusive Path A / Path B determination per CONTEXT §Recomposition.

    Args:
        floor_result: ``(floor_pass, failures)`` from :func:`floor_clears`.
        hurdle_result: ``(hurdle_pass, msgs)`` from :func:`hurdle_clears`.

    Returns:
        ``{"path_a_eligible": bool, "path_b_inevitable": bool}`` with the
        invariant ``path_a_eligible XOR path_b_inevitable == True``.

    Plan 42-03 is the authoritative path-materialisation step; this function
    emits the pre-computation eligibility only.
    """
    floor_pass, _ = floor_result
    hurdle_pass, _ = hurdle_result
    path_a_eligible = bool(floor_pass and hurdle_pass)
    return {
        "path_a_eligible": path_a_eligible,
        "path_b_inevitable": not path_a_eligible,
    }


def _unwrap_brier(per_slice: dict[str, Any]) -> dict[str, float]:
    """Unwrap nested per-slice metric dict → flat ``{slc: brier_score}``."""
    out: dict[str, float] = {}
    for slc in PER_SLICE_KEYS:
        v = per_slice[slc]
        if isinstance(v, dict):
            out[slc] = float(v["brier_score"]) if v["brier_score"] is not None else float("nan")
        else:
            out[slc] = float(v)
    return out


def _unwrap_accuracy(per_slice: dict[str, Any]) -> dict[str, float]:
    """Unwrap nested per-slice metric dict → flat ``{slc: accuracy}``."""
    out: dict[str, float] = {}
    for slc in PER_SLICE_KEYS:
        v = per_slice[slc]
        if isinstance(v, dict):
            out[slc] = float(v["accuracy"]) if v["accuracy"] is not None else float("nan")
        else:
            out[slc] = float(v)
    return out


def build_report(
    *,
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    gate_contract_ref: str,
    xgb_sha: str,
    meta_sha: str,
    extra: dict | None = None,
) -> dict:
    """Compose the TRAVEL_COMPOSITION_V25_REPORT dict consumed by Plan 42-03.

    Args:
        baseline: per-slice baseline metrics
            ``{slc: {"brier_score": float, "accuracy": float, ...}}``.
        candidate: per-slice candidate metrics (same shape).
        gate_contract_ref: path string to the gate_contract JSON used
            (recorded for audit).
        xgb_sha: ``models/xgb_v2.joblib`` SHA256 at report-build time.
        meta_sha: ``models/meta/meta_v2.joblib`` SHA256 at report-build time.
        extra: optional dict merged into the top-level report (e.g.
            ``cached_fixtures_mode``, ``candidate_substrate_warning``).

    Returns:
        Plain dict with native Python types (no numpy / no datetime) ready
        for ``json.dumps``.
    """
    baseline_brier_flat = _unwrap_brier(baseline)
    baseline_acc_flat = _unwrap_accuracy(baseline)
    candidate_brier_flat = _unwrap_brier(candidate)
    candidate_acc_flat = _unwrap_accuracy(candidate)

    floor_result = floor_clears(
        baseline_brier_flat, candidate_brier_flat, candidate_acc_flat,
    )
    hurdle_result = hurdle_clears(baseline_brier_flat, candidate_brier_flat)
    path = determine_path(floor_result, hurdle_result)

    per_slice_out: dict[str, dict[str, Any]] = {}
    for slc in PER_SLICE_KEYS:
        b = baseline_brier_flat[slc]
        c = candidate_brier_flat[slc]
        ba = baseline_acc_flat[slc]
        ca = candidate_acc_flat[slc]
        if math.isnan(b) or math.isnan(c):
            delta = float("nan")
        else:
            delta = float(b - c)
        # Per-slice floor: candidate brier ≤ baseline brier AND candidate acc ≥ 0.70
        slc_floor_brier_ok = (
            (not math.isnan(b))
            and (not math.isnan(c))
            and c <= b
        )
        slc_floor_acc_ok = (not math.isnan(ca)) and ca >= FLOOR_ACCURACY_MIN
        slc_floor_ok = bool(slc_floor_brier_ok and slc_floor_acc_ok)
        # Per-slice hurdle: delta ≥ 0.003 (NaN delta fails)
        slc_hurdle_ok = bool(
            (not math.isnan(delta)) and delta >= HURDLE_BRIER_MIN
        )
        per_slice_out[slc] = {
            "baseline_brier": float(b) if not math.isnan(b) else None,
            "candidate_brier": float(c) if not math.isnan(c) else None,
            "baseline_accuracy": float(ba) if not math.isnan(ba) else None,
            "candidate_accuracy": float(ca) if not math.isnan(ca) else None,
            "delta_brier": float(delta) if not math.isnan(delta) else None,
            "floor_clears": slc_floor_ok,
            "hurdle_clears": slc_hurdle_ok,
        }
        # Pass-through optional ``n`` and ``n_after_nan_drop`` fields if
        # the caller supplied them (Pitfall 9 transparency).
        for opt_key in ("n", "n_after_nan_drop", "n_baseline", "n_candidate"):
            if opt_key in baseline.get(slc, {}):
                per_slice_out[slc][f"n_baseline_{opt_key}"] = baseline[slc][opt_key]
            if opt_key in candidate.get(slc, {}):
                per_slice_out[slc][f"n_candidate_{opt_key}"] = candidate[slc][opt_key]

    floor_pass, floor_failures = floor_result
    hurdle_pass, hurdle_msgs = hurdle_result

    report: dict[str, Any] = {
        "phase": "42",
        "plan": "02",
        "substrate_version": SUBSTRATE_VERSION,
        "gate_contract_ref": gate_contract_ref,
        "formula_hash": FORMULA_HASH,
        "produced_at": datetime.now(tz=UTC).isoformat(),
        "xgb_v2_sha256": str(xgb_sha),
        "meta_v2_sha256": str(meta_sha),
        "baseline": _coerce_for_json(baseline),
        "candidate": _coerce_for_json(candidate),
        "per_slice": per_slice_out,
        "floor_clears_all_three": bool(floor_pass),
        "floor_failures": list(floor_failures),
        "hurdle_clears_majority": bool(hurdle_pass),
        "hurdle_per_slice_msgs": list(hurdle_msgs),
        "path_a_eligible": bool(path["path_a_eligible"]),
        "path_b_inevitable": bool(path["path_b_inevitable"]),
        "context_d18": {
            "floor_accuracy_min": float(FLOOR_ACCURACY_MIN),
            "hurdle_brier_min": float(HURDLE_BRIER_MIN),
            "hurdle_majority_threshold": int(HURDLE_MAJORITY_THRESHOLD),
            "note": (
                "0.70 is the COARSE CONTEXT D-18 lock; per-slice "
                "gate_contract_v2.3.json accuracy_min values are "
                "0.7814 / 0.7627 / 0.7812 (recorded under "
                "gate_contract_per_slice for operator transparency, "
                "NOT used as the binding floor for Plan 42-02)."
            ),
        },
    }
    if extra:
        for k, v in extra.items():
            report[k] = _coerce_for_json(v)
    return report


def _coerce_for_json(d: Any) -> Any:
    """Recursively coerce numpy / datetime / set → JSON-friendly Python."""
    if isinstance(d, dict):
        return {k: _coerce_for_json(v) for k, v in d.items()}
    if isinstance(d, (list, tuple, set)):
        return [_coerce_for_json(x) for x in d]
    if isinstance(d, np.integer):
        return int(d)
    if isinstance(d, np.floating):
        v = float(d)
        return v if not math.isnan(v) else None
    if isinstance(d, np.bool_):
        return bool(d)
    if isinstance(d, np.ndarray):
        return [_coerce_for_json(x) for x in d.tolist()]
    if isinstance(d, float) and math.isnan(d):
        return None
    if isinstance(d, datetime):
        return d.isoformat()
    if isinstance(d, date):
        return d.isoformat()
    return d


# ────────────────────── SHA helpers (AUDIT-01) ───────────────────────────────

def _sha256_path(path: Path) -> str:
    """Compute SHA256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_canonical_shas() -> tuple[str, str]:
    """AUDIT-01 byte-identity preflight: assert xgb_v2 + meta_v2 SHAs.

    Raises AssertionError on drift (HALT immediately).
    Returns ``(xgb_sha, meta_sha)`` on success.
    """
    xgb_sha = _sha256_path(XGB_V2_PATH)
    meta_sha = _sha256_path(META_V2_PATH)
    assert xgb_sha == EXPECTED_XGB_V2_SHA256, (
        f"AUDIT-01 violation: xgb_v2 SHA drifted to {xgb_sha[:12]}... "
        f"expected {EXPECTED_XGB_V2_SHA256[:12]}..."
    )
    assert meta_sha == EXPECTED_META_V2_SHA256, (
        f"AUDIT-01 violation: meta_v2 SHA drifted to {meta_sha[:12]}... "
        f"expected {EXPECTED_META_V2_SHA256[:12]}..."
    )
    return xgb_sha, meta_sha


# ────────────────────── Data loading helpers ─────────────────────────────────

def _load_assembled_data_v25_travel(
    cutoff_date_iso: str = EXPECTED_CUTOFF_DATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Load fights + assemble the 92-col v2.5-travel feature matrix from live DB.

    Mirrors ``scripts.compose_v23_meta._load_assembled_data_v22`` shape but
    requests ``feature_set='v2.5-travel'`` (Plan 42-01 dispatch).
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
        fighter_physicals, fight_records, cutoff_date_obj,
    )

    config = MLConfig(cutoff_date=cutoff_date_iso)
    assembler = FeatureMatrixAssembler(config)
    X_v25, y, fight_dates = assembler.assemble(
        fight_records, elo_features, computed_features,
        fighter_physicals, division_medians, round_stats,
        pre_ufc_records=pre_ufc,
        fight_odds=fight_odds,
        feature_set="v2.5-travel",
    )
    assert X_v25.shape[1] == 92, (
        f"v2.5-travel assembled matrix shape mismatch: "
        f"got {X_v25.shape[1]} expected 92"
    )
    return X_v25, y, fight_dates, fight_records


def _compute_elo_prob_for_fight(fight: dict, elo_features: dict) -> float:
    """As-of-fight-date Elo P(A wins). Verbatim port of Phase 32 helper."""
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


# ────────────────────── Multi-seed trainer ───────────────────────────────────

def _train_candidate_multi_seed(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    fight_dates_eval: np.ndarray,
    seeds: list[int],
) -> tuple[Any, dict[str, dict[str, float]]]:
    """Fit MetaLearnerLogistic per-seed; return (chosen_model, per-slice median).

    Returns a tuple ``(chosen_model_seed42, median_per_slice)`` — the seed=42
    model is returned for in-memory use by callers that need a probe model
    (Plan 42-02 does NOT persist the model — that's Plan 42-03 Path A only).
    """
    from ufc_prediction.ml.evaluator import evaluate_per_slice
    from ufc_prediction.ml.meta_learner import MetaLearnerLogistic
    from ufc_prediction.ml.trainer import median_metrics

    per_seed_results: dict[int, dict] = {}
    per_seed_models: dict[int, Any] = {}
    for seed in seeds:
        model = MetaLearnerLogistic(random_state=seed).fit(X_train, y_train)
        per_seed_results[seed] = evaluate_per_slice(
            model, X_eval, y_eval, fight_dates_eval,
        )
        per_seed_models[seed] = model
    median = median_metrics(list(per_seed_results.values()))
    chosen = per_seed_models[seeds[0]]
    return chosen, median


# ────────────────────── Baseline + candidate evaluators ──────────────────────

def _evaluate_baseline_meta_v22_calib(
    *,
    base_meta_v22: np.ndarray,    # 13-col META-V22 Level-1 substrate (eval slice)
    y_eval: np.ndarray,
    fight_dates_eval: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Evaluate the META-V22 + CALIB baseline (= ``models/meta/meta_v2.joblib``).

    NB on substrate: ``models/meta/meta_v2.joblib`` was persisted under the
    13-col META-V22 Level-1 set (see ``models/meta/meta_v2_meta.json``
    ``meta_feature_columns``). The pipeline includes ``PolynomialFeatures
    (degree=2, interaction_only=True, include_bias=False)`` which expects
    exactly 13 input features. We feed the full 13-col matrix as-is.

    [Rule 1 - Bug] Earlier draft of this helper sliced ``base_meta_v22[:, :3]``
    under the false assumption that meta_v2 used the minimal META-01 3-col
    substrate. That triggered ``ValueError: X has 3 features, but
    PolynomialFeatures is expecting 13 features as input`` against the
    persisted model. Fixed to pass the full 13-col input the model was
    trained on.
    """
    import joblib

    from ufc_prediction.ml.evaluator import evaluate_per_slice

    meta_v2_model = joblib.load(META_V2_PATH)
    assert base_meta_v22.shape[1] == 13, (
        f"meta_v2.joblib expects 13 META-V22 cols; got {base_meta_v22.shape[1]}"
    )
    return evaluate_per_slice(
        meta_v2_model, base_meta_v22, y_eval, fight_dates_eval,
    )


# ────────────────────── Main pipeline ────────────────────────────────────────

def _build_synthetic_v25(n: int = 600) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Synthetic 92-col v2.5-travel fixture for --dry-run."""
    import datetime as _datetime
    rng = np.random.default_rng(42)
    X_v25 = rng.standard_normal((n, 92))
    y = rng.integers(0, 2, size=n)
    today = date.today()
    dates: list[date] = []
    fight_ids: list[int] = []
    for i in range(n):
        if i < n // 3:
            d = date(2020 + (i % 3), 6, 1 + (i % 28))
        elif i < 2 * n // 3:
            d = date(2024, 1 + (i % 12), 1 + (i % 28))
        else:
            d = today - _datetime.timedelta(days=int(rng.integers(1, 364)))
        dates.append(d)
        fight_ids.append(i)
    fight_records = [
        {
            "fight_id": fight_ids[i],
            "event_date": dates[i],
            "fighter_a_id": i * 2,
            "fighter_b_id": i * 2 + 1,
        }
        for i in range(n)
    ]
    return X_v25, y, np.array(dates), fight_records


def run_composition(args) -> dict:  # noqa: C901, PLR0912, PLR0915
    """Pipeline driver: AUDIT-01 preflight → load data → baseline+candidate →
    floor/hurdle → emit TRAVEL_COMPOSITION_V25_REPORT.json.

    Returns the full report dict. Per Plan 42-02 spec: does NOT persist the
    candidate blender (Plan 42-03 Path A conditional).
    """
    from ufc_prediction.ml.config import FEATURE_COLUMNS_V22
    from ufc_prediction.ml.gate_contract import load_gate_contract
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

    NAN_DROP_POLICY = "per_feature_strict_baseline"
    BASELINE_COLS = ("xgb_oof_prob", "elo_prob")

    print(f"[compose_v25] args: {vars(args)}")

    # ── AUDIT-01 preflight ──
    xgb_sha, meta_sha = _assert_canonical_shas()
    print(
        f"[compose_v25] AUDIT-01 preflight OK "
        f"(xgb_v2={xgb_sha[:12]}..., meta_v2={meta_sha[:12]}...)"
    )

    # ── Load gate contract (D-18 hash check internal) ──
    contract = load_gate_contract(version="v2.3")
    assert contract.formula_hash == FORMULA_HASH, (
        f"D-18 formula hash drift: contract={contract.formula_hash} "
        f"expected={FORMULA_HASH}"
    )
    print(
        f"[compose_v25] gate contract v2.3 loaded "
        f"(formula_hash={contract.formula_hash[:12]}...)"
    )

    # ── Step 0: Load 92-col v2.5-travel feature matrix ──
    cached_fixtures_mode = bool(args.cached_fixtures_only)
    if args.dry_run:
        print("[compose_v25] DRY-RUN: synthetic 92-col v2.5-travel fixture")
        X_v25, y, fight_dates, fight_records = _build_synthetic_v25()
    else:
        print(
            "[compose_v25] Loading 92-col v2.5-travel matrix from DB..."
        )
        X_v25, y, fight_dates, fight_records = _load_assembled_data_v25_travel()
    print(
        f"[compose_v25] X_v25.shape={X_v25.shape}, "
        f"n_records={len(fight_records)}"
    )

    # The first 90 cols are the V22 substrate (xgb_v2 input shape preserved).
    X_v22 = X_v25[:, :90]
    # The last 2 cols are the NEW TRAVEL-V25 cols.
    travel_v25_cols = X_v25[:, 90:92]
    assert travel_v25_cols.shape[1] == 2, (
        f"TRAVEL-V25 col count drift: got {travel_v25_cols.shape[1]} expected 2"
    )

    # ── Three-way split (Phase 32-proven temporal split) ──
    base_train_fights, meta_train_fights, meta_eval_fights = make_three_way_split(
        fight_records,
        base_cutoff=date.fromisoformat(EXPECTED_CUTOFF_DATE),
        meta_eval_window_days=365,
        today=date.today(),
    )
    print(
        f"[compose_v25] split: base={len(base_train_fights)} "
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

    if len(meta_eval_idx) == 0:
        # Degenerate slice — emit Path B report immediately.
        print(
            "[compose_v25] WARNING: 0 meta_eval fights — synthetic fixture "
            "may not span the post-cutoff window. Emitting Path B."
        )

    # ── OOF base XGB on 72-col view ──
    X_oof = X_v22[meta_train_idx][:, :72]
    assert X_oof.shape[1] == 72, (
        f"OOF base XGB requires 72-col view; got {X_oof.shape[1]}"
    )
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

    # ── Elo probabilities ──
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

    # ── Build base META-V22 Level-1 (13 cols) for both splits ──
    print("[compose_v25] Building base META-V22 Level-1 (1 base train + Elo lookups)...")
    base_estimator = _make_oof_estimator(seed=42)
    base_estimator.fit(X_v22[meta_train_idx][:, :72], y[meta_train_idx])
    xgb_eval_prob = base_estimator.predict_proba(X_v22[meta_eval_idx][:, :72])[:, 1]

    base_train_meta_v22 = build_meta_features_v22(
        xgb_oof_aligned, elo_prob_train, X_v22[meta_train_idx],
    )
    base_eval_meta_v22 = build_meta_features_v22(
        xgb_eval_prob, elo_prob_eval, X_v22[meta_eval_idx],
    )

    seeds = list(args.seeds)

    # ── Apples-to-apples NaN-drop on BOTH baseline + candidate eval sets ──
    # Baseline (META-V22 + CALIB) gets the META-V22 NaN-drop discipline.
    # Candidate (META-V22 + CALIB + TRAVEL-V25) gets a CUMULATIVE NaN-drop
    # on the 15-col matrix (13 META + 2 TRAVEL-V25). To preserve apples-to-
    # apples we report effective n on BOTH paths per slice.
    meta_feature_cols = list(META_V22_FEATURE_COLUMNS)
    travel_v25_train = travel_v25_cols[meta_train_idx]
    travel_v25_eval = travel_v25_cols[meta_eval_idx]

    # Baseline NaN-drop (13-col META-V22 substrate).
    base_train_mask = apply_nan_drop_policy(
        base_train_meta_v22, meta_feature_cols,
        policy=NAN_DROP_POLICY, baseline_columns=BASELINE_COLS,
    )
    base_eval_mask = apply_nan_drop_policy(
        base_eval_meta_v22, meta_feature_cols,
        policy=NAN_DROP_POLICY, baseline_columns=BASELINE_COLS,
    )

    # Candidate matrix (13 META + 2 TRAVEL-V25).
    cand_feature_cols = meta_feature_cols + [
        "travel_distance_km", "tz_shift_hours",
    ]
    cand_train_matrix = np.column_stack([base_train_meta_v22, travel_v25_train])
    cand_eval_matrix = np.column_stack([base_eval_meta_v22, travel_v25_eval])
    cand_train_mask = apply_nan_drop_policy(
        cand_train_matrix, cand_feature_cols,
        policy=NAN_DROP_POLICY, baseline_columns=BASELINE_COLS,
    )
    cand_eval_mask = apply_nan_drop_policy(
        cand_eval_matrix, cand_feature_cols,
        policy=NAN_DROP_POLICY, baseline_columns=BASELINE_COLS,
    )

    # ── Median-impute non-baseline NaN cols on both matrices ──
    # (Phase 29 EVAL-V23-01 pattern — preserves apples-to-apples.)
    def _impute_train_eval(X_train: np.ndarray, X_eval: np.ndarray, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
        Xt = X_train.copy()
        Xe = X_eval.copy()
        non_baseline_idx = [
            i for i, c in enumerate(cols) if c not in BASELINE_COLS
        ]
        for idx in non_baseline_idx:
            col_t = Xt[:, idx] if Xt.size else np.array([])
            finite = col_t[~np.isnan(col_t)] if col_t.size else np.array([])
            median_val = float(np.median(finite)) if finite.size else 0.0
            if Xt.size:
                m = np.isnan(Xt[:, idx])
                if m.any():
                    Xt[m, idx] = median_val
            if Xe.size:
                m = np.isnan(Xe[:, idx])
                if m.any():
                    Xe[m, idx] = median_val
        return Xt, Xe

    X_base_train_clean = base_train_meta_v22[base_train_mask]
    y_base_train_clean = y[meta_train_idx][base_train_mask]
    X_base_eval_clean = base_eval_meta_v22[base_eval_mask]
    y_base_eval_clean = y[meta_eval_idx][base_eval_mask]
    fight_dates_base_eval = fight_dates[meta_eval_idx][base_eval_mask]
    X_base_train_clean, X_base_eval_clean = _impute_train_eval(
        X_base_train_clean, X_base_eval_clean, meta_feature_cols,
    )

    X_cand_train_clean = cand_train_matrix[cand_train_mask]
    y_cand_train_clean = y[meta_train_idx][cand_train_mask]
    X_cand_eval_clean = cand_eval_matrix[cand_eval_mask]
    y_cand_eval_clean = y[meta_eval_idx][cand_eval_mask]
    fight_dates_cand_eval = fight_dates[meta_eval_idx][cand_eval_mask]
    X_cand_train_clean, X_cand_eval_clean = _impute_train_eval(
        X_cand_train_clean, X_cand_eval_clean, cand_feature_cols,
    )

    n_base_eval = int(base_eval_mask.sum())
    n_cand_eval = int(cand_eval_mask.sum())
    print(
        f"[compose_v25] NaN-drop: baseline_eval_n={n_base_eval}/"
        f"{len(base_eval_mask)} candidate_eval_n={n_cand_eval}/"
        f"{len(cand_eval_mask)}"
    )

    # ── Evaluate baseline (META-V22 + CALIB via meta_v2.joblib) ──
    # NB: meta_v2.joblib uses the full 13-col META-V22 Level-1 substrate
    # (PolynomialFeatures degree=2 interaction_only=True expects 13 cols).
    # Pass X_base_eval_clean as-is — no slicing.
    print("[compose_v25] Evaluating baseline (META-V22 + CALIB via meta_v2.joblib)...")
    baseline_per_slice: dict[str, dict[str, float]] = {}
    if n_base_eval > 0:
        baseline_per_slice = _evaluate_baseline_meta_v22_calib(
            base_meta_v22=X_base_eval_clean,
            y_eval=y_base_eval_clean,
            fight_dates_eval=fight_dates_base_eval,
        )
    else:
        baseline_per_slice = {
            slc: {"brier_score": float("nan"), "accuracy": float("nan")}
            for slc in PER_SLICE_KEYS
        }

    # ── Train + evaluate candidate (META-V22 + CALIB + TRAVEL-V25) ──
    print(
        f"[compose_v25] Training candidate blender (15-col Level-1, "
        f"seeds={seeds})..."
    )
    candidate_per_slice: dict[str, dict[str, float]] = {}
    chosen_model: Any = None
    if X_cand_train_clean.shape[0] > 0 and X_cand_eval_clean.shape[0] > 0:
        chosen_model, candidate_per_slice = _train_candidate_multi_seed(
            X_train=X_cand_train_clean,
            y_train=y_cand_train_clean,
            X_eval=X_cand_eval_clean,
            y_eval=y_cand_eval_clean,
            fight_dates_eval=fight_dates_cand_eval,
            seeds=seeds,
        )
    else:
        candidate_per_slice = {
            slc: {"brier_score": float("nan"), "accuracy": float("nan")}
            for slc in PER_SLICE_KEYS
        }

    # ── Annotate per-slice with effective n + nan_drop ratio ──
    for slc in PER_SLICE_KEYS:
        # baseline / candidate per_slice are nested {slice: {brier_score:..,
        # accuracy:..}}; we sneak in n + n_after_nan_drop fields for
        # Pitfall 9 transparency. evaluate_per_slice already computed
        # effective per-slice n; we record the GROSS n (before NaN-drop) as
        # the eval-mask sum, and effective n is what evaluate_per_slice saw.
        baseline_per_slice[slc].setdefault("n_baseline_n", n_base_eval)
        candidate_per_slice[slc].setdefault("n_candidate_n", n_cand_eval)

    # Pitfall 9 substrate warning.
    substrate_warning: str | None = None
    if n_base_eval > 0 and n_cand_eval < int(SMALL_EFFECTIVE_N_FRACTION * n_base_eval):
        substrate_warning = "small_effective_n"
        print(
            f"[compose_v25] PITFALL 9 WARNING: candidate effective n "
            f"({n_cand_eval}) < {SMALL_EFFECTIVE_N_FRACTION:.0%} of baseline "
            f"({n_base_eval}). substrate_warning=small_effective_n"
        )

    # ── Compose gate_contract per-slice transparency record ──
    gate_contract_per_slice = {
        slc: {
            "brier_max": float(contract.per_slice[slc].brier_max),
            "accuracy_min": float(contract.per_slice[slc].accuracy_min),
        }
        for slc in PER_SLICE_KEYS
    }

    extra = {
        "n_seeds": len(seeds),
        "seeds": [int(s) for s in seeds],
        "cached_fixtures_mode": bool(cached_fixtures_mode),
        "dry_run": bool(args.dry_run),
        "n_base_eval_after_nan_drop": n_base_eval,
        "n_candidate_eval_after_nan_drop": n_cand_eval,
        "n_base_train_after_nan_drop": int(base_train_mask.sum()),
        "n_candidate_train_after_nan_drop": int(cand_train_mask.sum()),
        "gate_contract_per_slice": gate_contract_per_slice,
        "feature_columns_baseline": list(META_V22_FEATURE_COLUMNS),
        "feature_columns_candidate": cand_feature_cols,
        "nan_drop_policy": NAN_DROP_POLICY,
        "candidate_substrate_warning": substrate_warning,
        "split": {
            "base_train_n": len(base_train_fights),
            "meta_train_n": len(meta_train_fights),
            "meta_eval_n": len(meta_eval_fights),
            "base_cutoff": EXPECTED_CUTOFF_DATE,
            "meta_eval_window_days": 365,
        },
    }

    report = build_report(
        baseline=baseline_per_slice,
        candidate=candidate_per_slice,
        gate_contract_ref=str(GATE_CONTRACT_V23_PATH),
        xgb_sha=xgb_sha,
        meta_sha=meta_sha,
        extra=extra,
    )

    # ── Re-verify AUDIT-01 byte-identity at write time ──
    xgb_sha_end, meta_sha_end = _assert_canonical_shas()
    assert xgb_sha_end == xgb_sha, (
        "xgb_v2 SHA drifted mid-pipeline (AUDIT-01 violation)"
    )
    assert meta_sha_end == meta_sha, (
        "meta_v2 SHA drifted mid-pipeline (AUDIT-01 violation)"
    )

    _emit_report(report)

    # ── Plan 42-03 conditional persistence (Path A only) ──
    # Per Plan 42-03 spec + CONTEXT §Recomposition:
    # - meta_v22_travel.joblib is a SIBLING of meta_v2.joblib; NOT canonical.
    # - In-process persistence: the same model object that produced the
    #   in-report candidate per-slice metrics is the one written to disk.
    #   No re-train + tolerance check required because there is no
    #   intermediate process boundary that could introduce drift.
    if getattr(args, "persist_on_path_a", False):
        if not report["path_a_eligible"]:
            print(
                "[compose_v25] --persist-on-path-a: report.path_a_eligible=False "
                "→ SKIP persistence (Path B documented gap, no candidate sibling)."
            )
        elif chosen_model is None:
            print(
                "[compose_v25] --persist-on-path-a: chosen_model is None "
                "(degenerate train/eval split) → SKIP persistence."
            )
        else:
            _persist_path_a_sibling(
                chosen_model=chosen_model,
                candidate_per_slice=candidate_per_slice,
                baseline_per_slice=baseline_per_slice,
                feature_columns=cand_feature_cols,
                xgb_sha=xgb_sha,
                meta_sha=meta_sha,
                gate_contract_ref=str(GATE_CONTRACT_V23_PATH),
            )
            # Defensive re-verification: canonical SHAs unchanged after
            # SIBLING write — protects against accidental overwrite of
            # canonical artifacts.
            xgb_sha_post, meta_sha_post = _assert_canonical_shas()
            assert xgb_sha_post == xgb_sha, (
                "xgb_v2 SHA drifted after sibling persistence (AUDIT-01)"
            )
            assert meta_sha_post == meta_sha, (
                "meta_v2 SHA drifted after sibling persistence (AUDIT-01)"
            )

    return report


def _persist_path_a_sibling(
    *,
    chosen_model: Any,
    candidate_per_slice: dict[str, dict[str, float]],
    baseline_per_slice: dict[str, dict[str, float]],
    feature_columns: list[str],
    xgb_sha: str,
    meta_sha: str,
    gate_contract_ref: str,
) -> None:
    """Plan 42-03 Task 3 Path A: write meta_v22_travel.joblib SIBLING + sidecar JSON.

    Writes to a DIFFERENT path than canonical meta_v2.joblib by construction —
    save_meta_model variants in meta_persistence.py are pattern-matched; this
    helper writes BY HAND to keep the bespoke sidecar schema documented in
    Plan 42-03 ``<interfaces>``.

    NB on cross-cut with meta_persistence.save_meta_model: that helper has a
    fixed JSON schema that does NOT include the Phase 42-specific fields
    (``meta_v2_base_sha256``, ``formula_hash``, ``substrate_version``,
    ``path_materialized``, etc.). Writing the sidecar by hand here keeps
    Plan 42-03's bespoke schema verbatim — Phase 45 sidecar will use the
    canonical save_meta_model path when it promotes meta_v3.
    """
    import joblib

    out_joblib = Path("models/meta/meta_v22_travel.joblib")
    out_sidecar = Path("models/meta/meta_v22_travel_meta.json")
    out_joblib.parent.mkdir(parents=True, exist_ok=True)

    # ── Defensive: refuse to overwrite canonical meta_v2.joblib ──
    canonical = Path("models/meta/meta_v2.joblib")
    assert out_joblib.resolve() != canonical.resolve(), (
        "Refusing to write SIBLING to canonical meta_v2.joblib path (AUDIT-01)"
    )

    joblib.dump(chosen_model, out_joblib)
    print(f"[compose_v25] SIBLING blender → {out_joblib}")

    # ── Per-slice metric dump for the sidecar ──
    slice_metrics = {
        slc: {
            "brier_score": (
                float(candidate_per_slice[slc]["brier_score"])
                if candidate_per_slice[slc].get("brier_score") is not None
                else None
            ),
            "accuracy": (
                float(candidate_per_slice[slc]["accuracy"])
                if candidate_per_slice[slc].get("accuracy") is not None
                else None
            ),
        }
        for slc in PER_SLICE_KEYS
    }
    baseline_slice_metrics = {
        slc: {
            "brier_score": (
                float(baseline_per_slice[slc]["brier_score"])
                if baseline_per_slice[slc].get("brier_score") is not None
                else None
            ),
            "accuracy": (
                float(baseline_per_slice[slc]["accuracy"])
                if baseline_per_slice[slc].get("accuracy") is not None
                else None
            ),
        }
        for slc in PER_SLICE_KEYS
    }

    sidecar = {
        "meta_version": "v22_travel",
        "meta_kind": "logistic",
        "base_model_version": "v2",
        "base_model_sha256": xgb_sha,
        "meta_v2_base_sha256": meta_sha,
        "phase": "42",
        "trained_by_script": "scripts/compose_v25_travel.py",
        "feature_columns": list(feature_columns),
        "slice_metrics": slice_metrics,
        "baseline_slice_metrics": baseline_slice_metrics,
        "gate_contract_ref": gate_contract_ref,
        "formula_hash": FORMULA_HASH,
        "substrate_version": SUBSTRATE_VERSION,
        "path_materialized": "path_a",
        "produced_at": datetime.now(tz=UTC).isoformat(),
        "sibling_of": "models/meta/meta_v2.joblib",
        "canonical_status": "candidate_sibling_NOT_canonical",
        "phase_45_input_stack": "[META-V22, CALIB, TRAVEL]",
        "operator_caveat": (
            "Baseline Brier in baseline_slice_metrics is from runtime XGB OOF "
            "regeneration on v2.5 substrate (--no-cache-oof), not the "
            "training-time OOF that meta_v2.joblib was calibrated against. "
            "Delta vs slice_metrics is apples-to-apples on the SAME eval matrix; "
            "absolute magnitudes are NOT comparable to Phase 26 / Phase 32 "
            "published baselines. Phase 45 should reproduce on training-time "
            "OOF before any canonical-promotion decision."
        ),
    }

    out_sidecar.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(f"[compose_v25] SIBLING sidecar → {out_sidecar}")


def _emit_report(report: dict) -> None:
    """Write TRAVEL_COMPOSITION_V25_REPORT.json."""
    TRAVEL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAVEL_REPORT_PATH.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[compose_v25] TRAVEL_COMPOSITION_V25_REPORT → {TRAVEL_REPORT_PATH}")


# ────────────────────── CLI entry ────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 42 Plan 42-02 — TRAVEL recomposition spike "
            "(META-V22+CALIB+TRAVEL-V25 candidate blender)"
        ),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(SEEDS_DEFAULT),
        help="Random seeds for MetaLearnerLogistic (default: 42 43 44 45 46)",
    )
    parser.add_argument(
        "--cached-fixtures-only", action="store_true",
        help=(
            "Reuse Phase 26 OOF parquet cache; flag report "
            "cached_fixtures_mode=true so Plan 42-03 can request a "
            "live-DB rerun before path materialization."
        ),
    )
    parser.add_argument(
        "--no-cache-oof", action="store_true",
        help="Force OOF parquet rebuild (ignores cache_path)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Synthetic 92-col v2.5-travel smoke (no DB / no real xgb_v2 inference)",
    )
    parser.add_argument(
        "--gate-contract-ref", type=str, default=str(GATE_CONTRACT_V23_PATH),
        help="Override gate_contract path (recorded in report; loader uses load_gate_contract(version='v2.3'))",
    )
    parser.add_argument(
        "--persist-on-path-a", action="store_true",
        help=(
            "Plan 42-03 conditional persistence: if the run produces "
            "path_a_eligible=True, write the candidate blender as "
            "models/meta/meta_v22_travel.joblib SIBLING (NOT canonical) "
            "and the bespoke sidecar JSON at "
            "models/meta/meta_v22_travel_meta.json. NO promotion to "
            "canonical (Phase 45 meta_v3 is the canonical-change vehicle)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = run_composition(args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
