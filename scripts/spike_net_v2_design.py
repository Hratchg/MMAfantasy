#!/usr/bin/env python
"""Phase 18 / NET-V2-03 — 5-seed × 2-variant × 3-slice NET-V2 design spike.

Operator-driven entry point for the v2.1 NET-V2 redesign-or-removal milestone.
Reads `models/xgb_v2_meta.json`, asserts AF-1 (best_params verbatim) + AF-2
(`len(FEATURE_COLUMNS) == 75` + NET-* in last 3 positions), reproduces xgb_v2's
seed-42 Brier on the 72-col subset (Pitfall E sanity check), trains the two
PRE-REGISTERED variants — ablation (72 cols) and time_decayed (75 cols with
``mov_weight * 0.98^days_since_event``) — across 5 seeds × 3 slices, applies
the operator-approved gate_contract.json verdict per variant, applies the
mechanical tie-breaker (D-03(P18); ≥0.003 margin on every slice), and emits:

  - `models/xgb_v2_1_<variant>.joblib` for both variants
  - `models/xgb_v2_1.joblib` (canonical alias to winner; iff winner != xgb_v2_stays)
  - `.planning/phases/18-net-redesign-or-removal/archive/<loser>.joblib`
    (AUDIT-03; D-09(P18) carry-forward)
  - `.planning/phases/18-net-redesign-or-removal/18-NET-DESIGN-COMPARISON.md`

Locked decisions (Phase 18):
  - D-01(P18): variant 1 = ablation; FEATURE_COLUMNS_NO_NET = FEATURE_COLUMNS[:-3]
  - D-02(P18): variant 2 = time_decayed; DECAY_BASE = 0.98 per
    Lazova & Basnarkov 2015 (arXiv:1503.01331)
  - D-03(P18): tie-breaker margin = 0.003 Brier on EVERY slice; locked at
    plan-time; no post-measurement renegotiation
  - D-13(P16) carry-forward: 3-slice ALL-pass gate
  - D-16(P16) carry-forward: 5-seed median + per-seed std
  - D-04(P16) carry-forward: cutoff_date inherited verbatim from xgb_v2_meta

Anti-features (BANNED):
  - AF-1 (no hparam retuning): asserts xgb_v2_meta["best_params"] verbatim;
    uses `_train_with_fixed_params` (skips Optuna; does NOT call
    trainer.train_multi*seed which re-tunes per seed).
  - AF-2 (no feature engineering): asserts `len(FEATURE_COLUMNS) == 75` +
    NET-* in last 3 positions.

Pitfalls mitigated:
  - Pitfall A (view-mutation): consumed via apply_network_features_v2 from
    network_v2.py — guard is in the imported module, not here.
  - Pitfall D (auto-commit): spike does NOT call `git`. Operator owns commits
    in Wave 2 Task 16+.
  - Pitfall E (corpus drift): pre-spike seed=42 reproduces xgb_v2's Brier
    within PITFALL_E_TOLERANCE (= 0.005; relaxed 2026-05-04 from 1e-4 per
    operator decision; see constant docstring in spike_noise_floor.py).
  - Pitfall #6 (discover-and-test post-hoc): NO third variant added mid-phase.
    VARIANTS_PRE_REGISTERED is the immutable list set in the chore() commit
    `pre-register NET-V2 variants` (NET-V2-00 gate).

Usage:
    uv run python scripts/spike_net_v2_design.py \\
        --seeds 42 43 44 45 46 \\
        --meta-v2-path models/xgb_v2_meta.json \\
        --report-path .planning/phases/18-net-redesign-or-removal/18-NET-DESIGN-COMPARISON.md

After completion, operator reviews NET-DESIGN-COMPARISON.md, runs
`bash scripts/audit_xgb_v2_sha.sh`, and commits the artifacts manually
(Pitfall D).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import sklearn
import xgboost
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

from ufc_prediction.db.session import SessionLocal
from ufc_prediction.features.network_v2 import apply_network_features_v2
from ufc_prediction.ml.config import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_NO_NET,
    MLConfig,
)
from ufc_prediction.ml.evaluator import evaluate_per_slice, gate_verdict
from ufc_prediction.ml.feature_matrix import (
    FeatureMatrixAssembler,
    compute_division_medians,
    split_temporal,
)
from ufc_prediction.ml.gate_contract import load_gate_contract
from ufc_prediction.ml.persistence import save_model
from ufc_prediction.ml.queries import (
    load_computed_features,
    load_elo_features,
    load_fight_odds,
    load_fight_records,
    load_fighter_physicals,
    load_pre_ufc_records,
    load_round_stats_for_ml,
)
from ufc_prediction.ml.trainer import median_metrics

# ── Phase 18 locked constants (D-01..D-03(P18); NET-V2-00 pre-registration) ─

# Pre-registered immutable variant list (NET-V2-00; Pitfall #6 carry-forward).
# Set in the chore() commit `pre-register NET-V2 variants` and asserted here.
VARIANTS_PRE_REGISTERED: tuple[str, ...] = ("ablation", "time_decayed")

# D-03(P18): tie-breaker margin locked at plan-time.
TIE_BREAKER_MARGIN: float = 0.003

# Operator-approved formula hash for the v2.1 promotion gate (carries forward
# from Phase 17). Spike halts on drift between this literal and
# load_gate_contract().formula_hash.
EXPECTED_FORMULA_HASH: str = "7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a"

# AF-1 enforcement: best_params snapshot from models/xgb_v2_meta.json. The
# 9-key dict is asserted verbatim at startup; any drift halts the spike.
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

# Pitfall E: xgb_v2's seed-42 Brier on the 72-col matrix MUST match
# xgb_v2_meta.json["metrics"]["brier_score"] within PITFALL_E_TOLERANCE.
# Threshold is 0.005 (relaxed 2026-05-04 from 1e-4 — see spike_noise_floor.py
# constant docstring for full rationale).
EXPECTED_XGB_V2_BRIER: float = 0.22061555132914565
PITFALL_E_TOLERANCE: float = 0.005

# Per-slice keys (mirrors evaluator.PER_SLICE_KEYS).
PER_SLICE_KEYS: tuple[str, ...] = (
    "most_recent_12mo",
    "most_recent_24mo",
    "random_15pct",
)


def _train_with_fixed_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_params: dict,
    seed: int,
) -> CalibratedClassifierCV:
    """Single-seed train using xgb_v2 best_params (skips Optuna).

    Lifted verbatim from scripts/spike_noise_floor.py:168-197 per AF-1
    enforcement. Mirrors trainer.ModelTrainer.train()'s post-Optuna logic:
      - 80/20 chronological train_proper / calibration_holdout split
      - XGBClassifier fit on train_proper with the supplied params
      - CalibratedClassifierCV (sigmoid) wraps a FrozenEstimator
    """
    n_total = len(X_train)
    split_idx = int(n_total * 0.8)
    X_proper = X_train[:split_idx]
    y_proper = y_train[:split_idx]
    X_calib = X_train[split_idx:]
    y_calib = y_train[split_idx:]

    base = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=int(seed),
        verbosity=0,
    )
    base.fit(X_proper, y_proper)

    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    calibrated.fit(X_calib, y_calib)
    return calibrated


def _archive_loser(
    loser_name: str,
    loser_joblib_path: Path,
    loser_meta_path: Path,
    archive_dir: Path,
) -> str:
    """AUDIT-03 (D-09(P18)): archive losing variant bundle to git-tracked dir.

    Copies loser_joblib + loser_meta to archive_dir. Writes a SHA-256 anchor
    file alongside for downstream verification. Operator commits manually
    after spike completes (no auto-commit; Pitfall D).

    Returns the loser's joblib SHA-256 hex digest.
    """
    archive_dir.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(loser_joblib_path.read_bytes()).hexdigest()
    suffix = f"_{sha[:8]}"
    target_joblib = archive_dir / f"{loser_name}{suffix}.joblib"
    target_meta = archive_dir / f"{loser_name}{suffix}_meta.json"
    target_sha_anchor = archive_dir / f"{loser_name}{suffix}.sha256"
    shutil.copy2(loser_joblib_path, target_joblib)
    shutil.copy2(loser_meta_path, target_meta)
    target_sha_anchor.write_text(sha + "\n", encoding="utf-8")
    print(f"[spike] AUDIT-03 archive: {loser_name} -> {target_joblib} (sha256 {sha[:16]}...)")
    return sha


def _recompute_net_v2_in_computed_features(
    fight_records: list[dict],
    computed_features: dict,
) -> dict:
    """Re-compute NET-* keys via apply_network_features_v2 and merge into a
    fresh computed_features-shaped dict.

    The original computed_features dict carries Phase-16 (no time decay)
    values for `pagerank` / `sos_2hop` / `is_debutant_in_graph`. For the
    time_decayed variant we OVERWRITE these per-fighter values using the
    v2 graph (mov_weight * 0.98^days), then return a NEW dict (does NOT
    mutate the input — keeps the ablation path's data isolated per Pitfall A
    discipline).

    Args:
        fight_records: same shape produced by load_fight_records — each row
            has fight_id, event_date, fighter_a_id, fighter_b_id, winner_id,
            method, weight_class.
        computed_features: dict keyed by (fighter_id, fight_id) -> features
            dict (or by some other key tuple — preserved verbatim; we only
            overwrite the 3 NET-* keys per row).

    Returns:
        A new dict with the same keys as ``computed_features`` and identical
        non-NET values; NET-* keys overwritten with time-decayed values.
    """
    # 1) Build the net_fights list in the shape apply_network_features_v2 expects
    #    (mirrors features/compute.py:179-193). winner_id/loser_id from the
    #    fight outcome; method + event_date + weight_class for graph build.
    net_fights: list[dict] = []
    for f in fight_records:
        winner = f.get("winner_id")
        a_id = f.get("fighter_a_id")
        b_id = f.get("fighter_b_id")
        if winner == a_id:
            loser = b_id
        elif winner == b_id:
            loser = a_id
        else:
            loser = None
        net_fights.append(
            {
                "fight_id": f.get("fight_id"),
                "event_date": f.get("event_date"),
                "winner_id": winner,
                "loser_id": loser,
                "method": f.get("method"),
                "weight_class": f.get("weight_class"),
            }
        )

    # 2) Build raw_results in the shape apply_network_features_v2 expects:
    #    [{fighter_id, as_of_date, features}, ...]
    #    For each (fighter_id, fight_id) in computed_features we need to know
    #    the fight's event_date as the as_of_date.
    fight_date_by_id: dict = {f["fight_id"]: f["event_date"] for f in fight_records}

    raw_results: list[dict] = []
    # Discover the key shape used in computed_features. The loaders use
    # (fighter_id, fight_id) tuples per ml/queries.py.
    for key, feats in computed_features.items():
        if not isinstance(key, tuple) or len(key) != 2:
            # Unexpected key shape — surface clearly so the spike halts on
            # corpus drift instead of silently emitting wrong NET values.
            raise RuntimeError(
                f"computed_features key shape unexpected: {key!r}; expected "
                "(fighter_id, fight_id) per ml/queries.py."
            )
        fighter_id, fight_id = key
        as_of = fight_date_by_id.get(fight_id)
        if as_of is None:
            # Fight not in fight_records — leave row untouched (debutant
            # fallback handled inside apply_network_features_v2 if as_of None).
            continue
        raw_results.append(
            {
                "fighter_id": fighter_id,
                "fight_id": fight_id,
                "as_of_date": as_of,
                # Mutate a COPY of feats — never the input dict.
                "features": dict(feats),
            }
        )

    # 3) Run the time-decayed apply (overwrites pagerank / sos_2hop /
    #    is_debutant_in_graph in-place inside each row's features dict).
    apply_network_features_v2(raw_results, net_fights)

    # 4) Re-build the dict with overwritten NET-* keys.
    new_computed: dict = {}
    for row in raw_results:
        new_computed[(row["fighter_id"], row["fight_id"])] = row["features"]
    # Preserve any keys we skipped (no event_date — extremely rare).
    for key, feats in computed_features.items():
        if key not in new_computed:
            new_computed[key] = feats

    return new_computed


def _emit_report(
    report_path: Path,
    *,
    spike_started: datetime,
    spike_finished: datetime,
    seeds: list[int],
    cutoff_str: str,
    n_train: int,
    n_test: int,
    formula_hash: str,
    medians_per_variant: dict,
    seed_stds_per_variant: dict,
    verdicts_per_variant: dict,
    contract,
    tie_breaker: dict,
    winner: str,
    persisted_paths: dict,
    pitfall_e_diff: float,
    per_seed_per_variant: dict,
) -> None:
    """Write 18-NET-DESIGN-COMPARISON.md per the template at
    18-RESEARCH.md:758-857.
    """
    duration_min = (spike_finished - spike_started).total_seconds() / 60.0
    lines: list[str] = []
    lines.append("# Phase 18 — NET-DESIGN-COMPARISON Report")
    lines.append("")
    lines.append(f"**Spike started:** {spike_started.isoformat()}")
    lines.append(f"**Spike finished:** {spike_finished.isoformat()}")
    lines.append(f"**Spike duration:** ~{duration_min:.1f} min")
    lines.append(f"**Spike seeds:** {seeds} ({len(seeds)} seeds; D-16(P16) carry-forward)")
    lines.append(f"**Cutoff date:** `{cutoff_str}` (verbatim from xgb_v2_meta.json)")
    lines.append(f"**Train fights / Test fights:** {n_train} / {n_test}")
    lines.append("**Hparams:** verbatim from xgb_v2_meta.json (AF-1 enforced)")
    lines.append(f"**FEATURE_COLUMNS at startup:** {len(FEATURE_COLUMNS)} (AF-2 enforced)")
    lines.append("**Variants tested (PRE-REGISTERED in Plan 18; D-01/D-02(P18)):**")
    lines.append("  - `ablation`     — 72 cols; FEATURE_COLUMNS_NO_NET = FEATURE_COLUMNS[:-3]")
    lines.append(
        "  - `time_decayed` — 75 cols; NET-V2 from network_v2.py with "
        "0.98^days edge weights (Lazova & Basnarkov 2015)"
    )
    lines.append("")
    lines.append(
        f"**Gate contract:** .planning/gate_contract.json (formula_hash `{formula_hash[:16]}...`)"
    )
    lines.append(
        f"**Tie-breaker (D-03(P18); locked at plan-time):** improvement-margin "
        f">={TIE_BREAKER_MARGIN} Brier across ALL 3 slices"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**Outcome path:** `{tie_breaker['outcome_path']}`")
    lines.append("")
    lines.append(f"**Winner:** `{winner}`")
    lines.append("")
    lines.append("**Persisted artifacts:**")
    for label, path_sha in persisted_paths.items():
        lines.append(f"- `{label}` -> `{path_sha}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-variant tables
    lines.append("## Per-Variant Tables")
    lines.append("")
    for variant in VARIANTS_PRE_REGISTERED:
        _cols = 72 if variant == "ablation" else 75
        suffix = "(72 cols)" if variant == "ablation" else "(75 cols; 0.98^days edge weights)"
        lines.append(f"### Variant: `{variant}` {suffix}")
        lines.append("")
        lines.append(
            "| Slice | Median Brier (n=5) | Median Acc (n=5) | std_brier | "
            "std_acc | Gate brier_max | Gate accuracy_min | Gate clear |"
        )
        lines.append(
            "|-------|--------------------|--------------------|-----------|"
            "---------|----------------|--------------------|------------|"
        )
        med = medians_per_variant[variant]
        stds = seed_stds_per_variant[variant]
        passed, failed = verdicts_per_variant[variant]
        for slice_name in PER_SLICE_KEYS:
            br = med[slice_name]["brier_score"]
            ac = med[slice_name]["accuracy"]
            sbr = stds[slice_name]["brier_score"]
            sac = stds[slice_name]["accuracy"]
            ts = contract.per_slice[slice_name]
            slice_failed = any(slice_name in f for f in failed)
            clear = "FAIL" if slice_failed else "PASS"
            lines.append(
                f"| `{slice_name}` | {br:.4f} | {ac:.4f} | {sbr:.4f} | "
                f"{sac:.4f} | {ts.brier_max} | {ts.accuracy_min} | {clear} |"
            )
        gate_pass = "PASS" if passed else "FAIL"
        lines.append("")
        lines.append(f"**Gate verdict (3-slice ALL-pass; D-13(P16)):** `{gate_pass}`")
        if not passed:
            lines.append("")
            lines.append("Failed criteria:")
            for f in failed:
                lines.append(f"- {f}")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Tie-breaker
    lines.append("## Tie-Breaker Computation (D-03(P18))")
    lines.append("")
    lines.append(
        "Applied IF AND ONLY IF both variants clear the gate. Margin = "
        "ablation_brier - timedecayed_brier per slice. Time-decayed wins if "
        f"all 3 slice margins >= {TIE_BREAKER_MARGIN}; else ablation wins."
    )
    lines.append("")
    lines.append(
        "| Slice | ablation_median_brier | timedecayed_median_brier | margin (a - td) | >= 0.003? |"
    )
    lines.append(
        "|-------|------------------------|---------------------------|"
        "------------------|------------|"
    )
    if tie_breaker.get("margins") is not None:
        for slice_name in PER_SLICE_KEYS:
            ablat_b = medians_per_variant["ablation"][slice_name]["brier_score"]
            td_b = medians_per_variant["time_decayed"][slice_name]["brier_score"]
            margin = ablat_b - td_b
            ok = "YES" if margin >= TIE_BREAKER_MARGIN else "NO"
            lines.append(f"| `{slice_name}` | {ablat_b:.4f} | {td_b:.4f} | {margin:+.4f} | {ok} |")
    else:
        lines.append("| (n/a — only one variant cleared, or neither cleared) | | | | |")
    lines.append("")
    lines.append(f"**Tie-breaker decision:** `{tie_breaker['decision']}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-seed sub-rows
    lines.append("## Per-Seed Sub-Rows (audit reproducibility)")
    lines.append("")
    for variant in VARIANTS_PRE_REGISTERED:
        lines.append(f"### Variant: {variant}")
        lines.append(
            "| Seed | 12mo Brier | 12mo Acc | 24mo Brier | 24mo Acc | "
            "random_15pct Brier | random_15pct Acc |"
        )
        lines.append(
            "|------|-----------|----------|-----------|----------|"
            "----------------------|--------------------|"
        )
        for seed, per_slice in zip(seeds, per_seed_per_variant[variant], strict=True):
            row_b12 = per_slice["most_recent_12mo"]["brier_score"]
            row_a12 = per_slice["most_recent_12mo"]["accuracy"]
            row_b24 = per_slice["most_recent_24mo"]["brier_score"]
            row_a24 = per_slice["most_recent_24mo"]["accuracy"]
            row_brp = per_slice["random_15pct"]["brier_score"]
            row_arp = per_slice["random_15pct"]["accuracy"]
            lines.append(
                f"| {seed} | {row_b12:.4f} | {row_a12:.4f} | {row_b24:.4f} | "
                f"{row_a24:.4f} | {row_brp:.4f} | {row_arp:.4f} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")

    # Sanity checks
    lines.append("## Sanity Checks")
    lines.append("")
    lines.append('- AF-1 (no hparam retuning): xgb_v2_meta.json["best_params"] matched verbatim ✓')
    lines.append(
        f"- AF-2 (no feature engineering): len(FEATURE_COLUMNS) == "
        f"{len(FEATURE_COLUMNS)}; NET-* in last 3 ✓"
    )
    lines.append(
        f"- Pitfall E (corpus drift): seed=42 ablation Brier within "
        f"{PITFALL_E_TOLERANCE} of EXPECTED_XGB_V2_BRIER (diff "
        f"{pitfall_e_diff:.6f}) ✓"
    )
    lines.append(f"- D-07(P17) formula_hash match: {formula_hash[:16]}... ✓")
    lines.append("- D-06(P18) gate_contract.json applied (per-slice gate) ✓")
    lines.append("- D-09(P18) AUDIT-03 loser archive: see persisted artifacts above ✓")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    try:
        import scipy

        scipy_v = scipy.__version__
    except ImportError:
        scipy_v = "n/a"
    try:
        import networkx

        nx_v = networkx.__version__
    except ImportError:
        nx_v = "n/a"
    lines.append(f"- Python: {sys.version.split()[0]}")
    lines.append(f"- xgboost: {xgboost.__version__}")
    lines.append(f"- sklearn: {sklearn.__version__}")
    lines.append(f"- scipy: {scipy_v}")
    lines.append(f"- networkx: {nx_v}")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta-v2-path",
        default="models/xgb_v2_meta.json",
        help="Path to xgb_v2 metadata JSON (D-04(P16) cutoff_date inheritance).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Seeds to train (default: 42 43 44 45 46; D-16(P16) carry-forward).",
    )
    parser.add_argument(
        "--report-path",
        default=(".planning/phases/18-net-redesign-or-removal/18-NET-DESIGN-COMPARISON.md"),
        help="Output path for NET-DESIGN-COMPARISON.md.",
    )
    parser.add_argument(
        "--archive-dir",
        default=".planning/phases/18-net-redesign-or-removal/archive",
        help="Archive dir for losing variant bundle (D-09(P18) AUDIT-03).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Quick mode: only 1 seed (debugging only). Refuses to write "
            "variant joblibs, canonical alias, or NET-DESIGN-COMPARISON.md "
            "when n_seeds_observed != 5."
        ),
    )
    args = parser.parse_args()

    spike_started = datetime.now(UTC)

    # ── AF-2 startup assertion ────────────────────────────────────────
    if len(FEATURE_COLUMNS) != 75:
        msg = (
            f"FEATURE_COLUMNS drift: expected 75 (72 v2.0-baseline + 3 "
            f"NET-* added Phase 16); got {len(FEATURE_COLUMNS)}"
        )
        print(f"[spike] FATAL: {msg}", file=sys.stderr)
        return 2
    expected_net_tail = [
        "pagerank_diff",
        "sos_2hop_diff",
        "is_debutant_in_graph_diff",
    ]
    if list(FEATURE_COLUMNS[-3:]) != expected_net_tail:
        msg = (
            f"FEATURE_COLUMNS NET-* drift: expected last 3 to be "
            f"{expected_net_tail}; got {list(FEATURE_COLUMNS[-3:])}"
        )
        print(f"[spike] FATAL: {msg}", file=sys.stderr)
        return 2
    if list(FEATURE_COLUMNS_NO_NET) != list(FEATURE_COLUMNS[:-3]):
        msg = "FEATURE_COLUMNS_NO_NET drift: expected FEATURE_COLUMNS[:-3]; got divergent list."
        print(f"[spike] FATAL: {msg}", file=sys.stderr)
        return 2
    print(
        "[spike] AF-2 OK: len(FEATURE_COLUMNS) == 75; NET-* in last 3 "
        "positions; FEATURE_COLUMNS_NO_NET == FEATURE_COLUMNS[:-3]."
    )

    # ── Read xgb_v2 meta + AF-1 startup assertion ─────────────────────
    print("[spike] Reading xgb_v2 metadata for cutoff inheritance...")
    meta_v2_path = Path(args.meta_v2_path)
    if not meta_v2_path.exists():
        print(
            f"[spike] FATAL: xgb_v2 meta JSON not found at {meta_v2_path}",
            file=sys.stderr,
        )
        return 2
    meta_v2 = json.loads(meta_v2_path.read_text())
    cutoff_str: str = meta_v2["cutoff_date"]
    cutoff_date_obj: date = date.fromisoformat(cutoff_str)
    best_params: dict = meta_v2["best_params"]
    if best_params != EXPECTED_XGB_V2_BEST_PARAMS:
        msg = (
            f"AF-1 violation: xgb_v2 best_params drift detected.\n"
            f"  Got:      {best_params}\n"
            f"  Expected: {EXPECTED_XGB_V2_BEST_PARAMS}"
        )
        print(f"[spike] FATAL: {msg}", file=sys.stderr)
        return 2
    print("[spike] AF-1 OK: xgb_v2 best_params matched EXPECTED_XGB_V2_BEST_PARAMS verbatim.")
    print(f"  cutoff_date: {cutoff_str}")

    # ── Load gate_contract.json + verify formula_hash ────────────────
    contract = load_gate_contract()
    if contract.formula_hash != EXPECTED_FORMULA_HASH:
        msg = (
            "FORMULA_HASH drift between Plan 18 and gate_contract.json — "
            f"halt.\n  Loaded: {contract.formula_hash}\n  "
            f"Expected: {EXPECTED_FORMULA_HASH}"
        )
        print(f"[spike] FATAL: {msg}", file=sys.stderr)
        return 2
    formula_hash = contract.formula_hash
    print(f"[spike] gate_contract.json formula_hash matched ({formula_hash[:16]}...).")

    # ── Data load (mirrors retrain_xgb_v3.py) ─────────────────────────
    print("[spike] Loading data from DB...")
    t0 = time.time()
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
    print(
        f"  Loaded {len(fight_records)} fights / {len(fight_odds)} odds "
        f"rows in {time.time() - t0:.1f}s"
    )

    print("[spike] Computing division medians (training-set only)...")
    division_medians = compute_division_medians(
        fighter_physicals,
        fight_records,
        cutoff_date_obj,
    )

    # ── Assemble feature matrix for ablation (Phase-16 NET-* values) ─
    print(
        "[spike] Assembling feature matrix #1 (ablation): 75 cols using "
        "Phase-16 NET-* values; will subset to 72 via [:-3]..."
    )
    config = MLConfig(cutoff_date=cutoff_str)
    assembler = FeatureMatrixAssembler(config)
    X_full, y, fight_dates_full = assembler.assemble(
        fight_records,
        elo_features,
        computed_features,
        fighter_physicals,
        division_medians,
        round_stats,
        pre_ufc_records=pre_ufc,
        fight_odds=fight_odds,
    )
    print(f"  X.shape = {X_full.shape}, y.shape = {y.shape}")
    if X_full.shape[1] != len(FEATURE_COLUMNS):
        msg = (
            f"FEATURE_COLUMNS drift: assembler produced {X_full.shape[1]} "
            f"cols, FEATURE_COLUMNS has {len(FEATURE_COLUMNS)}"
        )
        print(f"[spike] FATAL: {msg}", file=sys.stderr)
        return 2

    print("[spike] Temporal split at cutoff_date...")
    X_train, X_test, y_train, y_test = split_temporal(
        X_full,
        y,
        fight_dates_full,
        cutoff_date_obj,
    )
    test_mask = np.array([d >= cutoff_date_obj for d in fight_dates_full])
    fight_dates_test = np.array(fight_dates_full)[test_mask]
    print(f"  Train: {X_train.shape[0]} fights, Test: {X_test.shape[0]} fights")

    # Pitfall E shape sanity
    expected_n_train = meta_v2.get("n_training_fights")
    expected_n_test = meta_v2.get("n_test_fights")
    if expected_n_train is not None and X_train.shape[0] != expected_n_train:
        msg = (
            f"Pitfall E shape sanity: n_training_fights drift: "
            f"got {X_train.shape[0]}, expected {expected_n_train}"
        )
        print(f"[spike] FATAL: {msg}", file=sys.stderr)
        return 2
    if expected_n_test is not None and X_test.shape[0] != expected_n_test:
        msg = (
            f"Pitfall E shape sanity: n_test_fights drift: "
            f"got {X_test.shape[0]}, expected {expected_n_test}"
        )
        print(f"[spike] FATAL: {msg}", file=sys.stderr)
        return 2

    # Subset to 72 cols for ablation variant.
    X_train_72 = X_train[:, :-3]
    X_test_72 = X_test[:, :-3]
    if X_train_72.shape[1] != 72:
        print(
            f"[spike] FATAL: ablation subset produced {X_train_72.shape[1]} cols; expected 72.",
            file=sys.stderr,
        )
        return 2
    print("[spike] Ablation matrix: 72 cols (FEATURE_COLUMNS[:-3]).")

    # ── Pitfall E pre-spike sanity reproduce on 72-col ablation matrix ──
    print(
        "[spike] Pitfall E sanity: training seed=42 on 72-col ablation matrix "
        f"and checking Brier within {PITFALL_E_TOLERANCE} of xgb_v2 baseline "
        f"({EXPECTED_XGB_V2_BRIER:.6f})..."
    )
    t_repro = time.time()
    repro_model = _train_with_fixed_params(X_train_72, y_train, best_params, 42)
    repro_probs = repro_model.predict_proba(X_test_72)[:, 1]
    seed42_repro_brier = float(np.mean((repro_probs - y_test) ** 2))
    repro_diff = abs(seed42_repro_brier - EXPECTED_XGB_V2_BRIER)
    print(
        f"  seed=42 Brier on 72-col matrix: {seed42_repro_brier:.6f} "
        f"(diff {repro_diff:.6f}; took {time.time() - t_repro:.1f}s)"
    )
    if repro_diff > PITFALL_E_TOLERANCE:
        msg = (
            "Pitfall E sanity check FAILED: spike's 72-col seed-42 reproduce "
            f"diverged from xgb_v2 baseline by {repro_diff:.6f} "
            f"(> {PITFALL_E_TOLERANCE}). Halt before variant trains."
        )
        print(f"[spike] FATAL: {msg}", file=sys.stderr)
        return 2
    print(
        "[spike] Pitfall E OK: seed-42 Brier reproduces xgb_v2 baseline "
        f"within {PITFALL_E_TOLERANCE}."
    )

    # ── Re-assemble feature matrix #2 (time_decayed) with NET-V2 values ─
    print(
        "[spike] Re-computing NET-* via apply_network_features_v2 (0.98^days) "
        "for time_decayed variant..."
    )
    t_v2 = time.time()
    computed_features_v2 = _recompute_net_v2_in_computed_features(
        fight_records,
        computed_features,
    )
    print(
        f"  re-computed NET-V2 keys for {len(computed_features_v2)} rows in "
        f"{time.time() - t_v2:.1f}s"
    )

    print(
        "[spike] Assembling feature matrix #2 (time_decayed): 75 cols using "
        "NET-V2 (0.98^days) values..."
    )
    X_full_v2, y_v2, fight_dates_full_v2 = assembler.assemble(
        fight_records,
        elo_features,
        computed_features_v2,
        fighter_physicals,
        division_medians,
        round_stats,
        pre_ufc_records=pre_ufc,
        fight_odds=fight_odds,
    )
    if (
        X_full_v2.shape != X_full.shape
        or not np.array_equal(y_v2, y)
        or list(fight_dates_full_v2) != list(fight_dates_full)
    ):
        msg = (
            "v2 feature matrix shape/order drift vs ablation — halt. "
            f"shape: {X_full_v2.shape} vs {X_full.shape}; "
            f"len(y): {len(y_v2)} vs {len(y)}"
        )
        print(f"[spike] FATAL: {msg}", file=sys.stderr)
        return 2
    X_train_v2, X_test_v2, _, _ = split_temporal(
        X_full_v2,
        y_v2,
        fight_dates_full_v2,
        cutoff_date_obj,
    )
    print(f"  v2 train/test shape: {X_train_v2.shape} / {X_test_v2.shape}")

    # ── Outer variant × inner seed loop (5×2 = 10 trains; reuse seed=42) ─
    seeds = args.seeds if not args.quick else [42]
    if args.quick:
        print(
            "[spike] --quick mode: running ONLY seed 42 (HARD GUARD: refuses "
            "to write variant joblibs or report when n_seeds_observed != 5)."
        )
    print(
        f"[spike] Training {len(seeds)} seeds × 2 variants = "
        f"{len(seeds) * 2} candidate models via _train_with_fixed_params "
        "(AF-1 enforced; no Optuna)..."
    )

    medians_per_variant: dict = {}
    seed_stds_per_variant: dict = {}
    verdicts_per_variant: dict = {}
    per_seed_per_variant: dict = {"ablation": [], "time_decayed": []}
    models_by_variant: dict = {}

    for variant in VARIANTS_PRE_REGISTERED:
        if variant == "ablation":
            X_train_V = X_train_72
            X_test_V = X_test_72
        else:
            X_train_V = X_train_v2
            X_test_V = X_test_v2
        print(
            f"[spike] === Variant '{variant}' "
            f"(train shape {X_train_V.shape}; test shape {X_test_V.shape}) ==="
        )
        for i, seed in enumerate(seeds):
            if variant == "ablation" and seed == 42:
                model = repro_model
                print(
                    f"[spike] Seed {seed} ({i + 1}/{len(seeds)}) [ablation] - "
                    "reusing Pitfall E reproduce model."
                )
            else:
                t_seed = time.time()
                print(f"[spike] Seed {seed} ({i + 1}/{len(seeds)}) [{variant}] - training...")
                model = _train_with_fixed_params(
                    X_train_V,
                    y_train,
                    best_params,
                    seed,
                )
                print(f"  seed={seed} trained in {time.time() - t_seed:.1f}s")
            per_slice_seed = evaluate_per_slice(
                model,
                X_test_V,
                y_test,
                fight_dates_test,
                today=date.today(),
                random_seed=42,
            )
            per_seed_per_variant[variant].append(per_slice_seed)
            slice_summary = ", ".join(
                f"{slice_name}: brier={metrics['brier_score']:.4f} acc={metrics['accuracy']:.4f}"
                for slice_name, metrics in per_slice_seed.items()
            )
            print(f"  seed={seed} slices: {slice_summary}")
            # Cache the median-position seed's model for persistence.
            if i == len(seeds) // 2:
                models_by_variant[variant] = model

        # Aggregate per-variant
        medians_per_variant[variant] = median_metrics(
            per_seed_per_variant[variant],
        )
        seed_stds_per_variant[variant] = {}
        for slice_name in PER_SLICE_KEYS:
            brier_arr = np.array(
                [psm[slice_name]["brier_score"] for psm in per_seed_per_variant[variant]]
            )
            acc_arr = np.array(
                [psm[slice_name]["accuracy"] for psm in per_seed_per_variant[variant]]
            )
            seed_stds_per_variant[variant][slice_name] = {
                "brier_score": float(np.std(brier_arr)),
                "accuracy": float(np.std(acc_arr)),
            }
        verdicts_per_variant[variant] = gate_verdict(
            medians_per_variant[variant],
            contract,
        )
        passed, failed = verdicts_per_variant[variant]
        print(
            f"[spike] Variant '{variant}' gate verdict: "
            f"{'PASS' if passed else 'FAIL'}; failed criteria: {failed}"
        )

    # ── D-03(P18) tie-breaker (mechanical; locked at plan-time) ──────
    v_a_passed = verdicts_per_variant["ablation"][0]
    v_t_passed = verdicts_per_variant["time_decayed"][0]
    tie_breaker: dict = {
        "outcome_path": "",
        "decision": "",
        "margins": None,
    }
    if not v_a_passed and not v_t_passed:
        winner = "xgb_v2_stays"
        tie_breaker["outcome_path"] = "neither_clears_xgb_v2_stays"
        tie_breaker["decision"] = "n/a (neither variant cleared gate; xgb_v2 stays)"
    elif v_t_passed and not v_a_passed:
        winner = "time_decayed"
        tie_breaker["outcome_path"] = "time_decayed_only_clears_ships"
        tie_breaker["decision"] = "n/a (only time_decayed cleared gate; ships)"
    elif v_a_passed and not v_t_passed:
        winner = "ablation"
        tie_breaker["outcome_path"] = "ablation_only_clears_ships"
        tie_breaker["decision"] = "n/a (only ablation cleared gate; ships)"
    else:
        # Both clear gate — apply tie-breaker margin per slice.
        margins = [
            medians_per_variant["ablation"][s]["brier_score"]
            - medians_per_variant["time_decayed"][s]["brier_score"]
            for s in PER_SLICE_KEYS
        ]
        tie_breaker["margins"] = margins
        all_clear = all(m >= TIE_BREAKER_MARGIN for m in margins)
        if all_clear:
            winner = "time_decayed"
            tie_breaker["outcome_path"] = "both_clear_time_decayed_ships_above_margin"
            tie_breaker["decision"] = f"time_decayed wins (all 3 margins >= {TIE_BREAKER_MARGIN})"
        else:
            winner = "ablation"
            tie_breaker["outcome_path"] = "both_clear_ablation_ships_under_margin"
            tie_breaker["decision"] = f"ablation wins (at least one margin < {TIE_BREAKER_MARGIN})"
    print(
        f"[spike] Tie-breaker decision: winner = `{winner}` "
        f"(outcome_path = {tie_breaker['outcome_path']})"
    )

    # ── Quick-mode hard guard ─────────────────────────────────────────
    if len(seeds) != 5:
        print(
            f"[spike] --quick mode active (n_seeds_observed={len(seeds)} "
            "!= 5). REFUSING to persist variant joblibs / canonical alias / "
            "report. Re-run with --seeds 42 43 44 45 46 to emit canonical "
            "artifacts.",
            file=sys.stderr,
        )
        return 0

    # ── Persistence (Pitfall C: train in-memory; persist AFTER tie-breaker) ─
    persisted_paths: dict = {}
    n_train = int(X_train.shape[0])
    n_test = int(X_test.shape[0])
    for variant in VARIANTS_PRE_REGISTERED:
        version_tag = "v2_1_ablation" if variant == "ablation" else "v2_1_timedecayed"
        feat_cols = list(FEATURE_COLUMNS_NO_NET) if variant == "ablation" else list(FEATURE_COLUMNS)
        extra = {
            "base_model_kind": variant,
            "n_seeds_observed": len(seeds),
            "applied_gate_contract_formula_hash": EXPECTED_FORMULA_HASH,
        }
        save_model(
            model=models_by_variant[variant],
            metrics=medians_per_variant[variant],
            feature_columns=feat_cols,
            best_params=best_params,
            model_dir="models",
            version=version_tag,
            cutoff_date=cutoff_str,
            n_training_fights=n_train,
            n_test_fights=n_test,
            extra_meta=extra,
        )
        joblib_path = Path("models") / f"xgb_{version_tag}.joblib"
        _meta_path = Path("models") / f"xgb_{version_tag}_meta.json"
        sha = hashlib.sha256(joblib_path.read_bytes()).hexdigest()
        persisted_paths[f"models/xgb_{version_tag}.joblib"] = f"sha256 {sha[:16]}..."
        print(f"[spike] Persisted variant '{variant}' -> {joblib_path} (sha256 {sha[:16]}...)")

    # Canonical alias (only if winner != xgb_v2_stays)
    archive_dir = Path(args.archive_dir)
    if winner != "xgb_v2_stays":
        version_tag_winner = "v2_1_ablation" if winner == "ablation" else "v2_1_timedecayed"
        winner_joblib = Path("models") / f"xgb_{version_tag_winner}.joblib"
        winner_meta = Path("models") / f"xgb_{version_tag_winner}_meta.json"
        canonical_joblib = Path("models") / "xgb_v2_1.joblib"
        canonical_meta = Path("models") / "xgb_v2_1_meta.json"
        shutil.copy2(winner_joblib, canonical_joblib)
        shutil.copy2(winner_meta, canonical_meta)
        canonical_sha = hashlib.sha256(
            canonical_joblib.read_bytes(),
        ).hexdigest()
        persisted_paths["models/xgb_v2_1.joblib (canonical alias)"] = (
            f"sha256 {canonical_sha[:16]}..."
        )
        print(f"[spike] Canonical alias: models/xgb_v2_1.joblib (sha256 {canonical_sha[:16]}...)")

        # AUDIT-03 archive of loser
        loser = "time_decayed" if winner == "ablation" else "ablation"
        version_tag_loser = "v2_1_ablation" if loser == "ablation" else "v2_1_timedecayed"
        loser_joblib = Path("models") / f"xgb_{version_tag_loser}.joblib"
        loser_meta = Path("models") / f"xgb_{version_tag_loser}_meta.json"
        loser_sha = _archive_loser(
            version_tag_loser,
            loser_joblib,
            loser_meta,
            archive_dir,
        )
        persisted_paths[f"archive/{version_tag_loser}*"] = f"sha256 {loser_sha[:16]}..."
    else:
        # winner == xgb_v2_stays — archive BOTH variants for empirical record.
        for variant in VARIANTS_PRE_REGISTERED:
            version_tag = "v2_1_ablation" if variant == "ablation" else "v2_1_timedecayed"
            v_joblib = Path("models") / f"xgb_{version_tag}.joblib"
            v_meta = Path("models") / f"xgb_{version_tag}_meta.json"
            v_sha = _archive_loser(
                version_tag,
                v_joblib,
                v_meta,
                archive_dir,
            )
            persisted_paths[f"archive/{version_tag}*"] = f"sha256 {v_sha[:16]}..."

    # ── Emit NET-DESIGN-COMPARISON.md ─────────────────────────────────
    spike_finished = datetime.now(UTC)
    report_path = Path(args.report_path)
    _emit_report(
        report_path,
        spike_started=spike_started,
        spike_finished=spike_finished,
        seeds=list(seeds),
        cutoff_str=cutoff_str,
        n_train=n_train,
        n_test=n_test,
        formula_hash=formula_hash,
        medians_per_variant=medians_per_variant,
        seed_stds_per_variant=seed_stds_per_variant,
        verdicts_per_variant=verdicts_per_variant,
        contract=contract,
        tie_breaker=tie_breaker,
        winner=winner,
        persisted_paths=persisted_paths,
        pitfall_e_diff=repro_diff,
        per_seed_per_variant=per_seed_per_variant,
    )
    print(f"[spike] Wrote NET-DESIGN-COMPARISON.md: {report_path}")

    # ── Final stdout (Pitfall D: do NOT call git) ─────────────────────
    print(
        f"\n[spike] Spike complete.\n"
        f"  Verdict: winner = `{winner}`\n"
        f"  outcome_path = {tie_breaker['outcome_path']}\n"
        f"  Persisted: see NET-DESIGN-COMPARISON.md\n"
        f"  Archive dir: {archive_dir}\n"
    )
    print(
        "[spike] Operator: review NET-DESIGN-COMPARISON.md, "
        "run AUDIT-01 sanity, then commit manually."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
