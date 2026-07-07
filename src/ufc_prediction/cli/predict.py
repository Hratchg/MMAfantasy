"""CLI predict sub-app for ML prediction commands.

Per D-10: `ufc predict A vs B` returns win probability from trained model.
Per D-11: Manual retrain via CLI command only.
Per D-12: Model coexists with Elo predictions.

Phase 31 / GATE-V23-01..04: `ufc predict gate-spike --feature-set v2.3
--seeds 10 --bootstrap` runs the 10-seed × 3-slice bootstrap noise-floor
spike on the populated REF/TRAVEL substrate and emits
`.planning/gate_contract_v2.3.json` with Path A 0.70 operator floor
pre-committed. See module-level subcommand `predict_gate_spike` below.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ufc_prediction.db.session import SessionLocal
from ufc_prediction.ml.config import FEATURE_COLUMNS, MLConfig
from ufc_prediction.ml.evaluator import evaluate_model, format_evaluation_report
from ufc_prediction.ml.feature_matrix import (
    FeatureMatrixAssembler,
    compute_division_medians,
    split_temporal,
)
from ufc_prediction.ml.order_invariant import predict_order_invariant
from ufc_prediction.ml.persistence import load_metadata, load_model, save_model
from ufc_prediction.ml.predictor import ModelPredictor
from ufc_prediction.ml.queries import (
    load_computed_features,
    load_elo_features,
    load_fight_odds,
    load_fight_records,
    load_fighter_physicals,
    load_pre_ufc_records,
    load_round_stats_for_ml,
)
from ufc_prediction.ml.trainer import ModelTrainer

predict_app = typer.Typer(name="predict", help="ML prediction commands")
console = Console()


# ── Phase 31 module-level constants (GATE-V23-01..04 bindings) ────────────

# WR-05 fix: resolve paths via Path(__file__).resolve().parents[3] (the repo
# root) so the CLI does not silently break when invoked from a directory
# other than the repo root. Mirrors the pattern used in
# `src/ufc_prediction/ml/gate_contract.py:43-51`.
_REPO_ROOT: Path = Path(__file__).resolve().parents[3]

PHASE_31_DIR: Path = (
    _REPO_ROOT / ".planning" / "phases" / "31-gate-v23-re-derivation-on-populated-substrate"
)
V23_CONTRACT_PATH_DEFAULT: Path = _REPO_ROOT / ".planning" / "gate_contract_v2.3.json"

# CONTEXT D-01 Path A — operator empirical floor pre-committed BEFORE spike runs.
OPERATOR_FLOOR_V23: float = 0.70

# AUDIT-01 baseline SHA — models/xgb_v2.joblib re-baselined to the corrected
# dedup + Sherdog-seeded substrate (2026-07-06 promotion). MUST stay byte-identical
# end-to-end at this new baseline.
EXPECTED_XGB_V2_SHA: str = "0b0b40afc8ec41d87508745a9b5f40a46f7d86c054b1ab2acece03d319f6fecd"

# D-18 formula hash binding — carried forward from v2.1 + v2.2; no post-measurement
# renegotiation.
EXPECTED_FORMULA_HASH: str = "7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a"

# Phase 21 BFO backfill commit timestamp — carry-forward from v2.2 contract verbatim.
_BFO_COMMIT_AT_FALLBACK: str = "2026-05-15T15:06:49-07:00"
_PHASE_21_BFO_DIR: str = ".planning/phases/21-bfo-coverage-backfill/"

# Phase 28 INGEST_COVERAGE.json commit timestamp — NEW v2.3-required field.
_INGEST_COMMIT_AT_FALLBACK: str = "2026-05-20T01:59:08-07:00"
_PHASE_28_INGEST_COVERAGE_PATH: str = (
    ".planning/phases/28-referee-venue-ingestion-pipeline/INGEST_COVERAGE.json"
)


# ── Phase 31 module-level injection points (test seams) ───────────────────
#
# These are module-level names so unit tests can monkeypatch them to stub out
# the heavy spike + DB / variance harness paths. Production callers see the
# real implementations.

# IN-01 fix: scripts/ path was previously inserted independently by three
# helpers, each call accumulating a duplicate sys.path entry. Consolidate to
# a single module-level constant + idempotent injector so we never push more
# than one copy.
_SCRIPTS_DIR: str = str(_REPO_ROOT / "scripts")


def _ensure_scripts_on_path() -> None:
    """Inject scripts/ into sys.path exactly once (idempotent)."""
    if _SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _SCRIPTS_DIR)


def _spike_main(argv):
    """Delegate to scripts/spike_noise_floor_v23.main; resolved lazily.

    Lazy import keeps CLI startup cheap and avoids `scripts/` sys.path
    side effects until the gate-spike subcommand actually fires.
    """
    _ensure_scripts_on_path()
    from spike_noise_floor_v23 import main as spike_main

    return spike_main(argv)


def _assert_xgb_v2_sha(label: str) -> str:
    """AUDIT-01 byte-identity assertion; reuses spike_v23 helper.

    Reads models/xgb_v2.joblib bytes, computes sha256, raises SystemExit on
    drift. Module-level seam so unit tests can stub without touching disk.
    """
    _ensure_scripts_on_path()
    from spike_noise_floor_v23 import _assert_xgb_v2_sha as spike_assert

    return spike_assert(label)


def _compute_v23_variance_dict(
    seed_list: list[int],
    *,
    bootstrap: bool = True,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Run variance harness in-process; return (aggregated, warnings) + medians.

    Per Pattern 4 (31-RESEARCH.md): the CLI re-runs the multi-seed harness
    a second time so it has typed-dict access to per-slice variance + per-seed
    medians (the spike's stdout/markdown report would require fragile parsing).
    Wall-clock budget per CONTEXT specifics §5: ≤2 minutes for 10 seeds; in
    practice ~3 seconds against Phase 30 cached fixtures.

    Module-level seam so unit tests can stub this with synthetic data and
    avoid the live DB / META load.
    """
    _ensure_scripts_on_path()
    from spike_noise_floor_v23 import (
        _load_meta_train_eval_matrices,
        _meta_fit_fn,
        _no_bootstrap_metrics,
    )

    from ufc_prediction.ml.variance import (
        aggregate_variance,
        bootstrap_resample,
        multi_seed_metrics,
    )

    (
        X_meta_train,
        y_meta_train,
        X_meta_eval,
        y_meta_eval,
        fight_dates_eval,
    ) = _load_meta_train_eval_matrices(dry_run=False)

    # WR-01 fix: honor --no-bootstrap. The CLI's outer spike subprocess
    # already routes through `_no_bootstrap_metrics` when bootstrap=False
    # (see spike_v23.main); the in-process re-run MUST take the same path
    # so contract values match the report values. Mixing harnesses across
    # the two runs silently drifts the contract from the report.
    if bootstrap:
        per_seed = multi_seed_metrics(
            X_meta_train,
            y_meta_train,
            X_meta_eval,
            y_meta_eval,
            fight_dates_eval,
            seeds=seed_list,
            fit_fn=_meta_fit_fn,
        )
        # Representative model: first-seed bootstrap fit (mirrors v22 spike line 1100).
        Xb, yb = bootstrap_resample(
            X_meta_train,
            y_meta_train,
            seed=seed_list[0],
        )
        representative = _meta_fit_fn(Xb, yb, seed_list[0])
    else:
        # Deterministic single-fit path (ResearchQ10 lineage anchor); fits
        # _meta_fit_fn on the RAW (X_meta_train, y_meta_train) per seed.
        per_seed = _no_bootstrap_metrics(
            X_meta_train,
            y_meta_train,
            X_meta_eval,
            y_meta_eval,
            fight_dates_eval,
            seeds=seed_list,
        )
        # Representative model: first-seed deterministic fit (no resample).
        representative = _meta_fit_fn(
            X_meta_train,
            y_meta_train,
            seed_list[0],
        )
    aggregated, warnings = aggregate_variance(
        per_seed,
        representative_model=representative,
        X_eval=X_meta_eval,
        y_eval=y_meta_eval,
        fight_dates_eval=fight_dates_eval,
    )
    # Add per-slice median Brier + accuracy across seeds (needed for formula).
    medians = _compute_per_seed_medians(per_seed)
    for slc, slc_med in medians.items():
        aggregated[slc].update(slc_med)
    return aggregated, warnings


def _read_project_md() -> str:
    """Read .planning/PROJECT.md; module-level seam for D-24 precondition test.

    WR-05 fix: resolve PROJECT.md via the repo-root anchor so the CLI works
    regardless of the operator's CWD (matches `gate_contract.py:43-51`
    pattern). Previously this used a CWD-relative path that silently broke
    when the CLI was invoked from outside the repo root and surfaced as an
    uncaught FileNotFoundError rather than the clean SystemExit(2) the
    Pitfall-1 precondition test expects.
    """
    return (_REPO_ROOT / ".planning" / "PROJECT.md").read_text(encoding="utf-8")


# ── Phase 31 helpers (formula + floor + HALT + contract emission) ─────────


def _compute_per_seed_medians(
    per_seed: dict[int, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """Per-slice median Brier + Acc across seeds.

    Mirrors spike_noise_floor_v22.py lines around 1075 (per-slice seed arrays
    + np.median reduction). Returns:
      {slice_name: {"median_brier": float, "median_acc": float}}
    """
    seeds = list(per_seed.keys())
    out: dict[str, dict[str, float]] = {}
    for slc in ("most_recent_12mo", "most_recent_24mo", "random_15pct"):
        briers = np.array(
            [per_seed[s][slc]["brier_score"] for s in seeds],
            dtype=np.float64,
        )
        accs = np.array(
            [per_seed[s][slc]["accuracy"] for s in seeds],
            dtype=np.float64,
        )
        out[slc] = {
            "median_brier": float(np.median(briers)),
            "median_acc": float(np.median(accs)),
        }
    return out


def _build_per_slice_thresholds_v23(
    aggregated: dict[str, dict[str, float]],
):
    """Apply FORMULA_SOURCE mechanically per slice per metric (D-18 binding).

    Verbatim adaptation of spike_noise_floor_v22.py:_build_per_slice_thresholds
    (lines 324-389). Formula:
      gate_brier_max    = round(median_brier - 1 * std_brier_used, 4)
      gate_accuracy_min = round(median_acc   + 1 * std_acc_used,   4)

    aggregated[slc] must contain median_brier, median_acc, seed_std_brier,
    seed_std_acc, bootstrap_ci_half_brier, bootstrap_ci_half_acc,
    std_brier_used, std_acc_used (populated by _compute_v23_variance_dict).

    Returns a dict[slc -> PerSliceThresholds] PRE-floor (floor application
    happens in _apply_operator_floor_and_detect_breach).
    """
    from ufc_prediction.ml.gate_contract import PerSliceThresholds

    per_slice: dict[str, PerSliceThresholds] = {}
    for slc in ("most_recent_12mo", "most_recent_24mo", "random_15pct"):
        a = aggregated[slc]
        median_brier = float(a["median_brier"])
        median_acc = float(a["median_acc"])
        std_brier_used = float(a["std_brier_used"])
        std_acc_used = float(a["std_acc_used"])
        gate_brier_max = round(median_brier - 1 * std_brier_used, 4)
        gate_accuracy_min = round(median_acc + 1 * std_acc_used, 4)
        per_slice[slc] = PerSliceThresholds(
            brier_max=gate_brier_max,
            accuracy_min=gate_accuracy_min,
            median_brier_xgb_v2=median_brier,
            median_acc_xgb_v2=median_acc,
            seed_std_brier=float(a["seed_std_brier"]),
            seed_std_acc=float(a["seed_std_acc"]),
            bootstrap_ci_half_brier=float(a["bootstrap_ci_half_brier"]),
            bootstrap_ci_half_acc=float(a["bootstrap_ci_half_acc"]),
            std_brier_used=std_brier_used,
            std_acc_used=std_acc_used,
        )
    return per_slice


def _apply_operator_floor_and_detect_breach(
    per_slice,
    *,
    floor: float = OPERATOR_FLOOR_V23,
):
    """Apply Path A operator floor + detect breach (CONTEXT D-01).

    For each slice:
      pre_floor_accuracy_min = per_slice[slice].accuracy_min  (formula output)
      final_accuracy_min     = max(pre_floor_accuracy_min, floor)
      breach[slice]          = pre_floor_accuracy_min < floor

    Path A semantic (per 31-RESEARCH.md Pattern 6, NOT v2.2 accept_truth):
      operator_floor_applied = TRUE  on slices that clear the floor (floor LIFTED
                                     the formula output to 0.70)
      operator_floor_applied = FALSE on slices that breach (formula < 0.70)

    Returns (floored_per_slice, breach_diagnostic).
    """
    from ufc_prediction.ml.gate_contract import PerSliceThresholds

    floored: dict[str, PerSliceThresholds] = {}
    breach: dict[str, dict] = {}
    for slc, ts in per_slice.items():
        pre = float(ts.accuracy_min)
        post = max(pre, floor)
        is_breach = pre < floor
        gap = max(floor - pre, 0.0)
        floored[slc] = dataclasses.replace(
            ts,
            accuracy_min=post,
            # Path A: floor APPLIED on non-breach (floor lifted formula to 0.70).
            #         floor NOT applied on breach (HALT triggers; contract not emitted).
            operator_floor_applied=(not is_breach),
            operator_floor_value=floor,
            operator_decision=("path_a_floor_applied" if not is_breach else "path_a_breach_halt"),
        )
        breach[slc] = {
            "pre_floor_accuracy_min": pre,
            "post_floor_accuracy_min": post,
            "breach": is_breach,
            "gap": gap,
        }
    return floored, breach


def _git_log_first_iso(path: str, fallback: str) -> str:
    """`git log --format=%aI -1 -- <path>` with fallback on failure (Pitfall 7)."""
    try:
        out = subprocess.run(
            ["git", "log", "--format=%aI", "-1", "--", path],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out or fallback
    except (FileNotFoundError, subprocess.CalledProcessError):
        return fallback


def _emit_v23_contract(
    per_slice_floored,
    seed_list: list[int],
    *,
    contract_path: Path,
) -> None:
    """Write .planning/gate_contract_v2.3.json (Path A only; v2.2 schema + ingest field).

    Mirrors spike_noise_floor_v22.py lines 1207-1253 verbatim with v2.3 deltas:
      - version = "v2.3" (was "v2.2")
      - supersedes = ["D-21(v2.2, GATE)", "v2.2-gate"]
      - feature_columns_hash from FEATURE_COLUMNS_V22 (v2.3 column space ≡ v2.2 per
        CONTEXT A1; substrate fill, not schema change — confirmed via Open Questions
        Q3 fallback)
      - ingest_completed_at from Phase 28 INGEST_COVERAGE.json git log (Pitfall 7)
      - bfo_backfill_committed_at carried forward from Phase 21 dir git log

    WR-04 fix: the previous signature accepted an `aggregated` parameter
    that was never referenced; secondary_metrics_observed is hard-coded to
    {} in this contract version because the on-disk v2.3 contract was
    already committed with that shape (CONTEXT: DO NOT regenerate emit-
    time fields). Removed the dead parameter so a future reviewer doesn't
    assume it is wired. If a later version wants v2.2-style AUC/ECE per-
    slice medians, plumb `per_seed` through a NEW parameter — re-using
    the dropped name would silently regress the audit lineage.
    """
    from dataclasses import asdict

    from ufc_prediction.ml.config import get_feature_columns
    from ufc_prediction.ml.gate_contract import GateContract

    # feature_columns_hash — Pitfall 3: sort_keys=False mandatory.
    # v2.3 column space ≡ v2.2 (substrate fill, not schema change; per CONTEXT
    # A1 + 31-RESEARCH.md Open Q3). Plan 31-01 does NOT add a v2.3 dispatch to
    # get_feature_columns — deferred to Phase 32. The hash matches v2.2's.
    v23_cols = get_feature_columns(feature_set="v2.2")
    fc_hash = hashlib.sha256(
        json.dumps(v23_cols, sort_keys=False).encode("utf-8"),
    ).hexdigest()

    bfo_ts = _git_log_first_iso(_PHASE_21_BFO_DIR, _BFO_COMMIT_AT_FALLBACK)
    ingest_ts = _git_log_first_iso(
        _PHASE_28_INGEST_COVERAGE_PATH,
        _INGEST_COMMIT_AT_FALLBACK,
    )

    contract = GateContract(
        version="v2.3",
        derived_at=date.today().isoformat(),
        n_seeds_observed=len(seed_list),
        base_features_set="FEATURE_COLUMNS_V22",
        n_features=len(v23_cols),  # 90 unchanged
        k_value=1,
        formula_hash=EXPECTED_FORMULA_HASH,
        cutoff_date="2023-01-01",
        per_slice=per_slice_floored,
        secondary_metrics_observed={},
        supersedes=["D-21(v2.2, GATE)", "v2.2-gate"],
        notes=(
            "v2.3 gate contract — mechanically re-derived on the POPULATED "
            "REF/TRAVEL substrate (Phase 28 ingestion close) using the same "
            "formula hash as v2.1+v2.2 (D-18 binding). Operator floor "
            "accuracy_min >= 0.70 PRE-COMMITTED at Path A (CONTEXT D-01) "
            "and applied via max(formula_output, 0.70) per slice. "
            "HALT-AND-DECIDE artifact emitted if any slice pre-floor < 0.70 "
            "(D-06 binding). v2.1 contract at .planning/gate_contract.json "
            "PRESERVED. v2.2 contract at .planning/gate_contract_v2.2.json "
            "PRESERVED. "
            "Formula: gate_brier_max = round(median_brier - 1 * "
            "max(seed_std_brier, bootstrap_ci_half_brier), 4); "
            "gate_accuracy_min = round(median_acc + 1 * "
            "max(seed_std_acc, bootstrap_ci_half_acc), 4)"
        ),
        feature_columns_hash=fc_hash,
        bfo_backfill_committed_at=bfo_ts,
        ingest_completed_at=ingest_ts,
        operator_decision={
            "decision": "path_a_operator_floor_pre_committed_at_milestone_open",
            "decided_at": "2026-05-22",
            "rationale": (
                "v2.3 is the first public partner release; Path A pre-commit "
                "reinforces the 0.70 partner-facing floor. D-18 binding: no "
                "post-measurement renegotiation."
            ),
            "phase_32_implication": (
                "Forward-stepwise composition (Phase 32) gates against "
                "accuracy_min post-floor (i.e., the >= 0.70 surface). "
                "REF/TRAVEL re-tested on populated substrate against this gate."
            ),
        },
    )
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(
        json.dumps(asdict(contract), indent=2) + "\n",
        encoding="utf-8",
    )


def _render_31_halt_and_decide(
    breach_diagnostic: dict[str, dict],
    *,
    triggered_at: str,
) -> str:
    """Render 31-HALT-AND-DECIDE.md per CONTEXT D-01 Path D pre-templates.

    Verbatim fork of 24-HALT-AND-DECIDE.md with phase-id substitutions + the
    three Path D outcome-path IDs from 31-RESEARCH.md Pattern 7.

    IN-04 fix: previously accepted a `per_slice_raw` argument that the body
    never referenced (the v2.2 spike's renderer used it for a "PRE-floor
    formula output" diagnostic column; Phase 31's version dropped that
    column but forgot to drop the parameter). Removed to avoid misleading
    a future reviewer.
    """
    lines: list[str] = [
        "---",
        "phase: 31-gate-v23-re-derivation-on-populated-substrate",
        "artifact: HALT-AND-DECIDE",
        f"triggered_at: {triggered_at}",
        'trigger_condition: "Pre-floor formula_output < OPERATOR_ACCURACY_FLOOR'
        ' (0.70) on at least one slice"',
        "operator_decision_required: true",
        "outcome_paths:",
        "  - id: v23_substrate_fills_but_gate_lt_floor_path_d_delay_ship",
        '    description: "Delay v2.3 public ship; reopen partner-readiness'
        ' gate; either deepen substrate or expand corpus"',
        "  - id: v23_substrate_fills_but_gate_lt_floor_path_d_ship_v22_publicly",
        '    description: "Ship v2.2 publicly (already gate-validated); v2.3'
        ' internal-only milestone"',
        "  - id: v23_substrate_fills_but_gate_lt_floor_path_d_accept_lower_floor",
        '    description: "Accept lower empirical floor with explicit'
        " governance amendment (D-18 override; requires operator-signed"
        ' amendment)"',
        "---",
        "",
        "# Phase 31 HALT-AND-DECIDE",
        "",
        "**The v2.3 gate spike found at least one slice where the mechanically-",
        "derived `accuracy_min` (formula output) fell BELOW the operator empirical",
        "floor of `0.70`. Per CONTEXT D-01 (Path A pre-commit) + D-18 binding,",
        "the spike does NOT silently apply the floor — operator decision is required",
        "at the partner-readiness gate.**",
        "",
        "## Per-Slice Gap Diagnostic",
        "",
        "| Slice | pre_floor_accuracy_min | post_floor (clamped) | gap | breach? |",
        "|-------|------------------------|---------------------|-----|---------|",
    ]
    for slc in ("most_recent_12mo", "most_recent_24mo", "random_15pct"):
        # WR-02 fix: defensive against missing slice keys. The HALT writer
        # is the operator's only safety net at Path D — a KeyError here
        # would leave the operator with no HALT artifact AND no contract.
        # Under normal flow all three slices are present (built by
        # _apply_operator_floor_and_detect_breach), but a future helper
        # refactor or a stubbed-test variance with fewer slices must
        # render rather than crash.
        d = breach_diagnostic.get(slc)
        if d is None:
            lines.append(f"| {slc} | (missing) | (missing) | n/a | UNKNOWN |")
            continue
        is_breach = "YES" if d["breach"] else "NO"
        lines.append(
            f"| {slc} | {d['pre_floor_accuracy_min']:.4f} | "
            f"{d['post_floor_accuracy_min']:.4f} | {d['gap']:.4f} | {is_breach} |"
        )
    lines.extend(
        [
            "",
            "## Pre-Templated Outcome Paths (Path D materialization)",
            "",
            "- **v23_substrate_fills_but_gate_lt_floor_path_d_delay_ship:** Hold v2.3 "
            "public ship. Re-open partner-readiness gate. Either widen substrate "
            "(additional ingestion / corpus growth) OR re-derive on a longer eval window.",
            "",
            "- **v23_substrate_fills_but_gate_lt_floor_path_d_ship_v22_publicly:** Ship "
            "v2.2 publicly (already gate-validated at v2.2 Phase 26 close). Tag v2.3 as "
            "internal-only milestone. v2.4+ continues the partner pitch.",
            "",
            "- **v23_substrate_fills_but_gate_lt_floor_path_d_accept_lower_floor:** "
            "Operator-signed amendment overriding D-18; accept empirical truth as new "
            "public floor; SIGN GOVERNANCE AMENDMENT before contract emit.",
            "",
            "## Operator Action Required",
            "",
            "Set `actual_outcome` in `31-SUMMARY.md` frontmatter to one of:",
            "- `v23_substrate_fills_but_gate_lt_floor_path_d_delay_ship`",
            "- `v23_substrate_fills_but_gate_lt_floor_path_d_ship_v22_publicly`",
            "- `v23_substrate_fills_but_gate_lt_floor_path_d_accept_lower_floor`",
            "",
            "Then run `/gsd-execute-phase 31` resumption.",
            "",
        ]
    )
    return "\n".join(lines)


# ── @predict_app.command("gate-spike") subcommand ─────────────────────────


@predict_app.command("gate-spike")
def predict_gate_spike(
    feature_set: str = typer.Option(
        "v2.3",
        "--feature-set",
        help=(
            "Feature-column space identifier. v2.3 column space ≡ v2.2 "
            "(substrate fill, not schema change). Forward-compat flag; "
            "Phase 32 may extend get_feature_columns() with a true v2.3 "
            "dispatch."
        ),
    ),
    seeds: int = typer.Option(
        10,
        "--seeds",
        help=(
            "Number of bootstrap seeds. CLI receives a COUNT; translates "
            "to deterministic seed-list [42..41+seeds] matching spike_v23 "
            "SEEDS_DEFAULT (Pitfall 2)."
        ),
    ),
    bootstrap: bool = typer.Option(
        True,
        "--bootstrap/--no-bootstrap",
        help=(
            "Enable bootstrap row-resampling for per-seed variance "
            "(D-01 of Phase 30; REQUIRED for empirically-honest variance)."
        ),
    ),
    contract_path: str = typer.Option(
        str(V23_CONTRACT_PATH_DEFAULT),
        "--contract-path",
        help="Output path for v2.3 gate contract JSON.",
    ),
) -> None:
    """Phase 31 / GATE-V23-01..04: 10-seed × 3-slice bootstrap noise-floor
    spike on the populated REF/TRAVEL substrate. Emits gate_contract_v2.3.json
    with Path A operator floor (accuracy_min = max(formula, 0.70)).

    Pre-conditions (HARD-ORDERED per CONTEXT SC4 binding):
      1. D-24(v2.3, GATE) PROJECT.md row committed
      2. load_gate_contract(version="v2.3") dispatch landed (Plan 31-01 Task 2)
      3. THIS CLI body landed (Plan 31-01 Task 3)

    Exit codes:
      0 = Path A success (all 3 slices clear 0.70 pre-floor) — contract emitted
      3 = Path D HALT (any slice formula_output < 0.70) — 31-HALT-AND-DECIDE.md emitted
      2 = Fatal pre-spike invariant violation (D-24 row missing,
          AUDIT-01 SHA mismatch, etc.)
    """
    # ── 1. D-24 precondition (Pitfall 1; GATE-V23-04 SC4 binding) ────
    # WR-06 fix: require the bold-pipe table row format
    # `| **D-24(v2.3, GATE)** |` rather than a plain substring match. A bare
    # substring passes on commented-out lines, quoted historical references,
    # and "deferred to Phase NN" narrative mentions — none of which satisfy
    # the SC4 intent that the DECISION ROW is committed and ACTIVE.
    project_md = _read_project_md()
    if "| **D-24(v2.3, GATE)** |" not in project_md:
        console.print(
            "[red]Pre-condition violated: D-24(v2.3, GATE) decision row missing "
            "from .planning/PROJECT.md (must appear as a bold-pipe table row: "
            "`| **D-24(v2.3, GATE)** |`). Plan 31-01 task 1 must land before "
            "spike runs (GATE-V23-04 binding).[/red]"
        )
        raise SystemExit(2)

    # ── 2. AUDIT-01 STARTUP SHA check ────────────────────────────────
    xgb_v2_sha_start = _assert_xgb_v2_sha("PHASE-31-STARTUP")
    console.print(f"[green]AUDIT-01 STARTUP SHA OK: {xgb_v2_sha_start[:16]}...[/green]")

    # ── 3. Translate --seeds N → seed-list [42..41+N] (Pitfall 2) ────
    seed_list = list(range(42, 42 + int(seeds)))
    console.print(f"Seed translation: --seeds {seeds} → {seed_list}")

    # ── 4. Invoke spike_v23 with Phase 31 artifact paths (Pitfall 8) ─
    spike_argv: list[str] = ["--seeds", *(str(s) for s in seed_list)]
    if not bootstrap:
        spike_argv.append("--no-bootstrap")
    spike_argv += [
        "--report-path",
        str(PHASE_31_DIR / "31-VARIANCE-REPORT.md"),
        "--halt-path",
        str(PHASE_31_DIR / "31-VARIANCE-HALT.md"),
        "--sha-end-path",
        str(PHASE_31_DIR / "31-XGB-V2-SHA-PHASE-31-END-SPIKE.txt"),
    ]
    console.print(f"[yellow]Invoking spike_v23.main({spike_argv})...[/yellow]")
    spike_exit = _spike_main(spike_argv)
    if spike_exit == 3:
        console.print(
            "[red]spike_v23 exited 3 (variance collapse). Resolve "
            "31-VARIANCE-HALT.md before re-running.[/red]"
        )
        raise SystemExit(3)
    if spike_exit != 0:
        console.print(f"[red]spike_v23 exited {spike_exit} (fatal).[/red]")
        raise SystemExit(spike_exit)

    # ── 5. Re-compute variance dict in-process (Pattern 4) ───────────
    aggregated, warnings = _compute_v23_variance_dict(
        seed_list,
        bootstrap=bootstrap,
    )
    for w in warnings:
        console.print(f"[yellow]variance warning: {w}[/yellow]")

    # ── 6. Apply formula + operator floor (Pattern 6) ────────────────
    per_slice_raw = _build_per_slice_thresholds_v23(aggregated)
    per_slice_floored, diagnostic = _apply_operator_floor_and_detect_breach(
        per_slice_raw,
        floor=OPERATOR_FLOOR_V23,
    )
    for slc, d in diagnostic.items():
        marker = " (BREACH)" if d["breach"] else ""
        console.print(
            f"  floor-apply {slc}: pre={d['pre_floor_accuracy_min']:.4f} -> "
            f"post={d['post_floor_accuracy_min']:.4f} "
            f"(gap={d['gap']:.4f}){marker}"
        )

    halt_required = any(d["breach"] for d in diagnostic.values())

    # ── 7. Path branch: HALT (Path D) OR contract emit (Path A) ──────
    if halt_required:
        halt_body = _render_31_halt_and_decide(
            diagnostic,
            triggered_at=datetime.now(UTC).isoformat(),
        )
        halt_path = PHASE_31_DIR / "31-HALT-AND-DECIDE.md"
        halt_path.parent.mkdir(parents=True, exist_ok=True)
        halt_path.write_text(halt_body, encoding="utf-8")
        # END SHA assertion runs but artifact NOT written (Pitfall 5).
        _assert_xgb_v2_sha("PHASE-31-END")
        console.print(
            f"[red]Path D HALT: 31-HALT-AND-DECIDE.md emitted -> {halt_path}. "
            "Operator action required.[/red]"
        )
        raise SystemExit(3)

    # Path A: emit contract + END SHA artifact + clear loader cache (Pitfall 6).
    _emit_v23_contract(
        per_slice_floored,
        seed_list,
        contract_path=Path(contract_path),
    )
    xgb_v2_sha_end = _assert_xgb_v2_sha("PHASE-31-END")
    sha_end_path = PHASE_31_DIR / "31-XGB-V2-SHA-PHASE-31-END.txt"
    sha_end_path.parent.mkdir(parents=True, exist_ok=True)
    sha_end_path.write_text(xgb_v2_sha_end + "\n", encoding="utf-8")

    from ufc_prediction.ml.gate_contract import load_gate_contract

    load_gate_contract.cache_clear()  # Pitfall 6

    console.print(
        f"[green]Phase 31 spike complete (Path A). Wrote:\n"
        f"  {contract_path}\n"
        f"  {sha_end_path}\n"
        f"Review {contract_path} and commit artifacts manually "
        "(Pitfall D — CLI does not call git).[/green]"
    )


def _enforce_accuracy_gate(
    metrics: dict,
    brier_max: float = 0.2202,
    acc_min: float = 0.6391,
) -> None:
    """Hard-block retrain on metric regression (Phase 15: D-07, D-08).

    Raises ``SystemExit(1)`` with a red message naming each missed
    target and the delta. On pass, prints a green PASSED message
    including both metrics. The check runs BEFORE ``save_model`` so a
    regressed model is never persisted to disk (D-09 implicit;
    T-15-02-01 mitigation).
    """
    brier = metrics["brier_score"]
    acc = metrics["accuracy"]

    failures: list[str] = []
    if brier > brier_max:
        failures.append(
            f"Brier {brier:.4f} > target {brier_max:.4f} (missed by {brier - brier_max:+.4f})"
        )
    if acc < acc_min:
        failures.append(
            f"Accuracy {acc:.4f} < target {acc_min:.4f} (missed by {acc - acc_min:+.4f})"
        )

    if failures:
        console.print("[red bold]Phase 15 accuracy gate FAILED:[/red bold]")
        for f in failures:
            console.print(f"  [red]{f}[/red]")
        console.print(
            "\n[yellow]Investigate: odds coverage gaps, feature leak, "
            "training-data sparsity (Pitfall 4 in 15-RESEARCH.md).[/yellow]"
        )
        raise SystemExit(1)

    console.print(
        f"[green]Accuracy gate PASSED: "
        f"Brier={brier:.4f} <= {brier_max}, "
        f"Accuracy={acc:.4f} >= {acc_min}[/green]"
    )


def _reevaluate_v1_baseline(
    X_test,
    y_test,
    model_dir: str = "models",
) -> dict:
    """Re-evaluate xgb_v1 on a given test fold for relaxation-anchor metrics.

    Per CONTEXT D-11(P15.1): xgb_v1 stays untouched (D-09(P15) byte-identical
    SHA invariant); only its measured metrics get a fresh entry against the
    test fold for the Path B relaxed-gate derivation (D-09(P15.1)/D-10(P15.1)).

    This function is READ-ONLY. It MUST NEVER call ``save_model`` or modify
    ``models/xgb_v1.*`` in any way. The D-09(P15) invariant is enforced by
    the Plan-02 RUN-LOG SHA-256 audits at Task 0, Task 5, and Task 6.

    Schema-compatibility shim (Rule 1 deviation, captured in 15.1-02-SUMMARY):
    xgb_v1 was trained against a 67-column FEATURE_COLUMNS schema; the current
    FEATURE_COLUMNS has grown to 72 columns (added 5 odds-derived features in
    Phase 15). The helper reads ``xgb_v1_meta.json["feature_columns"]`` and
    slices ``X_test`` to the v1 column subset (by name -> current-index lookup)
    before calling ``evaluate_model``. xgb_v1 itself is unmodified.

    Args:
        X_test: Test feature matrix (numpy array, shape: [n_test, current_n_features]).
        y_test: Test labels (numpy array, shape: [n_test]).
        model_dir: Directory containing ``xgb_v1.joblib`` (default: ``"models"``).

    Returns:
        Metrics dict from ``evaluate_model``. Contains at least
        ``"brier_score"`` and ``"accuracy"`` keys (per evaluator.py contract).
    """
    v1_model = load_model(model_dir, "v1")  # READ-ONLY — joblib.load only
    # Schema shim: slice X_test to v1's feature columns (Rule 1 fix for
    # 67-vs-72 column drift between Phase 14 v1 and Phase 15 FEATURE_COLUMNS).
    try:
        v1_meta = load_metadata(model_dir, "v1")
        v1_cols = v1_meta.get("feature_columns")
    except FileNotFoundError:
        v1_cols = None
    if v1_cols and hasattr(X_test, "shape") and X_test.shape[1] != len(v1_cols):
        v1_indices = [FEATURE_COLUMNS.index(c) for c in v1_cols]
        X_test = X_test[:, v1_indices]
    return evaluate_model(v1_model, X_test, y_test)


def _derive_relaxed_thresholds(v1_metrics: dict) -> tuple[float, float]:
    """Derive Path B relaxed thresholds from v1 baseline metrics (D-10(P15.1)).

    Per CONTEXT D-09(P15.1): principle = beat xgb_v1 by margin.
    Per CONTEXT D-10(P15.1): Brier margin = -0.010, Acc margin = +0.020.

    Args:
        v1_metrics: Dict with "brier_score" and "accuracy" keys
                    (output of evaluate_model on xgb_v1).

    Returns:
        (new_brier_max, new_acc_min) tuple. v2 gate passes iff
        v2_brier <= new_brier_max AND v2_acc >= new_acc_min.
    """
    new_brier_max = v1_metrics["brier_score"] - 0.010
    new_acc_min = v1_metrics["accuracy"] + 0.020
    return (new_brier_max, new_acc_min)


@predict_app.command("relax-gate")
def predict_relax_gate(
    model_dir: str = typer.Option(
        "models",
        help="Model directory containing xgb_v1.joblib",
    ),
) -> None:
    """Path B (Phase 15.1 D-04 / D-09 / D-10 / D-11): re-evaluate xgb_v1 on the
    existing test fold, derive relaxed thresholds, and (on operator confirmation)
    persist them as the new D-07 hard gate constants in source.

    NO --force / --yes bypass. Operator MUST type 'y' interactively per
    Phase 15.1 D-02 (operator-decides) + RESEARCH.md Pitfall 7.
    """
    from datetime import datetime
    from pathlib import Path

    # Step 1: rebuild the same test fold predict_train uses (no --train-lower
    # per orchestrator CRITICAL CONTEXT #2)
    config = MLConfig()
    cutoff = date.fromisoformat(config.cutoff_date)

    console.print("[bold]Loading data for v1 re-evaluation (D-11(P15.1) anchor)...[/bold]")
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
        cutoff,
    )
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
    )
    # No train_lower — Path B uses the EXISTING Phase 15 test fold (CRITICAL CONTEXT #2)
    _, X_test, _, y_test = split_temporal(X, y, fight_dates, cutoff)
    console.print(f"  Test fold: {len(X_test)} fights")

    # Step 2: Re-evaluate v1 (D-11(P15.1)) — Task 3 helper
    console.print(
        "[bold]Re-evaluating xgb_v1 (READ-ONLY, D-09(P15) byte-identity preserved)...[/bold]"
    )
    v1_metrics = _reevaluate_v1_baseline(X_test, y_test, model_dir=model_dir)

    # Step 3: Derive new thresholds (D-10(P15.1))
    new_brier_max, new_acc_min = _derive_relaxed_thresholds(v1_metrics)

    # Step 4: Surface to operator
    panel_body = (
        f"[bold]v1 baseline metrics on Phase 15 test fold (D-11(P15.1) anchor):[/bold]\n"
        f"  Brier: {v1_metrics['brier_score']:.4f} / Acc: {v1_metrics['accuracy']:.4f}\n\n"
        f"[bold]Proposed relaxed thresholds (D-10(P15.1): Brier <= v1-0.010, Acc >= v1+0.020):[/bold]\n"
        f"  new_brier_max = {v1_metrics['brier_score']:.4f} - 0.010 = {new_brier_max:.4f}\n"
        f"  new_acc_min   = {v1_metrics['accuracy']:.4f} + 0.020 = {new_acc_min:.4f}\n"
    )
    console.print(
        Panel(
            panel_body,
            title="Phase 15.1 Path B — Relaxed Gate Derivation",
            border_style="cyan",
        )
    )

    # Step 5: Operator-decides (D-02(P15.1)). NO --yes bypass (RESEARCH Pitfall 7).
    try:
        typer.confirm(
            "Persist these as the new D-07 hard gate?",
            default=False,
            abort=True,
        )
    except typer.Abort:
        console.print("[yellow]Aborted — original gate preserved.[/yellow]")
        raise SystemExit(0) from None

    # Step 6: In-place source edit (D-04(P15.1) persistence)
    # FOUR substitutions in one atomic read-write cycle:
    #   (a) helper defaults block (lines 42-46)
    #   (b) Typer Option brier_max block
    #   (c) Typer Option acc_min block
    #   (d) docstring (Warning 6 fix — supersession trace)
    predict_py_path = Path(__file__)  # this very file
    original_text = predict_py_path.read_text(encoding="utf-8")
    new_text = original_text

    # (a) Helper defaults (predict.py:42-46) — bare floats
    new_text = new_text.replace(
        "    brier_max: float = 0.21,\n    acc_min: float = 0.65,\n",
        f"    brier_max: float = {new_brier_max:.4f},\n    acc_min: float = {new_acc_min:.4f},\n",
    )
    # (b) Typer Option brier_max
    new_text = new_text.replace(
        '    brier_max: float = typer.Option(\n        0.21,\n        "--brier-max",\n        help="Hard upper bound for Brier score (D-07).",\n    ),',
        f'    brier_max: float = typer.Option(\n        {new_brier_max:.4f},\n        "--brier-max",\n        help="Hard upper bound for Brier score (Phase 15.1 D-04 supersedes D-07).",\n    ),',
    )
    # (c) Typer Option acc_min
    new_text = new_text.replace(
        '    acc_min: float = typer.Option(\n        0.65,\n        "--acc-min",\n        help="Hard lower bound for accuracy (D-07).",\n    ),',
        f'    acc_min: float = typer.Option(\n        {new_acc_min:.4f},\n        "--acc-min",\n        help="Hard lower bound for accuracy (Phase 15.1 D-04 supersedes D-07).",\n    ),',
    )
    # (d) Docstring (Warning 6 fix) — must include literal new threshold values
    new_text = new_text.replace(
        "Hard accuracy gate (Brier <= 0.2202 AND accuracy >= 0.6391, D-04(P15.1) supersedes D-07(P15))",
        f"Hard accuracy gate (Brier <= {new_brier_max:.4f} AND accuracy >= {new_acc_min:.4f}, D-04(P15.1) supersedes D-07(P15))",
    )

    if new_text == original_text:
        console.print(
            "[red]ERROR: source-edit produced zero changes. Constants may already "
            "be relaxed or block format mismatched. Aborting without write.[/red]"
        )
        raise SystemExit(1)

    predict_py_path.write_text(new_text, encoding="utf-8")
    console.print(
        "[green]Persisted relaxed thresholds + docstring update to predict.py "
        "per D-04(P15.1) supersession of D-07(P15).[/green]"
    )

    # Step 7: Append derivation to RUN-LOG
    runlog_path = Path(".planning/phases/15.1-odds-03-coverage-closure/15.1-RUN-LOG.md")
    runlog = runlog_path.read_text(encoding="utf-8")
    ts = datetime.now().isoformat(timespec="seconds")
    addendum = (
        f"\n\nDerivation timestamp: {ts}\n"
        f"v1 baseline (D-11(P15.1)): "
        f"Brier={v1_metrics['brier_score']:.4f}, "
        f"Acc={v1_metrics['accuracy']:.4f}\n"
        f"D-10(P15.1) formula: (Brier-0.010, Acc+0.020)\n"
        f"new_brier_max = {new_brier_max:.4f}\n"
        f"new_acc_min   = {new_acc_min:.4f}\n"
        f"Operator confirmation: YES\n"
        f"predict.py constants swapped (D-04(P15.1) supersession of D-07(P15)): COMPLETE\n"
        f"predict.py:120 docstring rewritten with literal relaxed thresholds: COMPLETE\n"
    )
    runlog_path.write_text(
        runlog.replace(
            "### Task 4: Relaxed-gate derivation + operator confirmation (D-04(P15.1), D-10(P15.1))",
            "### Task 4: Relaxed-gate derivation + operator confirmation (D-04(P15.1), D-10(P15.1))"
            + addendum,
        ),
        encoding="utf-8",
    )


@predict_app.command("train")
def predict_train(
    trials: int = typer.Option(50, help="Number of Optuna trials"),
    version: str = typer.Option("v2", help="Model version tag (Phase 15: default v2)"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Skip the 50% odds-coverage threshold check (D-13).",
    ),
    brier_max: float = typer.Option(
        0.2202,
        "--brier-max",
        help="Hard upper bound for Brier score (Phase 15.1 D-04 supersedes D-07).",
    ),
    acc_min: float = typer.Option(
        0.6391,
        "--acc-min",
        help="Hard lower bound for accuracy (Phase 15.1 D-04 supersedes D-07).",
    ),
    train_lower: str | None = typer.Option(
        None,
        "--train-lower",
        help="ISO date (YYYY-MM-DD) lower bound for train fold "
        "(Phase 15.1 D-05/D-06/D-07). Overrides MLConfig.train_lower_bound_date "
        "if provided. Fights before this date are DROPPED from train and "
        "never appear in test.",
    ),
) -> None:
    """Train XGBoost model with Optuna hyperparameter tuning (D-11).

    Phase 15 contract:
      - 50% odds-coverage pre-check on training set (Claude's discretion);
        bypass with --force.
      - Hard accuracy gate (Brier <= 0.2202 AND accuracy >= 0.6391, D-04(P15.1) supersedes D-07(P15))
        runs after evaluate_model and BEFORE save_model so regressed
        models are never persisted.
      - Default --version is 'v2' (D-09); 'v1' artifacts remain untouched.
    """
    config = MLConfig(n_optuna_trials=trials)
    # Phase 15.1 D-05/D-06/D-07: --train-lower CLI flag overrides MLConfig field if set.
    effective_train_lower_str = train_lower or config.train_lower_bound_date
    train_lower_obj: date | None = (
        date.fromisoformat(effective_train_lower_str) if effective_train_lower_str else None
    )
    if train_lower_obj is not None:
        console.print(
            f"[cyan]Phase 15.1 train_lower_bound_date active: "
            f"{train_lower_obj.isoformat()} "
            f"(fights before this date dropped from train fold)[/cyan]"
        )

    console.print("[bold]Loading data...[/bold]")
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

    console.print(f"  Loaded {len(fight_records)} fights")
    console.print(f"  Loaded {len(pre_ufc)} pre-UFC records")
    console.print(f"  Loaded {len(fight_odds)} fight_odds rows")

    # Phase 15 (Claude's discretion): training-set odds coverage check.
    # Bypass with --force.
    cutoff_date_obj = date.fromisoformat(config.cutoff_date)
    fights_with_odds = {fid for (_fighter_id, fid) in fight_odds}
    training_total = sum(1 for f in fight_records if f["event_date"] < cutoff_date_obj)
    training_with_odds = sum(
        1
        for f in fight_records
        if f["event_date"] < cutoff_date_obj and f["fight_id"] in fights_with_odds
    )
    coverage = training_with_odds / max(training_total, 1)
    console.print(f"  Odds coverage on training set: {coverage:.1%}")
    if coverage < 0.50 and not force:
        console.print(
            f"[red]Coverage {coverage:.1%} < 50% threshold. "
            f"Pass --force to retrain on sparse data, or run "
            f"`ufc scrape odds` first.[/red]"
        )
        raise SystemExit(1)

    # Compute division medians using only training data
    cutoff = date.fromisoformat(config.cutoff_date)
    division_medians = compute_division_medians(
        fighter_physicals,
        fight_records,
        cutoff,
    )

    # Assemble feature matrix
    console.print("[bold]Assembling feature matrix...[/bold]")
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
    )
    console.print(f"  Matrix shape: {X.shape}")

    # Temporal split (Phase 15.1: 3-bucket if train_lower_obj set, else 2-bucket)
    X_train, X_test, y_train, y_test = split_temporal(
        X, y, fight_dates, cutoff, train_lower=train_lower_obj
    )
    console.print(f"  Training: {len(X_train)} fights, Test: {len(X_test)} fights")

    # Train
    console.print(f"[bold]Training XGBoost with {trials} Optuna trials...[/bold]")
    trainer = ModelTrainer(config)
    calibrated_model, best_params, importances = trainer.train(X_train, y_train)

    # Evaluate on test set
    console.print("[bold]Evaluating on test set...[/bold]")
    metrics = evaluate_model(calibrated_model, X_test, y_test)

    # Phase 15 (D-07, D-08): enforce hard accuracy gate BEFORE save_model.
    # T-15-02-01: a regressed model must NEVER reach disk. The gate
    # logic lives in the module-level helper ``_enforce_accuracy_gate``.
    _enforce_accuracy_gate(metrics, brier_max=brier_max, acc_min=acc_min)

    # Save model
    model_path = save_model(
        model=calibrated_model,
        metrics=metrics,
        feature_columns=FEATURE_COLUMNS,
        best_params=best_params,
        model_dir=config.model_dir,
        version=version,
        cutoff_date=config.cutoff_date,
        n_training_fights=len(X_train),
        n_test_fights=len(X_test),
    )
    console.print(f"[green]Model saved to {model_path}[/green]")

    # Display results
    _display_training_summary(metrics, importances, best_params, len(X_train), len(X_test))


@predict_app.command("evaluate")
def predict_evaluate(
    version: str = typer.Option("v2", help="Model version to evaluate (Phase 15: default v2)"),
    model_dir: str = typer.Option("models", help="Model directory"),
) -> None:
    """Evaluate trained model on held-out test set (D-09)."""
    from ufc_prediction.ml.persistence import load_model

    config = MLConfig()

    console.print(f"[bold]Loading model {version}...[/bold]")
    model = load_model(model_dir, version)
    _ = load_metadata(model_dir, version)  # Validate metadata exists

    console.print("[bold]Loading data...[/bold]")
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

    cutoff = date.fromisoformat(config.cutoff_date)
    division_medians = compute_division_medians(
        fighter_physicals,
        fight_records,
        cutoff,
    )

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
    )
    _, X_test, _, y_test = split_temporal(X, y, fight_dates, cutoff)

    console.print(f"[bold]Evaluating on {len(X_test)} test fights...[/bold]")
    metrics = evaluate_model(model, X_test, y_test)

    # Get feature importances from metadata or model
    feature_importances = _extract_importances(model)

    console.print(format_evaluation_report(metrics, feature_importances))

    # Display calibration curve table
    _display_calibration_table(metrics)


@predict_app.command("coverage")
def predict_coverage(
    threshold: float = typer.Option(
        0.70,
        "--threshold",
        help="Coverage threshold (Phase 15.1 D-07); first year >= this is the recommended lower bound.",
    ),
) -> None:
    """Per-year BFO odds-coverage report (Phase 15.1 D-06).

    Read-only diagnostic. Computes per-year totals + with-odds counts
    from fight_records and fight_odds, prints a Rich table marking
    years that meet the threshold, and surfaces the recommended
    train_lower_bound_date (first year reaching >= threshold).
    """
    from collections import defaultdict

    console.print("[bold]Loading data...[/bold]")
    session = SessionLocal()
    try:
        fight_records = load_fight_records(session)
        fight_odds = load_fight_odds(session)
    finally:
        session.close()

    fights_with_odds = {fid for (_fighter_id, fid) in fight_odds}
    by_year: dict[int, dict[str, int]] = defaultdict(lambda: {"total": 0, "with_odds": 0})
    for fight in fight_records:
        year = fight["event_date"].year
        by_year[year]["total"] += 1
        if fight["fight_id"] in fights_with_odds:
            by_year[year]["with_odds"] += 1

    table = Table(title="BFO Odds Coverage by Year (Phase 15.1 D-06)")
    table.add_column("Year", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("With Odds", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column(f">= {threshold:.0%}", justify="center")

    lower_bound_year: int | None = None
    for year in sorted(by_year):
        stats = by_year[year]
        cov = stats["with_odds"] / max(stats["total"], 1)
        meets = cov >= threshold
        if meets and lower_bound_year is None:
            lower_bound_year = year
        table.add_row(
            str(year),
            str(stats["total"]),
            str(stats["with_odds"]),
            f"{cov:.1%}",
            "[green]yes[/green]" if meets else "",
        )
    console.print(table)
    if lower_bound_year is not None:
        console.print(
            f"[green]Lower bound (D-06/D-07): "
            f"{lower_bound_year}-01-01 "
            f"(first year with coverage >= {threshold:.0%})[/green]"
        )
    else:
        console.print(
            f"[red]No year reaches {threshold:.1%} coverage "
            f"-- Path B (Plan-02) likely required[/red]"
        )


@predict_app.command("matchup")
def predict_matchup(
    fighter_a: str = typer.Argument(..., help="Fighter A name"),
    vs: str = typer.Argument(..., help="Literal 'vs'"),
    fighter_b: str = typer.Argument(..., help="Fighter B name"),
    version: str = typer.Option("v2", help="Model version"),
    model_dir: str = typer.Option("models", help="Model directory"),
    no_use_meta: bool = typer.Option(
        False,
        "--no-use-meta",
        help="Disable meta-learner blender; return base XGBoost prob (D-05(P19) kill switch).",
    ),
) -> None:
    """Predict fight outcome: ufc predict matchup 'Fighter A' vs 'Fighter B' (D-10)."""
    if vs.lower() != "vs":
        console.print(f"[red]Expected 'vs' between fighter names, got '{vs}'[/red]")
        raise typer.Exit(code=1)

    try:
        predictor = ModelPredictor(
            model_dir=model_dir,
            version=version,
            meta_dir=None if no_use_meta else "models/meta",
        )
    except FileNotFoundError:
        console.print("[red]No trained model found. Run 'ufc predict train' first.[/red]")
        raise typer.Exit(code=1) from None

    session = SessionLocal()
    try:
        result = predict_order_invariant(predictor, session, fighter_a, fighter_b)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e
    finally:
        session.close()

    if no_use_meta:
        # D-05(P19) — distinguish CLI override from naturally-absent artifact
        result["meta_skipped_reason"] = "cli_override"

    _display_prediction(result)


def _display_training_summary(
    metrics: dict,
    importances: dict,
    best_params: dict,
    n_train: int,
    n_test: int,
) -> None:
    """Display Rich training summary."""
    # Metrics table
    table = Table(title="Model Evaluation Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Brier Score", f"{metrics['brier_score']:.4f}")
    table.add_row("AUC-ROC", f"{metrics['auc_roc']:.4f}")
    table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
    table.add_row("Training Fights", str(n_train))
    table.add_row("Test Fights", str(n_test))
    console.print(table)

    # Feature importances table
    imp_table = Table(title="Top 10 Feature Importances (Gain)")
    imp_table.add_column("Feature", style="cyan")
    imp_table.add_column("Importance", justify="right")
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for name, score in sorted_feats[:10]:
        imp_table.add_row(name, f"{score:.4f}")
    console.print(imp_table)


def _display_calibration_table(metrics: dict) -> None:
    """Display calibration curve as a Rich table."""
    if "calibration_curve" not in metrics:
        return

    cal = metrics["calibration_curve"]
    table = Table(title="Calibration Curve")
    table.add_column("Predicted", justify="right")
    table.add_column("Observed", justify="right")
    for pred, obs in zip(
        cal["mean_predicted_value"],
        cal["fraction_of_positives"],
        strict=True,
    ):
        table.add_row(f"{pred:.4f}", f"{obs:.4f}")
    console.print(table)


def _display_prediction(result: dict) -> None:
    """Display prediction result as a Rich panel."""
    fighter_a = result["fighter_a"]
    fighter_b = result["fighter_b"]

    table = Table(show_header=True)
    table.add_column("", style="bold")
    table.add_column(fighter_a, justify="center", style="cyan")
    table.add_column(fighter_b, justify="center", style="magenta")

    table.add_row(
        "ML Model",
        f"{result['model_probability_a']:.1%}",
        f"{result['model_probability_b']:.1%}",
    )
    table.add_row(
        "Elo Rating",
        f"{result['elo_a']:.0f}",
        f"{result['elo_b']:.0f}",
    )
    table.add_row(
        "Elo Prob",
        f"{result['elo_probability_a']:.1%}",
        f"{result['elo_probability_b']:.1%}",
    )

    panel = Panel(table, title="Fight Prediction", border_style="green")
    console.print(panel)

    # Top contributing features
    if result.get("feature_importances"):
        imp_table = Table(title="Top Contributing Features")
        imp_table.add_column("Feature", style="cyan")
        imp_table.add_column("Importance", justify="right")
        for name, score in result["feature_importances"].items():
            imp_table.add_row(name, f"{score:.4f}")
        console.print(imp_table)


def _extract_importances(model) -> dict[str, float]:
    """Extract feature importances from a calibrated model."""
    try:
        base_estimator = model.calibrated_classifiers_[0].estimator.estimator
        raw = base_estimator.feature_importances_
        return dict(zip(FEATURE_COLUMNS, [float(v) for v in raw], strict=True))
    except (AttributeError, IndexError):
        return {col: 0.0 for col in FEATURE_COLUMNS}
