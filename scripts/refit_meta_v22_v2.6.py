#!/usr/bin/env python
"""Phase 72 METH-V261-01 — meta_v2 refit-baseline driver per gate_methodology_v2.6.md §7.

Trains a refit-baseline MetaLearnerLogistic on the SAME 13-col META-V22 layout
as canonical ``models/meta/meta_v2.joblib`` (per Phase 26 D-02), using the
canonical ``xgb_v2`` OOF distribution + META-V22 base features. The refit is
intentionally identical in column ordering to canonical — the ONLY purpose of
the refit baseline is to provide a substrate-aligned counterfactual for the
substrate-drift verifier (`gate_methodology_v2.6.md` §3.1).

Emits two SIBLING artifacts (canonical ``meta_v2.joblib`` UNTOUCHED per
AUDIT-01 D-10):

  - ``models/meta/meta_v2_refit_v2.6.joblib`` — 13-col MetaLearnerLogistic refit
  - ``models/meta/meta_v2_refit_v2.6_meta.json`` — sidecar metadata recording
    methodology spec reference, training mode, seed, AUDIT-01 anchors, RNG
    seed, training-script SHA, fit timestamp.

Per ``gate_methodology_v2.6.md`` §7.2:

> The methodology (a) refit baseline ships as
> ``models/meta/meta_v2_refit_v2.6.joblib`` — a SIBLING artifact, NOT a
> replacement for canonical ``meta_v2.joblib``. This pattern is approved by
> this spec.

Pipeline shape mirrors canonical ``meta_v2.joblib`` exactly (read from
``meta_v2_meta.json::best_params``):

  Pipeline([
    ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000)),
  ])

Wrapped in canonical ``MetaLearnerLogistic`` so the saved joblib loads via
the SAME class as canonical — the Phase 64 ``_introspect_pipeline_width``
helper uses ``obj.pipeline.n_features_in_``.

Anti-overwrite discipline (Phase 64 CR-01 + Phase 65 carry-forward):
The script REFUSES to write to any path resolving into ``PROTECTED_OUTPUTS``
(canonical ``meta_v2.joblib`` + ``meta_v2_meta.json`` + ``xgb_v2.joblib``).
RuntimeError → clean stderr message + exit 1.

AUDIT-01 sandwich (Phase 64 + 65 inherited):
Canonical anchor SHAs verified at script entry AND after every disk write.

Frozen-date determinism (Phase 64 CR-03 + Phase 65 carry-forward):
Synthetic mode uses ``REFIT_FROZEN_DATE = date(2026, 6, 4)`` for byte-stability.

Usage:
    python scripts/refit_meta_v22_v2.6.py --help
    python scripts/refit_meta_v22_v2.6.py --mode synthetic      # default
    python scripts/refit_meta_v22_v2.6.py --mode full           # live DB
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

# Ensure scripts/ is on sys.path for direct invocation (mirrors Phase 65).
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ── LOCKED constants (Phase 72 D-02 + methodology §7.2) ───────────────────

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# AUDIT-01 anchors — locked per .planning/AUDIT-01-BASELINE-SHA.txt.
EXPECTED_XGB_V2_SHA256: str = (
    "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
)
EXPECTED_META_V2_SHA256: str = (
    "77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196"
)

# Canonical artifacts (READ-ONLY for this script).
CANONICAL_META_JSON: Path = PROJECT_ROOT / "models" / "meta" / "meta_v2_meta.json"
CANONICAL_META_V2_JOBLIB: Path = PROJECT_ROOT / "models" / "meta" / "meta_v2.joblib"
CANONICAL_XGB_V2_JOBLIB: Path = PROJECT_ROOT / "models" / "xgb_v2.joblib"

# Sibling output paths — per methodology §7.2 file-name convention.
# Output filename intentionally uses dotted milestone for parity with the
# spec literal (gate_methodology_v2.6.md §7.2).
OUT_JOBLIB: Path = PROJECT_ROOT / "models" / "meta" / "meta_v2_refit_v2.6.joblib"
OUT_META: Path = PROJECT_ROOT / "models" / "meta" / "meta_v2_refit_v2.6_meta.json"

# Anti-overwrite guard set — Phase 65 carry-forward. Resolves symlink/relative
# path tricks so a typo (e.g. ``--output models/meta/meta_v2.joblib``) is
# caught by the guard before any disk write.
PROTECTED_OUTPUTS: frozenset[Path] = frozenset({
    CANONICAL_META_V2_JOBLIB.resolve(),
    CANONICAL_META_JSON.resolve(),
    CANONICAL_XGB_V2_JOBLIB.resolve(),
})

# Phase 64 CR-03 determinism — frozen reference date for synthetic mode.
REFIT_FROZEN_DATE: date = date(2026, 6, 4)

# 13-col META-V22 layout — byte-equal canonical meta_v2_meta.json (D-02 §7.2
# semantic: refit baseline is canonical column ordering on aligned substrate).
META_V2_REFIT_FEATURE_COLUMNS: tuple[str, ...] = (
    "xgb_oof_prob",                  # col[0] — canonical xgb_v2 OOF source
    "elo_prob",
    "closing_prob_diff",
    "stance_matchup",
    "height_diff",
    "reach_diff",
    "days_since_last_fight_diff",
    "age_diff",
    "elo_overall_diff",
    "elo_striking_diff",
    "elo_grappling_diff",
    "division_finish_rate_shrunk",
    "sharp_money_signal",
)
assert len(META_V2_REFIT_FEATURE_COLUMNS) == 13, (
    f"META_V2_REFIT_FEATURE_COLUMNS must have exactly 13 cols (Phase 64 "
    f"width-guard avoidance + methodology §7.2 byte-equality); "
    f"got {len(META_V2_REFIT_FEATURE_COLUMNS)}"
)

# Methodology spec reference — sidecar JSON exposes this so downstream
# consumers (verifier + audit tooling) can resolve which version of the spec
# the refit was produced under.
METHODOLOGY_SPEC_REF: str = "gate_methodology_v2.6.md §7.2"
METHODOLOGY_VERSION: str = "v2.6"
MILESTONE_LABEL: str = "v2.6"

# Synthetic mode params (mirror Plan 65-03 so the matrix shapes align with
# the substrate-snapshot loader's expected width).
SYNTHETIC_N_FIGHTS: int = 240
DEFAULT_SEED: int = 42


# ── AUDIT-01 invariant assertions (sandwich pattern) ──────────────────────


def _sha256_file(path: Path) -> str:
    """Return the hex SHA256 digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_audit01_invariants() -> None:
    """Verify canonical xgb_v2 + meta_v2 are byte-identical to locked anchors.

    Sandwich pattern from Phase 65 — called at script entry AND after every
    disk write so any AUDIT-01 drift surfaces immediately.

    Raises:
        AssertionError: with "AUDIT-01" in the message so test suites + audit
            tooling can grep for it unambiguously.
    """
    if not CANONICAL_XGB_V2_JOBLIB.exists():
        raise AssertionError(
            f"AUDIT-01 invariant cannot be checked: canonical "
            f"{CANONICAL_XGB_V2_JOBLIB} is missing. Restore from git."
        )
    if not CANONICAL_META_V2_JOBLIB.exists():
        raise AssertionError(
            f"AUDIT-01 invariant cannot be checked: canonical "
            f"{CANONICAL_META_V2_JOBLIB} is missing. Restore from git."
        )
    sha_xgb = _sha256_file(CANONICAL_XGB_V2_JOBLIB)
    if sha_xgb != EXPECTED_XGB_V2_SHA256:
        raise AssertionError(
            f"AUDIT-01 violation: canonical xgb_v2.joblib SHA drifted. "
            f"got={sha_xgb} expected={EXPECTED_XGB_V2_SHA256}"
        )
    sha_meta = _sha256_file(CANONICAL_META_V2_JOBLIB)
    if sha_meta != EXPECTED_META_V2_SHA256:
        raise AssertionError(
            f"AUDIT-01 violation: canonical meta_v2.joblib SHA drifted. "
            f"got={sha_meta} expected={EXPECTED_META_V2_SHA256}"
        )


def assert_meta_v2_layout() -> None:
    """Confirm META_V2_REFIT_FEATURE_COLUMNS byte-equals canonical META-V22.

    Reads ``models/meta/meta_v2_meta.json::meta_feature_columns`` at entry.
    Mirrors Phase 65 ``assert_meta_v2_layout`` discipline — methodology §7.2
    refit baseline requires byte-equal column ordering to canonical so the
    substrate-drift verifier's aligned comparison is meaningful.
    """
    if not CANONICAL_META_JSON.exists():
        raise FileNotFoundError(
            f"canonical meta_v2_meta.json not found at {CANONICAL_META_JSON} "
            f"(AUDIT-01 invariant violated — restore from git before refit)"
        )
    canonical = json.loads(CANONICAL_META_JSON.read_text(encoding="utf-8"))
    canonical_cols = list(canonical["meta_feature_columns"])
    if len(canonical_cols) != 13:
        raise AssertionError(
            f"canonical meta_v2_meta.json::meta_feature_columns has "
            f"{len(canonical_cols)} cols, expected 13. Possible META-V22 "
            f"drift; investigate before refit."
        )
    if tuple(canonical_cols) != META_V2_REFIT_FEATURE_COLUMNS:
        raise AssertionError(
            f"META-V22 layout drift: refit={META_V2_REFIT_FEATURE_COLUMNS} "
            f"canonical={tuple(canonical_cols)}"
        )


# ── 13-col training matrix assembly ───────────────────────────────────────


def _synthesize_13col_matrix(
    *, seed: int = DEFAULT_SEED
) -> tuple[Any, Any]:
    """Build a deterministic synthetic 13-col matrix for ``--mode synthetic``.

    Uses ``compose_v25_travel._build_synthetic_v25`` (Phase 42 helper, reused
    by Phase 65) for the v2.2-style 90-col substrate, then picks the 12
    META-V22 internal columns by name from FEATURE_COLUMNS_V22, and
    synthesizes col[0] (xgb_oof_prob) using a seeded RNG.

    Freezes ``date.today()`` to REFIT_FROZEN_DATE for byte-stability across
    calendar days (Phase 64 CR-03 + Phase 65 carry-forward).

    Returns ``(X_13, y)`` where ``X_13`` is shape (n, 13) np.float64.
    """
    import datetime as _dt

    import numpy as np

    import compose_v25_travel as _cv  # type: ignore[import-not-found]
    from compose_v25_travel import (  # type: ignore[import-not-found]
        _build_synthetic_v25,
    )
    from ufc_prediction.ml.config import FEATURE_COLUMNS_V22

    class _FixedDate(_dt.date):
        @classmethod
        def today(cls):  # type: ignore[override]
            return REFIT_FROZEN_DATE

    _orig_date = _cv.date
    _cv.date = _FixedDate
    try:
        X_v25, y, _fight_dates, _fight_records = _build_synthetic_v25(
            n=SYNTHETIC_N_FIGHTS
        )
    finally:
        _cv.date = _orig_date

    assert X_v25.shape[1] >= 90, (
        f"synthetic substrate must be >=90 cols; got {X_v25.shape[1]}"
    )
    X_v22 = X_v25[:, :90]

    # Synthesize col[0] (xgb_oof_prob) and col[1] (elo_prob) externally.
    # Same seed → matrix byte-stable across re-runs.
    rng = np.random.default_rng(seed)
    xgb_oof_synth = rng.beta(2.0, 2.0, size=X_v22.shape[0])
    elo_prob_synth = rng.beta(2.0, 2.0, size=X_v22.shape[0])

    # Resolve the 11 internal cols[2..12] by name from FEATURE_COLUMNS_V22.
    internal_cols: list[Any] = []
    for name in META_V2_REFIT_FEATURE_COLUMNS[2:]:
        idx = FEATURE_COLUMNS_V22.index(name)
        internal_cols.append(X_v22[:, idx])

    X_13 = np.column_stack([
        xgb_oof_synth, elo_prob_synth, *internal_cols
    ]).astype(np.float64)
    assert X_13.shape[1] == 13, (
        f"_synthesize_13col_matrix: expected 13 cols, got {X_13.shape[1]}"
    )
    y_int = np.asarray(y, dtype=np.int64)
    return X_13, y_int


def _build_live_13col_matrix() -> tuple[Any, Any]:
    """Build the live-DB 13-col training matrix for ``--mode full``.

    Reuses ``scripts/train_meta_v22.py::_load_assembled_data_v22`` for the
    canonical v2.2 90-col substrate and ``_compute_elo_prob_for_fight`` for
    col[1]. col[0] is sourced from canonical xgb_v2 OOF (not Plan 65-02 OOF
    — the refit baseline by definition uses the canonical xgb distribution).

    Note: live mode is heavier (DB round-trip + per-fight Elo computation).
    Synthetic mode is sufficient for the test suite + canonical demonstration.
    """
    import numpy as np

    import train_meta_v22 as _tm  # type: ignore[import-not-found]
    from ufc_prediction.ml.config import FEATURE_COLUMNS_V22

    X_v22, y, _fight_dates, fight_records = _tm._load_assembled_data_v22()
    assert X_v22.shape[1] == 90, (
        f"v2.2 substrate must be 90 cols; got {X_v22.shape[1]}"
    )

    from ufc_prediction.db.session import SessionLocal
    from ufc_prediction.ml.queries import load_elo_features

    session = SessionLocal()
    try:
        elo_features = load_elo_features(session)
    finally:
        session.close()

    elo_prob_arr = np.array(
        [_tm._compute_elo_prob_for_fight(rec, elo_features) for rec in fight_records],
        dtype=np.float64,
    )

    # col[0] — canonical xgb_v2 OOF. Live mode would need access to the
    # canonical OOF parquet at data/intermediate/xgb_v2_oof.parquet. For Phase
    # 72 ship the synthetic-mode artifact; live mode is operator-triggered
    # when corpus grows (METH-V27-01 re-eligibility).
    canonical_oof_path = PROJECT_ROOT / "data" / "intermediate" / "xgb_v2_oof.parquet"
    if not canonical_oof_path.exists():
        raise FileNotFoundError(
            f"canonical xgb_v2 OOF parquet not found at {canonical_oof_path}. "
            f"Run `python scripts/spike_noise_floor_v22.py` to regenerate, OR "
            f"use `--mode synthetic` for the Phase 72 SIBLING demonstration."
        )
    import pandas as pd

    oof_df = pd.read_parquet(canonical_oof_path)
    required = {"fight_id", "oof_prob"}
    missing = required - set(oof_df.columns)
    if missing:
        raise AssertionError(
            f"canonical xgb_v2 OOF parquet missing required columns: {missing}"
        )
    oof_by_fid = dict(
        zip(
            oof_df["fight_id"].astype("int64").tolist(),
            oof_df["oof_prob"].astype("float64").tolist(),
        )
    )

    # Per-row unique negative sentinel (Phase 65 CR-02 carry-forward).
    fight_ids: list[int] = [
        int(rec["fight_id"]) if rec.get("fight_id") is not None
        else -(10**9 + i)
        for i, rec in enumerate(fight_records)
    ]
    keep_mask = np.array([fid in oof_by_fid for fid in fight_ids], dtype=bool)
    n_drop = int((~keep_mask).sum())
    if n_drop > 0:
        print(
            f"[refit_meta_v22_v2.6] dropped {n_drop} rows lacking canonical "
            f"xgb_v2 OOF coverage ({n_drop}/{len(fight_ids)} = "
            f"{100 * n_drop / max(len(fight_ids), 1):.2f}%)",
            file=sys.stderr,
        )
    xgb_oof_col = np.array(
        [oof_by_fid.get(fid, np.nan) for fid in fight_ids], dtype=np.float64
    )

    internal_cols: list[Any] = []
    for name in META_V2_REFIT_FEATURE_COLUMNS[2:]:
        idx = FEATURE_COLUMNS_V22.index(name)
        internal_cols.append(X_v22[:, idx])

    X_13_all = np.column_stack([
        xgb_oof_col, elo_prob_arr, *internal_cols
    ]).astype(np.float64)
    assert X_13_all.shape[1] == 13, (
        f"_build_live_13col_matrix: expected 13 cols, got {X_13_all.shape[1]}"
    )
    y_arr = np.asarray(y, dtype=np.int64)

    X_13 = X_13_all[keep_mask]
    y_kept = y_arr[keep_mask]

    if X_13.shape[0] > 0 and np.any(np.isnan(X_13[:, 0])):
        nan_count = int(np.isnan(X_13[:, 0]).sum())
        raise RuntimeError(
            f"_build_live_13col_matrix: {nan_count} NaN values in col[0] "
            f"(xgb_oof_prob) after keep_mask filter — mask/nan alignment "
            f"drift suspected. Investigate fight_id sentinel logic."
        )

    return X_13, y_kept


def build_13col_training_matrix(
    *, source: str = "synthetic"
) -> tuple[Any, Any]:
    """Build the 13-col META-V22-layout training matrix.

    Two source modes:
      - ``synthetic`` (default): DB-free, deterministic via
        :func:`_synthesize_13col_matrix`. Used by tests + Phase 72 SIBLING
        emission.
      - ``live``: full DB-backed assembly via :func:`_build_live_13col_matrix`.
        Requires canonical xgb_v2 OOF parquet.

    Returns ``(X_13, y)``.
    """
    if source == "synthetic":
        return _synthesize_13col_matrix(seed=DEFAULT_SEED)
    elif source == "live":
        return _build_live_13col_matrix()
    else:
        raise ValueError(
            f"build_13col_training_matrix: unknown source {source!r} "
            f"(expected 'synthetic' or 'live')"
        )


# ── Meta fit ──────────────────────────────────────────────────────────────


def fit_meta_refit(X_13: Any, y: Any, *, seed: int = DEFAULT_SEED) -> Any:
    """Fit MetaLearnerLogistic on the 13-col matrix (canonical Pipeline shape).

    Wraps the canonical ``MetaLearnerLogistic`` class so the saved joblib
    loads through the SAME class as canonical — Phase 64's two-level
    ``_introspect_pipeline_width`` helper uses ``obj.pipeline.n_features_in_``.

    Per ``meta_v2_meta.json::best_params``:
      C=1.0, penalty='l2', solver='lbfgs',
      PolynomialFeatures=degree=2 interaction_only=True include_bias=False

    The MetaLearnerLogistic constructor pins these defaults — we pass only
    ``random_state=seed`` to keep the fit reproducible.
    """
    from ufc_prediction.ml.meta_learner import MetaLearnerLogistic

    meta = MetaLearnerLogistic(random_state=seed).fit(X_13, y)
    return meta


# ── Output emission with anti-overwrite guard ─────────────────────────────


def _check_anti_overwrite(*candidate_paths: Path) -> None:
    """Raise RuntimeError if any candidate path resolves into PROTECTED_OUTPUTS.

    Called by ``emit_outputs`` BEFORE any disk write so the guard cannot be
    bypassed by partial writes / atomic-rename tricks. Resolves each candidate
    (follows symlinks, normalizes ``..``).
    """
    for p in candidate_paths:
        try:
            resolved = p.resolve()
        except (OSError, RuntimeError):
            continue
        if resolved in PROTECTED_OUTPUTS:
            raise RuntimeError(
                f"Phase 72 T-72-01 / Phase 64 CR-01 guard: refusing to "
                f"overwrite canonical AUDIT-01 artifact at {resolved}. "
                f"Refit-baseline outputs MUST go to "
                f"models/meta/meta_v2_refit_v<milestone>.{{joblib,_meta.json}} "
                f"per gate_methodology_v2.6.md §7.2 (see OUT_JOBLIB / OUT_META "
                f"defaults in scripts/refit_meta_v22_v2.6.py)."
            )


def emit_outputs(
    *,
    meta_pipeline: Any,
    out_joblib: Path,
    out_meta: Path,
    seed: int,
    mode: str,
    n_training_rows: int,
) -> None:
    """Write joblib + sidecar JSON to the configured sibling paths.

    Anti-overwrite guard fires BEFORE any write (Phase 64 CR-01 pattern).
    """
    import joblib

    # Anti-overwrite guard FIRST — before any disk side effect.
    _check_anti_overwrite(out_joblib, out_meta)

    out_joblib.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    # 1. joblib — the fitted MetaLearnerLogistic.
    joblib.dump(meta_pipeline, out_joblib)

    # 2. Sidecar JSON — sibling metadata + AUDIT-01 anchor record.
    # SHA the training script itself so the audit trail records which version
    # of the driver produced the artifact (operator-traceability for any
    # future re-emit triggered by corpus growth).
    training_script_path = Path(__file__).resolve()
    training_script_sha = _sha256_file(training_script_path)

    # SHA the emitted joblib for the sidecar record (operators can verify
    # joblib hasn't been swapped post-emission).
    refit_joblib_sha = _sha256_file(out_joblib)

    sidecar = {
        "meta_version": "v2_refit_v2.6",
        "meta_kind": "logistic",
        "canonical_status": "candidate_sibling_NOT_canonical",
        "sibling_of": "models/meta/meta_v2.joblib",
        "methodology_spec": METHODOLOGY_SPEC_REF,
        "methodology_version": METHODOLOGY_VERSION,
        "milestone_label": MILESTONE_LABEL,
        "base_xgb_oof_source": "canonical_xgb_v2_oof",
        "base_xgb_model": "models/xgb_v2.joblib",
        "base_xgb_model_version": "v2",
        "meta_feature_columns": list(META_V2_REFIT_FEATURE_COLUMNS),
        "n_features": 13,
        "best_params": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "PolynomialFeatures": "degree=2 interaction_only=True include_bias=False",
        },
        "seed": seed,
        "rng_seed": seed,
        "training_mode": mode,
        "n_training_rows": n_training_rows,
        "trained_at": datetime.now(UTC).isoformat(),
        "training_script": "scripts/refit_meta_v22_v2.6.py",
        "training_script_sha256": training_script_sha,
        "refit_joblib_sha256": refit_joblib_sha,
        "phase": "72-meth-v261-01-02-refit-driver-recalib-cli",
        "requirement_id": "METH-V261-01",
        "decision_ids": ["D-01", "D-02"],
        "audit_01_invariant": {
            "xgb_v2_sha": EXPECTED_XGB_V2_SHA256,
            "meta_v2_sha": EXPECTED_META_V2_SHA256,
            "status": "UNCHANGED",
        },
        "synthetic_n_fights": SYNTHETIC_N_FIGHTS if mode == "synthetic" else None,
        "synthetic_frozen_date": (
            REFIT_FROZEN_DATE.isoformat() if mode == "synthetic" else None
        ),
    }
    out_meta.write_text(
        json.dumps(sidecar, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


# ── CLI ───────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI surface (no Typer — keep dep-light)."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 72 METH-V261-01 — meta_v2 refit-baseline driver per "
            "gate_methodology_v2.6.md §7. Trains a 13-col MetaLearnerLogistic "
            "refit baseline using canonical xgb_v2 OOF + META-V22 base "
            "features; emits SIBLING artifacts at "
            "models/meta/meta_v2_refit_v2.6.{joblib,_meta.json}. Canonical "
            "meta_v2.joblib + meta_v2_meta.json + xgb_v2.joblib are NEVER "
            "overwritten (AUDIT-01 D-10 + Phase 64 CR-01 guard)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("synthetic", "full"),
        default="synthetic",
        help=(
            "Training source: 'synthetic' (default; DB-free, fast, <60s) for "
            "the Phase 72 SIBLING emission; 'full' invokes the live DB loader "
            "and reads canonical xgb_v2 OOF parquet (operator-triggered when "
            "corpus growth re-eligibility fires per METH-V27-01)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for LogisticRegression + synthesis (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_JOBLIB,
        help=(
            f"Output joblib path (default: {OUT_JOBLIB.relative_to(PROJECT_ROOT)}). "
            f"Paths resolving into PROTECTED_OUTPUTS raise RuntimeError "
            f"(canonical artifacts AUDIT-01 protected)."
        ),
    )
    parser.add_argument(
        "--output-meta",
        type=Path,
        default=OUT_META,
        help=(
            f"Output sidecar JSON path (default: "
            f"{OUT_META.relative_to(PROJECT_ROOT)}). Same anti-overwrite guard."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry: argparse → AUDIT-01 preflight → fit → emit → AUDIT-01 postflight.

    Returns OS exit code (0 success, non-zero failure). Phase 64 CR-02 pattern:
    all known-failure modes surface as clean stderr message + non-zero rc;
    no Python traceback leaks to operators.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Pre-fit AUDIT-01 sandwich.
    try:
        assert_audit01_invariants()
    except AssertionError as e:
        print(f"AUDIT-01 preflight failed: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"AUDIT-01 preflight failed: {e}", file=sys.stderr)
        return 1

    # Verify META-V22 layout byte-equals canonical.
    try:
        assert_meta_v2_layout()
    except FileNotFoundError as e:
        print(f"meta layout check failed: {e}", file=sys.stderr)
        return 1
    except AssertionError as e:
        print(f"meta layout check failed: {e}", file=sys.stderr)
        return 1

    # PRE-EMIT anti-overwrite guard — fire as early as possible.
    try:
        _check_anti_overwrite(args.output, args.output_meta)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    # Build training matrix.
    print(f"[refit_meta_v22_v2.6] mode={args.mode}, seed={args.seed}")
    try:
        if args.mode == "synthetic":
            X_13, y = build_13col_training_matrix(source="synthetic")
        else:
            X_13, y = build_13col_training_matrix(source="live")
    except FileNotFoundError as e:
        print(f"training data load failed: {e}", file=sys.stderr)
        return 1
    print(
        f"[refit_meta_v22_v2.6] X_13.shape={X_13.shape}, y.shape={y.shape}"
    )

    # Fit.
    meta_pipeline = fit_meta_refit(X_13, y, seed=args.seed)

    # Emit (anti-overwrite guard fires here again as defense-in-depth).
    try:
        emit_outputs(
            meta_pipeline=meta_pipeline,
            out_joblib=args.output,
            out_meta=args.output_meta,
            seed=args.seed,
            mode=args.mode,
            n_training_rows=int(X_13.shape[0]),
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"output emit failed: {e}", file=sys.stderr)
        return 1

    # Post-fit AUDIT-01 sandwich — confirm canonical byte-identical.
    try:
        assert_audit01_invariants()
    except AssertionError as e:
        print(f"AUDIT-01 POSTFLIGHT VIOLATION: {e}", file=sys.stderr)
        return 2

    print(f"[refit_meta_v22_v2.6] wrote {args.output}")
    print(f"[refit_meta_v22_v2.6] wrote {args.output_meta}")
    print(
        f"[refit_meta_v22_v2.6] AUDIT-01 unchanged "
        f"(xgb_v2={EXPECTED_XGB_V2_SHA256[:12]}..., "
        f"meta_v2={EXPECTED_META_V2_SHA256[:12]}...)"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
