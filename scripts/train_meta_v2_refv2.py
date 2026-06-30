#!/usr/bin/env python
"""Phase 65 Plan 65-03 (FEAT-V261-02) — meta_v2_refv2 train script.

Trains a logistic-regression meta candidate over the same 13-col META-V22
layout as canonical ``models/meta/meta_v2.joblib`` (per Phase 26 D-02). The
ONLY difference from canonical is col[0]: ``xgb_v2_refv2_oof`` (Plan 65-02's
candidate OOF) instead of canonical ``xgb_oof_prob`` (Phase 26 canonical OOF).
Cols[1..12] mirror canonical META-V22 byte-for-byte:

  elo_prob, closing_prob_diff, stance_matchup, height_diff, reach_diff,
  days_since_last_fight_diff, age_diff, elo_overall_diff, elo_striking_diff,
  elo_grappling_diff, division_finish_rate_shrunk, sharp_money_signal

Same Pipeline shape as canonical ``meta_v2.joblib`` (read from
``meta_v2_meta.json::best_params``):

  Pipeline([
    ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000)),
  ])

Wrapped in the canonical ``MetaLearnerLogistic`` class so the saved joblib
loads via the SAME class as canonical ``meta_v2.joblib`` — the Phase 64
``_introspect_pipeline_width`` helper uses the two-level
``obj.pipeline.n_features_in_`` introspection.

Emits two SIBLING artifacts (canonical ``meta_v2.joblib`` UNTOUCHED per
AUDIT-01 D-10):

  - ``models/meta/meta_v2_refv2.joblib`` — 13-col MetaLearnerLogistic candidate
  - ``models/meta/meta_v2_refv2_meta.json`` — sidecar metadata with
    ``canonical_status="candidate_sibling_NOT_canonical"``, ``sibling_of``,
    ``base_xgb_oof_source``, ``base_xgb_model``

Per Phase 65 D-03a + CONTEXT line 73: keeping the candidate 13-wide AVOIDS
Phase 64's width-mismatch guard. The substrate-drift confound Plan 65-05
expects is at col[0] OOF distribution shift, NOT at width.

Anti-overwrite discipline (Phase 64 CR-01 + Plan 65-02 carry-forward):
The script REFUSES to write to any path resolving into ``PROTECTED_OUTPUTS``
(canonical ``meta_v2.joblib`` + ``meta_v2_meta.json`` + ``xgb_v2.joblib``).
RuntimeError converted to a clean stderr message + exit-1; no traceback.

Frozen-date determinism (Phase 64 CR-03 + Plan 65-02 carry-forward):
Synthetic mode uses ``META_REFV2_FROZEN_DATE = date(2026, 6, 4)`` to
monkeypatch ``date.today()`` callers in the upstream ``compose_v25_travel``
helper so the synthetic eval fixture is byte-stable across re-runs on
different calendar days.

FileNotFoundError handling (Phase 64 CR-02 + Plan 65-02 carry-forward):
Missing canonical meta JSON, missing canonical joblib, or missing Plan 65-02
OOF parquet (``--mode full``) raise a clean stderr message + exit-1; no
Python traceback leaks to operators.

Usage:
    python scripts/train_meta_v2_refv2.py --help
    python scripts/train_meta_v2_refv2.py --mode synthetic     # fast, deterministic
    python scripts/train_meta_v2_refv2.py --mode full \\
        --xgb-refv2-oof data/intermediate/xgb_v2_refv2_oof.parquet
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

# Ensure the scripts/ directory is on sys.path so we can import
# ``compose_v25_travel`` helpers when this script is invoked directly
# (not as a package). Mirrors the Phase 64 builder + Plan 65-02 pattern.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ── LOCKED constants (Phase 65 D-03a + D-10 AUDIT-01) ─────────────────────

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# AUDIT-01 anchors — locked per .planning/AUDIT-01-BASELINE-SHA.txt.
EXPECTED_XGB_V2_SHA256: str = "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
EXPECTED_META_V2_SHA256: str = "77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196"

# Canonical artifacts (READ-ONLY for this script).
CANONICAL_META_JSON: Path = PROJECT_ROOT / "models" / "meta" / "meta_v2_meta.json"
CANONICAL_META_V2_JOBLIB: Path = PROJECT_ROOT / "models" / "meta" / "meta_v2.joblib"
CANONICAL_XGB_V2_JOBLIB: Path = PROJECT_ROOT / "models" / "xgb_v2.joblib"

# Plan 65-02 base artifacts (READ-ONLY).
XGB_REFV2_OOF_PATH: Path = PROJECT_ROOT / "data" / "intermediate" / "xgb_v2_refv2_oof.parquet"
XGB_REFV2_JOBLIB: Path = PROJECT_ROOT / "models" / "xgb_v2_refv2.joblib"

# Sibling output paths — DEFAULT only; CLI ``--output*`` overrides allowed
# everywhere EXCEPT into PROTECTED_OUTPUTS.
OUT_JOBLIB: Path = PROJECT_ROOT / "models" / "meta" / "meta_v2_refv2.joblib"
OUT_META: Path = PROJECT_ROOT / "models" / "meta" / "meta_v2_refv2_meta.json"

# Anti-overwrite guard set — Phase 64 CR-01 + Plan 65-02 carry-forward;
# T-65-10 mitigation in this phase. Resolved paths so a symlinked or
# relative-style operator argv cannot bypass the guard. Includes the
# canonical xgb joblib too: any operator typo (e.g.
# ``--output models/xgb_v2.joblib``) is caught by the same guard.
PROTECTED_OUTPUTS: frozenset[Path] = frozenset(
    {
        CANONICAL_META_V2_JOBLIB.resolve(),
        CANONICAL_META_JSON.resolve(),
        CANONICAL_XGB_V2_JOBLIB.resolve(),
    }
)

# Phase 64 CR-03 determinism — frozen reference date for any synthetic-mode
# ``date.today()`` callers in the upstream compose_v25_travel helper.
META_REFV2_FROZEN_DATE: date = date(2026, 6, 4)

# Plan 65-03 D-03a 13-col META-V22 layout — col[0] is the candidate OOF
# (NOT canonical xgb_v2 OOF); cols[1..12] mirror canonical META-V22 byte-for-
# byte (asserted at runtime against meta_v2_meta.json::meta_feature_columns).
META_V2_REFV2_FEATURE_COLUMNS: tuple[str, ...] = (
    "xgb_v2_refv2_oof",  # col[0] — Plan 65-02 OOF source
    "elo_prob",  # cols[1..12] — mirror canonical META-V22
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
assert len(META_V2_REFV2_FEATURE_COLUMNS) == 13, (
    f"META_V2_REFV2_FEATURE_COLUMNS must have exactly 13 cols "
    f"(D-03a + Phase 64 width-guard avoidance); "
    f"got {len(META_V2_REFV2_FEATURE_COLUMNS)}"
)

# Synthetic mode params — same SYNTHETIC_N_FIGHTS as Plan 65-02 so the
# 13-col matrix shapes match downstream.
SYNTHETIC_N_FIGHTS: int = 240
DEFAULT_SEED: int = 42


# ── AUDIT-01 invariant assertions (sandwich pattern) ──────────────────────


def _sha256_file(path: Path) -> str:
    """Return the hex SHA256 digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_audit01_invariants() -> None:
    """Verify canonical ``xgb_v2.joblib`` + ``meta_v2.joblib`` are byte-identical.

    Sandwich pattern from Plan 65-02 ``scripts/retrain_xgb_v2_refv2.py`` —
    called at script entry AND after every disk write so any AUDIT-01 drift
    surfaces immediately.

    Raises:
        AssertionError: if either canonical SHA does not match the locked
            constant. Message contains the literal string ``AUDIT-01`` so
            operators (and the test suite) can grep for it unambiguously.
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
    """Confirm cols[1..12] of our 13-col layout mirror canonical META-V22.

    Reads ``models/meta/meta_v2_meta.json::meta_feature_columns`` at script
    entry. ANY drift in canonical ordering produces a loud failure with a
    diff — mitigates T-65-13 (canonical-meta-JSON corruption silently
    producing drifted cols[1..12]).

    Wrapped in try/except FileNotFoundError per Phase 64 CR-02 — caller
    converts the FileNotFoundError to a clean stderr message + non-zero rc.
    """
    if not CANONICAL_META_JSON.exists():
        raise FileNotFoundError(
            f"canonical meta_v2_meta.json not found at {CANONICAL_META_JSON} "
            f"(AUDIT-01 invariant violated — restore from git before train)"
        )
    canonical = json.loads(CANONICAL_META_JSON.read_text(encoding="utf-8"))
    canonical_cols = list(canonical["meta_feature_columns"])
    if len(canonical_cols) != 13:
        raise AssertionError(
            f"canonical meta_v2_meta.json::meta_feature_columns has "
            f"{len(canonical_cols)} cols, expected 13. Possible META-V22 "
            f"drift; investigate before training."
        )
    expected_tail = list(canonical_cols[1:])
    actual_tail = list(META_V2_REFV2_FEATURE_COLUMNS[1:])
    if actual_tail != expected_tail:
        raise AssertionError(
            f"META-V22 cols[1..12] drift: refv2={actual_tail} canonical={expected_tail}"
        )


# ── Plan 65-02 OOF parquet loader ─────────────────────────────────────────


def load_xgb_refv2_oof(path: Path) -> "Any":
    """Read Plan 65-02 OOF parquet from disk into a DataFrame.

    Schema (per Plan 65-02 SUMMARY): ``{fight_id int64, oof_prob float64,
    event_date object}``. Plan 65-04 substrate builder + Plan 65-03 col[0]
    consume this artifact.

    Raises FileNotFoundError if the parquet is missing — the caller (main)
    converts this to a clean stderr exit per Phase 64 CR-02.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Plan 65-02 OOF parquet not found at {path}. The parquet is "
            f"gitignored (regeneratable). Run "
            f"`python scripts/retrain_xgb_v2_refv2.py --full` to recreate it."
        )
    import pandas as pd

    df = pd.read_parquet(path)
    required = {"fight_id", "oof_prob", "event_date"}
    missing = required - set(df.columns)
    if missing:
        raise AssertionError(
            f"Plan 65-02 OOF parquet at {path} missing required columns: "
            f"{missing}. Schema drift suspected."
        )
    return df


# ── 13-col training matrix assembly ───────────────────────────────────────


def _synthesize_13col_matrix(*, seed: int = DEFAULT_SEED) -> tuple[Any, Any]:
    """Build a deterministic synthetic 13-col training matrix for ``--mode synthetic``.

    Uses ``compose_v25_travel._build_synthetic_v25`` (Phase 42-shipped
    helper, reused by Plan 65-02) for the v2.2-style 90-col substrate, then
    picks the 12 META-V22-internal columns by name from FEATURE_COLUMNS_V22,
    synthesizes a candidate-OOF column at col[0] using a seeded RNG (same
    distribution shape as the Plan 65-02 OOF parquet).

    Freezes ``date.today()`` to META_REFV2_FROZEN_DATE for byte-stability
    across calendar days (Phase 64 CR-03 + Plan 65-02 carry-forward).

    Returns ``(X_13, y)`` where ``X_13`` has shape (n, 13) np.float64.
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
            return META_REFV2_FROZEN_DATE

    _orig_date = _cv.date
    _cv.date = _FixedDate
    try:
        X_v25, y, _fight_dates, _fight_records = _build_synthetic_v25(n=SYNTHETIC_N_FIGHTS)
    finally:
        _cv.date = _orig_date

    # Plan 65-02 ships a (n, 92) v2.5-travel matrix; the first 90 cols are
    # the v2.2 substrate per FEATURE_COLUMNS_V22 ordering. We pick 12 of
    # those 90 cols by name (the canonical META-V22 cols[2..12] are all in
    # FEATURE_COLUMNS_V22; col[1] elo_prob is external — synthesized below).
    assert X_v25.shape[1] >= 90, f"synthetic substrate must be >=90 cols; got {X_v25.shape[1]}"
    X_v22 = X_v25[:, :90]

    # Synthesize col[0] (candidate OOF) and col[1] (elo_prob) externally.
    # Same seed makes the matrix byte-stable across re-runs (determinism gate).
    rng = np.random.default_rng(seed)
    # Bound to (0, 1) so they look like probabilities — beta(2, 2) gives a
    # mean of 0.5 and stdev ~0.22, matching the Plan 65-02 OOF
    # distribution (mean 0.502, std 0.187).
    xgb_refv2_oof_synth = rng.beta(2.0, 2.0, size=X_v22.shape[0])
    elo_prob_synth = rng.beta(2.0, 2.0, size=X_v22.shape[0])

    # Resolve the 11 internal cols[2..12] by name.
    internal_cols: list[Any] = []
    for name in META_V2_REFV2_FEATURE_COLUMNS[2:]:
        idx = FEATURE_COLUMNS_V22.index(name)
        internal_cols.append(X_v22[:, idx])

    X_13 = np.column_stack([xgb_refv2_oof_synth, elo_prob_synth, *internal_cols]).astype(np.float64)
    assert X_13.shape[1] == 13, f"_synthesize_13col_matrix: expected 13 cols, got {X_13.shape[1]}"
    y_int = np.asarray(y, dtype=np.int64)
    return X_13, y_int


def _build_live_13col_matrix(*, xgb_refv2_oof_df: Any) -> tuple[Any, Any]:
    """Build the live-DB 13-col training matrix for ``--mode full``.

    Reuses ``scripts/train_meta_v22.py::_load_assembled_data_v22`` for the
    v2.2 90-col substrate and ``_compute_elo_prob_for_fight`` for the
    elo_prob column. Overrides col[0] with Plan 65-02's candidate-OOF
    column joined on ``fight_id``.

    Note: live mode is heavier (DB round-trip + per-fight Elo computation)
    and is intended for the actual sibling-artifact emission. Synthetic mode
    is sufficient for the test suite.

    Returns ``(X_13, y)`` aligned with the rows present in the OOF parquet.
    """
    import numpy as np

    import train_meta_v22 as _tm  # type: ignore[import-not-found]
    from ufc_prediction.ml.config import FEATURE_COLUMNS_V22

    # 1. v2.2 90-col matrix + fight_records.
    X_v22, y, fight_dates, fight_records = _tm._load_assembled_data_v22()
    assert X_v22.shape[1] == 90, f"v2.2 substrate must be 90 cols; got {X_v22.shape[1]}"

    # 2. Elo P(A wins) per fight (canonical helper).
    from ufc_prediction.ml.queries import load_elo_features
    from ufc_prediction.db.session import SessionLocal

    session = SessionLocal()
    try:
        elo_features = load_elo_features(session)
    finally:
        session.close()

    elo_prob_arr = np.array(
        [_tm._compute_elo_prob_for_fight(rec, elo_features) for rec in fight_records],
        dtype=np.float64,
    )

    # 3. Join Plan 65-02 OOF on fight_id. Drop rows not in the OOF parquet
    # (the OOF was trained on the same ~8473 corpus so coverage should be
    # near-complete; surface the drop count to stderr if non-zero).
    oof_by_fid = dict(
        zip(
            xgb_refv2_oof_df["fight_id"].astype("int64").tolist(),
            xgb_refv2_oof_df["oof_prob"].astype("float64").tolist(),
        )
    )

    # Phase 65 CR-02 fix: per-row unique negative sentinel for records that
    # lack a fight_id key. The old fallback was a constant -1 — if by
    # database accident the OOF parquet ever contained fight_id == -1 the
    # missing-id record would silently absorb that fight's OOF probability
    # (a silent data corruption). Per-row unique negatives (-(10**9 + i))
    # are guaranteed not to collide with any real positive fight_id AND
    # different missing-id rows cannot collide with each other in
    # oof_by_fid.get(...). The keep_mask construction below still drops
    # these rows (no real fight_id ever equals the synthesized sentinel).
    fight_ids: list[int] = [
        int(rec["fight_id"]) if rec.get("fight_id") is not None else -(10**9 + i)
        for i, rec in enumerate(fight_records)
    ]
    keep_mask = np.array([fid in oof_by_fid for fid in fight_ids], dtype=bool)
    n_drop = int((~keep_mask).sum())
    if n_drop > 0:
        print(
            f"[train_meta_v2_refv2] dropped {n_drop} rows lacking Plan 65-02 "
            f"OOF coverage ({n_drop}/{len(fight_ids)} = "
            f"{100 * n_drop / max(len(fight_ids), 1):.2f}%)",
            file=sys.stderr,
        )
    xgb_refv2_col = np.array([oof_by_fid.get(fid, np.nan) for fid in fight_ids], dtype=np.float64)

    # 4. Pull the 11 META-V22 internal cols by name (cols[2..12]).
    internal_cols: list[Any] = []
    for name in META_V2_REFV2_FEATURE_COLUMNS[2:]:
        idx = FEATURE_COLUMNS_V22.index(name)
        internal_cols.append(X_v22[:, idx])

    X_13_all = np.column_stack([xgb_refv2_col, elo_prob_arr, *internal_cols]).astype(np.float64)
    assert X_13_all.shape[1] == 13, (
        f"_build_live_13col_matrix: expected 13 cols, got {X_13_all.shape[1]}"
    )
    y_arr = np.asarray(y, dtype=np.int64)

    # Apply keep_mask to align with the OOF parquet.
    X_13 = X_13_all[keep_mask]
    y_kept = y_arr[keep_mask]

    # Phase 65 CR-02 fix: defensive NaN guard. By construction every row
    # in X_13 maps to an OOF parquet entry (keep_mask filters on the same
    # oof_by_fid containment check), so col[0] (xgb_v2_refv2_oof) should
    # never contain NaN after the mask. If this ever fires it indicates a
    # refactor has drifted the mask/nan alignment — silent NaN coefs from
    # the downstream LogisticRegression fit would otherwise corrupt the
    # candidate sibling. Catch it loudly here instead.
    if X_13.shape[0] > 0 and np.any(np.isnan(X_13[:, 0])):
        nan_count = int(np.isnan(X_13[:, 0]).sum())
        raise RuntimeError(
            f"_build_live_13col_matrix: {nan_count} NaN values in col[0] "
            f"(xgb_v2_refv2_oof) after keep_mask filter — mask/nan alignment "
            f"drift suspected. Every row in X_13 should map to an OOF parquet "
            f"entry. Check that fight_id sentinel logic + oof_by_fid keys "
            f"have not diverged."
        )

    return X_13, y_kept


def build_13col_training_matrix(
    *, source: str = "synthetic", xgb_refv2_oof_df: Any = None
) -> tuple[Any, Any]:
    """Build the 13-col META-V22-layout training matrix.

    Two source modes:
      - ``synthetic`` (default): DB-free, deterministic via
        :func:`_synthesize_13col_matrix`. Used by the unit test suite +
        ``--mode synthetic`` CLI.
      - ``live``: full DB-backed assembly via :func:`_build_live_13col_matrix`.
        Requires the Plan 65-02 OOF parquet (``xgb_refv2_oof_df`` argument).

    Returns ``(X_13, y)``.
    """
    if source == "synthetic":
        return _synthesize_13col_matrix(seed=DEFAULT_SEED)
    elif source == "live":
        if xgb_refv2_oof_df is None:
            raise ValueError(
                "build_13col_training_matrix(source='live') requires "
                "xgb_refv2_oof_df from load_xgb_refv2_oof(...)"
            )
        return _build_live_13col_matrix(xgb_refv2_oof_df=xgb_refv2_oof_df)
    else:
        raise ValueError(
            f"build_13col_training_matrix: unknown source {source!r} "
            f"(expected 'synthetic' or 'live')"
        )


# ── Meta fit ──────────────────────────────────────────────────────────────


def fit_meta_refv2(X_13: Any, y: Any, *, seed: int = DEFAULT_SEED) -> Any:
    """Fit MetaLearnerLogistic on the 13-col matrix.

    Wraps the canonical ``ufc_prediction.ml.meta_learner.MetaLearnerLogistic``
    class so the saved joblib loads through the SAME class as canonical
    ``meta_v2.joblib`` — this is what Phase 64's two-level
    ``_introspect_pipeline_width`` helper expects (``obj.pipeline.n_features_in_``).

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
    bypassed by partial writes / atomic-rename tricks. The check resolves
    each candidate (follows symlinks, normalizes ``..``).
    """
    for p in candidate_paths:
        try:
            resolved = p.resolve()
        except (OSError, RuntimeError):
            # If the path cannot be resolved, treat it as a non-protected
            # location (the subsequent write will surface the real OS error).
            continue
        if resolved in PROTECTED_OUTPUTS:
            raise RuntimeError(
                f"Phase 65 T-65-10 / Phase 64 CR-01 guard: refusing to "
                f"overwrite canonical AUDIT-01 artifact at {resolved}. "
                f"Sibling outputs MUST go to *_refv2.* paths (see "
                f"OUT_JOBLIB / OUT_META defaults in "
                f"scripts/train_meta_v2_refv2.py)."
            )


def emit_outputs(
    *,
    meta_pipeline: Any,
    out_joblib: Path,
    out_meta: Path,
    seed: int,
    mode: str,
) -> None:
    """Write joblib + sidecar JSON to the configured sibling paths.

    Anti-overwrite guard fires BEFORE any write (Phase 64 CR-01 pattern).
    """
    import joblib

    # Anti-overwrite guard FIRST — before any disk side effect.
    _check_anti_overwrite(out_joblib, out_meta)

    # Ensure output dirs exist.
    out_joblib.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    # 1. joblib — the fitted MetaLearnerLogistic.
    joblib.dump(meta_pipeline, out_joblib)

    # 2. Sidecar JSON — sibling metadata + AUDIT-01 anchor record.
    sidecar = {
        "meta_version": "v2_refv2",
        "meta_kind": "logistic",
        "canonical_status": "candidate_sibling_NOT_canonical",
        "sibling_of": "models/meta/meta_v2.joblib",
        "base_xgb_oof_source": "data/intermediate/xgb_v2_refv2_oof.parquet",
        "base_xgb_model": "models/xgb_v2_refv2.joblib",
        "base_xgb_model_version": "v2_refv2",
        "meta_feature_columns": list(META_V2_REFV2_FEATURE_COLUMNS),
        "n_features": 13,
        "best_params": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "PolynomialFeatures": "degree=2 interaction_only=True include_bias=False",
        },
        "seed": seed,
        "training_mode": mode,
        "trained_at": datetime.now(UTC).isoformat(),
        "phase": "65-ref-feat-v261-02-implementation-ml-spike-verifier-run",
        "decision_ids": ["D-03a", "D-10"],
        "audit_01_invariant": {
            "xgb_v2_sha": EXPECTED_XGB_V2_SHA256,
            "meta_v2_sha": EXPECTED_META_V2_SHA256,
            "status": "UNCHANGED",
        },
        "synthetic_n_fights": SYNTHETIC_N_FIGHTS if mode == "synthetic" else None,
        "synthetic_frozen_date": (
            META_REFV2_FROZEN_DATE.isoformat() if mode == "synthetic" else None
        ),
    }
    out_meta.write_text(json.dumps(sidecar, indent=2, sort_keys=False) + "\n", encoding="utf-8")


# ── CLI ───────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI surface (kept dep-light — no Typer)."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 65 Plan 65-03 (FEAT-V261-02) — meta_v2_refv2 train. "
            "Trains a 13-col MetaLearnerLogistic candidate sibling of "
            "canonical meta_v2.joblib; col[0] = xgb_v2_refv2_oof (Plan 65-02 "
            "candidate OOF), cols[1..12] mirror canonical META-V22. Canonical "
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
            "plumbing tests; 'full' invokes the live DB loader + joins Plan "
            "65-02 OOF parquet."
        ),
    )
    parser.add_argument(
        "--xgb-refv2-oof",
        type=Path,
        default=XGB_REFV2_OOF_PATH,
        help=(
            f"Plan 65-02 OOF parquet path (default: "
            f"{XGB_REFV2_OOF_PATH.relative_to(PROJECT_ROOT)}). Used by --mode full."
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

    Returns the OS exit code (0 success, non-zero failure). Phase 64 CR-02
    pattern: all known-failure modes (anti-overwrite guard, missing canonical
    files, missing OOF parquet) surface as a clean stderr message + non-zero
    rc; no Python traceback leaks to operators.
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

    # Verify META-V22 cols[1..12] layout matches canonical.
    try:
        assert_meta_v2_layout()
    except FileNotFoundError as e:
        print(f"meta layout check failed: {e}", file=sys.stderr)
        return 1
    except AssertionError as e:
        print(f"meta layout check failed: {e}", file=sys.stderr)
        return 1

    # PRE-EMIT anti-overwrite guard — fire as early as possible so an
    # invalid --output argv doesn't waste minutes of train time before
    # surfacing. The guard also fires inside emit_outputs as defense-in-depth.
    try:
        _check_anti_overwrite(args.output, args.output_meta)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    # Build training matrix.
    print(f"[train_meta_v2_refv2] mode={args.mode}, seed={args.seed}")
    try:
        if args.mode == "synthetic":
            X_13, y = build_13col_training_matrix(source="synthetic")
        else:
            # --mode full requires Plan 65-02 OOF parquet.
            try:
                oof_df = load_xgb_refv2_oof(args.xgb_refv2_oof)
            except FileNotFoundError as e:
                print(f"OOF parquet load failed: {e}", file=sys.stderr)
                return 1
            X_13, y = build_13col_training_matrix(source="live", xgb_refv2_oof_df=oof_df)
    except FileNotFoundError as e:
        print(f"training data load failed: {e}", file=sys.stderr)
        return 1
    print(f"[train_meta_v2_refv2] X_13.shape={X_13.shape}, y.shape={y.shape}")

    # Fit.
    meta_pipeline = fit_meta_refv2(X_13, y, seed=args.seed)

    # Emit (anti-overwrite guard fires here again as defense-in-depth).
    try:
        emit_outputs(
            meta_pipeline=meta_pipeline,
            out_joblib=args.output,
            out_meta=args.output_meta,
            seed=args.seed,
            mode=args.mode,
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"output emit failed: {e}", file=sys.stderr)
        return 1

    # Post-fit AUDIT-01 sandwich — confirm canonical artifacts byte-identical.
    try:
        assert_audit01_invariants()
    except AssertionError as e:
        # Catastrophic — somehow the fit clobbered canonical. Loud failure.
        print(f"AUDIT-01 POSTFLIGHT VIOLATION: {e}", file=sys.stderr)
        return 2

    print(f"[train_meta_v2_refv2] wrote {args.output}")
    print(f"[train_meta_v2_refv2] wrote {args.output_meta}")
    print(
        f"[train_meta_v2_refv2] AUDIT-01 unchanged (xgb_v2={EXPECTED_XGB_V2_SHA256[:12]}..., "
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
        # Defense-in-depth: any unhandled exception surfaces as exit 1
        # with a traceback so operators see the root cause but the process
        # rc is still well-defined.
        traceback.print_exc()
        sys.exit(1)
