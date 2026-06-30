#!/usr/bin/env python
"""Phase 65 Plan 65-02 (FEAT-V261-02) — xgb_v2_refv2 retrain script.

Trains an xgboost candidate sibling of canonical ``xgb_v2.joblib`` on the
v2.2 90-col feature space PLUS the 2 NEW REF v2 columns
(``ref_v2_finish_rate_shrunk``, ``ref_v2_decision_rate_shrunk``) computed
via Plan 65-01's ``compute_ref_v2_features_shrunk`` — 92-col xgb input.

Emits three SIBLING artifacts (canonical ``xgb_v2.joblib`` UNTOUCHED per
AUDIT-01 D-10):

  - ``models/xgb_v2_refv2.joblib`` — 92-col xgb candidate
  - ``models/xgb_v2_refv2_meta.json`` — sidecar metadata with
    ``canonical_status="candidate_sibling_NOT_canonical"`` +
    ``base_model_sha256`` referencing canonical xgb_v2
  - ``data/intermediate/xgb_v2_refv2_oof.parquet`` — OOF predictions
    (``fight_id``, ``oof_prob``, ``event_date``); the col[0] source for
    Plan 65-03 meta candidate AND Plan 65-04 substrate parquet.

Hyperparameters are READ verbatim from ``models/xgb_v2_meta.json::best_params``
per Phase 65 D-03: "Same xgboost hyperparameters as canonical xgb_v2.joblib".
ONLY the feature columns differ — this is the deliberate REF v2 vs canonical
ablation that the GATE-V26-02 verifier's refit_baseline path is designed to
measure.

Note on feature-column count: canonical ``xgb_v2.joblib`` was trained on a
72-col subset of the v2.2 90-col substrate (``xgb_v2_meta.json::feature_columns``
length 72). Phase 65 D-03 specifies the RETRAIN runs on the FULL v2.2 90-col
substrate + the 2 REF v2 cols → 92-col input. This is INTENTIONAL: the
comparison is "what happens if we add REF v2 to the FULL v2.2 substrate,
using the same hyperparameters?" not "what happens if we add REF v2 to
xgb_v2's exact 72-col subset?" The sidecar JSON documents both column lists
under ``feature_columns`` (the 92-col input) + ``canonical_xgb_v2_columns``
(the 72-col reference) so audit trail is unambiguous.

Anti-overwrite discipline (Phase 64 CR-01 carry-forward):
The script REFUSES to write to any path resolving into ``PROTECTED_OUTPUTS``
(canonical ``xgb_v2.joblib`` + ``xgb_v2_meta.json``). RuntimeError converted
to a clean stderr message + exit-1; no traceback. Verified by
``tests/unit/scripts/test_retrain_xgb_v2_refv2.py``.

Frozen-date determinism (Phase 64 CR-03 carry-forward):
Synthetic mode uses ``XGB_REFV2_FROZEN_DATE = date(2026, 6, 4)`` to monkeypatch
``date.today()`` callers in the upstream ``compose_v25_travel`` helper so the
synthetic eval fixture is byte-stable across re-runs on different calendar
days.

FileNotFoundError handling (Phase 64 CR-02 carry-forward):
Missing canonical meta JSON OR missing canonical joblib raise a clean stderr
message + exit-1; no Python traceback leaks to operators.

Usage:
    python scripts/retrain_xgb_v2_refv2.py --help
    python scripts/retrain_xgb_v2_refv2.py --dry-run    # synthetic, fast
    python scripts/retrain_xgb_v2_refv2.py --full       # full retrain (slow)
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
# (not as a package). Mirrors the Phase 64 builder pattern.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ── LOCKED constants (Phase 65 D-03 + D-10 AUDIT-01) ──────────────────────

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# AUDIT-01 anchors — locked per .planning/AUDIT-01-BASELINE-SHA.txt.
EXPECTED_XGB_V2_SHA256: str = "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
EXPECTED_META_V2_SHA256: str = "77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196"

# Canonical hyperparameter source (READ-ONLY for this script).
CANONICAL_XGB_META: Path = PROJECT_ROOT / "models" / "xgb_v2_meta.json"
CANONICAL_XGB_JOBLIB: Path = PROJECT_ROOT / "models" / "xgb_v2.joblib"
CANONICAL_META_V2_JOBLIB: Path = PROJECT_ROOT / "models" / "meta" / "meta_v2.joblib"

# Sibling output paths — DEFAULT only; CLI ``--output*`` overrides allowed
# everywhere EXCEPT into PROTECTED_OUTPUTS.
OUT_JOBLIB: Path = PROJECT_ROOT / "models" / "xgb_v2_refv2.joblib"
OUT_META: Path = PROJECT_ROOT / "models" / "xgb_v2_refv2_meta.json"
OUT_OOF: Path = PROJECT_ROOT / "data" / "intermediate" / "xgb_v2_refv2_oof.parquet"

# Anti-overwrite guard set — Phase 64 CR-01 pattern + Phase 65 T-65-05 mitigation.
# Resolved paths so a symlinked or relative-style operator argv cannot bypass
# the guard. The guard fires BEFORE any disk write.
PROTECTED_OUTPUTS: frozenset[Path] = frozenset(
    {
        CANONICAL_XGB_JOBLIB.resolve(),
        CANONICAL_XGB_META.resolve(),
    }
)

# Phase 64 CR-03 determinism — frozen reference date for any synthetic-mode
# ``date.today()`` callers in the upstream compose_v25_travel helper.
XGB_REFV2_FROZEN_DATE: date = date(2026, 6, 4)

# The two NEW REF v2 cols (Plan 65-01 contract).
REF_V2_COLS: tuple[str, ...] = (
    "ref_v2_finish_rate_shrunk",
    "ref_v2_decision_rate_shrunk",
)

# Default KFold seed for OOF generation. Canonical xgb_v2_meta.json does NOT
# record a seed (we verified: ``meta["seed"] is None``); we pin seed=42 as
# the project-wide default and document it in the sidecar JSON.
DEFAULT_OOF_SEED: int = 42
DEFAULT_OOF_FOLDS: int = 5

# Synthetic mode params for the --dry-run path (smaller than the 600-row
# Phase 64 default so the spike fits in the unit-tier <60s budget but still
# leaves enough rows per KFold split for a stable fit).
SYNTHETIC_N_FIGHTS: int = 240

# Phase 26 D-02 rule-of-thumb shrinkage strength (passed through to
# compute_ref_v2_features_shrunk). Not CV-tuned (Phase 26 D-02 banned).
REF_V2_K_SHRINK: float = 50.0


# ── AUDIT-01 invariant assertions (sandwich pattern) ──────────────────────


def _sha256_file(path: Path) -> str:
    """Return the hex SHA256 digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_audit01_invariants() -> None:
    """Verify canonical ``xgb_v2.joblib`` + ``meta_v2.joblib`` are byte-identical.

    Sandwich pattern from ``train_meta_v22.py`` — called at script entry
    AND after every disk write so any AUDIT-01 drift surfaces immediately.

    Raises:
        AssertionError: if either canonical SHA does not match the locked
            constant. Message contains the literal string ``AUDIT-01`` so
            operators (and the test suite) can grep for it unambiguously.
    """
    if not CANONICAL_XGB_JOBLIB.exists():
        raise AssertionError(
            f"AUDIT-01 invariant cannot be checked: canonical "
            f"{CANONICAL_XGB_JOBLIB} is missing. Restore from git."
        )
    if not CANONICAL_META_V2_JOBLIB.exists():
        raise AssertionError(
            f"AUDIT-01 invariant cannot be checked: canonical "
            f"{CANONICAL_META_V2_JOBLIB} is missing. Restore from git."
        )
    sha_xgb = _sha256_file(CANONICAL_XGB_JOBLIB)
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


def load_canonical_hyperparams() -> tuple[dict, list[str]]:
    """Read canonical ``xgb_v2_meta.json`` → ``(best_params, feature_columns)``.

    Per Phase 65 D-03: the retrain script reuses canonical xgboost
    hyperparameters byte-for-byte. ONLY the feature columns differ between
    canonical and refv2.

    Returns:
        ``(best_params, canonical_xgb_v2_columns)`` where ``best_params``
        is the dict at ``meta["best_params"]`` (n_estimators, max_depth,
        learning_rate, etc.) and ``canonical_xgb_v2_columns`` is the
        72-col list at ``meta["feature_columns"]`` (documented in the
        sidecar JSON for audit trail).

    Raises:
        FileNotFoundError: when the canonical meta JSON is missing — caller
            is expected to convert this to a clean stderr exit (Phase 64
            CR-02 pattern; see ``main``).
    """
    if not CANONICAL_XGB_META.exists():
        raise FileNotFoundError(
            f"canonical xgb_v2_meta.json not found at {CANONICAL_XGB_META} "
            f"(AUDIT-01 invariant violated — restore from git before retrain)"
        )
    meta = json.loads(CANONICAL_XGB_META.read_text(encoding="utf-8"))
    best_params = dict(meta["best_params"])
    canonical_cols = list(meta["feature_columns"])
    return best_params, canonical_cols


# ── 92-col training matrix assembly ───────────────────────────────────────


def _synthesize_cohort_keys(n: int, seed: int) -> tuple[list[str], list[str]]:
    """Deterministic per-row (event_country_bucket, scoring_regime) labels.

    Synthetic mode does not carry venue.country information through the
    upstream ``compose_v25_travel._build_synthetic_v25`` fixture; we
    synthesize cohort assignments deterministically (seeded RNG) so the
    Beta-binomial shrinkage exercises ALL 12 cohorts (2 regimes × 6
    countries) and the OOF parquet has non-degenerate REF v2 columns.

    Note: this is for PLUMBING coverage only — Phase 65 D-04 states
    explicitly that synthetic-mode REF v2 numbers carry no signal. The
    full retrain path (``--full``) uses real venue.country values.
    """
    import numpy as np

    from ufc_prediction.features.referee_v2 import EVENT_COUNTRY_BUCKETS

    rng = np.random.default_rng(seed)
    countries = list(rng.choice(EVENT_COUNTRY_BUCKETS, size=n))
    regimes = list(rng.choice(["pre_2017", "unified_post_2017"], size=n))
    return countries, regimes


def _compute_ref_v2_columns(
    *,
    event_dates: list[date],
    event_countries: list[str],
    scoring_regimes: list[str],
    y: list[int] | Any,
    fight_records: list[dict],
) -> tuple[list[float], list[float]]:
    """Compute the 2 REF v2 columns for every training row.

    Iterates fights in chronological order so each row's
    ``compute_ref_v2_features_shrunk`` sees ONLY past entries in
    ``cohort_history`` (strict pre-fight discipline per Plan 65-01
    docstring + Pitfall #5).

    Args:
        event_dates: per-row event date.
        event_countries: per-row country bucket (already mapped via
            :func:`derive_event_country_bucket`).
        scoring_regimes: per-row regime label (from
            :func:`derive_scoring_regime`).
        y: per-row binary outcome — used as a proxy for finish/decision
            in synthetic mode. In live mode the fight_records carry the
            actual method strings.
        fight_records: optional list of per-row dicts containing a
            ``method`` key (live mode) — if absent, the proxy from ``y``
            is used.

    Returns:
        ``(finish_col, decision_col)`` aligned with the input row order.
    """
    from ufc_prediction.features.referee_v2 import (
        RefereeV2Stratification,
        compute_ref_v2_features_shrunk,
    )
    from ufc_prediction.ml.features_v22.ref import classify_outcome

    n = len(event_dates)
    assert len(event_countries) == n
    assert len(scoring_regimes) == n
    assert len(y) == n
    assert len(fight_records) == n

    # Chronological order ensures cohort_history at row i contains ONLY
    # rows j with event_date_j < event_date_i (Pitfall #5).
    order = sorted(range(n), key=lambda i: event_dates[i])

    cohort_history: dict[RefereeV2Stratification, list[dict[str, Any]]] = {}
    finish_col: list[float | None] = [None] * n
    decision_col: list[float | None] = [None] * n

    # Phase 65 CR-01 fix: global_finish_rate is computed from a strict
    # pre-fight rolling window — only fights with event_date < event_dates[i]
    # contribute to the Bayesian prior at row i. This mirrors the same
    # temporal discipline the per-cohort shrinkage already enforces inside
    # compute_ref_v2_features_shrunk (see referee_v2.py line ~242).
    #
    # Convention: in BOTH live and synthetic modes, the denominator counts
    # every chronologically-prior fight (finishes + decisions + no_action).
    # In live mode classify_outcome("KO/TKO" / "Submission" / etc.) → "finish";
    # all other method strings (Decision-*, "No Contest", DQ, etc.) → not
    # finish but still contribute to n_total. In synthetic mode the y binary
    # serves as a finish/non-finish proxy with the same denominator semantics
    # (y==1 → finish; y==0 → non-finish, still counted in n_total).
    n_finish_running = 0
    n_total_running = 0

    for i in order:
        # global_finish_rate at row i = finishes / total over fights with
        # event_date < event_dates[i]. For the very first chronological row
        # n_total_running == 0 → fall back to 0.5 (uninformative Beta-binomial
        # prior; matches `cohort_history` being empty at this point so the
        # shrinkage formula collapses to the prior either way).
        if n_total_running > 0:
            global_finish_rate = n_finish_running / n_total_running
        else:
            global_finish_rate = 0.5

        feats = compute_ref_v2_features_shrunk(
            event_country=event_countries[i],
            scoring_regime=scoring_regimes[i],
            as_of_date=event_dates[i],
            cohort_history=cohort_history,
            global_finish_rate=global_finish_rate,
            k_shrink=REF_V2_K_SHRINK,
        )
        finish_col[i] = feats["ref_v2_finish_rate_shrunk"]
        decision_col[i] = feats["ref_v2_decision_rate_shrunk"]

        # Append this row to cohort_history AFTER its features are computed
        # (strict pre-fight discipline; mirrors the as_of_date < event_date
        # filter enforced inside compute_ref_v2_features_shrunk).
        key = RefereeV2Stratification(
            event_country=event_countries[i],
            scoring_regime=scoring_regimes[i],
        )
        # Phase 65 WR-02 fix: drop dead `if fight_records else None` guard —
        # the assert at function entry guarantees len(fight_records) == n,
        # so fight_records[i] is always valid in this loop.
        method_str = fight_records[i].get("method")
        if method_str is None:
            # Synthetic: encode outcome as a finish/decision proxy via the
            # method string the classify_outcome helper recognises.
            method_str = "KO/TKO" if int(y[i]) == 1 else "Decision - Unanimous"
        cohort_history.setdefault(key, []).append(
            {"event_date": event_dates[i], "method": method_str}
        )

        # Update running global counters AFTER feature compute so row i's
        # own outcome cannot leak into row i's prior. classify_outcome
        # treats "KO/TKO"/"Submission"/etc. → "finish"; both synthetic and
        # live paths feed the same string-based classification (live: real
        # method strings; synthetic: synthesized "KO/TKO" or
        # "Decision - Unanimous" above) so the two modes use byte-identical
        # accounting.
        cat = classify_outcome(method_str)
        if cat == "finish":
            n_finish_running += 1
        n_total_running += 1

    # mypy: by construction every slot is filled.
    return [float(v) for v in finish_col], [float(v) for v in decision_col]  # type: ignore[arg-type]


def build_92col_training_matrix(*, source: str = "synthetic") -> tuple[Any, Any, Any, Any]:
    """Build the (X_92, y, fight_ids, event_dates) training matrix.

    Two source modes:
      - ``synthetic`` (default): reuses ``compose_v25_travel._build_synthetic_v25``
        (Phase 42-shipped helper) for the v2.2 90-col substrate, freezes
        ``date.today()`` to ``XGB_REFV2_FROZEN_DATE`` for determinism (Phase 64
        CR-03 pattern), synthesizes per-row (country, regime) cohort labels,
        and appends the 2 REF v2 cols → 92-col output.
      - ``live``: invokes ``_load_assembled_data_v25_travel`` against the
        live PostgreSQL DB, then joins ``event.venue_id → venue.country`` to
        derive REAL country buckets, and appends the 2 REF v2 cols using
        the actual fight method strings from ``fight_records``.

    Returns:
        ``(X_92, y, fight_ids, event_dates)`` where ``X_92`` has shape
        ``(n, 92)`` (numpy.ndarray), ``y`` shape ``(n,)`` int, ``fight_ids``
        a list of ints, ``event_dates`` a list of ``datetime.date``.
    """
    import numpy as np

    if source == "synthetic":
        # Freeze date.today() — see Phase 64 CR-03 fix in
        # scripts/build_travel_substrate_v261.py for the same pattern.
        import datetime as _dt

        import compose_v25_travel as _cv  # type: ignore[import-not-found]
        from compose_v25_travel import (  # type: ignore[import-not-found]
            _build_synthetic_v25,
        )

        class _FixedDate(_dt.date):
            @classmethod
            def today(cls):  # type: ignore[override]
                return XGB_REFV2_FROZEN_DATE

        _orig_date = _cv.date
        _cv.date = _FixedDate
        try:
            X_v25, y, fight_dates, fight_records = _build_synthetic_v25(n=SYNTHETIC_N_FIGHTS)
        finally:
            _cv.date = _orig_date

        # Synthetic cohort assignment — seeded so re-runs are byte-stable.
        event_countries, scoring_regimes = _synthesize_cohort_keys(
            len(fight_records), seed=DEFAULT_OOF_SEED
        )

    elif source == "live":
        from compose_v25_travel import (  # type: ignore[import-not-found]
            _load_assembled_data_v25_travel,
        )
        from ufc_prediction.features.referee_v2 import (
            derive_event_country_bucket,
            derive_scoring_regime,
        )

        X_v25, y, fight_dates, fight_records = _load_assembled_data_v25_travel()
        # Live mode: every record carries the venue.country join already
        # (compose_v25_travel._load_assembled_data_v25_travel records the
        # join in fight_records when the upstream loader is wired). If the
        # records don't carry it, fall back to UNKNOWN — the substrate
        # builder coverage gate (Plan 65-04) will surface that.
        event_countries = [
            derive_event_country_bucket(rec.get("venue_country")) for rec in fight_records
        ]
        scoring_regimes = [derive_scoring_regime(d) for d in fight_dates]
    else:
        raise ValueError(
            f"build_92col_training_matrix: unknown source {source!r} "
            f"(expected 'synthetic' or 'live')"
        )

    assert X_v25.shape[1] == 92, (
        f"build_92col_training_matrix: expected 92-col v2.5-travel matrix, "
        f"got {X_v25.shape[1]} cols. The first 90 cols are the v2.2 substrate; "
        f"we override the trailing 2 (travel_distance_km, tz_shift_hours) "
        f"with the 2 REF v2 cols."
    )

    # Strip the trailing 2 v2.5-travel cols and use the first 90 as the
    # v2.2 substrate — Plan 65-02 D-03 specifies "v2.2 90-col + 2 REF v2 cols".
    X_v22 = X_v25[:, :90]

    # Compute the 2 REF v2 cols chronologically.
    fight_dates_list: list[date] = [d for d in fight_dates]
    finish_col, decision_col = _compute_ref_v2_columns(
        event_dates=fight_dates_list,
        event_countries=event_countries,
        scoring_regimes=scoring_regimes,
        y=list(y),
        fight_records=fight_records,
    )

    # Stack into 92-col matrix.
    X_92 = np.column_stack(
        [
            X_v22,
            np.asarray(finish_col, dtype=np.float64),
            np.asarray(decision_col, dtype=np.float64),
        ]
    )
    assert X_92.shape[1] == 92, (
        f"build_92col_training_matrix: expected 92-col output, got {X_92.shape[1]}"
    )

    fight_ids = [int(rec.get("fight_id", i)) for i, rec in enumerate(fight_records)]
    y_int = np.asarray(y, dtype=np.int64)

    return X_92, y_int, fight_ids, fight_dates_list


# ── xgboost fit + OOF generation ──────────────────────────────────────────


def fit_xgb_refv2(
    X_92: Any,
    y: Any,
    best_params: dict,
    *,
    seed: int = DEFAULT_OOF_SEED,
    n_splits: int = DEFAULT_OOF_FOLDS,
) -> tuple[Any, Any]:
    """Train xgboost on the 92-col input with the canonical best_params.

    Returns ``(final_model, oof_predictions)`` where ``oof_predictions`` is
    a 1-D ``np.ndarray`` of length ``X_92.shape[0]`` produced by KFold
    cross-validated training (sklearn KFold with shuffle=True, seed pinned).

    The FINAL model is fit on the full data with the same hyperparameters
    so the saved joblib can score arbitrary new inputs.

    Args:
        X_92: 92-col feature matrix (n_rows, 92).
        y: binary outcome vector (n_rows,).
        best_params: hyperparameter dict from canonical xgb_v2_meta.json.
        seed: KFold + xgboost random_state.
        n_splits: KFold splits (default 5).
    """
    import numpy as np
    from sklearn.model_selection import KFold
    from xgboost import XGBClassifier

    n_rows = X_92.shape[0]
    oof = np.full(n_rows, np.nan, dtype=np.float64)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_92)):
        fold_model = XGBClassifier(
            **best_params,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=int(seed),
            verbosity=0,
        )
        fold_model.fit(X_92[train_idx], y[train_idx])
        oof[val_idx] = fold_model.predict_proba(X_92[val_idx])[:, 1]

    # Final fit on full data.
    final_model = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=int(seed),
        verbosity=0,
    )
    final_model.fit(X_92, y)
    return final_model, oof


# ── Output emission with anti-overwrite guard ─────────────────────────────


def _check_anti_overwrite(*candidate_paths: Path) -> None:
    """Raise RuntimeError if any candidate path resolves into PROTECTED_OUTPUTS.

    Called by ``emit_outputs`` BEFORE any disk write so the guard cannot be
    bypassed by partial writes / atomic-rename tricks. The check resolves
    each candidate (follows symlinks, normalizes ``..``) so a sneaky
    operator argv like ``--output models/../models/xgb_v2.joblib`` is also
    caught.
    """
    for p in candidate_paths:
        try:
            resolved = p.resolve()
        except (OSError, RuntimeError):
            # ``resolve(strict=False)`` is the default — OSError here would
            # be unusual. If the path cannot be resolved, treat it as a
            # non-protected location (the subsequent write will surface the
            # real OS error).
            continue
        if resolved in PROTECTED_OUTPUTS:
            raise RuntimeError(
                f"Phase 65 T-65-05 / Phase 64 CR-01 guard: refusing to "
                f"overwrite canonical AUDIT-01 artifact at {resolved}. "
                f"Sibling outputs MUST go to *_refv2.* paths (see OUT_JOBLIB / "
                f"OUT_META defaults in scripts/retrain_xgb_v2_refv2.py)."
            )


def emit_outputs(
    *,
    model: Any,
    oof: Any,
    fight_ids: list[int],
    event_dates: list[date],
    best_params: dict,
    canonical_xgb_v2_columns: list[str],
    out_joblib: Path,
    out_meta: Path,
    out_oof: Path,
    seed: int,
    n_splits: int,
    mode: str,
) -> None:
    """Write joblib + sidecar JSON + OOF parquet to the configured sibling paths.

    Anti-overwrite guard fires BEFORE any write (Phase 64 CR-01 pattern).
    """
    import joblib
    import pandas as pd

    # Anti-overwrite guard FIRST — before any disk side effect.
    _check_anti_overwrite(out_joblib, out_meta, out_oof)

    # Resolve the 92-col feature_columns list. The canonical xgb_v2 used
    # 72 cols (read above). The refv2 input uses the FULL 90-col v2.2
    # substrate + 2 REF v2 cols. We import FEATURE_COLUMNS_V22 to name
    # those 90 cols by their exact identifiers.
    from ufc_prediction.ml.config import FEATURE_COLUMNS_V22

    feature_columns_92: list[str] = list(FEATURE_COLUMNS_V22) + list(REF_V2_COLS)
    assert len(feature_columns_92) == 92, (
        f"emit_outputs: expected 92 feature_columns, got {len(feature_columns_92)}"
    )

    # Ensure output dirs exist.
    out_joblib.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_oof.parent.mkdir(parents=True, exist_ok=True)

    # 1. joblib — the final xgb model.
    joblib.dump(model, out_joblib)

    # 2. sidecar JSON — sibling metadata + AUDIT-01 anchor record.
    sidecar = {
        "version": "v2_refv2",
        "meta_kind": "xgb",
        "canonical_status": "candidate_sibling_NOT_canonical",
        "base_model_version": "v2",
        "base_model_sha256": EXPECTED_XGB_V2_SHA256,
        "source_meta": "models/xgb_v2_meta.json",
        "feature_columns": feature_columns_92,
        "canonical_xgb_v2_columns": canonical_xgb_v2_columns,
        "n_features": 92,
        "best_params": dict(best_params),
        "trained_at": datetime.now(UTC).isoformat(),
        "phase": "65-ref-feat-v261-02-implementation-ml-spike-verifier-run",
        "decision_ids": ["D-03", "D-10"],
        "training_mode": mode,
        "oof_seed": seed,
        "oof_n_splits": n_splits,
        "audit_01_invariant": {
            "xgb_v2_sha": EXPECTED_XGB_V2_SHA256,
            "meta_v2_sha": EXPECTED_META_V2_SHA256,
            "status": "UNCHANGED",
        },
        "ref_v2_k_shrink": REF_V2_K_SHRINK,
        "synthetic_n_fights": SYNTHETIC_N_FIGHTS if mode == "synthetic" else None,
        "synthetic_frozen_date": (
            XGB_REFV2_FROZEN_DATE.isoformat() if mode == "synthetic" else None
        ),
    }
    out_meta.write_text(json.dumps(sidecar, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    # 3. OOF parquet — Plan 65-03 meta candidate reads col[0] from here.
    df_oof = pd.DataFrame(
        {
            "fight_id": pd.array(fight_ids, dtype="int64"),
            "oof_prob": pd.array(oof, dtype="float64"),
            "event_date": pd.array(event_dates, dtype="object"),
        }
    )
    df_oof.to_parquet(out_oof, index=False)


# ── CLI ───────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI surface (kept dep-light — no Typer)."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 65 Plan 65-02 (FEAT-V261-02) — xgb_v2_refv2 retrain. "
            "Trains an xgboost candidate sibling on the v2.2 90-col substrate "
            "+ the 2 NEW REF v2 cols → 92-col input. Canonical xgb_v2.joblib "
            "is NEVER overwritten (AUDIT-01 D-10 + Phase 64 CR-01 guard)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("synthetic", "full"),
        default="synthetic",
        help=(
            "Training source: 'synthetic' (default; DB-free, fast, ≤60s) "
            "for plumbing tests; 'full' invokes the live DB loader and "
            "may take minutes."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_const",
        dest="mode",
        const="synthetic",
        help="Alias for --mode synthetic.",
    )
    parser.add_argument(
        "--full",
        action="store_const",
        dest="mode",
        const="full",
        help="Alias for --mode full.",
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
    parser.add_argument(
        "--output-oof",
        type=Path,
        default=OUT_OOF,
        help=(
            f"Output OOF parquet path (default: "
            f"{OUT_OOF.relative_to(PROJECT_ROOT)}). Gitignored — regeneratable."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_OOF_SEED,
        help=f"KFold + xgboost random_state (default: {DEFAULT_OOF_SEED}).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_OOF_FOLDS,
        help=f"KFold split count (default: {DEFAULT_OOF_FOLDS}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry: argparse → AUDIT-01 preflight → fit → emit → AUDIT-01 postflight.

    Returns the OS exit code (0 success, non-zero failure). Phase 64 CR-02
    pattern: all known-failure modes (anti-overwrite guard, missing canonical
    files) surface as a clean stderr message + non-zero rc; no Python
    traceback leaks to operators.
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
        # Phase 64 CR-02 pattern — clean stderr, no traceback.
        print(f"AUDIT-01 preflight failed: {e}", file=sys.stderr)
        return 1

    # Load canonical hyperparameters.
    try:
        best_params, canonical_cols = load_canonical_hyperparams()
    except FileNotFoundError as e:
        print(f"canonical hyperparam load failed: {e}", file=sys.stderr)
        return 1

    # Build 92-col training matrix.
    # CLI mode "synthetic" → matrix source "synthetic"; CLI mode "full"
    # → matrix source "live" (the matrix builder distinguishes "synthetic"
    # vs "live" data sources; the CLI exposes the operator-facing
    # vocabulary "dry-run"/"full" → mode "synthetic"/"full").
    matrix_source = "synthetic" if args.mode == "synthetic" else "live"
    print(f"[retrain_xgb_v2_refv2] mode={args.mode}, seed={args.seed}")
    try:
        X_92, y, fight_ids, event_dates = build_92col_training_matrix(source=matrix_source)
    except FileNotFoundError as e:
        print(f"training data load failed: {e}", file=sys.stderr)
        return 1
    print(
        f"[retrain_xgb_v2_refv2] X_92.shape={X_92.shape}, "
        f"y.shape={y.shape}, n_fights={len(fight_ids)}"
    )

    # Fit + OOF.
    model, oof = fit_xgb_refv2(X_92, y, best_params, seed=args.seed, n_splits=args.n_splits)

    # Emit (anti-overwrite guard fires here BEFORE writes).
    try:
        emit_outputs(
            model=model,
            oof=oof,
            fight_ids=fight_ids,
            event_dates=event_dates,
            best_params=best_params,
            canonical_xgb_v2_columns=canonical_cols,
            out_joblib=args.output,
            out_meta=args.output_meta,
            out_oof=args.output_oof,
            seed=args.seed,
            n_splits=args.n_splits,
            mode=args.mode,
        )
    except RuntimeError as e:
        # Phase 64 CR-01 pattern — RuntimeError from the anti-overwrite
        # guard surfaces as a clean stderr message + rc=1 (no traceback).
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

    print(f"[retrain_xgb_v2_refv2] wrote {args.output}")
    print(f"[retrain_xgb_v2_refv2] wrote {args.output_meta}")
    print(f"[retrain_xgb_v2_refv2] wrote {args.output_oof}")
    print(
        f"[retrain_xgb_v2_refv2] AUDIT-01 unchanged (xgb_v2={EXPECTED_XGB_V2_SHA256[:12]}..., "
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
