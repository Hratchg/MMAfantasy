#!/usr/bin/env python
"""Phase 66 Plan 66-01 (FEAT-V261-03) — xgb_v2_netd retrain script.

Trains an xgboost candidate sibling of canonical ``xgb_v2.joblib`` on the
v2.2 90-col feature space PLUS the 2 NET v2 columns
(``net_v2_pagerank_at``, ``net_v2_2hop_sos_at``) computed via Phase 60's
shipped ``network_v2.compute_pagerank_at_v2`` +
``network_v2.compute_2hop_sos_at_v2`` — 92-col xgb input.

The ``_netd`` suffix = **net**work **t**ime-**d**ecayed. ``DECAY_BASE``
is locked at ``0.98`` at the ``network_v2`` module level per Phase 66
D-01 (Lazova & Basnarkov 2015 ~34.3-day half-life); the ML spike does
NOT expose it via CLI.

Emits three SIBLING artifacts (canonical ``xgb_v2.joblib`` UNTOUCHED per
AUDIT-01 D-10):

  - ``models/xgb_v2_netd.joblib`` — 92-col xgb candidate
  - ``models/xgb_v2_netd_meta.json`` — sidecar metadata with
    ``canonical_status="candidate_sibling_NOT_canonical"`` +
    ``base_model_sha256`` referencing canonical xgb_v2 +
    ``decay_base: 0.98`` audit field
  - ``data/intermediate/xgb_v2_netd_oof.parquet`` — OOF predictions
    (``fight_id``, ``oof_prob``, ``event_date``); the col[0] source for
    Plan 66-02 meta candidate AND Plan 66-03 substrate parquet.

Hyperparameters are READ verbatim from ``models/xgb_v2_meta.json::best_params``
per Phase 66 D-03 (mirrors Phase 65 D-03 — "Same xgboost hyperparameters as
canonical xgb_v2.joblib; only the feature columns differ"). This is the
deliberate NET-v2 vs canonical ablation that the GATE-V26-02 verifier's
``refit_baseline`` path is designed to measure (Plan 66-04 substrate-drift
auto-detection per Phase 66 CONTEXT line 40).

Note on feature-column count: canonical ``xgb_v2.joblib`` was trained on a
72-col subset of the v2.2 90-col substrate (``xgb_v2_meta.json::feature_columns``
length 72). Phase 66 D-03 specifies the RETRAIN runs on the FULL v2.2 90-col
substrate + the 2 NET v2 cols → 92-col input. INTENTIONAL: the comparison is
"what happens if we add NET v2 to the FULL v2.2 substrate, using the same
hyperparameters?" not "what happens if we add NET v2 to xgb_v2's exact
72-col subset?" The sidecar JSON documents both column lists under
``feature_columns`` (the 92-col input) + ``canonical_xgb_v2_columns``
(the 72-col reference) so audit trail is unambiguous.

Anti-overwrite discipline (Phase 64 CR-01 + Phase 65 CR-extended carry-forward):
The script REFUSES to write to any path resolving into ``PROTECTED_OUTPUTS``
(canonical ``xgb_v2.joblib`` + ``xgb_v2_meta.json`` + Phase 65 siblings
``xgb_v2_refv2.joblib`` + ``xgb_v2_refv2_meta.json``). RuntimeError
converted to a clean stderr message + exit-1; no traceback. Verified by
``tests/unit/scripts/test_retrain_xgb_v2_netd.py``.

Frozen-date determinism (Phase 64 CR-03 carry-forward):
Synthetic mode uses ``XGB_NETD_FROZEN_DATE = date(2026, 6, 6)`` to
monkeypatch ``date.today()`` callers in the upstream ``compose_v25_travel``
helper. Phase 65 used 2026-06-04; Phase 66 advances to 2026-06-06 so a
Phase 66 synthetic substrate does NOT collide with Phase 65's by
frozen-date.

FileNotFoundError handling (Phase 64 CR-02 carry-forward):
Missing canonical meta JSON OR missing canonical joblib raise a clean
stderr message + exit-1; no Python traceback leaks to operators.

Pre-fight discipline (Pitfall #9 — NET v2 temporal-leakage mitigation):
NET v2 cols are computed by passing the fight's ``event_date`` as
``as_of_date`` to ``compute_pagerank_at_v2`` /
``compute_2hop_sos_at_v2``. The shared ``_decay_subgraph`` helper inside
``network_v2.py`` filters ``earliest_date < as_of_date`` (strict — see
``network_v2.py`` line ~100), so the per-row PageRank/SoS computation
ONLY sees edges from chronologically-prior fights. NET-03 temporal-leakage
regression test on network_v2.py guards this contract.

NaN handling: PageRank / 2hop-SoS return ``None`` for debutants (fighters
absent from the graph or with zero in-neighbors). We map None → NaN; xgb
handles NaN natively (``missing=NaN`` default). This is the same
debutant-handling discipline as the canonical 4th-pass integration helper
``apply_network_features_v2`` (see network_v2.py line ~210). The sidecar
JSON records ``nan_handling: "xgb_native_missing"`` for audit trail.

Usage:
    python scripts/retrain_xgb_v2_netd.py --help
    python scripts/retrain_xgb_v2_netd.py --dry-run    # synthetic, fast
    python scripts/retrain_xgb_v2_netd.py --full       # full retrain (slow)
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
# (not as a package). Mirrors the Phase 64 / Phase 65 builder pattern.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ── LOCKED constants (Phase 66 D-01 + D-03 + D-10 AUDIT-01) ───────────────

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# AUDIT-01 anchors — locked per .planning/AUDIT-01-BASELINE-SHA.txt.
EXPECTED_XGB_V2_SHA256: str = "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
EXPECTED_META_V2_SHA256: str = "77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196"

# Canonical hyperparameter source (READ-ONLY for this script).
CANONICAL_XGB_META: Path = PROJECT_ROOT / "models" / "xgb_v2_meta.json"
CANONICAL_XGB_JOBLIB: Path = PROJECT_ROOT / "models" / "xgb_v2.joblib"
CANONICAL_META_V2_JOBLIB: Path = PROJECT_ROOT / "models" / "meta" / "meta_v2.joblib"

# Phase 65 siblings — also AUDIT-01-protected siblings (do NOT clobber the
# refv2 leg with the netd leg either; both are independent candidate
# experiments and each owns its own joblib path).
PHASE65_XGB_REFV2_JOBLIB: Path = PROJECT_ROOT / "models" / "xgb_v2_refv2.joblib"
PHASE65_XGB_REFV2_META: Path = PROJECT_ROOT / "models" / "xgb_v2_refv2_meta.json"

# Sibling output paths — DEFAULT only; CLI ``--output*`` overrides allowed
# everywhere EXCEPT into PROTECTED_OUTPUTS.
OUT_JOBLIB: Path = PROJECT_ROOT / "models" / "xgb_v2_netd.joblib"
OUT_META: Path = PROJECT_ROOT / "models" / "xgb_v2_netd_meta.json"
OUT_OOF: Path = PROJECT_ROOT / "data" / "intermediate" / "xgb_v2_netd_oof.parquet"

# Anti-overwrite guard set — Phase 64 CR-01 + Phase 65 T-65-05 carry-forward,
# extended in Phase 66 to also cover Phase 65's sibling outputs so a future
# operator running an old --output argv against the netd script does NOT
# accidentally clobber refv2 either. Resolved paths so a symlinked or
# relative-style operator argv cannot bypass the guard.
PROTECTED_OUTPUTS: frozenset[Path] = frozenset(
    {
        CANONICAL_XGB_JOBLIB.resolve(),
        CANONICAL_XGB_META.resolve(),
        PHASE65_XGB_REFV2_JOBLIB.resolve(),
        PHASE65_XGB_REFV2_META.resolve(),
    }
)

# Phase 64 CR-03 determinism — frozen reference date for any synthetic-mode
# ``date.today()`` callers in the upstream compose_v25_travel helper. Phase
# 65 used date(2026, 6, 4); Phase 66 advances to date(2026, 6, 6) so a
# Phase 66 synthetic substrate does NOT collide with Phase 65's by
# frozen-date.
XGB_NETD_FROZEN_DATE: date = date(2026, 6, 6)

# The two NEW NET v2 cols (Phase 60 / D-02(P18) contract). Order is locked —
# tests pin ``NET_V2_COLS[0] == "net_v2_pagerank_at"`` and ``NET_V2_COLS[1]
# == "net_v2_2hop_sos_at"``. Plan 66-02 + 66-03 substrate readers assume
# this ordering.
NET_V2_COLS: tuple[str, ...] = (
    "net_v2_pagerank_at",
    "net_v2_2hop_sos_at",
)

# DECAY_BASE is locked in network_v2.py at the module level per D-01; we
# record it in the sidecar JSON for audit trail but do NOT re-define it
# here (single source of truth — see network_v2.DECAY_BASE).
# (Import happens lazily inside _compute_net_v2_columns to keep the module
# import cheap for the test tier.)

# Default KFold seed for OOF generation. Canonical xgb_v2_meta.json does NOT
# record a seed (verified: ``meta.get("seed") is None``); we pin seed=42 as
# the project-wide default and document it in the sidecar JSON.
DEFAULT_OOF_SEED: int = 42
DEFAULT_OOF_FOLDS: int = 5

# Synthetic mode params for the --dry-run path (mirror Phase 65's 240-row
# choice so the spike fits in the unit-tier <60s budget while leaving enough
# rows per KFold split for a stable fit).
SYNTHETIC_N_FIGHTS: int = 240


# ── AUDIT-01 invariant assertions (sandwich pattern) ──────────────────────


def _sha256_file(path: Path) -> str:
    """Return the hex SHA256 digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_audit01_invariants() -> None:
    """Verify canonical ``xgb_v2.joblib`` + ``meta_v2.joblib`` are byte-identical.

    Sandwich pattern from ``train_meta_v22.py`` (Phase 64) — called at
    script entry AND after every disk write so any AUDIT-01 drift surfaces
    immediately.

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

    Per Phase 66 D-03: the retrain script reuses canonical xgboost
    hyperparameters byte-for-byte. ONLY the feature columns differ between
    canonical and netd.

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


def _compute_net_v2_columns(
    *,
    event_dates: list[date],
    fight_records: list[dict],
    y: list[int] | Any,
) -> tuple[list[float], list[float]]:
    """Compute the 2 NET v2 columns for every training row.

    Iterates fights and computes PageRank / 2hop-SoS for fighter_a at each
    fight's ``event_date`` — strict pre-fight: ``_decay_subgraph`` filters
    ``earliest_date < as_of_date`` inside ``network_v2`` (Pitfall #9
    mitigation; see network_v2.py line ~100). The 2 NET v2 cols are
    computed for the FIGHTER-side (left/fighter_a); the v2.2 substrate
    already carries fighter-side differentials in its 90 cols, so the
    spike measures whether the NET v2 cols add signal beyond the substrate
    differentials.

    Debutant handling: ``compute_pagerank_at_v2`` and
    ``compute_2hop_sos_at_v2`` return ``None`` for debutants (fighter not
    in graph) or for fighters with zero in-neighbors. We map None → NaN
    (np.nan) — xgb handles missing values natively via ``missing=NaN``
    default. This is the same debutant-handling discipline as the
    canonical 4th-pass helper ``apply_network_features_v2``.

    Args:
        event_dates: per-row event date (used as ``as_of_date`` per row).
        fight_records: per-row dict carrying ``fighter_a_id``,
            ``fighter_b_id``, ``event_date``, and optionally ``winner_id``
            + ``loser_id``. The graph is built from records that DO carry
            winner_id/loser_id; synthetic-mode records that lack them get
            winner/loser synthesized from the ``y`` binary label.
        y: per-row binary outcome — used in synthetic mode to synthesize
            winner_id/loser_id when fight_records lack them so the graph
            has non-degenerate structure.

    Returns:
        ``(pagerank_col, sos_col)`` aligned with the input row order.
        Length matches ``len(fight_records)``. Entries are ``float``
        (with ``np.nan`` for debutants).
    """
    import numpy as np

    from ufc_prediction.features.network_v2 import (
        build_fight_graph_v2,
        compute_2hop_sos_at_v2,
        compute_pagerank_at_v2,
    )

    n = len(fight_records)
    assert len(event_dates) == n
    assert len(y) == n

    # Build the graph from records with explicit winner/loser fields. For
    # synthetic-mode records (no method/winner_id), synthesize winner/loser
    # from the y binary: y==1 → fighter_a wins, y==0 → fighter_b wins.
    # This gives the synthetic graph non-degenerate structure so
    # PageRank/SoS have something to converge to.
    decisive: list[dict] = []
    for i, fight in enumerate(fight_records):
        winner_id = fight.get("winner_id")
        loser_id = fight.get("loser_id")
        if winner_id is None or loser_id is None:
            # Synthesize from y proxy: y==1 → fighter_a wins (synthetic mode).
            if int(y[i]) == 1:
                winner_id = fight.get("fighter_a_id")
                loser_id = fight.get("fighter_b_id")
            else:
                winner_id = fight.get("fighter_b_id")
                loser_id = fight.get("fighter_a_id")
            if winner_id is None or loser_id is None:
                # Skip rows that lack fighter identities entirely (defensive;
                # the synthetic generator always supplies both).
                continue
        # build_fight_graph_v2 expects the keys winner_id, loser_id,
        # event_date, method (optional — falls back to default MOV
        # multiplier when absent).
        decisive.append(
            {
                "winner_id": winner_id,
                "loser_id": loser_id,
                "event_date": fight.get("event_date") or event_dates[i],
                "method": fight.get("method"),
            }
        )

    graph = build_fight_graph_v2(decisive)

    # Per-row pre-fight PageRank / 2hop-SoS. Pitfall #9: pass event_date as
    # as_of_date; _decay_subgraph filters earliest_date < as_of_date so the
    # row's own outcome is excluded from its own feature value.
    pagerank_col: list[float] = [float("nan")] * n
    sos_col: list[float] = [float("nan")] * n
    for i, fight in enumerate(fight_records):
        as_of = event_dates[i]
        fa_id = fight.get("fighter_a_id")
        if fa_id is None:
            continue
        pr = compute_pagerank_at_v2(graph, fa_id, as_of)
        sos = compute_2hop_sos_at_v2(graph, fa_id, as_of)
        if pr is not None:
            pagerank_col[i] = float(pr)
        if sos is not None:
            sos_col[i] = float(sos)

    # numpy nan is float; cast explicitly so the downstream column_stack
    # produces a uniform float64 ndarray.
    return [float(v) for v in pagerank_col], [float(v) for v in sos_col]


def build_92col_training_matrix(*, source: str = "synthetic") -> tuple[Any, Any, Any, Any]:
    """Build the (X_92, y, fight_ids, event_dates) training matrix.

    Two source modes:
      - ``synthetic`` (default): reuses ``compose_v25_travel._build_synthetic_v25``
        (Phase 42-shipped helper) for the v2.2 90-col substrate, freezes
        ``date.today()`` to ``XGB_NETD_FROZEN_DATE`` for determinism (Phase 64
        CR-03 pattern), and appends the 2 NET v2 cols → 92-col output.
        Winner/loser ids are synthesized from y for the NET graph (the
        synthetic generator emits fighter_a_id + fighter_b_id but no
        method/winner field).
      - ``live``: invokes ``_load_assembled_data_v25_travel`` against the
        live PostgreSQL DB, then computes the 2 NET v2 cols from the
        records' actual winner_id/loser_id fields.

    Returns:
        ``(X_92, y, fight_ids, event_dates)`` where ``X_92`` has shape
        ``(n, 92)`` (numpy.ndarray), ``y`` shape ``(n,)`` int, ``fight_ids``
        a list of ints, ``event_dates`` a list of ``datetime.date``.
    """
    import numpy as np

    if source == "synthetic":
        # Freeze date.today() — Phase 64 CR-03 pattern (see
        # scripts/build_travel_substrate_v261.py).
        import datetime as _dt

        import compose_v25_travel as _cv  # type: ignore[import-not-found]
        from compose_v25_travel import (  # type: ignore[import-not-found]
            _build_synthetic_v25,
        )

        class _FixedDate(_dt.date):
            @classmethod
            def today(cls):  # type: ignore[override]
                return XGB_NETD_FROZEN_DATE

        _orig_date = _cv.date
        _cv.date = _FixedDate
        try:
            X_v25, y, fight_dates, fight_records = _build_synthetic_v25(n=SYNTHETIC_N_FIGHTS)
        finally:
            _cv.date = _orig_date

    elif source == "live":
        from compose_v25_travel import (  # type: ignore[import-not-found]
            _load_assembled_data_v25_travel,
        )

        X_v25, y, fight_dates, fight_records = _load_assembled_data_v25_travel()
    else:
        raise ValueError(
            f"build_92col_training_matrix: unknown source {source!r} "
            f"(expected 'synthetic' or 'live')"
        )

    assert X_v25.shape[1] == 92, (
        f"build_92col_training_matrix: expected 92-col v2.5-travel matrix, "
        f"got {X_v25.shape[1]} cols. The first 90 cols are the v2.2 substrate; "
        f"we override the trailing 2 (travel_distance_km, tz_shift_hours) "
        f"with the 2 NET v2 cols."
    )

    # Strip the trailing 2 v2.5-travel cols and use the first 90 as the
    # v2.2 substrate — Plan 66-01 D-03 specifies "v2.2 90-col + 2 NET v2 cols".
    X_v22 = X_v25[:, :90]

    # Compute the 2 NET v2 cols.
    fight_dates_list: list[date] = [d for d in fight_dates]
    pagerank_col, sos_col = _compute_net_v2_columns(
        event_dates=fight_dates_list,
        fight_records=fight_records,
        y=list(y),
    )

    # Stack into 92-col matrix.
    X_92 = np.column_stack(
        [
            X_v22,
            np.asarray(pagerank_col, dtype=np.float64),
            np.asarray(sos_col, dtype=np.float64),
        ]
    )
    assert X_92.shape[1] == 92, (
        f"build_92col_training_matrix: expected 92-col output, got {X_92.shape[1]}"
    )

    fight_ids = [int(rec.get("fight_id", i)) for i, rec in enumerate(fight_records)]
    y_int = np.asarray(y, dtype=np.int64)

    return X_92, y_int, fight_ids, fight_dates_list


# ── xgboost fit + OOF generation ──────────────────────────────────────────


def fit_xgb_netd(
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
        X_92: 92-col feature matrix (n_rows, 92). May contain NaN in the
            trailing 2 NET v2 cols for debutants — xgboost handles missing
            natively via ``missing=NaN`` default.
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
                f"Phase 66 T-66-01 / Phase 65 T-65-05 / Phase 64 CR-01 guard: "
                f"refusing to overwrite canonical AUDIT-01 artifact at "
                f"{resolved}. Sibling outputs MUST go to *_netd.* paths (see "
                f"OUT_JOBLIB / OUT_META defaults in "
                f"scripts/retrain_xgb_v2_netd.py)."
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

    from ufc_prediction.features.network_v2 import DECAY_BASE as _DECAY_BASE

    # Anti-overwrite guard FIRST — before any disk side effect.
    _check_anti_overwrite(out_joblib, out_meta, out_oof)

    # Resolve the 92-col feature_columns list. The canonical xgb_v2 used
    # 72 cols (read above). The netd input uses the FULL 90-col v2.2
    # substrate + 2 NET v2 cols. We import FEATURE_COLUMNS_V22 to name
    # those 90 cols by their exact identifiers.
    from ufc_prediction.ml.config import FEATURE_COLUMNS_V22

    feature_columns_92: list[str] = list(FEATURE_COLUMNS_V22) + list(NET_V2_COLS)
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
        "version": "v2_netd",
        "meta_kind": "xgb",
        "canonical_status": "candidate_sibling_NOT_canonical",
        "base_model_version": "v2",
        "base_model_sha256": EXPECTED_XGB_V2_SHA256,
        "source_meta": "models/xgb_v2_meta.json",
        "feature_columns": feature_columns_92,
        "canonical_xgb_v2_columns": canonical_xgb_v2_columns,
        "n_features": 92,
        "best_params": dict(best_params),
        # Locked DECAY_BASE from network_v2.py (D-01) — recorded here for
        # audit trail even though the value is not configurable from this
        # script. Pre-commit / verifier auditors can grep for this field.
        "decay_base": float(_DECAY_BASE),
        "trained_at": datetime.now(UTC).isoformat(),
        "frozen_date_synthetic_mode": XGB_NETD_FROZEN_DATE.isoformat(),
        "phase": "66-net-feat-v261-03-time-decay-implementation-ml-spike-verifier-run",
        "decision_ids": ["D-01", "D-03", "D-10"],
        "training_mode": mode,
        "oof_seed": seed,
        "oof_n_splits": n_splits,
        "source_features_module": "src/ufc_prediction/features/network_v2.py",
        "nan_handling": "xgb_native_missing",
        "audit_01_invariant": {
            "xgb_v2_sha": EXPECTED_XGB_V2_SHA256,
            "meta_v2_sha": EXPECTED_META_V2_SHA256,
            "status": "UNCHANGED",
        },
        "synthetic_n_fights": SYNTHETIC_N_FIGHTS if mode == "synthetic" else None,
    }
    out_meta.write_text(json.dumps(sidecar, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    # 3. OOF parquet — Plan 66-02 meta candidate reads col[0] from here;
    #    Plan 66-03 substrate parquet also consumes col[0].
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
            "Phase 66 Plan 66-01 (FEAT-V261-03) — xgb_v2_netd retrain. "
            "Trains an xgboost candidate sibling on the v2.2 90-col substrate "
            "+ the 2 NEW NET v2 cols (time-decayed PageRank + 2hop-SoS, "
            "DECAY_BASE=0.98 locked at network_v2 module level per D-01) → "
            "92-col input. Canonical xgb_v2.joblib is NEVER overwritten "
            "(AUDIT-01 D-10 + Phase 64 CR-01 guard)."
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

    # Build 92-col training matrix. CLI mode "synthetic" → matrix source
    # "synthetic"; CLI mode "full" → matrix source "live".
    matrix_source = "synthetic" if args.mode == "synthetic" else "live"
    print(f"[retrain_xgb_v2_netd] mode={args.mode}, seed={args.seed}")
    try:
        X_92, y, fight_ids, event_dates = build_92col_training_matrix(source=matrix_source)
    except FileNotFoundError as e:
        print(f"training data load failed: {e}", file=sys.stderr)
        return 1
    print(
        f"[retrain_xgb_v2_netd] X_92.shape={X_92.shape}, "
        f"y.shape={y.shape}, n_fights={len(fight_ids)}"
    )

    # Fit + OOF.
    model, oof = fit_xgb_netd(X_92, y, best_params, seed=args.seed, n_splits=args.n_splits)

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

    print(f"[retrain_xgb_v2_netd] wrote {args.output}")
    print(f"[retrain_xgb_v2_netd] wrote {args.output_meta}")
    print(f"[retrain_xgb_v2_netd] wrote {args.output_oof}")
    print(
        f"[retrain_xgb_v2_netd] AUDIT-01 unchanged (xgb_v2={EXPECTED_XGB_V2_SHA256[:12]}..., "
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
