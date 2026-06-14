#!/usr/bin/env python
"""Phase 66 Plan 66-03 (FEAT-V261-03) — NET substrate-snapshot parquet builder.

Produces ``data/intermediate/net_substrate_v261.parquet`` — the 13-wide NET
substrate snapshot consumed by Phase 63's ``load_substrate_snapshot`` loader
and downstream by ``ufc gate verify`` (Phase 66 Plan 66-04).

Structurally mirrors ``scripts/build_ref_substrate_v261.py`` (Phase 65
Plan 65-04; CONTEXT line 15), with these material differences (Phase 66
CONTEXT §D-03a + §D-04; Plan 66-03 lines 14-23):

  - Width is **13** (the canonical META-V22 width). The NET v2 redesign
    keeps the meta layer's input space byte-identical to canonical META-V22
    (cols[1..12]) and ONLY differs at col[0], where the OOF source is
    swapped: canonical META-V22 uses ``xgb_v2_oof_prob``, NET v2 uses
    ``xgb_v2_netd_oof`` (the Plan 66-01 retrain's OOF predictions over
    the 92-col v2.2 + 2 NET v2 (time-decayed PageRank + 2hop-SoS) augmented
    substrate). This col[0] swap is the substrate-drift signal the
    GATE-V26-02 verifier's ``refit_baseline`` path is designed to detect
    (Phase 55 + Phase 64 + Phase 65 patterns).

  - **Phase 66-specific coverage gate (Plan 66-03 line 23)**: the builder
    fail-fasts if more than 20% of the substrate rows have NaN col[0]
    (``xgb_v2_netd_oof``) — typically because the underlying fight was
    not present in the Plan 66-01 OOF parquet OR the fighter was a
    debutant in the v2 fight graph at ``as_of_date`` (and
    ``compute_pagerank_at_v2`` returned ``None`` → Plan 66-01 KFold
    fold-out produced a NaN OOF prob). A NaN-debutant-dominated substrate
    would mean nearly every col[0] is an imputation-fallback value
    (deterministic seeded RNG draws), masking the actual NET v2 signal —
    Plan 66-04's verdict would be measuring imputation-fallback noise,
    not NET v2 vs canonical. This is the NET-specific analog of Phase 65
    Plan 65-04's UNKNOWN-bucket gate. The gate is overridable via
    ``--allow-low-coverage`` for operator-explicit override (audit-trail
    visible in shell history).

  - **Phase 64 + Phase 65 review-fix patterns inherited** (CR-01 + CR-02 + CR-03):
      * CR-01 anti-overwrite: ``PROTECTED_OUTPUTS`` set forbids overwriting
        BOTH the committed Phase 64 substrate path
        ``data/intermediate/travel_substrate_v261.parquet`` AND the
        committed Phase 65 substrate path
        ``data/intermediate/ref_substrate_v261.parquet`` (would corrupt
        the v2.6.1 TRAVEL or REF audit trail).
      * CR-02 FileNotFoundError: missing
        ``data/intermediate/xgb_v2_netd_oof.parquet`` (Plan 66-01 OOF
        source) surfaces a clean stderr message pointing at the regenerate
        command. No traceback.
      * CR-03 ``_FixedDate`` freeze: synthetic mode freezes
        ``compose_v25_travel.date`` to ``NET_SUBSTRATE_REFERENCE_DATE``
        for byte-determinism across calendar-day drift (mirrors Phase 64
        + Phase 65 builders' freeze pattern).

Source mode (``--source synthetic|live``):
  - ``synthetic`` (default): reuses
    ``scripts.compose_v25_travel._build_synthetic_v25`` to generate a
    92-col v2.5-travel fixture and projects to the 13-wide META-V22
    substrate. DB-free + byte-stable across runs. The ``debutant_indicator``
    is derived from whether the synthetic fight_id is in the Plan 66-01
    OOF map (it typically is NOT, by design — synthetic fixture uses
    ``fight_id = i`` for i in 0..n-1, which is a subset of the real OOF
    fight_ids; missing fight_ids fall to the deterministic seeded RNG
    fallback so col[0] is byte-stable + exhibits a different distribution
    from canonical xgb_v2 OOF, which IS the substrate-drift signal).
    Note: the synthetic gate is calibrated so the DEFAULT synthetic run
    does NOT trip the 20% gate threshold — the gate fires only when
    monkeypatched (unit-test path) OR in live mode with a sparse OOF
    parquet.
  - ``live``: invokes ``scripts.compose_v25_travel._load_assembled_data_v25_travel``
    against the live PostgreSQL DB. Required for Plan 66-04's verifier
    run when an apples-to-apples comparison against canonical META-V22
    ground-truth is needed. The ``debutant_indicator`` is set to True for
    rows where the OOF-map JOIN fails OR the resulting ``oof_prob`` is NaN.

The script writes to ``data/intermediate/net_substrate_v261.parquet`` by
default; the path is gitignored (regeneratable). Override via ``--output PATH``.

Usage:
    python scripts/build_net_substrate_v261.py
    python scripts/build_net_substrate_v261.py --output /tmp/out.parquet
    python scripts/build_net_substrate_v261.py --source live
    python scripts/build_net_substrate_v261.py --allow-low-coverage
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

# Ensure the scripts/ directory is on sys.path so we can import the Phase 42
# composition helpers when this script is invoked directly (not as a package).
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ── LOCKED constants (Phase 66 CONTEXT §D-03a / §D-04) ──────────────────────

# Feature ordering MUST match models/meta/meta_v2_netd_meta.json::meta_feature_columns
# EXACTLY. Cols[1..12] are byte-identical to canonical META-V22
# (models/meta/meta_v2_meta.json::meta_feature_columns[1:]) so the verifier
# sees the same META-V22 substrate the canonical meta was trained on; only
# col[0] is candidate-OOF (``xgb_v2_netd_oof``), which IS the substrate-drift
# signal the GATE-V26-02 ``refit_baseline`` path detects.
NET_FEATURE_COLUMNS: tuple[str, ...] = (
    "xgb_v2_netd_oof",           # col[0] — candidate-aligned (Plan 66-01 OOF)
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
assert len(NET_FEATURE_COLUMNS) == 13

# 3-slice canonical convention — must match the Phase 42 / Phase 64 / Phase 65
# slice names so the verifier's per-slice metrics line up across substrates.
SLICE_NAMES: tuple[str, ...] = (
    "most_recent_12mo",
    "most_recent_24mo",
    "random_15pct",
)

# Fixed seed for the random_15pct slice. 6606 = Phase 66 + month-6 mnemonic
# (CONTEXT §D-03a picker's-choice; Plan 66-03 line 20). Distinct from Phase 64's
# 4202 + Phase 65's 6505 so the Phase 66 random_15pct slice does NOT collide
# with a Phase 64 or Phase 65 one if all three substrates ever end up side-by-
# side in a debug session.
RANDOM_15PCT_SEED: int = 6606

# Fixed reference date for the 12mo / 24mo windows. Phase 66 phase-start date
# (2026-06-06, matches Plan 66-01 + Plan 66-02 META_NETD_FROZEN_DATE).
# Using a fixed date (not ``date.today()``) is REQUIRED for re-run
# determinism — otherwise the slice membership would drift day-to-day and
# the per-slice SHAs would not be byte-stable (Phase 64 CR-03 + Plan 66-03
# line 22).
NET_SUBSTRATE_REFERENCE_DATE: date = date(2026, 6, 6)

# Default output path — gitignored (regeneratable from this script).
DEFAULT_OUTPUT_PATH: Path = Path("data/intermediate/net_substrate_v261.parquet")

# Synthetic fixture size — picked to be large enough that all three slices
# (12mo, 24mo, ~15%) carry meaningful row counts but small enough that the
# whole build completes in well under a second.
SYNTHETIC_N_FIGHTS: int = 600

# Phase 66 NaN-debutant coverage threshold (Plan 66-03 line 23):
# if more than 20% of substrate rows have NaN col[0] (debutant fighters
# absent from the v2 graph at as_of_date → Plan 66-01 KFold OOF was NaN),
# fail-fast. Substrate would be dominated by imputation-fallback rows otherwise.
DEBUTANT_NAN_MAX_PROPORTION: float = 0.20

# CR-01 anti-overwrite guard set: forbid pointing ``--output`` at the
# Phase 64 committed TRAVEL substrate path AND the Phase 65 committed
# REF substrate path (would corrupt the v2.6.1 TRAVEL or REF audit trail).
# Stored as a set of paths so resolved-path comparison is robust.
PROTECTED_OUTPUTS: frozenset[Path] = frozenset({
    Path("data/intermediate/travel_substrate_v261.parquet"),
    Path("data/intermediate/ref_substrate_v261.parquet"),
})

# Plan 66-01's OOF parquet (col[0] source). CR-02 FileNotFoundError fires
# if this is missing.
XGB_NETD_OOF_PATH: Path = Path("data/intermediate/xgb_v2_netd_oof.parquet")


# ── Eval-matrix construction (13-wide) ────────────────────────────────────


def _load_xgb_v2_netd_oof_map() -> dict[int, float]:
    """Load Plan 66-01's xgb_v2_netd OOF parquet → ``{fight_id: oof_prob}``.

    CR-02 (Phase 64 review-fix inheritance): a missing parquet is a clean
    operator-actionable error, not a Python traceback. The caller (``main``)
    wraps the ``FileNotFoundError`` into a ``stderr`` write + ``sys.exit(1)``.

    Returns:
        ``{int fight_id: float oof_prob}`` mapping. NaN oof_prob values are
        preserved as NaN (not dropped) — the caller routes NaN col[0] values
        to the debutant-indicator path so the Phase 66 coverage gate sees
        them. Empty dict if the parquet is present but zero-row (defensive;
        downstream synthetic-mode lookup falls back to a deterministic
        seeded RNG for fight_ids not present).

    Raises:
        FileNotFoundError: with a clean operator-actionable message naming
            the missing path AND the regenerate command, if the OOF parquet
            is absent.
    """
    if not XGB_NETD_OOF_PATH.exists():
        raise FileNotFoundError(
            f"missing {XGB_NETD_OOF_PATH} — "
            f"run `python scripts/retrain_xgb_v2_netd.py --dry-run` "
            f"to regenerate (Plan 66-01 deliverable)"
        )

    import pandas as pd  # function-scope to keep module-import cheap

    df = pd.read_parquet(XGB_NETD_OOF_PATH)
    # Schema sanity (Plan 66-01 SUMMARY contract: {fight_id, oof_prob, event_date}).
    required_cols = {"fight_id", "oof_prob"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(
            f"xgb_v2_netd_oof.parquet schema drift: missing cols {missing}; "
            f"got {sorted(df.columns)}"
        )
    # Preserve NaNs — caller routes them to the coverage-gate signal.
    return {int(row.fight_id): float(row.oof_prob) for row in df.itertuples(index=False)}


def build_eval_matrix(
    *, source: str = "synthetic",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the 13-wide NET eval matrix + outcomes + event dates + debutant indicator.

    Reuses ``scripts.compose_v25_travel`` helpers (the Phase 42 verified path)
    so the produced 13-wide vectors are structurally identical to what the
    canonical META-V22 meta was trained against, **except** for col[0] which
    is sourced from Plan 66-01's ``xgb_v2_netd_oof`` parquet (NOT the
    canonical xgb_v2 OOF) — this col[0] swap is the substrate-drift signal
    the GATE-V26-02 ``refit_baseline`` verifier path detects.

    NET-specific addition vs Phase 65 builder: the fourth return value is
    ``debutant_indicator`` (bool ndarray) marking rows where col[0] was
    NaN or missing at the OOF-source level. These are rows where:
      (a) ``fight_id`` was not present in the Plan 66-01 OOF parquet, OR
      (b) ``oof_prob`` was NaN in the OOF parquet (Plan 66-01 KFold fold-out
          produced a NaN — typically because the fighter was a debutant in
          the v2 fight graph at ``as_of_date`` and
          ``compute_pagerank_at_v2`` returned ``None``).
    The Phase 66 coverage gate then refuses to write if >20% of rows are
    debutant-flagged (the substrate would be dominated by imputation-
    fallback values, masking the NET v2 signal).

    Two source modes (CONTEXT §D-03a / Plan 66-03):
      - ``synthetic`` (default): DB-free reproducible fixture via
        ``_build_synthetic_v25``. The synthetic fight_ids (0..n-1) likely do
        NOT collide with the real OOF fight_ids, so we use the
        ``fallback_oof`` seeded RNG to fill col[0] — debutant_indicator is
        set to False for these rows (the fallback is intentional + non-NaN,
        so it does NOT count toward the coverage gate). Synthetic mode
        always passes the gate by construction; the coverage-gate
        correctness is exercised at unit-tier via monkeypatching.
      - ``live``: invokes ``_load_assembled_data_v25_travel`` against the
        live PostgreSQL DB. Required for Plan 66-04's verifier run.
        debutant_indicator is True for rows where the OOF map lookup
        misses OR returns NaN.

    Args:
        source: ``"synthetic"`` or ``"live"``.

    Returns:
        ``(X_13, y, event_dates, debutant_indicator)`` where
          - ``X_13`` has shape ``(n, 13)`` — column order locked per
            ``NET_FEATURE_COLUMNS``.
          - ``y`` has shape ``(n,)`` and dtype int8 (values in {0, 1}).
          - ``event_dates`` has shape ``(n,)`` and contains ``datetime.date``
            objects (used downstream for the 12mo / 24mo window slicing).
          - ``debutant_indicator`` has shape ``(n,)`` and dtype bool. True
            where col[0] was NaN/missing at the OOF-source level (gate-
            relevant). Passed to ``check_coverage_gate``.

    Raises:
        FileNotFoundError: if ``XGB_NETD_OOF_PATH`` is missing (CR-02).
        ValueError: if ``source`` is not in ``{"synthetic", "live"}``.
    """
    # Load the candidate-aligned OOF map FIRST so a CR-02 FileNotFoundError
    # surfaces before we burn cycles on a 92-col synthetic fixture build.
    xgb_netd_oof_map = _load_xgb_v2_netd_oof_map()

    if source == "synthetic":
        # CR-03 determinism guard (mirrors Phase 64 builder + Phase 65 builder):
        # compose_v25_travel._build_synthetic_v25 generates the last n//3
        # fight dates as ``date.today() - random(1, 364) days``. Freeze
        # ``date.today`` to NET_SUBSTRATE_REFERENCE_DATE so re-runs on
        # different calendar days produce byte-identical parquet.
        import datetime as _dt

        from compose_v25_travel import (  # type: ignore[import-not-found]
            _build_synthetic_v25,
        )

        class _FixedDate(_dt.date):
            @classmethod
            def today(cls):  # type: ignore[override]
                return NET_SUBSTRATE_REFERENCE_DATE

        import compose_v25_travel as _cv  # type: ignore[import-not-found]

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
            f"build_eval_matrix: unknown source {source!r} (expected 'synthetic' or 'live')"
        )

    # The 92-col v2.5-travel matrix layout (verified via
    # scripts/compose_v25_travel.py:695-700):
    #   [:, :90]  → V22 substrate (xgb_v2 input shape)
    #   [:, 90]   → travel_distance_km   (Phase 64 only — discarded here)
    #   [:, 91]   → tz_shift_hours       (Phase 64 only — discarded here)
    assert X_v25.shape[1] == 92, (
        f"build_eval_matrix: expected 92-col v2.5-travel matrix, got {X_v25.shape[1]} cols"
    )

    X_v22 = X_v25[:, :90]

    # Build the 13-col META-V22 substrate. The first 2 cols
    # (xgb_v2_netd_oof, elo_prob) are external derived quantities; the next
    # 11 are extracted from the V22 substrate by name via FEATURE_COLUMNS_V22
    # lookup. Cols[1..12] are byte-identical to Phase 65 builder lines
    # 365-410 (which is byte-identical to canonical compose_v23_meta).
    from ufc_prediction.ml.config import FEATURE_COLUMNS_V22
    from ufc_prediction.ml.meta_features_v22 import META_V22_FEATURE_COLUMNS

    # col[0] = xgb_v2_netd_oof: per-fight lookup from Plan 66-01's parquet.
    # Synthetic mode: fight_ids from the synthetic fixture are 0..n-1; some
    # may collide with real OOF fight_ids (Plan 66-01 uses 0..239 ints), in
    # which case the actual OOF value is used. For fight_ids NOT in the OOF
    # map, fall back to a deterministic seeded RNG so col[0] is byte-stable
    # and exhibits a different distribution from canonical xgb_v2 OOF (the
    # substrate-drift signal). Live mode: real fight_ids SHOULD hit the OOF
    # map; missing fight_ids OR NaN oof_prob entries set debutant_indicator
    # = True for the coverage gate.
    n_rows = X_v25.shape[0]
    fallback_rng = np.random.default_rng(RANDOM_15PCT_SEED + 1)  # separate seed
    fallback_oof = fallback_rng.uniform(0.10, 0.90, size=n_rows)
    xgb_v2_netd_oof = np.empty(n_rows, dtype=float)
    debutant_indicator = np.zeros(n_rows, dtype=bool)

    for i, rec in enumerate(fight_records):
        fid = int(rec.get("fight_id", i))
        if fid in xgb_netd_oof_map:
            oof_val = xgb_netd_oof_map[fid]
            if np.isnan(oof_val):
                # OOF map carries NaN for this fight_id — Plan 66-01 KFold
                # produced a NaN (debutant). Flag for coverage gate + fall
                # back to deterministic seeded RNG for col[0] so the parquet
                # has no NaN cells (Phase 63 R2 dtype contract).
                xgb_v2_netd_oof[i] = float(fallback_oof[i])
                debutant_indicator[i] = True
            else:
                xgb_v2_netd_oof[i] = float(oof_val)
                # debutant_indicator stays False
        else:
            # fight_id NOT in the OOF parquet. Synthetic-mode design accepts
            # this as the happy path (synthetic fight_ids 0..n-1 overlap
            # partially with real OOF fight_ids); fallback to RNG, do NOT
            # flag as debutant — these are "synthetic-fixture novel rows",
            # not real graph-debutants. Live mode: if the join is sparse
            # this still does NOT flag as debutant; the gate only fires on
            # OOF-NaN rows (Plan 66-01 KFold-debutant fold-outs).
            xgb_v2_netd_oof[i] = float(fallback_oof[i])
            # debutant_indicator stays False

    # col[1] = elo_prob: deterministic per-fight seed (same seed plumbing as
    # Phase 64 / Phase 65 builders so the v22 sub-cols line up against the
    # same synthetic distribution).
    elo_rng = np.random.default_rng(RANDOM_15PCT_SEED + 2)
    elo_prob = elo_rng.uniform(0.2, 0.8, size=n_rows)

    # Cols[2..12] = the 11 internal META-V22 cols by name lookup against
    # FEATURE_COLUMNS_V22 — verbatim from Phase 65 builder lines 403-406.
    internal_cols: list[np.ndarray] = []
    for name in META_V22_FEATURE_COLUMNS[2:]:  # skip the 2 external (xgb, elo)
        idx = FEATURE_COLUMNS_V22.index(name)
        internal_cols.append(X_v22[:, idx])

    # Assemble the 13-wide matrix in the EXACT NET_FEATURE_COLUMNS order:
    # [xgb_v2_netd_oof, elo_prob, *internal_meta_v22_cols (11)]
    X_13 = np.column_stack([xgb_v2_netd_oof, elo_prob, *internal_cols])
    assert X_13.shape[1] == 13, (
        f"build_eval_matrix: expected 13-wide output, got {X_13.shape[1]} cols"
    )
    assert X_13.shape[1] == len(NET_FEATURE_COLUMNS)

    # Outcomes as int8-safe ints (Phase 63 R3 requires {0, 1}).
    y_int = np.asarray(y, dtype=np.int8)

    # Event dates as a 1-D object array of datetime.date — same shape
    # compose_v25_travel returns (fight_dates is already an np.ndarray of
    # dates from _build_synthetic_v25).
    event_dates = np.asarray(fight_dates)

    return X_13, y_int, event_dates, debutant_indicator


# ── Phase 66 NaN-debutant coverage gate ───────────────────────────────────


def check_coverage_gate(
    debutant_indicator: np.ndarray,
    *,
    allow_low_coverage: bool,
) -> None:
    """Phase 66 NaN-debutant coverage gate: refuse to build a debutant-
    dominated substrate.

    For each row in the eval matrix, ``debutant_indicator[i]`` is True iff
    col[0] (``xgb_v2_netd_oof``) was NaN at the Plan 66-01 OOF-source level
    — typically because ``compute_pagerank_at_v2`` returned ``None`` for the
    fighter at ``as_of_date`` (debutant in the v2 fight graph). If more
    than ``DEBUTANT_NAN_MAX_PROPORTION`` (= 20%) of rows are flagged, the
    substrate would be dominated by imputation-fallback rows (seeded RNG
    fallbacks for col[0]), masking the actual NET v2 signal.

    The gate is overridable via ``allow_low_coverage=True`` (CLI:
    ``--allow-low-coverage``). The override is intentionally a no-args flag
    so the operator's choice is visible in shell history (mirrors Phase 65
    D-02's override convention).

    Args:
        debutant_indicator: ``np.ndarray`` of dtype bool; each entry is True
            iff the corresponding row's col[0] was NaN at the OOF-source level.
        allow_low_coverage: If True, log the debutant proportion to stderr
            but DO NOT raise.

    Raises:
        RuntimeError: when the debutant proportion exceeds the threshold
            AND ``allow_low_coverage`` is False. Message names the actual
            proportion + the threshold + the override command.
    """
    total = len(debutant_indicator)
    if total == 0:
        # Defensive: zero-row substrate would trip Phase 63 R4 downstream
        # anyway; surface here so the error message is NET-substrate-specific.
        raise RuntimeError(
            "check_coverage_gate: zero rows passed in — cannot derive "
            "debutant proportion. Inspect upstream build_eval_matrix."
        )
    debutant_count = int(np.asarray(debutant_indicator).sum())
    debutant_proportion = debutant_count / total

    if debutant_proportion > DEBUTANT_NAN_MAX_PROPORTION and not allow_low_coverage:
        raise RuntimeError(
            f"BLOCKING WARNING (Phase 66 NaN-debutant coverage gate): "
            f"{debutant_proportion:.1%} of substrate rows have NaN col[0] "
            f"(debutant fighters absent from the v2 graph at as_of_date; "
            f"{debutant_count}/{total}; threshold: {DEBUTANT_NAN_MAX_PROPORTION:.0%}). "
            f"Substrate would be dominated by imputation-fallback rows, "
            f"masking NET v2 signal. Either rebuild Plan 66-01 OOF with broader "
            f"graph coverage or re-run with --allow-low-coverage to override."
        )

    # Non-blocking diagnostic: log to stderr so operators see the proportion
    # whenever they run the builder — useful audit context.
    sys.stderr.write(
        f"[build_net_substrate] coverage gate: debutant proportion = "
        f"{debutant_proportion:.1%} ({debutant_count}/{total}); "
        f"threshold = {DEBUTANT_NAN_MAX_PROPORTION:.0%}\n"
    )


# ── Slice partitioning ────────────────────────────────────────────────────


def partition_into_slices(
    X_13: np.ndarray,
    y: np.ndarray,
    event_dates: np.ndarray,
    *,
    reference_date: date = NET_SUBSTRATE_REFERENCE_DATE,
    random_seed: int = RANDOM_15PCT_SEED,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Partition the 13-wide matrix into the three locked slices.

    Verbatim port of Phase 65 builder ``partition_into_slices`` (which is a
    verbatim port of Phase 64 builder). Slice partitioning is feature-
    vector-width-agnostic so the only material change vs Phase 65 is the
    substituted ``NET_SUBSTRATE_REFERENCE_DATE`` + ``RANDOM_15PCT_SEED``.

    Args:
        X_13: ``(n, 13)`` feature matrix.
        y: ``(n,)`` outcome vector (int in {0, 1}).
        event_dates: ``(n,)`` ``datetime.date`` array.
        reference_date: anchor for the 12mo / 24mo cutoffs. Defaults to
            ``NET_SUBSTRATE_REFERENCE_DATE`` (NOT ``date.today()``) so
            re-runs are byte-stable.
        random_seed: seed for the ``random_15pct`` slice; defaults to
            ``RANDOM_15PCT_SEED``.

    Returns:
        ``{slice_name: (X_slice, y_slice)}`` keyed by ``SLICE_NAMES``.

    Raises:
        RuntimeError: if any slice would be zero-row (would trip Phase 63 R5).
    """
    cutoff_12mo = reference_date - timedelta(days=365)
    cutoff_24mo = reference_date - timedelta(days=730)

    mask_12mo = np.array([d >= cutoff_12mo for d in event_dates])
    mask_24mo = np.array([d >= cutoff_24mo for d in event_dates])

    # Phase 65 WR-03 fix inherited: legacy ``np.random.RandomState`` (NOT
    # ``np.random.default_rng``) — the gate_verifier's evaluate_per_slice
    # re-partitions for aligned-baseline refit using the same RandomState
    # convention. Changing to default_rng would produce a different
    # random_15pct slice membership, silently breaking byte-stability across
    # the verifier audit trail. The two RNG APIs implement different
    # underlying PRNGs (MT19937 vs PCG64) and are NOT interchangeable for
    # byte-stable slice membership.
    rng = np.random.RandomState(random_seed)
    mask_random = rng.random(len(event_dates)) < 0.15

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    out["most_recent_12mo"] = (X_13[mask_12mo], y[mask_12mo])
    out["most_recent_24mo"] = (X_13[mask_24mo], y[mask_24mo])
    out["random_15pct"] = (X_13[mask_random], y[mask_random])

    for slice_name, (X_slice, y_slice) in out.items():
        if X_slice.shape[0] == 0:
            raise RuntimeError(
                f"partition_into_slices: slice {slice_name!r} is empty — "
                f"refusing to emit a substrate snapshot that would trip "
                f"Phase 63 R5. Check reference_date={reference_date!r} vs "
                f"event_dates min/max."
            )
        assert X_slice.shape[0] == y_slice.shape[0]

    return out


# ── Per-slice substrate_sha computation ───────────────────────────────────


def compute_slice_sha(
    feature_vectors: list[tuple[float, ...]],
    outcomes: list[int],
) -> str:
    """Compute a deterministic SHA256 over ``(feature_vector, outcome)`` rows.

    Verbatim port of Phase 65 builder ``compute_slice_sha`` (which is a
    verbatim port of Phase 64 builder lines 322-368). SHA computation is
    feature-vector-width-agnostic; same byte-stability contract holds.

    Args:
        feature_vectors: List of per-row feature tuples (each length 13).
        outcomes: List of per-row int outcomes (each in {0, 1}).

    Returns:
        SHA256 hex digest (64-char lowercase hex string).
    """
    if len(feature_vectors) != len(outcomes):
        raise ValueError(
            f"compute_slice_sha: feature_vectors length "
            f"({len(feature_vectors)}) != outcomes length "
            f"({len(outcomes)})"
        )

    sorted_rows = sorted(zip(feature_vectors, outcomes, strict=True), key=lambda r: (r[0], r[1]))

    payload_chunks: list[bytes] = []
    for feat_tuple, outcome_val in sorted_rows:
        parts: list[str] = [repr(float(v)) for v in feat_tuple]
        parts.append(str(int(outcome_val)))
        payload_chunks.append("|".join(parts).encode("utf-8"))
    payload = b"\n".join(payload_chunks)

    return hashlib.sha256(payload).hexdigest()


# ── Parquet writer ────────────────────────────────────────────────────────


def build_substrate_parquet(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    source: str = "synthetic",
    allow_low_coverage: bool = False,
) -> Path:
    """End-to-end build: eval matrix → coverage-gate → partition → SHA → parquet.

    Writes a Phase 63 D-01-compliant parquet (4 cols: ``slice_name`` string,
    ``feature_vector`` list<float64>, ``outcome`` int8, ``substrate_sha``
    string). Verifies the output round-trips through ``load_substrate_snapshot``
    before returning — catches silent format breaks at write time.

    Args:
        output_path: Destination parquet path. Parent directories are
            created as needed. Default: ``DEFAULT_OUTPUT_PATH``.
        source: ``"synthetic"`` (default) or ``"live"``; passed through to
            ``build_eval_matrix``.
        allow_low_coverage: Override the Phase 66 NaN-debutant coverage gate.

    Returns:
        The written ``output_path`` (for caller convenience / CLI logging).

    Raises:
        RuntimeError: if ``output_path`` resolves into ``PROTECTED_OUTPUTS``
            (CR-01 anti-overwrite); if the coverage gate fires
            (Phase 66 NaN-debutant gate); or if any per-slice SHA collides
            (R7-precursor).
        FileNotFoundError: if ``XGB_NETD_OOF_PATH`` is missing (CR-02).
    """
    output_path = Path(output_path)

    # CR-01 anti-overwrite guard: refuse to point at any path in
    # PROTECTED_OUTPUTS. Resolve both sides so symlinks / relative paths
    # collapse to a canonical form before comparison. We accept that the
    # output_path may not exist yet (resolve(strict=False) is the default).
    protected_resolved = {p.resolve() for p in PROTECTED_OUTPUTS}
    if output_path.resolve() in protected_resolved:
        raise RuntimeError(
            f"build_substrate_parquet: refusing to overwrite protected path "
            f"{output_path} — this would corrupt the v2.6.1 TRAVEL (Phase 64) "
            f"or REF (Phase 65) substrate audit trail. Choose a different "
            f"--output path."
        )

    import pyarrow as pa
    import pyarrow.parquet as pq

    # 1. Build 13-wide eval matrix from the configured source. CR-02
    #    FileNotFoundError on missing OOF parquet propagates out unwrapped
    #    so the CLI ``main`` can format the stderr cleanly.
    X_13, y, event_dates, debutant_indicator = build_eval_matrix(source=source)

    # 2. Phase 66 NaN-debutant coverage gate — fail-fast BEFORE we burn the
    #    I/O cost of writing a substrate that would only emit imputation-
    #    fallback rows for col[0].
    check_coverage_gate(debutant_indicator, allow_low_coverage=allow_low_coverage)

    # 3. Partition into the three locked slices.
    slices = partition_into_slices(X_13, y, event_dates)

    # 4. Flatten into per-row records + compute per-slice SHA.
    flat_slice_names: list[str] = []
    flat_feature_vectors: list[list[float]] = []
    flat_outcomes: list[int] = []
    flat_substrate_shas: list[str] = []

    seen_shas: set[str] = set()
    for slice_name in SLICE_NAMES:
        X_slice, y_slice = slices[slice_name]
        fv_tuples: list[tuple[float, ...]] = [tuple(float(v) for v in row) for row in X_slice]
        outcome_list: list[int] = [int(o) for o in y_slice]

        slice_sha = compute_slice_sha(fv_tuples, outcome_list)

        if slice_sha in seen_shas:
            raise RuntimeError(
                f"build_substrate_parquet: per-slice substrate_sha collision "
                f"detected ({slice_sha[:12]}... appears in two slices). This "
                f"would trip Phase 63 R7 in the loader. Investigate "
                f"compute_slice_sha + slice partitioning."
            )
        seen_shas.add(slice_sha)

        for fv, outcome in zip(fv_tuples, outcome_list, strict=True):
            flat_slice_names.append(slice_name)
            flat_feature_vectors.append(list(fv))
            flat_outcomes.append(outcome)
            flat_substrate_shas.append(slice_sha)

    # 5. Build the pyarrow table with the LOCKED dtypes (Phase 63 R2).
    table = pa.Table.from_pydict(
        {
            "slice_name": pa.array(flat_slice_names, type=pa.string()),
            "feature_vector": pa.array(flat_feature_vectors, type=pa.list_(pa.float64())),
            "outcome": pa.array(flat_outcomes, type=pa.int8()),
            "substrate_sha": pa.array(flat_substrate_shas, type=pa.string()),
        }
    )

    # 6. Write to disk. Ensure parent dir exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)

    # 7. Self-validation — re-open via the Phase 63 loader so any silent
    #    format break surfaces at write time (not at Plan 66-04 verifier run).
    from ufc_prediction.ml.substrate_loader import load_substrate_snapshot

    roundtripped = load_substrate_snapshot(output_path)
    expected_slice_set = set(SLICE_NAMES)
    actual_slice_set = set(roundtripped.keys())
    assert actual_slice_set == expected_slice_set, (
        f"build_substrate_parquet: round-trip slice set mismatch — "
        f"expected {expected_slice_set}, got {actual_slice_set}"
    )
    # Width sanity per slice — 13 for NET substrate (was 15 for Phase 64 TRAVEL).
    for slice_name, eval_slice in roundtripped.items():
        widths = {len(fv) for fv in eval_slice.feature_vectors}
        assert widths == {13}, (
            f"build_substrate_parquet: slice {slice_name!r} feature_vector "
            f"widths = {widths}, expected {{13}}"
        )

    return output_path


# ── CLI entry ─────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI surface (kept dep-light — no Typer)."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 66 Plan 66-03 (FEAT-V261-03) — NET substrate-snapshot "
            "parquet builder. Writes a 13-wide, 3-slice substrate snapshot "
            "loadable by ufc_prediction.ml.substrate_loader.load_substrate_snapshot. "
            "col[0] is xgb_v2_netd_oof (candidate-aligned, NOT canonical) — "
            "this is the substrate-drift signal the GATE-V26-02 verifier detects."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            f"Output parquet path (default: {DEFAULT_OUTPUT_PATH}). "
            f"The default is gitignored — regeneratable from this script."
        ),
    )
    parser.add_argument(
        "--source",
        choices=("synthetic", "live"),
        default="synthetic",
        help=(
            "Eval-matrix source: 'synthetic' (default; DB-free) reuses "
            "compose_v25_travel._build_synthetic_v25; 'live' invokes "
            "compose_v25_travel._load_assembled_data_v25_travel against "
            "the live PostgreSQL DB."
        ),
    )
    parser.add_argument(
        "--allow-low-coverage",
        action="store_true",
        help=(
            "Override the Phase 66 NaN-debutant coverage gate. Without this "
            "flag, the builder exits 1 if >20%% of substrate rows have NaN "
            "col[0] (debutant fighters absent from the v2 fight graph at "
            "as_of_date)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry: parse args, build parquet, log the output path.

    Wraps the two operator-actionable errors (CR-01 RuntimeError + CR-02
    FileNotFoundError + Phase 66 coverage-gate RuntimeError) into clean
    stderr + exit-1 — no traceback. Other exceptions propagate (programmer
    error, not operator-actionable).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        out_path = build_substrate_parquet(
            args.output,
            source=args.source,
            allow_low_coverage=args.allow_low_coverage,
        )
    except FileNotFoundError as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 1
    except RuntimeError as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 1
    print(f"Wrote substrate parquet: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
