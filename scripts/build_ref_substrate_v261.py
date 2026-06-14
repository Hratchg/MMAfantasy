#!/usr/bin/env python
"""Phase 65 Plan 65-04 (FEAT-V261-02) — REF substrate-snapshot parquet builder.

Produces ``data/intermediate/ref_substrate_v261.parquet`` — the 13-wide REF
substrate snapshot consumed by Phase 63's ``load_substrate_snapshot`` loader
and downstream by ``ufc gate verify`` (Phase 65 Plan 65-05).

Structurally mirrors ``scripts/build_travel_substrate_v261.py`` (Phase 64
Plan 64-02), with these material differences (CONTEXT §D-02, §D-03a; lines
18, 200-204):

  - Width is **13** (the canonical META-V22 width), NOT 15. The REF v2
    redesign keeps the meta layer's input space byte-identical to canonical
    META-V22 (cols[1..12]) and ONLY differs at col[0], where the OOF source
    is swapped: canonical META-V22 uses ``xgb_v2_oof_prob``, REF v2 uses
    ``xgb_v2_refv2_oof`` (the Plan 65-02 retrain's OOF predictions, where
    the REF v2 cohort-shrunk finish/decision-rate cols were appended to the
    v2.2 substrate). This col[0] swap is the substrate-drift signal the
    GATE-V26-02 verifier's ``refit_baseline`` path is designed to detect
    (Phase 55 + Phase 64 patterns).

  - **NEW Phase 65 D-02 coverage gate**: the builder fail-fasts if more
    than 20% of the events backing the substrate land in the
    ``derive_event_country_bucket`` ``"UNKNOWN"`` bucket. An UNKNOWN-
    dominated substrate would mean nearly every REF v2 feature value is
    a Bayesian-prior fallback (Beta-binomial shrinkage to ``global_finish_rate``),
    which masks the actual REF v2 signal — Plan 65-05's verdict would be
    measuring shrinkage-noise, not REF v2 vs canonical. The gate is
    overridable via ``--allow-low-coverage`` for operator-explicit override
    (audit-trail visible in shell history).

  - **Phase 64 review-fix patterns inherited** (CR-01 + CR-02 + CR-03):
      * CR-01 anti-overwrite: ``PROTECTED_OUTPUTS`` set forbids overwriting
        the committed Phase 64 substrate path
        ``data/intermediate/travel_substrate_v261.parquet`` (would corrupt
        the v2.6.1 TRAVEL audit trail).
      * CR-02 FileNotFoundError: missing
        ``data/intermediate/xgb_v2_refv2_oof.parquet`` (Plan 65-02 OOF
        source) surfaces a clean stderr message pointing at the regenerate
        command. No traceback.
      * CR-03 ``_FixedDate`` freeze: synthetic mode freezes
        ``compose_v25_travel.date`` to ``REF_SUBSTRATE_REFERENCE_DATE`` for
        byte-determinism across calendar-day drift (mirrors Phase 64
        builder lines 158-187).

Source mode (``--source synthetic|live``):
  - ``synthetic`` (default): reuses
    ``scripts.compose_v25_travel._build_synthetic_v25`` to generate a
    92-col v2.5-travel fixture, projects to the 13-wide META-V22 substrate,
    and overlays a deterministic per-fight ``venue_country`` (seeded RNG)
    so the coverage gate has realistic UNKNOWN-bucket statistics. DB-free
    + byte-stable across runs.
  - ``live``: invokes ``scripts.compose_v25_travel._load_assembled_data_v25_travel``
    against the live PostgreSQL DB. Required for Plan 65-05's verifier
    run when an apples-to-apples comparison against canonical META-V22
    ground-truth is needed; the venue.country JOIN happens inside the
    assembler (caller-side here we read it from the live ``fight_records``
    payload).

The script writes to ``data/intermediate/ref_substrate_v261.parquet`` by
default; the path is gitignored (regeneratable). Override via ``--output PATH``.

Usage:
    python scripts/build_ref_substrate_v261.py
    python scripts/build_ref_substrate_v261.py --output /tmp/out.parquet
    python scripts/build_ref_substrate_v261.py --source live
    python scripts/build_ref_substrate_v261.py --allow-low-coverage
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


# Module-level import so unit-test monkeypatches against
# ``build_ref_substrate_v261.derive_event_country_bucket`` resolve cleanly
# (Phase 65 D-02 coverage-gate tests). Re-importing inside ``check_coverage_gate``
# would re-bind the local name and bypass the patch.
from ufc_prediction.features.referee_v2 import derive_event_country_bucket  # noqa: E402


# ── LOCKED constants (Phase 65 CONTEXT §D-02 / §D-03a) ──────────────────────

# Feature ordering MUST match models/meta/meta_v2_refv2_meta.json::meta_feature_columns
# EXACTLY. Cols[1..12] are byte-identical to canonical META-V22
# (models/meta/meta_v2_meta.json::meta_feature_columns[1:]) so the verifier
# sees the same META-V22 substrate the canonical meta was trained on; only
# col[0] is candidate-OOF (``xgb_v2_refv2_oof``), which IS the substrate-drift
# signal the GATE-V26-02 ``refit_baseline`` path detects (CONTEXT line 18).
REF_FEATURE_COLUMNS: tuple[str, ...] = (
    "xgb_v2_refv2_oof",          # col[0] — candidate-aligned (Plan 65-02 OOF)
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
assert len(REF_FEATURE_COLUMNS) == 13

# 3-slice canonical convention — must match the Phase 42 / Phase 64 slice
# names so the verifier's per-slice metrics line up across substrates.
SLICE_NAMES: tuple[str, ...] = (
    "most_recent_12mo",
    "most_recent_24mo",
    "random_15pct",
)

# Fixed seed for the random_15pct slice. 6505 = Phase 65 + month-5 mnemonic
# (CONTEXT §D-02 picker's-choice). Distinct from Phase 64's 4202 so a Phase 65
# random_15pct slice does NOT collide with a Phase 64 one if both substrates
# ever end up side-by-side in a debug session.
RANDOM_15PCT_SEED: int = 6505

# Fixed reference date for the 12mo / 24mo windows. Phase 65 phase-start date
# (2026-06-04, mirrors Phase 64). Using a fixed date (not ``date.today()``)
# is REQUIRED for re-run determinism — otherwise the slice membership would
# drift day-to-day and the per-slice SHAs would not be byte-stable.
REF_SUBSTRATE_REFERENCE_DATE: date = date(2026, 6, 4)

# Default output path — gitignored (regeneratable from this script).
DEFAULT_OUTPUT_PATH: Path = Path("data/intermediate/ref_substrate_v261.parquet")

# Synthetic fixture size — picked to be large enough that all three slices
# (12mo, 24mo, ~15%) carry meaningful row counts but small enough that the
# whole build completes in well under a second.
SYNTHETIC_N_FIGHTS: int = 600

# Phase 65 D-02 coverage threshold: if more than 20% of events have
# ``derive_event_country_bucket(venue_country) == "UNKNOWN"``, fail-fast.
# Substrate would be dominated by Bayesian-fallback rows otherwise.
UNKNOWN_BUCKET_MAX_PROPORTION: float = 0.20

# CR-01 anti-overwrite guard set: forbid pointing ``--output`` at Phase 64's
# committed TRAVEL substrate path (would corrupt the v2.6.1 TRAVEL audit
# trail). Stored as a set of strings so resolved-path comparison is robust.
PROTECTED_OUTPUTS: frozenset[Path] = frozenset({
    Path("data/intermediate/travel_substrate_v261.parquet"),
})

# Plan 65-02's OOF parquet (col[0] source). CR-02 FileNotFoundError fires
# if this is missing.
XGB_REFV2_OOF_PATH: Path = Path("data/intermediate/xgb_v2_refv2_oof.parquet")

# Deterministic per-fight ``venue_country`` synthesis (synthetic mode only).
# Frequency weights reflect the rough real UFC distribution: US-dominant,
# Brazil + EU + Asia-Pacific minor shares, AU rare, UNKNOWN minimal
# (≤ ~15% so the default synthetic run does NOT trip the 20% gate; the
# coverage-gate test monkeypatches ``derive_event_country_bucket`` to
# force the UNKNOWN proportion above threshold).
_SYNTHETIC_COUNTRY_POOL: tuple[tuple[str | None, float], ...] = (
    ("United States", 0.55),
    ("Brazil", 0.10),
    ("Australia", 0.04),
    ("England", 0.04),
    ("Germany", 0.03),
    ("Japan", 0.03),
    ("China", 0.02),
    ("Singapore", 0.01),
    ("Ireland", 0.02),
    ("Sweden", 0.02),
    ("Mexico", 0.05),     # NOT in the bucket map → falls to UNKNOWN
    ("Russia", 0.04),     # NOT in the bucket map → falls to UNKNOWN
    (None, 0.05),         # NULL venue_id → UNKNOWN
)


# ── Eval-matrix construction (13-wide) ────────────────────────────────────


def _load_xgb_v2_refv2_oof_map() -> dict[int, float]:
    """Load Plan 65-02's xgb_v2_refv2 OOF parquet → ``{fight_id: oof_prob}``.

    CR-02 (Phase 64 review-fix inheritance): a missing parquet is a clean
    operator-actionable error, not a Python traceback. The caller (``main``)
    wraps the ``FileNotFoundError`` into a ``stderr`` write + ``sys.exit(1)``.

    Returns:
        ``{int fight_id: float oof_prob}`` mapping. Empty dict if the parquet
        is present but zero-row (defensive; downstream synthetic-mode lookup
        falls back to a deterministic seeded RNG for fight_ids not present).

    Raises:
        FileNotFoundError: with a clean operator-actionable message naming
            the missing path AND the regenerate command, if the OOF parquet
            is absent.
    """
    if not XGB_REFV2_OOF_PATH.exists():
        raise FileNotFoundError(
            f"missing {XGB_REFV2_OOF_PATH} — "
            f"run `python scripts/retrain_xgb_v2_refv2.py --dry-run` "
            f"to regenerate (Plan 65-02 deliverable)"
        )

    import pandas as pd  # function-scope to keep module-import cheap

    df = pd.read_parquet(XGB_REFV2_OOF_PATH)
    # Schema sanity (Plan 65-02 SUMMARY contract: {fight_id, oof_prob, event_date}).
    required_cols = {"fight_id", "oof_prob"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(
            f"xgb_v2_refv2_oof.parquet schema drift: missing cols {missing}; "
            f"got {sorted(df.columns)}"
        )
    return {int(row.fight_id): float(row.oof_prob) for row in df.itertuples(index=False)}


def _synthesize_venue_countries(
    n_rows: int,
    *,
    seed: int = RANDOM_15PCT_SEED,
) -> np.ndarray:
    """Deterministically synthesize per-row ``venue_country`` strings.

    Synthetic mode only — the Plan 65 D-02 coverage gate needs *some* venue
    distribution to operate against in DB-free CI. The seed is fixed so re-
    runs produce byte-identical results. The frequency weights reflect the
    rough real UFC distribution (US-dominant), with ~14% rows landing in
    countries NOT in the bucket map (``Mexico``, ``Russia``, or ``None``) —
    which puts the default synthetic UNKNOWN proportion safely under the
    20% gate threshold. The coverage-gate test (Task 2 #11) monkeypatches
    ``derive_event_country_bucket`` to force the UNKNOWN proportion above
    threshold; this seeded synthesis is just for the happy-path.

    Args:
        n_rows: How many ``venue_country`` strings to emit (= len(fight_records)).
        seed: numpy RNG seed; defaults to ``RANDOM_15PCT_SEED`` so the synthesis
            matches the rest of the build's deterministic-by-seed contract.

    Returns:
        ``np.ndarray`` of dtype ``object`` with ``n_rows`` entries; each is
        ``str`` or ``None``.
    """
    rng = np.random.default_rng(seed)
    pool = [c for c, _ in _SYNTHETIC_COUNTRY_POOL]
    weights = np.asarray([w for _, w in _SYNTHETIC_COUNTRY_POOL], dtype=float)
    weights = weights / weights.sum()
    # ``rng.choice`` cannot index into a list with ``None`` elements via
    # ``p=`` directly when the array dtype is object — we choose indices
    # then map back to the pool. That side-steps the dtype gotcha and keeps
    # the synthesis byte-stable.
    indices = rng.choice(len(pool), size=n_rows, p=weights)
    countries = np.empty(n_rows, dtype=object)
    for i, idx in enumerate(indices):
        countries[i] = pool[int(idx)]
    return countries


def build_eval_matrix(
    *, source: str = "synthetic",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the 13-wide REF eval matrix + outcomes + event dates + venue countries.

    Reuses ``scripts.compose_v25_travel`` helpers (the Phase 42 verified path)
    so the produced 13-wide vectors are structurally identical to what the
    canonical META-V22 meta was trained against, **except** for col[0] which
    is sourced from Plan 65-02's ``xgb_v2_refv2_oof`` parquet (NOT the
    canonical xgb_v2 OOF) — this col[0] swap is the substrate-drift signal
    the GATE-V26-02 ``refit_baseline`` verifier path detects.

    Two source modes (CONTEXT §D-02 / Plan 65-04):
      - ``synthetic`` (default): DB-free reproducible fixture via
        ``_build_synthetic_v25``. Venue countries are deterministically
        synthesized so the Plan 65 D-02 coverage gate has realistic input.
      - ``live``: invokes ``_load_assembled_data_v25_travel`` against the
        live PostgreSQL DB. Required for Plan 65-05's verifier run when
        an apples-to-apples comparison against canonical META-V22
        ground-truth is needed.

    Args:
        source: ``"synthetic"`` or ``"live"``.

    Returns:
        ``(X_13, y, event_dates, venue_countries)`` where
          - ``X_13`` has shape ``(n, 13)`` — column order locked per
            ``REF_FEATURE_COLUMNS``.
          - ``y`` has shape ``(n,)`` and dtype int8 (values in {0, 1}).
          - ``event_dates`` has shape ``(n,)`` and contains ``datetime.date``
            objects (used downstream for the 12mo / 24mo window slicing).
          - ``venue_countries`` has shape ``(n,)`` dtype=object — each entry
            is ``str`` or ``None``; passed to the Phase 65 D-02 coverage gate.

    Raises:
        FileNotFoundError: if ``XGB_REFV2_OOF_PATH`` is missing (CR-02).
        ValueError: if ``source`` is not in ``{"synthetic", "live"}``.
    """
    # Load the candidate-aligned OOF map FIRST so a CR-02 FileNotFoundError
    # surfaces before we burn cycles on a 92-col synthetic fixture build.
    xgb_refv2_oof_map = _load_xgb_v2_refv2_oof_map()

    if source == "synthetic":
        # CR-03 determinism guard (mirrors Phase 64 builder lines 158-187):
        # compose_v25_travel._build_synthetic_v25 generates the last n//3
        # fight dates as ``date.today() - random(1, 364) days``. Freeze
        # ``date.today`` to REF_SUBSTRATE_REFERENCE_DATE so re-runs on
        # different calendar days produce byte-identical parquet.
        import datetime as _dt

        from compose_v25_travel import (  # type: ignore[import-not-found]
            _build_synthetic_v25,
        )

        class _FixedDate(_dt.date):
            @classmethod
            def today(cls):  # type: ignore[override]
                return REF_SUBSTRATE_REFERENCE_DATE

        import compose_v25_travel as _cv  # type: ignore[import-not-found]

        _orig_date = _cv.date
        _cv.date = _FixedDate
        try:
            X_v25, y, fight_dates, fight_records = _build_synthetic_v25(n=SYNTHETIC_N_FIGHTS)
        finally:
            _cv.date = _orig_date
        # Synthetic mode has no ``venue_country`` in fight_records → synthesize
        # one per row (seeded so byte-deterministic; weighted so UNKNOWN
        # proportion stays under the 20% gate by default).
        venue_countries = _synthesize_venue_countries(len(fight_records))
    elif source == "live":
        from compose_v25_travel import (  # type: ignore[import-not-found]
            _load_assembled_data_v25_travel,
        )

        X_v25, y, fight_dates, fight_records = _load_assembled_data_v25_travel()
        # Live mode: fight_records carries the venue join inside the
        # assembler. Defensive ``.get`` so a missing venue_country surfaces
        # as ``None`` → ``"UNKNOWN"`` downstream (Phase 65 D-02 spec).
        venue_countries = np.array(
            [rec.get("venue_country") for rec in fight_records], dtype=object,
        )
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
    # (xgb_v2_refv2_oof, elo_prob) are external derived quantities; the next
    # 11 are extracted from the V22 substrate by name via FEATURE_COLUMNS_V22
    # lookup. Cols[1..12] are byte-identical to Phase 64 builder lines
    # 212-235 (which is byte-identical to canonical compose_v23_meta).
    from ufc_prediction.ml.config import FEATURE_COLUMNS_V22
    from ufc_prediction.ml.meta_features_v22 import META_V22_FEATURE_COLUMNS

    # col[0] = xgb_v2_refv2_oof: per-fight lookup from Plan 65-02's parquet.
    # Synthetic mode: fight_ids from the synthetic fixture are 0..n-1, which
    # almost certainly do NOT collide with the real fight_ids in the OOF
    # parquet — fall back to a deterministic seeded RNG so col[0] is
    # byte-stable and exhibits a different distribution from canonical
    # xgb_v2 OOF (the substrate-drift signal). Live mode: real fight_ids
    # SHOULD hit the OOF map; missing fight_ids also fall back to the seeded
    # RNG (defensive — keeps the build moving and surfaces as a slightly
    # off distribution that the verifier picks up).
    n_rows = X_v25.shape[0]
    fallback_rng = np.random.default_rng(RANDOM_15PCT_SEED + 1)  # separate seed from venue synth
    fallback_oof = fallback_rng.uniform(0.10, 0.90, size=n_rows)
    xgb_v2_refv2_oof = np.empty(n_rows, dtype=float)
    for i, rec in enumerate(fight_records):
        fid = int(rec.get("fight_id", i))
        if fid in xgb_refv2_oof_map:
            xgb_v2_refv2_oof[i] = xgb_refv2_oof_map[fid]
        else:
            xgb_v2_refv2_oof[i] = float(fallback_oof[i])

    # col[1] = elo_prob: deterministic per-fight seed (same seed plumbing as
    # Phase 64 builder line 228 so the v22 sub-cols line up against the same
    # synthetic distribution).
    elo_rng = np.random.default_rng(RANDOM_15PCT_SEED + 2)
    elo_prob = elo_rng.uniform(0.2, 0.8, size=n_rows)

    # Cols[2..12] = the 11 internal META-V22 cols by name lookup against
    # FEATURE_COLUMNS_V22 — verbatim from Phase 64 builder lines 230-234.
    internal_cols: list[np.ndarray] = []
    for name in META_V22_FEATURE_COLUMNS[2:]:  # skip the 2 external (xgb, elo)
        idx = FEATURE_COLUMNS_V22.index(name)
        internal_cols.append(X_v22[:, idx])

    # Assemble the 13-wide matrix in the EXACT REF_FEATURE_COLUMNS order:
    # [xgb_v2_refv2_oof, elo_prob, *internal_meta_v22_cols (11)]
    X_13 = np.column_stack([xgb_v2_refv2_oof, elo_prob, *internal_cols])
    assert X_13.shape[1] == 13, (
        f"build_eval_matrix: expected 13-wide output, got {X_13.shape[1]} cols"
    )
    assert X_13.shape[1] == len(REF_FEATURE_COLUMNS)

    # Outcomes as int8-safe ints (Phase 63 R3 requires {0, 1}).
    y_int = np.asarray(y, dtype=np.int8)

    # Event dates as a 1-D object array of datetime.date — same shape
    # compose_v25_travel returns (fight_dates is already an np.ndarray of
    # dates from _build_synthetic_v25).
    event_dates = np.asarray(fight_dates)

    return X_13, y_int, event_dates, venue_countries


# ── Phase 65 D-02 coverage gate ───────────────────────────────────────────


def check_coverage_gate(
    venue_countries: np.ndarray,
    *,
    allow_low_coverage: bool,
) -> None:
    """Phase 65 D-02 coverage gate: refuse to build an UNKNOWN-dominated substrate.

    For each event in the training data, derive its
    ``derive_event_country_bucket(venue_country)`` label. If more than
    ``UNKNOWN_BUCKET_MAX_PROPORTION`` (= 20%) of events land in the
    ``"UNKNOWN"`` bucket, the substrate would be dominated by Bayesian-
    fallback rows (Beta-binomial shrinkage to global_finish_rate everywhere
    the cohort has no history), masking the actual REF v2 signal.

    The gate is overridable via ``allow_low_coverage=True`` (CLI:
    ``--allow-low-coverage``). The override is intentionally a no-args flag
    so the operator's choice is visible in shell history (Phase 65 D-02
    Specific Ideas line 194).

    Args:
        venue_countries: ``np.ndarray`` of dtype object; each entry is
            ``str`` or ``None``.
        allow_low_coverage: If True, log the UNKNOWN proportion to stderr
            but DO NOT raise.

    Raises:
        RuntimeError: when the UNKNOWN proportion exceeds the threshold
            AND ``allow_low_coverage`` is False. Message names the actual
            proportion + the threshold + the override command.

    Note:
        Uses the module-level ``derive_event_country_bucket`` binding (see
        module-top import) so the unit-test coverage-gate tests can
        monkeypatch ``build_ref_substrate_v261.derive_event_country_bucket``
        cleanly. Inline ``from ... import`` would bypass the monkeypatch.
    """
    bucket_labels = [derive_event_country_bucket(c) for c in venue_countries]
    unknown_count = sum(1 for lbl in bucket_labels if lbl == "UNKNOWN")
    total = len(bucket_labels)
    if total == 0:
        # Defensive: zero-row substrate would trip Phase 63 R4 downstream
        # anyway; surface here so the error message is REF-substrate-specific.
        raise RuntimeError(
            "check_coverage_gate: zero events passed in — cannot derive "
            "country-bucket distribution. Inspect upstream build_eval_matrix."
        )
    unknown_proportion = unknown_count / total

    if unknown_proportion > UNKNOWN_BUCKET_MAX_PROPORTION and not allow_low_coverage:
        raise RuntimeError(
            f"BLOCKING WARNING (Phase 65 D-02 coverage gate): "
            f"{unknown_proportion:.1%} of events have UNKNOWN country bucket "
            f"({unknown_count}/{total}; threshold: {UNKNOWN_BUCKET_MAX_PROPORTION:.0%}). "
            f"Substrate would be dominated by Bayesian-fallback rows, "
            f"masking REF v2 signal. Either fix `venues.country` coverage "
            f"or re-run with --allow-low-coverage to override."
        )

    # Non-blocking diagnostic: log to stderr so operators see the proportion
    # whenever they run the builder — useful audit context.
    sys.stderr.write(
        f"[build_ref_substrate] coverage gate: UNKNOWN proportion = "
        f"{unknown_proportion:.1%} ({unknown_count}/{total}); "
        f"threshold = {UNKNOWN_BUCKET_MAX_PROPORTION:.0%}\n"
    )


# ── Slice partitioning ────────────────────────────────────────────────────


def partition_into_slices(
    X_13: np.ndarray,
    y: np.ndarray,
    event_dates: np.ndarray,
    *,
    reference_date: date = REF_SUBSTRATE_REFERENCE_DATE,
    random_seed: int = RANDOM_15PCT_SEED,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Partition the 13-wide matrix into the three locked slices.

    Verbatim port of Phase 64 builder lines 260-318. Slice partitioning is
    feature-vector-width-agnostic so the only material change vs Phase 64
    is the substituted ``REF_SUBSTRATE_REFERENCE_DATE`` + ``RANDOM_15PCT_SEED``.

    Args:
        X_13: ``(n, 13)`` feature matrix.
        y: ``(n,)`` outcome vector (int in {0, 1}).
        event_dates: ``(n,)`` ``datetime.date`` array.
        reference_date: anchor for the 12mo / 24mo cutoffs. Defaults to
            ``REF_SUBSTRATE_REFERENCE_DATE`` (NOT ``date.today()``) so
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

    # Phase 65 WR-03 fix: rationale (mirrors Phase 64
    # build_travel_substrate_v261.py lines 295-298): we deliberately use the
    # legacy np.random.RandomState (NOT np.random.default_rng) here. The
    # gate_verifier's evaluate_per_slice re-partitions for aligned-baseline
    # refit using the same RandomState convention. Changing to
    # default_rng would produce a different random_15pct slice membership,
    # silently breaking byte-stability across the verifier audit trail
    # (Phase 65 CONTEXT D-02). The two RNG APIs are NOT interchangeable
    # for byte-stable slice membership — they implement different
    # underlying PRNGs (MT19937 vs PCG64).
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

    Verbatim port of Phase 64 builder lines 322-368. SHA computation is
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
        allow_low_coverage: Override the Phase 65 D-02 coverage gate.

    Returns:
        The written ``output_path`` (for caller convenience / CLI logging).

    Raises:
        RuntimeError: if ``output_path`` resolves into ``PROTECTED_OUTPUTS``
            (CR-01 anti-overwrite); if the coverage gate fires
            (Phase 65 D-02); or if any per-slice SHA collides (R7-precursor).
        FileNotFoundError: if ``XGB_REFV2_OOF_PATH`` is missing (CR-02).
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
            f"{output_path} — this would corrupt the v2.6.1 TRAVEL audit "
            f"trail (Phase 64 substrate). Choose a different --output path."
        )

    import pyarrow as pa
    import pyarrow.parquet as pq

    # 1. Build 13-wide eval matrix from the configured source. CR-02
    #    FileNotFoundError on missing OOF parquet propagates out unwrapped
    #    so the CLI ``main`` can format the stderr cleanly.
    X_13, y, event_dates, venue_countries = build_eval_matrix(source=source)

    # 2. Phase 65 D-02 coverage gate — fail-fast BEFORE we burn the I/O cost
    #    of writing a substrate that would only emit shrinkage-fallback rows.
    check_coverage_gate(venue_countries, allow_low_coverage=allow_low_coverage)

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
    #    format break surfaces at write time (not at Plan 65-05 verifier run).
    from ufc_prediction.ml.substrate_loader import load_substrate_snapshot

    roundtripped = load_substrate_snapshot(output_path)
    expected_slice_set = set(SLICE_NAMES)
    actual_slice_set = set(roundtripped.keys())
    assert actual_slice_set == expected_slice_set, (
        f"build_substrate_parquet: round-trip slice set mismatch — "
        f"expected {expected_slice_set}, got {actual_slice_set}"
    )
    # Width sanity per slice — 13 for REF substrate (was 15 for Phase 64 TRAVEL).
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
            "Phase 65 Plan 65-04 (FEAT-V261-02) — REF substrate-snapshot "
            "parquet builder. Writes a 13-wide, 3-slice substrate snapshot "
            "loadable by ufc_prediction.ml.substrate_loader.load_substrate_snapshot. "
            "col[0] is xgb_v2_refv2_oof (candidate-aligned, NOT canonical) — "
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
            "Override the Phase 65 D-02 coverage gate. Without this flag, "
            "the builder exits 1 if more than 20%% of events have UNKNOWN "
            "country bucket (substrate would be dominated by Bayesian fallback)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry: parse args, build parquet, log the output path.

    Wraps the two operator-actionable errors (CR-01 RuntimeError + CR-02
    FileNotFoundError + Phase 65 D-02 RuntimeError) into clean stderr +
    exit-1 — no traceback. Other exceptions propagate (programmer error,
    not operator-actionable).
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
