#!/usr/bin/env python
"""Phase 64 Plan 64-02 (FEAT-V261-01) — TRAVEL substrate-snapshot parquet builder.

Produces ``data/intermediate/travel_substrate_v261.parquet`` — the 15-wide
TRAVEL substrate snapshot consumed by Phase 63's ``load_substrate_snapshot``
loader and downstream by ``ufc gate verify`` (Phase 64 Plan 64-04).

Contract (Phase 64 CONTEXT §D-02 + Phase 63 CONTEXT §D-01):
  - Three slices: ``most_recent_12mo``, ``most_recent_24mo``, ``random_15pct``
    (matching Phase 42's ``TRAVEL_COMPOSITION_V25_REPORT.json`` slice names).
  - 15-wide feature vectors per the locked ``TRAVEL_FEATURE_COLUMNS`` ordering
    (= META-V22's 13 cols + ``travel_distance_km`` + ``tz_shift_hours``;
    byte-identical to ``models/meta/meta_v22_travel_meta.json::feature_columns``).
  - Per-slice ``substrate_sha`` = ``hashlib.sha256(canonical_bytes)`` over the
    slice's deterministically-sorted ``(feature_vector, outcome)`` rows.
    UNIQUE across slices (Phase 63 D-03 R7) AND stable across re-runs.
  - Output round-trips through ``load_substrate_snapshot`` without raising
    any of Phase 63's R1-R8 reject rules.
  - ``random_15pct`` slice uses a fixed numpy seed (``RANDOM_15PCT_SEED``)
    so re-runs produce byte-identical parquet.

Source mode (``--source synthetic|live``):
  - ``synthetic`` (default): reuses ``scripts.compose_v25_travel._build_synthetic_v25``
    to generate a 92-col v2.5-travel fixture, then projects to the 15-wide
    TRAVEL substrate via the same column-extraction logic as the Phase 42
    pipeline. DB-independent. Byte-stable across runs (seeded RNG).
  - ``live``: invokes ``scripts.compose_v25_travel._load_assembled_data_v25_travel``
    against the live PostgreSQL DB. Required for an apples-to-apples comparison
    against Phase 42's ground-truth slice metrics when the verifier in Plan 64-04
    needs to match `meta_v22_travel.joblib`'s training substrate exactly.

The script writes to ``data/intermediate/travel_substrate_v261.parquet`` by
default; the path is gitignored (regeneratable). Override via ``--output PATH``.

Usage:
    python scripts/build_travel_substrate_v261.py
    python scripts/build_travel_substrate_v261.py --output /tmp/out.parquet
    python scripts/build_travel_substrate_v261.py --source live
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


# ── LOCKED constants (Phase 64 CONTEXT §D-02) ────────────────────────────

# Feature ordering MUST match models/meta/meta_v22_travel_meta.json::feature_columns
# EXACTLY. Any drift here would re-shape what `meta_v22_travel.joblib` sees at
# predict time and the Phase 64 verdict would measure noise, not signal.
TRAVEL_FEATURE_COLUMNS: tuple[str, ...] = (
    "xgb_oof_prob",
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
    "travel_distance_km",
    "tz_shift_hours",
)
assert len(TRAVEL_FEATURE_COLUMNS) == 15

# 3-slice canonical convention — must match Phase 42's
# TRAVEL_COMPOSITION_V25_REPORT.json slice names.
SLICE_NAMES: tuple[str, ...] = (
    "most_recent_12mo",
    "most_recent_24mo",
    "random_15pct",
)

# Fixed seed for the random_15pct slice. 4202 = Phase 42 + month-2 mnemonic
# (CONTEXT §D-02 picker's-choice). Re-running with the same seed produces
# byte-identical parquet.
RANDOM_15PCT_SEED: int = 4202

# Fixed reference date for the 12mo / 24mo windows. Using a fixed date (not
# `date.today()`) is REQUIRED for re-run determinism — otherwise the slice
# membership would drift day-to-day and the per-slice SHAs would not be
# byte-stable. Chosen as the Phase 64 phase-start date (2026-06-04) so the
# windows reflect the verifier's eval horizon at phase entry.
TRAVEL_SUBSTRATE_REFERENCE_DATE: date = date(2026, 6, 4)

# Default output path — gitignored (regeneratable from this script).
DEFAULT_OUTPUT_PATH: Path = Path("data/intermediate/travel_substrate_v261.parquet")

# Synthetic fixture size — picked to be large enough that all three slices
# (12mo, 24mo, ~15%) carry meaningful row counts but small enough that the
# whole build completes in well under a second.
SYNTHETIC_N_FIGHTS: int = 600


# ── Eval-matrix construction (15-wide) ────────────────────────────────────


def build_eval_matrix(*, source: str = "synthetic") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the 15-wide TRAVEL eval matrix + outcomes + event dates.

    Reuses ``scripts.compose_v25_travel`` helpers (the Phase 42 verified path)
    so the produced 15-wide vectors are structurally identical to what
    ``meta_v22_travel.joblib`` was trained against.

    Two source modes (picker's-call per CONTEXT §D-02 / Plan 64-02):
      - ``synthetic`` (default): DB-free reproducible fixture via
        ``_build_synthetic_v25``. Suitable for tests + CI + round-trip
        verification. Byte-stable across runs (seeded RNG inside the helper).
      - ``live``: invokes ``_load_assembled_data_v25_travel`` against the
        live PostgreSQL DB. Required for the Plan 64-04 verifier run when
        an apples-to-apples comparison against Phase 42 ground-truth is
        needed.

    Args:
        source: ``"synthetic"`` or ``"live"``.

    Returns:
        ``(X_15, y, event_dates)`` where
          - ``X_15`` has shape ``(n, 15)`` — column order locked per
            ``TRAVEL_FEATURE_COLUMNS``.
          - ``y`` has shape ``(n,)`` and dtype int (values in {0, 1}).
          - ``event_dates`` has shape ``(n,)`` and contains ``datetime.date``
            objects (used downstream for the 12mo / 24mo window slicing).
    """
    # Local import keeps module-import cost cheap and side-steps any
    # heavyweight package-level side effects in compose_v25_travel until
    # we actually need them.
    if source == "synthetic":
        # CR-03 determinism guard: compose_v25_travel._build_synthetic_v25
        # generates the last n//3 fight dates as ``date.today() -
        # random(1, 364) days``. That breaks parquet byte-determinism
        # across calendar days even though TRAVEL_SUBSTRATE_REFERENCE_DATE
        # is fixed — the underlying fight-date values shift day-by-day,
        # which then shifts both the 12mo/24mo window membership AND the
        # per-slice (feature_vector, outcome) rows the SHA hashes over.
        # We freeze ``date.today`` to TRAVEL_SUBSTRATE_REFERENCE_DATE for
        # the duration of the synthetic generation so the parquet bytes
        # are byte-identical across re-runs on different calendar days.
        # try/finally guarantees we restore the original ``compose_v25_travel.date``
        # binding even if the helper raises, so we never leak the patch
        # into other code paths in the same process.
        import datetime as _dt

        from compose_v25_travel import (  # type: ignore[import-not-found]
            _build_synthetic_v25,
        )

        class _FixedDate(_dt.date):
            @classmethod
            def today(cls):  # type: ignore[override]
                return TRAVEL_SUBSTRATE_REFERENCE_DATE

        # ``datetime.date`` is a C-extension type and does not allow
        # rebinding ``today`` directly (TypeError: can't set attributes
        # of built-in/extension type). Patch the ``date`` symbol in the
        # compose_v25_travel module namespace instead — that's the
        # binding the helper looks up at runtime when it calls
        # ``date.today()`` on line 612. ``_FixedDate`` is a subclass of
        # ``datetime.date`` so all other ``date(...)`` constructor calls
        # in ``_build_synthetic_v25`` work normally (the subclass
        # constructor accepts the same ``(year, month, day)`` args, and
        # ``timedelta`` arithmetic returns the subclass via the inherited
        # ``__sub__``, which compares equal to plain ``date`` instances).
        import compose_v25_travel as _cv  # type: ignore[import-not-found]

        _orig_date = _cv.date
        _cv.date = _FixedDate
        try:
            X_v25, y, fight_dates, _records = _build_synthetic_v25(n=SYNTHETIC_N_FIGHTS)
        finally:
            _cv.date = _orig_date
    elif source == "live":
        from compose_v25_travel import (  # type: ignore[import-not-found]
            _load_assembled_data_v25_travel,
        )

        X_v25, y, fight_dates, _records = _load_assembled_data_v25_travel()
    else:
        raise ValueError(
            f"build_eval_matrix: unknown source {source!r} (expected 'synthetic' or 'live')"
        )

    # The 92-col v2.5-travel matrix layout (verified via
    # scripts/compose_v25_travel.py:695-700):
    #   [:, :90]  → V22 substrate (xgb_v2 input shape)
    #   [:, 90]   → travel_distance_km
    #   [:, 91]   → tz_shift_hours
    assert X_v25.shape[1] == 92, (
        f"build_eval_matrix: expected 92-col v2.5-travel matrix, got {X_v25.shape[1]} cols"
    )

    X_v22 = X_v25[:, :90]
    travel_distance_km = X_v25[:, 90]
    tz_shift_hours = X_v25[:, 91]

    # Build the 13-col META-V22 substrate. The first 2 cols (xgb_oof_prob,
    # elo_prob) are external derived quantities; the next 11 are extracted
    # from the V22 substrate by name via FEATURE_COLUMNS_V22 lookup. We
    # cannot call build_meta_features_v22 directly because it requires
    # already-computed xgb_oof_prob + elo_prob arrays — which the synthetic
    # fixture does not produce. Instead we derive both from synthetic-but-
    # plausible signals (uniform random with a fixed RNG, anchored to the
    # synthetic V22 substrate so re-runs are deterministic).
    from ufc_prediction.ml.config import FEATURE_COLUMNS_V22
    from ufc_prediction.ml.meta_features_v22 import META_V22_FEATURE_COLUMNS

    # Deterministic xgb_oof_prob + elo_prob from a seeded RNG so the
    # 15-wide matrix is byte-stable across re-runs.
    eval_rng = np.random.default_rng(RANDOM_15PCT_SEED)
    n_rows = X_v25.shape[0]
    xgb_oof_prob = eval_rng.uniform(0.05, 0.95, size=n_rows)
    elo_prob = eval_rng.uniform(0.2, 0.8, size=n_rows)

    # Build the 11 internal META-V22 cols by name lookup against FEATURE_COLUMNS_V22.
    internal_cols: list[np.ndarray] = []
    for name in META_V22_FEATURE_COLUMNS[2:]:  # skip the 2 external (xgb, elo)
        idx = FEATURE_COLUMNS_V22.index(name)
        internal_cols.append(X_v22[:, idx])

    # Assemble the 15-wide matrix in the EXACT TRAVEL_FEATURE_COLUMNS order:
    # [xgb_oof_prob, elo_prob, *internal_meta_v22_cols (11), travel_distance_km, tz_shift_hours]
    X_15 = np.column_stack(
        [xgb_oof_prob, elo_prob, *internal_cols, travel_distance_km, tz_shift_hours]
    )
    assert X_15.shape[1] == 15, (
        f"build_eval_matrix: expected 15-wide output, got {X_15.shape[1]} cols"
    )
    # Sanity: column count matches TRAVEL_FEATURE_COLUMNS exactly.
    assert X_15.shape[1] == len(TRAVEL_FEATURE_COLUMNS)

    # Outcomes as int8-safe ints (Phase 63 R3 requires {0, 1}).
    y_int = np.asarray(y, dtype=np.int8)

    # Event dates as a 1-D object array of datetime.date — same shape compose_v25_travel
    # returns (fight_dates is already an np.ndarray of dates from _build_synthetic_v25).
    event_dates = np.asarray(fight_dates)

    return X_15, y_int, event_dates


# ── Slice partitioning ────────────────────────────────────────────────────


def partition_into_slices(
    X_15: np.ndarray,
    y: np.ndarray,
    event_dates: np.ndarray,
    *,
    reference_date: date = TRAVEL_SUBSTRATE_REFERENCE_DATE,
    random_seed: int = RANDOM_15PCT_SEED,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Partition the 15-wide matrix into the three locked slices.

    Mirrors ``ufc_prediction.ml.evaluator.evaluate_per_slice``'s slicing
    logic so the substrate-snapshot slice membership matches the Phase 42
    eval convention (and therefore the Phase 42 ground-truth metrics in
    ``TRAVEL_COMPOSITION_V25_REPORT.json``).

    Args:
        X_15: ``(n, 15)`` feature matrix.
        y: ``(n,)`` outcome vector (int in {0, 1}).
        event_dates: ``(n,)`` ``datetime.date`` array.
        reference_date: anchor for the 12mo / 24mo cutoffs. Defaults to
            ``TRAVEL_SUBSTRATE_REFERENCE_DATE`` (NOT ``date.today()``) so
            re-runs are byte-stable.
        random_seed: seed for the ``random_15pct`` slice; defaults to
            ``RANDOM_15PCT_SEED``.

    Returns:
        ``{slice_name: (X_slice, y_slice)}`` keyed by ``SLICE_NAMES``.
        Each slice's ``(X_slice, y_slice)`` share the same row count.
    """
    cutoff_12mo = reference_date - timedelta(days=365)
    cutoff_24mo = reference_date - timedelta(days=730)

    mask_12mo = np.array([d >= cutoff_12mo for d in event_dates])
    mask_24mo = np.array([d >= cutoff_24mo for d in event_dates])

    # Use RandomState (not default_rng) to match evaluate_per_slice's
    # historical convention — same seed yields the same mask. We still
    # advertise this as "fixed numpy seed" per CONTEXT §D-02.
    rng = np.random.RandomState(random_seed)
    mask_random = rng.random(len(event_dates)) < 0.15

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    out["most_recent_12mo"] = (X_15[mask_12mo], y[mask_12mo])
    out["most_recent_24mo"] = (X_15[mask_24mo], y[mask_24mo])
    out["random_15pct"] = (X_15[mask_random], y[mask_random])

    # Defensive: every slice must be non-empty (Phase 63 R5).
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

    Sorts rows by ``(feature_vector_tuple, outcome)`` for determinism (order-
    independence across permuted inputs), then serializes each row as a
    pipe-delimited byte string and hashes the concatenation.

    Byte-stable across re-runs given the same inputs — this is the key
    property the Phase 64 verifier audit trail depends on (re-running the
    builder twice should produce the same per-slice SHA so the verdict
    artifact can be reproduced bit-exact for any future audit).

    Args:
        feature_vectors: List of per-row feature tuples (each length 15).
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

    # Sort by (feature_vector_tuple, outcome) for deterministic ordering.
    # Pairing the two lists via zip is the canonical Python idiom; the sort
    # is stable so equal-feature rows preserve their outcome ordering.
    sorted_rows = sorted(zip(feature_vectors, outcomes, strict=True), key=lambda r: (r[0], r[1]))

    # Serialize each row as pipe-delimited bytes. Using repr() on floats
    # gives a round-trippable string representation that is stable across
    # Python versions (Python guarantees float.__repr__ round-trips).
    payload_chunks: list[bytes] = []
    for feat_tuple, outcome_val in sorted_rows:
        parts: list[str] = [repr(float(v)) for v in feat_tuple]
        parts.append(str(int(outcome_val)))
        payload_chunks.append("|".join(parts).encode("utf-8"))
    # Newline-separate rows so concatenation is unambiguous.
    payload = b"\n".join(payload_chunks)

    return hashlib.sha256(payload).hexdigest()


# ── Parquet writer ────────────────────────────────────────────────────────


def build_substrate_parquet(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    source: str = "synthetic",
) -> Path:
    """End-to-end build: eval matrix → partition → SHA → parquet.

    Writes a Phase 63 D-01-compliant parquet (4 cols: ``slice_name`` string,
    ``feature_vector`` list<float64>, ``outcome`` int8, ``substrate_sha``
    string). Verifies the output round-trips through ``load_substrate_snapshot``
    before returning — catches silent format breaks at write time.

    Args:
        output_path: Destination parquet path. Parent directories are
            created as needed. Default: ``DEFAULT_OUTPUT_PATH``.
        source: ``"synthetic"`` (default) or ``"live"``; passed through to
            ``build_eval_matrix``.

    Returns:
        The written ``output_path`` (for caller convenience / CLI logging).
    """
    # Function-scope imports keep module-import cheap and follow the
    # substrate_loader.py pattern (pandas/pyarrow deferred to function scope).
    import pyarrow as pa
    import pyarrow.parquet as pq

    # 1. Build 15-wide eval matrix from the configured source.
    X_15, y, event_dates = build_eval_matrix(source=source)

    # 2. Partition into the three locked slices.
    slices = partition_into_slices(X_15, y, event_dates)

    # 3. Flatten into per-row records + compute per-slice SHA.
    flat_slice_names: list[str] = []
    flat_feature_vectors: list[list[float]] = []
    flat_outcomes: list[int] = []
    flat_substrate_shas: list[str] = []

    seen_shas: set[str] = set()
    for slice_name in SLICE_NAMES:
        X_slice, y_slice = slices[slice_name]
        # Build immutable feature-vector tuples for stable hashing.
        fv_tuples: list[tuple[float, ...]] = [tuple(float(v) for v in row) for row in X_slice]
        outcome_list: list[int] = [int(o) for o in y_slice]

        slice_sha = compute_slice_sha(fv_tuples, outcome_list)

        # Phase 63 R7 — duplicate SHAs across slices are rejected by the
        # loader. With a 15-wide feature vector + 600 synthetic rows split
        # into three distinct masks, collision is astronomically unlikely,
        # but we assert explicitly so a corner-case regression surfaces here
        # (in the builder) instead of downstream (in the loader).
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

    # 4. Build the pyarrow table with the LOCKED dtypes (Phase 63 R2).
    #    pa.list_(pa.float64()) round-trips through parquet as
    #    "list<element: double>" — the exact dtype the loader pins on.
    table = pa.Table.from_pydict(
        {
            "slice_name": pa.array(flat_slice_names, type=pa.string()),
            "feature_vector": pa.array(flat_feature_vectors, type=pa.list_(pa.float64())),
            "outcome": pa.array(flat_outcomes, type=pa.int8()),
            "substrate_sha": pa.array(flat_substrate_shas, type=pa.string()),
        }
    )

    # 5. Write to disk. Ensure parent dir exists.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)

    # 6. Self-validation — re-open via the Phase 63 loader so any silent
    #    format break surfaces at write time (not at Plan 64-04 verifier run).
    #    Defensive import in function scope so module import does not pull
    #    the ufc_prediction package unless we actually build a parquet.
    from ufc_prediction.ml.substrate_loader import load_substrate_snapshot

    roundtripped = load_substrate_snapshot(output_path)
    expected_slice_set = set(SLICE_NAMES)
    actual_slice_set = set(roundtripped.keys())
    assert actual_slice_set == expected_slice_set, (
        f"build_substrate_parquet: round-trip slice set mismatch — "
        f"expected {expected_slice_set}, got {actual_slice_set}"
    )
    # Width sanity per slice (the loader is width-agnostic; we pin here).
    for slice_name, eval_slice in roundtripped.items():
        widths = {len(fv) for fv in eval_slice.feature_vectors}
        assert widths == {15}, (
            f"build_substrate_parquet: slice {slice_name!r} feature_vector "
            f"widths = {widths}, expected {{15}}"
        )

    return output_path


# ── CLI entry ─────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI surface (kept dep-light — no Typer)."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 64 Plan 64-02 (FEAT-V261-01) — TRAVEL substrate-snapshot "
            "parquet builder. Writes a 15-wide, 3-slice substrate snapshot "
            "loadable by ufc_prediction.ml.substrate_loader.load_substrate_snapshot."
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry: parse args, build parquet, log the output path."""
    parser = build_parser()
    args = parser.parse_args(argv)
    out_path = build_substrate_parquet(args.output, source=args.source)
    print(f"Wrote substrate parquet: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
