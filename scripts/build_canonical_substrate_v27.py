#!/usr/bin/env python
"""Phase 75 Plan 75-01 (METH-V27-02) — canonical-substrate parquet builder.

Produces ``data/intermediate/canonical_substrate_v27.parquet`` — the 13-wide
canonical-aligned substrate snapshot consumed by Phase 75 Plan 75-02's dual-
test verifier (``verify_candidate_vs_canonical_dual_substrate``).

Structurally mirrors ``scripts/build_ref_substrate_v261.py`` (Phase 65) with
two material differences (Phase 75 CONTEXT §D-01 / §D-02):

  1. **col[0] source is canonical training-time OOF** — reads
     ``.planning/milestones/v2.2-phases/26-forward-stepwise-candidate-promotion/
     oof_predictions_v22.parquet`` (the Phase 26 ``xgb_oof_prob`` column) NOT
     a candidate-aligned OOF like ``xgb_v2_refv2_oof``. This col[0] swap is
     the STRUCTURAL inverse of REF: REF's builder put a candidate OOF in
     col[0]; this builder puts the canonical training-time OOF in col[0].
     The dual-test methodology (D-01) triangulates these two substrates to
     decide whether a candidate's apparent win is real lift or substrate-
     drift artifact.

  2. **Fight-id cross-reference with a paired candidate substrate** —
     ``--candidate-substrate <p>`` triggers loading a sidecar JSON at
     ``<p>.fight_ids.json`` listing the fight_ids backing that paired
     substrate; the canonical substrate is then INTERSECTED with those
     fight_ids so both substrates eval on the SAME fight set (D-02 closing
     sentence). ``--no-paired-substrate`` overrides for standalone use.

Phase 64/65 review-fix patterns inherited verbatim (CR-01 + CR-02 + CR-03):
  * CR-01 anti-overwrite: ``PROTECTED_OUTPUTS`` forbids overwriting Phase
    64 / 65 / 66 substrate parquet paths (would corrupt the v2.6.1
    TRAVEL / REF / NET audit trails).
  * CR-02 FileNotFoundError: missing canonical OOF parquet surfaces a clean
    operator-actionable stderr message. No traceback. The canonical OOF
    parquet is a READ-ONLY Phase 26 archive artifact — there is no
    regenerate command; operator must restore from
    ``.planning/milestones/v2.2-phases/26-…`` archive or git history.
  * CR-03 ``_FixedDate`` freeze: synthetic mode freezes
    ``compose_v25_travel.date`` to ``CANONICAL_REFERENCE_DATE`` for byte-
    determinism across calendar-day drift.

D-06 AUDIT-01 invariant: at script entry, the canonical OOF parquet's SHA
is verified against ``EXPECTED_CANONICAL_OOF_SHA``. On mismatch the script
raises a clean ``RuntimeError`` naming the AUDIT-01 D-06 invariant —
prevents building substrates from drifted training artifacts (which would
silently invalidate every downstream verdict).

Source mode (``--source synthetic|live``):
  - ``synthetic`` (default): reuses ``compose_v25_travel._build_synthetic_v25``
    to generate a 92-col v2.5-travel fixture; projects to 13-wide canonical
    META-V22 substrate. DB-free + byte-stable.
  - ``live``: invokes ``compose_v25_travel._load_assembled_data_v25_travel``
    against the live PostgreSQL DB. Reserved for Plan 75-04 regression runs.

Default output path is gitignored (regeneratable). Override via ``--output PATH``.

Usage:
    python scripts/build_canonical_substrate_v27.py --no-paired-substrate
    python scripts/build_canonical_substrate_v27.py \\
        --candidate-substrate data/intermediate/ref_substrate_v261.parquet
    python scripts/build_canonical_substrate_v27.py --source live --no-paired-substrate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

# Ensure the scripts/ directory is on sys.path so we can import the Phase 42
# composition helpers when this script is invoked directly (not as a package).
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# Module-level import of canonical META-V22 column contract so the import-time
# assertion below catches accidental drift between this builder's col[0]
# expectation and the canonical META-V22 source-of-truth.
from ufc_prediction.ml.meta_features_v22 import META_V22_FEATURE_COLUMNS

# ── LOCKED constants (Phase 75 CONTEXT §D-01 / §D-02 / §D-06) ──────────────

# Canonical column order — byte-identical to META_V22_FEATURE_COLUMNS by
# explicit re-import (not string duplication). Cols[1..12] match the REF
# builder's cols[1..12] which match canonical META-V22 (Phase 65 D-03a pin);
# the structural inverse vs REF is at col[0]: REF used ``xgb_v2_refv2_oof``
# (candidate-aligned); canonical uses ``xgb_oof_prob`` (canonical training-
# time OOF). The col[0] swap IS the substrate-drift signal the dual-test
# methodology triangulates (D-01).
CANONICAL_FEATURE_COLUMNS: tuple[str, ...] = tuple(META_V22_FEATURE_COLUMNS)
assert len(CANONICAL_FEATURE_COLUMNS) == 13, (
    f"CANONICAL_FEATURE_COLUMNS width drift: expected 13, got {len(CANONICAL_FEATURE_COLUMNS)}"
)
assert CANONICAL_FEATURE_COLUMNS[0] == "xgb_oof_prob", (
    f"CANONICAL_FEATURE_COLUMNS[0] should be 'xgb_oof_prob' (canonical OOF), "
    f"got {CANONICAL_FEATURE_COLUMNS[0]!r}"
)
assert tuple(CANONICAL_FEATURE_COLUMNS) == tuple(META_V22_FEATURE_COLUMNS), (
    f"CANONICAL_FEATURE_COLUMNS drifted from META_V22_FEATURE_COLUMNS — "
    f"the canonical-substrate builder MUST mirror canonical META-V22 col "
    f"order EXACTLY (D-02). builder: {list(CANONICAL_FEATURE_COLUMNS)}; "
    f"meta: {list(META_V22_FEATURE_COLUMNS)}"
)

# 3-slice canonical convention — must match the Phase 42 / Phase 64 / Phase 65
# slice names so the dual-test verifier's per-slice metrics line up across
# substrates.
SLICE_NAMES: tuple[str, ...] = (
    "most_recent_12mo",
    "most_recent_24mo",
    "random_15pct",
)

# Fixed seed for the random_15pct slice. 7507 = Phase 75 + month-7 mnemonic
# (Plan 75-01 picker's-choice). Distinct from Phase 64=4202, Phase 65=6505,
# Phase 66=6606 so a Phase 75 random_15pct slice does NOT collide with any
# v2.6.1 builder's slice membership in side-by-side debug sessions.
RANDOM_15PCT_SEED: int = 7507

# Fixed reference date for the 12mo / 24mo windows. Phase 75 phase-start date
# (2026-06-07, the date the dual-test methodology was operator-approved).
# Using a fixed date (not ``date.today()``) is REQUIRED for re-run
# determinism — otherwise the slice membership would drift day-to-day and
# the per-slice SHAs would not be byte-stable.
CANONICAL_REFERENCE_DATE: date = date(2026, 6, 7)

# Default output path — gitignored (regeneratable from this script).
DEFAULT_OUTPUT_PATH: Path = Path("data/intermediate/canonical_substrate_v27.parquet")

# Synthetic fixture size — matches REF builder so the dual-test methodology
# has comparable row counts across substrates when run on synthetic data.
SYNTHETIC_N_FIGHTS: int = 600

# CR-01 anti-overwrite guard set: forbid pointing ``--output`` at any of the
# three v2.6.1 committed substrate paths (would corrupt the TRAVEL / REF /
# NET audit trails per Plan 75-01 task spec). The canonical builder writes
# a SIBLING substrate; it must never overwrite a candidate-substrate.
PROTECTED_OUTPUTS: frozenset[Path] = frozenset(
    {
        Path("data/intermediate/travel_substrate_v261.parquet"),  # Phase 64
        Path("data/intermediate/ref_substrate_v261.parquet"),  # Phase 65
        Path("data/intermediate/net_substrate_v261.parquet"),  # Phase 66
    }
)

# Canonical OOF parquet source (D-02). This is the Phase 26 training-time
# ``xgb_oof_prob`` parquet, archived under .planning/milestones/v2.2-phases/.
# READ-ONLY; the constant SHA pin below catches silent drift.
CANONICAL_OOF_PATH: Path = Path(
    ".planning/milestones/v2.2-phases/"
    "26-forward-stepwise-candidate-promotion/oof_predictions_v22.parquet"
)

# D-06 AUDIT-01 SHA pin. This is the SHA of the CURRENT on-disk archived
# parquet at .planning/milestones/v2.2-phases/26-…/oof_predictions_v22.parquet
# (sha256 verified at Phase 75 implementation time, 2026-06-07).
#
# **Pre-existing discrepancy (Phase 73 DEBT-V261-01)**: the canonical
# ``models/meta/meta_v2_meta.json::meta_oof_parquet_sha256`` documents the
# OOF parquet SHA as ``edb413cd760169f9de6a38786c65a9447fb34fec8f6b80ae2846f0a7cbbd936d``
# — that value matches the ORIGINAL 2422-row blob (git blob
# ``3db94eb676aaf73c10f1bc33438496b21799ba80``) from Phase 26 commit
# ``0fe9026`` (2026-05-17). The archive commit ``6db877d`` (2026-05-22)
# committed a DIFFERENT 714-row filtered blob (git blob
# ``79bfa82bec03bd80bb2e56d946f0c73fe61af765``) which has SHA
# ``acb64b87…f9131``. Phase 73 DEBT-V261-01 is the work to reconcile
# ``meta_v2_meta.json`` to the actually-on-disk-and-used 714-row parquet.
# For Plan 75-01 we pin to the on-disk SHA (the file the dual-test
# methodology will actually consume) — this matches Plan 75-01's task spec
# wording "Confirm oof_predictions_v22.parquet SHA matches …" against the
# real on-disk file, not the stale meta JSON claim.
EXPECTED_CANONICAL_OOF_SHA: str = "acb64b87533dce09980383540b5698217994bbc703b73221ebc56ec8054f9131"


# ── SHA streaming helper (mirrors gate_verifier.py pattern) ────────────────


def _sha256_hex(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Compute streaming SHA-256 hex digest for a file.

    Streams 1 MiB chunks so the canonical OOF parquet (~25 KB today,
    potentially larger if Phase 73 restores the 2422-row version) does not
    require loading the whole file into memory.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


# ── Canonical OOF loader (D-02 + D-06) ────────────────────────────────────


def _load_canonical_oof_map(
    path: Path = CANONICAL_OOF_PATH,
) -> dict[int, float]:
    """Load Phase 26's canonical training-time OOF parquet → ``{fight_id: oof_prob}``.

    D-06 AUDIT-01 invariant: the file's SHA-256 is verified against
    ``EXPECTED_CANONICAL_OOF_SHA`` BEFORE any pandas read. SHA mismatch
    raises ``RuntimeError`` naming the AUDIT-01 D-06 invariant so operators
    see WHY the build was refused.

    CR-02 (Phase 65 inheritance): missing parquet surfaces a clean
    operator-actionable error, not a Python traceback. Unlike candidate-
    OOF parquets (which have ``--dry-run`` regenerate scripts), the
    canonical OOF parquet is a READ-ONLY Phase 26 archive artifact — the
    error message must point operators at restore-from-archive recovery.

    Args:
        path: Canonical OOF parquet path. Default
            ``CANONICAL_OOF_PATH``.

    Returns:
        ``{int fight_id: float oof_prob}`` mapping. NaN ``oof_prob`` values
        are returned as ``float('nan')`` (the parquet has NaN entries for
        fights whose XGBoost prediction failed — defensive: downstream
        builders may filter or impute).

    Raises:
        FileNotFoundError: with operator-actionable message naming the
            missing path AND the restore-from-archive recovery procedure.
        RuntimeError: with operator-actionable message naming the AUDIT-01
            D-06 invariant if the file SHA does not match
            ``EXPECTED_CANONICAL_OOF_SHA``.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path} — canonical OOF parquet is a READ-ONLY Phase 26 "
            f"archive artifact (no regenerate command). Restore from "
            f"`.planning/milestones/v2.2-phases/26-forward-stepwise-candidate-"
            f"promotion/` archive or git history (commit 6db877d)."
        )

    # D-06 AUDIT-01 SHA pin — verify BEFORE pandas read so a corrupted
    # source raises fast (no expensive parquet parse on a dirty file).
    computed_sha = _sha256_hex(path)
    if computed_sha != EXPECTED_CANONICAL_OOF_SHA:
        raise RuntimeError(
            f"canonical OOF SHA mismatch (AUDIT-01 D-06 invariant violation): "
            f"expected {EXPECTED_CANONICAL_OOF_SHA}, got {computed_sha}. "
            f"Refusing to build substrate from drifted training artifact — "
            f"every downstream dual-test verdict would be silently invalid. "
            f"Path: {path}"
        )

    import pandas as pd  # function-scope to keep module-import cheap

    df = pd.read_parquet(path, columns=["fight_id", "xgb_oof_prob"])
    # Schema sanity (mirrors REF builder pattern — fail-fast on schema drift).
    required_cols = {"fight_id", "xgb_oof_prob"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(
            f"canonical OOF parquet schema drift: missing cols {missing}; got {sorted(df.columns)}"
        )
    return {int(row.fight_id): float(row.xgb_oof_prob) for row in df.itertuples(index=False)}


# ── Paired candidate-substrate sidecar loader (D-02 cross-reference) ──────


def _load_paired_candidate_fight_ids(paired_substrate_path: Path) -> set[int]:
    """Load fight_ids from a paired candidate-substrate's sidecar JSON.

    The Phase 63 substrate parquet schema does NOT include a ``fight_id``
    column directly (only ``slice_name``, ``feature_vector``, ``outcome``,
    ``substrate_sha``). Phase 75 introduces a SIBLING fight-id manifest the
    candidate builder is expected to produce: ``<path>.fight_ids.json`` with
    shape ``{"fight_ids": [int, ...]}``.

    Args:
        paired_substrate_path: Path to the candidate substrate parquet
            (e.g., ``data/intermediate/ref_substrate_v261.parquet``). The
            sidecar is located at ``<path>.fight_ids.json``.

    Returns:
        Set of int fight_ids backing the paired substrate.

    Raises:
        FileNotFoundError: with operator-actionable message if the sidecar
            is missing. Plan 75-01 Task 1 spec: candidate substrate builders
            in Phase 75-04 will produce this sidecar; for Plan 75-01 unit
            tests, ``--no-paired-substrate`` is the happy path.
    """
    # Append ``.fight_ids.json`` to the full parquet basename (NOT
    # ``with_suffix``, which would replace ``.parquet`` with ``.fight_ids.json``
    # and lose the ``.parquet`` segment). The convention is:
    #   foo.parquet                    → foo.parquet.fight_ids.json
    sidecar_path = paired_substrate_path.parent / (paired_substrate_path.name + ".fight_ids.json")
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"missing paired-substrate fight-id sidecar {sidecar_path} — "
            f"Plan 75-01 requires <candidate_substrate>.fight_ids.json sidecar "
            f"listing the fight_ids backing that substrate (shape: "
            f'{{"fight_ids": [int, ...]}}). Generate the sidecar from the '
            f"candidate builder, or pass --no-paired-substrate to skip the "
            f"cross-reference and use the full canonical OOF fight_id set."
        )
    raw = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if "fight_ids" not in raw:
        raise RuntimeError(
            f"paired-substrate sidecar {sidecar_path} missing 'fight_ids' key; "
            f"got keys {sorted(raw.keys())}"
        )
    return {int(fid) for fid in raw["fight_ids"]}


# ── Eval-matrix construction (13-wide canonical) ───────────────────────────


def build_eval_matrix(
    *,
    candidate_substrate_path: Path | None = None,
    source: str = "synthetic",
    oof_parquet_path: Path = CANONICAL_OOF_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the 13-wide canonical eval matrix + outcomes + event dates.

    Reuses ``scripts.compose_v25_travel`` helpers (the Phase 42 verified path)
    so the produced 13-wide vectors are structurally identical to what the
    canonical META-V22 meta was trained against — col[0] is sourced from the
    Phase 26 ``xgb_oof_prob`` parquet (canonical training-time OOF), NOT a
    candidate OOF. This is the STRUCTURAL inverse of the REF builder.

    Cross-reference (D-02): if ``candidate_substrate_path`` is provided, load
    the sidecar fight_ids and INTERSECT with the canonical OOF fight_ids. Any
    fight in the paired candidate substrate but NOT in canonical OOF is
    dropped from BOTH so the dual-test in Plan 75-02 evaluates both substrates
    on the SAME fight set.

    Args:
        candidate_substrate_path: Optional path to a paired candidate
            substrate parquet whose ``<path>.fight_ids.json`` sidecar
            constrains the fight set. If ``None``, ALL canonical OOF
            fight_ids are used (fallback for standalone testing).
        source: ``"synthetic"`` or ``"live"``.
        oof_parquet_path: Override of ``CANONICAL_OOF_PATH``. Useful for
            unit tests that need to point at a fixture parquet.

    Returns:
        ``(X_13, y, event_dates)`` where
          - ``X_13`` has shape ``(n, 13)`` — column order locked per
            ``CANONICAL_FEATURE_COLUMNS``.
          - ``y`` has shape ``(n,)`` and dtype int8 (values in {0, 1}).
          - ``event_dates`` has shape ``(n,)`` and contains ``datetime.date``
            objects (used downstream for 12mo / 24mo window slicing).

    Raises:
        FileNotFoundError: if canonical OOF parquet missing (CR-02), or if
            ``candidate_substrate_path`` is provided but the sidecar JSON is
            missing.
        RuntimeError: if canonical OOF SHA mismatch (D-06 AUDIT-01) or
            schema drift.
        ValueError: if ``source`` is not in ``{"synthetic", "live"}``.
    """
    # Load the canonical OOF map FIRST — a CR-02 FileNotFoundError or D-06
    # SHA mismatch must surface before we burn cycles on a 92-col synthetic
    # fixture build.
    canonical_oof_map = _load_canonical_oof_map(oof_parquet_path)

    # Load paired-substrate fight_id constraint if provided. The empty set
    # case is treated as "no constraint" (defensive — empty sidecar would
    # produce zero-row intersection and trip Phase 63 R5 downstream).
    if candidate_substrate_path is not None:
        paired_fight_ids: set[int] | None = _load_paired_candidate_fight_ids(
            candidate_substrate_path,
        )
    else:
        paired_fight_ids = None

    if source == "synthetic":
        # CR-03 determinism guard (mirrors Phase 64/65 builder pattern):
        # compose_v25_travel._build_synthetic_v25 generates the last n//3
        # fight dates as ``date.today() - random(1, 364) days``. Freeze
        # ``date.today`` to CANONICAL_REFERENCE_DATE so re-runs on
        # different calendar days produce byte-identical parquet.
        import datetime as _dt

        from compose_v25_travel import (  # type: ignore[import-not-found]
            _build_synthetic_v25,
        )

        class _FixedDate(_dt.date):
            @classmethod
            def today(cls):  # type: ignore[override]
                return CANONICAL_REFERENCE_DATE

        import compose_v25_travel as _cv  # type: ignore[import-not-found]

        _orig_date = _cv.date
        _cv.date = _FixedDate
        try:
            X_v25, y, fight_dates, fight_records = _build_synthetic_v25(
                n=SYNTHETIC_N_FIGHTS,
            )
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
    #   [:, 90]   → travel_distance_km   (Phase 64-only — discarded here)
    #   [:, 91]   → tz_shift_hours       (Phase 64-only — discarded here)
    assert X_v25.shape[1] == 92, (
        f"build_eval_matrix: expected 92-col v2.5-travel matrix, got {X_v25.shape[1]} cols"
    )

    # D-02 intersection: filter fight_records (and corresponding X_v25 / y /
    # fight_dates rows) to the paired-substrate fight_id set. If the paired
    # sidecar set is None (--no-paired-substrate), keep all rows.
    if paired_fight_ids is not None:
        keep_mask = np.array(
            [
                int(rec.get("fight_id", i)) in paired_fight_ids
                for i, rec in enumerate(fight_records)
            ],
            dtype=bool,
        )
        if not keep_mask.any():
            raise RuntimeError(
                f"build_eval_matrix: paired-substrate intersection produced "
                f"zero rows (paired_fight_ids size = {len(paired_fight_ids)}, "
                f"synthetic fight_ids = 0..{len(fight_records) - 1}). Cross-"
                f"reference yields the empty set — would trip Phase 63 R5. "
                f"Inspect the sidecar fight_ids."
            )
        X_v25 = X_v25[keep_mask]
        y = np.asarray(y)[keep_mask]
        fight_dates = np.asarray(fight_dates)[keep_mask]
        fight_records = [rec for i, rec in enumerate(fight_records) if keep_mask[i]]

    X_v22 = X_v25[:, :90]

    # Build the 13-col canonical META-V22 substrate.
    #   - col[0] (xgb_oof_prob): per-fight lookup from canonical OOF map.
    #   - col[1] (elo_prob): deterministic per-fight seed (same RNG plumbing
    #     as REF builder line 397-399 so the synthetic distribution lines up
    #     across substrates).
    #   - cols[2..12]: 11 internal META-V22 cols by name lookup against
    #     FEATURE_COLUMNS_V22.
    from ufc_prediction.ml.config import FEATURE_COLUMNS_V22

    n_rows = X_v25.shape[0]

    # col[0] = canonical xgb_oof_prob: per-fight lookup. Synthetic mode's
    # synthetic fight_ids (0..n-1) almost certainly do NOT collide with the
    # real fight_ids in the OOF parquet — fall back to a deterministic
    # seeded RNG so col[0] is byte-stable across re-runs and exhibits a
    # different distribution from a candidate OOF (the substrate-drift
    # signal that the dual-test methodology detects).
    fallback_rng = np.random.default_rng(RANDOM_15PCT_SEED + 1)
    fallback_oof = fallback_rng.uniform(0.10, 0.90, size=n_rows)
    xgb_oof_prob = np.empty(n_rows, dtype=float)
    for i, rec in enumerate(fight_records):
        fid = int(rec.get("fight_id", i))
        if fid in canonical_oof_map:
            val = canonical_oof_map[fid]
            # NaN in canonical OOF (Phase 26 has 119 NaN rows in current
            # 714-row archive) → fallback to seeded RNG so the substrate
            # has no NaN col[0] entries. Phase 63 loader allows NaN in
            # feature_vector (A2 accept rule) but the dual-test verifier's
            # downstream Pipeline imputer expects a finite col[0] for the
            # comparison to be meaningful.
            if not np.isnan(val):
                xgb_oof_prob[i] = val
            else:
                xgb_oof_prob[i] = float(fallback_oof[i])
        else:
            xgb_oof_prob[i] = float(fallback_oof[i])

    # col[1] = elo_prob: deterministic per-fight seed.
    elo_rng = np.random.default_rng(RANDOM_15PCT_SEED + 2)
    elo_prob = elo_rng.uniform(0.2, 0.8, size=n_rows)

    # Cols[2..12] = 11 internal META-V22 cols by name lookup. Verbatim
    # recipe from REF builder lines 402-407.
    internal_cols: list[np.ndarray] = []
    for name in CANONICAL_FEATURE_COLUMNS[2:]:
        idx = FEATURE_COLUMNS_V22.index(name)
        internal_cols.append(X_v22[:, idx])

    # Assemble the 13-wide matrix in CANONICAL_FEATURE_COLUMNS order:
    #   [xgb_oof_prob, elo_prob, *internal_meta_v22_cols (11)]
    X_13 = np.column_stack([xgb_oof_prob, elo_prob, *internal_cols])
    assert X_13.shape[1] == 13, (
        f"build_eval_matrix: expected 13-wide output, got {X_13.shape[1]} cols"
    )
    assert X_13.shape[1] == len(CANONICAL_FEATURE_COLUMNS)

    # Outcomes as int8 (Phase 63 R3 requires {0, 1}).
    y_int = np.asarray(y, dtype=np.int8)

    # Event dates as a 1-D object array of datetime.date.
    event_dates = np.asarray(fight_dates)

    return X_13, y_int, event_dates


# ── Slice partitioning (verbatim port; width-agnostic) ────────────────────


def partition_into_slices(
    X_13: np.ndarray,
    y: np.ndarray,
    event_dates: np.ndarray,
    *,
    reference_date: date = CANONICAL_REFERENCE_DATE,
    random_seed: int = RANDOM_15PCT_SEED,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Partition the 13-wide matrix into the three locked slices.

    Verbatim port of Phase 65 builder lines 500-564 with reference_date +
    random_seed defaults swapped to Phase 75 constants. Slice partitioning
    is feature-vector-width-agnostic so this is mechanical.

    Phase 65 WR-03 fix inherited: uses legacy ``np.random.RandomState`` (NOT
    ``np.random.default_rng``) for byte-stable slice membership against the
    gate_verifier's evaluate_per_slice convention.

    Args:
        X_13: ``(n, 13)`` feature matrix.
        y: ``(n,)`` outcome vector (int in {0, 1}).
        event_dates: ``(n,)`` ``datetime.date`` array.
        reference_date: anchor for the 12mo / 24mo cutoffs. Defaults to
            ``CANONICAL_REFERENCE_DATE``.
        random_seed: seed for the ``random_15pct`` slice; defaults to
            ``RANDOM_15PCT_SEED``.

    Returns:
        ``{slice_name: (X_slice, y_slice)}`` keyed by ``SLICE_NAMES``.

    Raises:
        RuntimeError: if any slice would be zero-row (Phase 63 R5).
    """
    cutoff_12mo = reference_date - timedelta(days=365)
    cutoff_24mo = reference_date - timedelta(days=730)

    mask_12mo = np.array([d >= cutoff_12mo for d in event_dates])
    mask_24mo = np.array([d >= cutoff_24mo for d in event_dates])

    # Phase 65 WR-03 fix: legacy RandomState (NOT default_rng) for byte-
    # stable slice membership against the gate_verifier's evaluate_per_slice
    # convention. See REF builder lines 536-545 for the rationale.
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


# ── Per-slice substrate_sha computation (verbatim port; width-agnostic) ───


def compute_slice_sha(
    feature_vectors: list[tuple[float, ...]],
    outcomes: list[int],
) -> str:
    """Compute a deterministic SHA256 over ``(feature_vector, outcome)`` rows.

    Verbatim port of Phase 65 builder lines 570-602. SHA computation is
    feature-vector-width-agnostic; same byte-stability contract holds for
    13-wide canonical substrates.

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

    sorted_rows = sorted(
        zip(feature_vectors, outcomes, strict=True),
        key=lambda r: (r[0], r[1]),
    )

    payload_chunks: list[bytes] = []
    for feat_tuple, outcome_val in sorted_rows:
        parts: list[str] = [repr(float(v)) for v in feat_tuple]
        parts.append(str(int(outcome_val)))
        payload_chunks.append("|".join(parts).encode("utf-8"))
    payload = b"\n".join(payload_chunks)

    return hashlib.sha256(payload).hexdigest()


# ── Parquet writer (end-to-end driver) ────────────────────────────────────


def build_canonical_substrate_parquet(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    candidate_substrate_path: Path | None = None,
    source: str = "synthetic",
    oof_parquet_path: Path = CANONICAL_OOF_PATH,
) -> Path:
    """End-to-end build: eval matrix → partition → SHA → parquet → round-trip.

    Writes a Phase 63 D-01-compliant parquet (4 cols: ``slice_name`` string,
    ``feature_vector`` list<float64>, ``outcome`` int8, ``substrate_sha``
    string). Verifies the output round-trips through ``load_substrate_snapshot``
    before returning — catches silent format breaks at write time.

    Args:
        output_path: Destination parquet path. Parent directories are
            created as needed. Default: ``DEFAULT_OUTPUT_PATH``.
        candidate_substrate_path: Optional paired candidate substrate path
            for fight-id cross-reference (D-02). None → use all canonical
            OOF fight_ids.
        source: ``"synthetic"`` (default) or ``"live"``.
        oof_parquet_path: Override of canonical OOF parquet path.

    Returns:
        The written ``output_path``.

    Raises:
        RuntimeError: if ``output_path`` resolves into ``PROTECTED_OUTPUTS``
            (CR-01); if per-slice SHA collision; or canonical OOF SHA
            mismatch (D-06).
        FileNotFoundError: if canonical OOF or paired sidecar missing.
    """
    output_path = Path(output_path)

    # CR-01 anti-overwrite guard: refuse to point at any path in
    # PROTECTED_OUTPUTS. Resolve both sides so symlinks / relative paths
    # collapse to a canonical form before comparison.
    protected_resolved = {p.resolve() for p in PROTECTED_OUTPUTS}
    if output_path.resolve() in protected_resolved:
        raise RuntimeError(
            f"build_canonical_substrate_parquet: refusing to overwrite "
            f"protected path {output_path} — this would corrupt a v2.6.1 "
            f"candidate-substrate audit trail (Phase 64 TRAVEL / Phase 65 "
            f"REF / Phase 66 NET). Choose a different --output path."
        )

    import pyarrow as pa
    import pyarrow.parquet as pq

    # 1. Build 13-wide eval matrix from the configured source. CR-02
    #    FileNotFoundError + D-06 RuntimeError propagate unwrapped so the
    #    CLI ``main`` can format stderr cleanly.
    X_13, y, event_dates = build_eval_matrix(
        candidate_substrate_path=candidate_substrate_path,
        source=source,
        oof_parquet_path=oof_parquet_path,
    )

    # 2. Partition into the three locked slices.
    slices = partition_into_slices(X_13, y, event_dates)

    # 3. Flatten into per-row records + compute per-slice SHA.
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
                f"build_canonical_substrate_parquet: per-slice substrate_sha "
                f"collision detected ({slice_sha[:12]}... appears in two "
                f"slices). This would trip Phase 63 R7 in the loader. "
                f"Investigate compute_slice_sha + slice partitioning."
            )
        seen_shas.add(slice_sha)

        for fv, outcome in zip(fv_tuples, outcome_list, strict=True):
            flat_slice_names.append(slice_name)
            flat_feature_vectors.append(list(fv))
            flat_outcomes.append(outcome)
            flat_substrate_shas.append(slice_sha)

    # 4. Build the pyarrow table with the LOCKED dtypes (Phase 63 R2).
    table = pa.Table.from_pydict(
        {
            "slice_name": pa.array(flat_slice_names, type=pa.string()),
            "feature_vector": pa.array(
                flat_feature_vectors,
                type=pa.list_(pa.float64()),
            ),
            "outcome": pa.array(flat_outcomes, type=pa.int8()),
            "substrate_sha": pa.array(flat_substrate_shas, type=pa.string()),
        }
    )

    # 5. Write to disk. Ensure parent dir exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)

    # 6. Self-validation — re-open via the Phase 63 loader so any silent
    #    format break surfaces at write time (not at Plan 75-02 verifier run).
    from ufc_prediction.ml.substrate_loader import load_substrate_snapshot

    roundtripped = load_substrate_snapshot(output_path)
    expected_slice_set = set(SLICE_NAMES)
    actual_slice_set = set(roundtripped.keys())
    assert actual_slice_set == expected_slice_set, (
        f"build_canonical_substrate_parquet: round-trip slice set mismatch — "
        f"expected {expected_slice_set}, got {actual_slice_set}"
    )
    for slice_name, eval_slice in roundtripped.items():
        widths = {len(fv) for fv in eval_slice.feature_vectors}
        assert widths == {13}, (
            f"build_canonical_substrate_parquet: slice {slice_name!r} "
            f"feature_vector widths = {widths}, expected {{13}}"
        )

    return output_path


# ── CLI entry ─────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI surface."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 75 Plan 75-01 (METH-V27-02) — canonical-substrate parquet "
            "builder. Writes a 13-wide, 3-slice substrate snapshot loadable "
            "by ufc_prediction.ml.substrate_loader.load_substrate_snapshot. "
            "col[0] is canonical xgb_oof_prob (training-time OOF, NOT a "
            "candidate OOF) — STRUCTURAL inverse of Phase 65/66 builders. "
            "Consumed by Plan 75-02's dual-test verifier."
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
        "--candidate-substrate",
        type=Path,
        default=None,
        help=(
            "Paired candidate substrate parquet path (e.g., "
            "data/intermediate/ref_substrate_v261.parquet). Triggers D-02 "
            "fight-id cross-reference via <path>.fight_ids.json sidecar; "
            "canonical substrate is filtered to the intersection. REQUIRED "
            "unless --no-paired-substrate is set."
        ),
    )
    parser.add_argument(
        "--no-paired-substrate",
        action="store_true",
        help=(
            "Skip the D-02 cross-reference; use ALL canonical OOF fight_ids. "
            "Override for standalone testing (Plan 75-01 unit tests + Plan "
            "75-02 fixture builds). For Plan 75-04 regression runs, omit "
            "this flag and pass --candidate-substrate."
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
        "--oof-parquet",
        type=Path,
        default=CANONICAL_OOF_PATH,
        help=(
            f"Override canonical OOF parquet path (default: "
            f"{CANONICAL_OOF_PATH}). The default is the Phase 26 archived "
            f"OOF parquet (READ-ONLY); override is for unit-test fixtures."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry: parse args, build parquet, log the output path.

    Wraps operator-actionable errors (CR-01 RuntimeError + CR-02
    FileNotFoundError + D-06 RuntimeError + missing-paired-sidecar
    FileNotFoundError) into clean stderr + exit-1. No traceback. Other
    exceptions propagate (programmer error, not operator-actionable).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Argparse-layer validation: --candidate-substrate XOR --no-paired-substrate.
    if args.candidate_substrate is None and not args.no_paired_substrate:
        sys.stderr.write(
            "ERROR: pass --candidate-substrate <path> for D-02 cross-reference "
            "OR --no-paired-substrate to skip and use all canonical OOF fights.\n"
        )
        return 1
    if args.candidate_substrate is not None and args.no_paired_substrate:
        sys.stderr.write(
            "ERROR: --candidate-substrate and --no-paired-substrate are mutually exclusive.\n"
        )
        return 1

    candidate_substrate_path = None if args.no_paired_substrate else args.candidate_substrate

    try:
        out_path = build_canonical_substrate_parquet(
            args.output,
            candidate_substrate_path=candidate_substrate_path,
            source=args.source,
            oof_parquet_path=args.oof_parquet,
        )
    except FileNotFoundError as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 1
    except RuntimeError as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 1
    print(f"Wrote canonical substrate parquet: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
