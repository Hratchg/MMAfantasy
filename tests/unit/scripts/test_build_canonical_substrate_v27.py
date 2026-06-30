"""Phase 75 Plan 75-01 (METH-V27-02) — unit tests for the canonical-substrate builder.

Targets ``scripts/build_canonical_substrate_v27.py`` (Phase 75 D-01 + D-02 + D-06
deliverable for the dual-test substrate-drift-immune methodology).

Mirrors ``tests/unit/scripts/test_build_ref_substrate_v261.py`` 1:1 with these
material differences (per Plan 75-01 task inventory):

  1. ``test_builder_writes_parquet_at_explicit_output_path`` — happy path
  2. ``test_output_roundtrips_through_substrate_loader`` — Phase 63 loader accepts
  3. ``test_feature_vector_width_is_13_for_all_rows`` — canonical META-V22 width
  4. ``test_feature_column_order_matches_meta_v22_feature_columns`` — canonical-aligned
  5. ``test_feature_col_0_is_canonical_xgb_oof_prob_not_candidate`` — substrate-drift
     STRUCTURAL signal (col[0] is xgb_oof_prob, NOT a candidate OOF name)
  6. ``test_per_slice_substrate_sha_is_distinct_across_slices`` — Phase 63 R7
  7. ``test_within_slice_substrate_sha_is_consistent`` — Phase 63 R6
  8. ``test_builder_is_deterministic_across_reruns`` — byte-identical re-runs
  9. ``test_oof_sha_mismatch_fails_fast`` — D-06 AUDIT-01 SHA pin
  10. ``test_oof_source_missing_fails_fast`` — CR-02 clean error path
  11. ``test_anti_overwrite_guard_blocks_phase_64_65_66_paths`` — CR-01 PROTECTED_OUTPUTS
      includes ALL three v2.6.1 substrate paths (TRAVEL + REF + NET)
  12. ``test_paired_substrate_intersection_drops_unpaired_fights`` — D-02 cross-reference
  13. ``test_slice_outcomes_are_int8_in_zero_one`` — Phase 63 R3 (defensive)
  14. ``test_canonical_oof_map_loads_real_parquet`` — sanity: actual on-disk OOF parquet
      loads via the builder's loader
  15. ``test_no_paired_substrate_flag_uses_all_oof_fights`` — fallback path

All tests use the synthetic source (DB-free) so they run in CI without a PostgreSQL
container. The cross-reference fixture (test 12) writes a synthetic sidecar JSON
listing a subset of synthetic-mode fight_ids; the synthetic fight_records use
fight_ids 0..n-1 (per ``compose_v25_travel._build_synthetic_v25`` lines 1186-1196),
so the cross-reference test enumerates a SUBSET of those synthetic ids.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# ── Module-import helpers ─────────────────────────────────────────────────

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# Direct import from scripts/ — Task 1 deliverable. Tests collect-fail with
# ImportError until the script exists (RED phase contract).
from build_canonical_substrate_v27 import (
    CANONICAL_FEATURE_COLUMNS,
    CANONICAL_OOF_PATH,
    CANONICAL_REFERENCE_DATE,
    DEFAULT_OUTPUT_PATH,
    PROTECTED_OUTPUTS,
    RANDOM_15PCT_SEED,
    SLICE_NAMES,
    _load_canonical_oof_map,
    build_canonical_substrate_parquet,
    main,
)

from ufc_prediction.ml.gate_verifier import EvalSlice
from ufc_prediction.ml.meta_features_v22 import META_V22_FEATURE_COLUMNS
from ufc_prediction.ml.substrate_loader import load_substrate_snapshot

# ── Shared fixture ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def built_parquet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a fresh canonical-substrate parquet under a module-scope tmp dir.

    Module-scope so the (cheap-but-non-trivial) builder runs once per module
    run, not once per test. Inputs are deterministic-by-construction
    (fixed seed + frozen CANONICAL_REFERENCE_DATE), so sharing across tests
    is safe — no test mutates the parquet.

    Uses ``--no-paired-substrate`` fallback so the canonical OOF map's full
    fight_id set drives the build (no cross-reference dependency).
    """
    tmp_dir = tmp_path_factory.mktemp("canonical_substrate_module")
    out_path = tmp_dir / "canonical_substrate_v27.parquet"
    written = build_canonical_substrate_parquet(
        out_path,
        candidate_substrate_path=None,
        source="synthetic",
    )
    assert written == out_path
    return written


# ── Tests ─────────────────────────────────────────────────────────────────


def test_builder_writes_parquet_at_explicit_output_path(tmp_path: Path) -> None:
    """Test #1 mirror: ``--output PATH`` lands the file on disk."""
    target = tmp_path / "explicit_output.parquet"
    assert not target.exists()
    out = build_canonical_substrate_parquet(
        target,
        candidate_substrate_path=None,
        source="synthetic",
    )
    assert out == target
    assert out.exists()
    assert out.stat().st_size > 0


def test_output_roundtrips_through_substrate_loader(built_parquet: Path) -> None:
    """Test #2: Phase 63 ``load_substrate_snapshot`` accepts the builder's output."""
    slices = load_substrate_snapshot(built_parquet)
    assert set(slices.keys()) == set(SLICE_NAMES)
    assert set(slices.keys()) == {
        "most_recent_12mo",
        "most_recent_24mo",
        "random_15pct",
    }
    for name, sl in slices.items():
        assert isinstance(sl, EvalSlice), (
            f"slice {name!r} should be EvalSlice, got {type(sl).__name__}"
        )
        assert len(sl.feature_vectors) > 0, (
            f"slice {name!r} has zero feature_vectors (Phase 63 R5 risk)"
        )
        assert len(sl.outcomes) == len(sl.feature_vectors)
        assert isinstance(sl.substrate_sha, str)
        assert len(sl.substrate_sha) == 64  # SHA256 hex digest length


def test_feature_vector_width_is_13_for_all_rows(built_parquet: Path) -> None:
    """Test #3: canonical META-V22 width is 13 across every row of every slice."""
    slices = load_substrate_snapshot(built_parquet)
    for name, sl in slices.items():
        widths = {len(fv) for fv in sl.feature_vectors}
        assert widths == {13}, (
            f"slice {name!r} has heterogeneous or non-13 feature_vector widths: {widths}"
        )


def test_feature_column_order_matches_meta_v22_feature_columns() -> None:
    """Test #4: CANONICAL_FEATURE_COLUMNS byte-identical to META_V22_FEATURE_COLUMNS.

    This is the regression guard for accidental col[0] swap drift — the
    canonical-substrate builder MUST mirror canonical META-V22 column order
    EXACTLY (D-02 closing sentence: "the col[0] swap is the substrate-drift
    signal; canonical builder uses canonical col name").
    """
    assert tuple(CANONICAL_FEATURE_COLUMNS) == tuple(META_V22_FEATURE_COLUMNS), (
        f"CANONICAL_FEATURE_COLUMNS drifted from META_V22_FEATURE_COLUMNS:\n"
        f"  builder: {list(CANONICAL_FEATURE_COLUMNS)}\n"
        f"  meta:    {list(META_V22_FEATURE_COLUMNS)}"
    )
    assert len(CANONICAL_FEATURE_COLUMNS) == 13


def test_feature_col_0_is_canonical_xgb_oof_prob_not_candidate() -> None:
    """Test #5: col[0] is the CANONICAL training-time OOF name (``xgb_oof_prob``),
    NOT a candidate-aligned OOF (e.g., ``xgb_v2_refv2_oof`` from Phase 65 or
    ``xgb_v2_netd_oof`` from Phase 66).

    This pins the substrate-drift design intent: the canonical-substrate
    builder swaps col[0] BACK to canonical so the dual-test methodology
    (D-01) can triangulate substrate drift between candidate (REF/NET) and
    canonical OOF sources.
    """
    assert CANONICAL_FEATURE_COLUMNS[0] == "xgb_oof_prob", (
        f"CANONICAL_FEATURE_COLUMNS[0] should be 'xgb_oof_prob' "
        f"(canonical training-time OOF), got {CANONICAL_FEATURE_COLUMNS[0]!r}"
    )
    # Defensive: must NOT be any of the known candidate OOF column names
    forbidden_candidate_names = {
        "xgb_v2_refv2_oof",  # Phase 65 REF candidate
        "xgb_v2_netd_oof",  # Phase 66 NET candidate
    }
    assert CANONICAL_FEATURE_COLUMNS[0] not in forbidden_candidate_names, (
        f"CANONICAL_FEATURE_COLUMNS[0] = {CANONICAL_FEATURE_COLUMNS[0]!r} matches a "
        f"candidate OOF name; canonical builder MUST use canonical col name"
    )


def test_per_slice_substrate_sha_is_distinct_across_slices(
    built_parquet: Path,
) -> None:
    """Test #6: Phase 63 D-03 R7 — three distinct per-slice SHAs."""
    slices = load_substrate_snapshot(built_parquet)
    shas = {name: sl.substrate_sha for name, sl in slices.items()}
    assert len(set(shas.values())) == 3, (
        f"Per-slice substrate_sha collision detected — Phase 63 R7 would "
        f"reject this parquet:\n{shas}"
    )


def test_within_slice_substrate_sha_is_consistent(built_parquet: Path) -> None:
    """Test #7: Phase 63 D-03 R6 — every row within a slice shares same SHA."""
    df: pd.DataFrame = pd.read_parquet(built_parquet, engine="pyarrow")
    for name in SLICE_NAMES:
        slice_df = df[df["slice_name"] == name]
        assert len(slice_df) > 0, f"slice {name!r} missing from raw parquet"
        unique_shas = slice_df["substrate_sha"].unique()
        assert len(unique_shas) == 1, (
            f"slice {name!r} has multiple substrate_sha values "
            f"({len(unique_shas)}): {sorted(unique_shas)}"
        )


def test_builder_is_deterministic_across_reruns(tmp_path: Path) -> None:
    """Test #8: two builds → byte-identical parquet (D-06 audit-trail integrity)."""
    p1 = build_canonical_substrate_parquet(
        tmp_path / "run1.parquet",
        candidate_substrate_path=None,
        source="synthetic",
    )
    p2 = build_canonical_substrate_parquet(
        tmp_path / "run2.parquet",
        candidate_substrate_path=None,
        source="synthetic",
    )
    sha1 = hashlib.sha256(p1.read_bytes()).hexdigest()
    sha2 = hashlib.sha256(p2.read_bytes()).hexdigest()
    assert sha1 == sha2, (
        f"Builder is non-deterministic across re-runs:\n"
        f"  run1: {sha1}\n"
        f"  run2: {sha2}\n"
        f"Check RANDOM_15PCT_SEED={RANDOM_15PCT_SEED} + "
        f"CANONICAL_REFERENCE_DATE={CANONICAL_REFERENCE_DATE} plumbing."
    )


def test_oof_sha_mismatch_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test #9: D-06 AUDIT-01 SHA pin — corrupted OOF source raises clean
    ``RuntimeError`` naming the AUDIT-01 invariant before any I/O cost.

    Monkeypatches the EXPECTED constant to a deliberately wrong value so
    the actual on-disk file (whose true SHA matches the constant in prod)
    appears as a mismatch. The error message must name the AUDIT-01 D-06
    invariant so operators see WHY the build failed.
    """
    import build_canonical_substrate_v27 as builder

    monkeypatch.setattr(
        builder,
        "EXPECTED_CANONICAL_OOF_SHA",
        "deadbeef" * 8,
    )
    with pytest.raises(RuntimeError) as excinfo:
        _load_canonical_oof_map()
    msg = str(excinfo.value).lower()
    # Operator-actionable substring: must name the audit invariant
    assert "audit-01" in msg or "d-06" in msg or "sha" in msg, (
        f"RuntimeError message must name the AUDIT-01/D-06/SHA invariant; got: {excinfo.value!r}"
    )


def test_oof_source_missing_fails_fast(tmp_path: Path) -> None:
    """Test #10: CR-02 clean error path — missing OOF parquet raises
    ``FileNotFoundError`` with operator-actionable message.
    """
    missing_path = tmp_path / "does_not_exist.parquet"
    assert not missing_path.exists()

    with pytest.raises(FileNotFoundError) as excinfo:
        _load_canonical_oof_map(missing_path)
    msg = str(excinfo.value)
    assert str(missing_path) in msg, f"FileNotFoundError must name the missing path; got: {msg!r}"


@pytest.mark.parametrize(
    "protected_path",
    [
        Path("data/intermediate/travel_substrate_v261.parquet"),
        Path("data/intermediate/ref_substrate_v261.parquet"),
        Path("data/intermediate/net_substrate_v261.parquet"),
    ],
)
def test_anti_overwrite_guard_blocks_phase_64_65_66_paths(
    protected_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test #11: CR-01 anti-overwrite — PROTECTED_OUTPUTS includes ALL THREE
    v2.6.1 substrate paths (TRAVEL + REF + NET). Pointing ``--output`` at any
    of them is refused with a clean ``RuntimeError`` + exit-1 (via CLI).

    Parametrized so the test fires once per protected path; a future
    maintainer who drops any path from PROTECTED_OUTPUTS will see one
    parametrize-id fail with the dropped path named.
    """
    assert protected_path in PROTECTED_OUTPUTS, (
        f"PROTECTED_OUTPUTS missing {protected_path}; set is: {PROTECTED_OUTPUTS}"
    )
    rc = main(
        [
            "--no-paired-substrate",
            "--source",
            "synthetic",
            "--output",
            str(protected_path),
        ]
    )
    assert rc == 1, f"Expected exit code 1 for protected path {protected_path}, got {rc}"
    captured = capsys.readouterr()
    assert "refusing to overwrite" in captured.err.lower(), (
        f"Expected 'refusing to overwrite' in stderr for {protected_path}; got: {captured.err!r}"
    )


def test_paired_substrate_intersection_drops_unpaired_fights(
    tmp_path: Path,
) -> None:
    """Test #12: D-02 cross-reference — paired candidate-substrate sidecar
    listing a SUBSET of fight_ids → produced canonical substrate covers ONLY
    the intersection (fewer rows than the no-paired baseline).

    The sidecar lives at ``<paired_substrate_path>.fight_ids.json`` with shape
    ``{"fight_ids": [int, ...]}`` per Plan 75-01 task spec.

    Synthetic mode: ``compose_v25_travel._build_synthetic_v25`` assigns
    fight_ids 0..n-1 (line 1186-1196 of that module's synthetic fixture
    code) with deterministic date distribution. Per inspection: SYNTHETIC_N_FIGHTS
    = 600 rows span 2020-2026 in cyclic year-distribution. We list a SUBSET
    of fight_ids spread across the date range (every 3rd id, ~200 ids out
    of 600) so all three slices remain non-empty — the goal is to prove
    intersection drops rows, NOT to test edge-case zero-rows.
    """
    # Build the no-paired baseline first to anchor the comparison.
    baseline_path = tmp_path / "baseline.parquet"
    build_canonical_substrate_parquet(
        baseline_path,
        candidate_substrate_path=None,
        source="synthetic",
    )
    baseline_df = pd.read_parquet(baseline_path, engine="pyarrow")
    baseline_rows = len(baseline_df)

    # Build paired-substrate sidecar with every 3rd fight_id (200 of 600).
    # This guarantees all three slices stay populated (no Phase 63 R5 trip)
    # while still demonstrably reducing row count from the baseline.
    paired_path = tmp_path / "paired_candidate.parquet"
    paired_path.write_bytes(b"placeholder")  # builder only reads the sidecar
    sidecar_path = paired_path.parent / (paired_path.name + ".fight_ids.json")
    every_third = list(range(0, 600, 3))  # [0, 3, 6, ..., 597] — 200 ids
    sidecar_path.write_text(json.dumps({"fight_ids": every_third}))

    intersected_path = tmp_path / "intersected.parquet"
    build_canonical_substrate_parquet(
        intersected_path,
        candidate_substrate_path=paired_path,
        source="synthetic",
    )
    intersected_df = pd.read_parquet(intersected_path, engine="pyarrow")
    intersected_rows = len(intersected_df)

    assert intersected_rows < baseline_rows, (
        f"Cross-reference did NOT drop unpaired fights: "
        f"baseline={baseline_rows} vs intersected={intersected_rows} "
        f"(expected intersected < baseline)"
    )
    # Defensive: intersection should also be non-trivial (not zero, not full)
    assert intersected_rows > 0, "Intersection produced zero rows"


def test_slice_outcomes_are_int8_in_zero_one(built_parquet: Path) -> None:
    """Test #13: Phase 63 D-03 R3 — outcomes in {0, 1}; loader rejects otherwise."""
    slices = load_substrate_snapshot(built_parquet)
    for name, sl in slices.items():
        for outcome in sl.outcomes:
            assert outcome in (0, 1), (
                f"slice {name!r} has outcome {outcome!r} outside {{0, 1}} (Phase 63 R3 violation)"
            )
        assert all(isinstance(o, int) for o in sl.outcomes), (
            f"slice {name!r} has non-int outcome values"
        )


def test_canonical_oof_map_loads_real_parquet() -> None:
    """Test #14: sanity — the on-disk Phase 26 OOF parquet loads via the
    builder's loader.

    Confirms:
      - ``CANONICAL_OOF_PATH`` constant points at the actual archived parquet
      - The constant SHA (``EXPECTED_CANONICAL_OOF_SHA``) matches the on-disk
        file SHA at test-time (no silent drift between runs)
      - The returned ``{fight_id: oof_prob}`` map is non-empty
    """
    assert CANONICAL_OOF_PATH.exists(), (
        f"Canonical OOF parquet missing on disk: {CANONICAL_OOF_PATH}. "
        f"Phase 26 archive integrity violated."
    )
    # Live SHA check (no monkeypatch) — should succeed under prod constants
    oof_map = _load_canonical_oof_map()
    assert len(oof_map) > 0, "Canonical OOF map is empty — parquet read failed"
    # Cheap sanity check: every key is an int, every value is a float (or NaN)
    sample_key = next(iter(oof_map.keys()))
    sample_val = oof_map[sample_key]
    assert isinstance(sample_key, int), (
        f"OOF map keys should be int, got {type(sample_key).__name__}"
    )
    assert isinstance(sample_val, float), (
        f"OOF map values should be float, got {type(sample_val).__name__}"
    )


def test_no_paired_substrate_flag_uses_all_oof_fights(tmp_path: Path) -> None:
    """Test #15: CLI ``--no-paired-substrate`` flag bypasses the cross-reference
    and uses the full canonical OOF fight_id set. End-to-end via ``main``.
    """
    target = tmp_path / "no_paired.parquet"
    rc = main(
        [
            "--no-paired-substrate",
            "--source",
            "synthetic",
            "--output",
            str(target),
        ]
    )
    assert rc == 0, f"main() exited {rc} with --no-paired-substrate"
    assert target.exists()
    # Round-trip sanity
    slices = load_substrate_snapshot(target)
    assert set(slices.keys()) == set(SLICE_NAMES)


def test_default_output_path_is_canonical_substrate_v27() -> None:
    """Test #16 (bonus): DEFAULT_OUTPUT_PATH points at the documented Phase 75
    canonical substrate path under data/intermediate/.

    Pins the .gitignore entry's target — a maintainer who renames the default
    path without updating .gitignore will see this fail before committing
    a substrate-binary artifact accidentally.
    """
    assert Path("data/intermediate/canonical_substrate_v27.parquet") == DEFAULT_OUTPUT_PATH, (
        f"DEFAULT_OUTPUT_PATH drifted from documented Plan 75-01 target: "
        f"got {DEFAULT_OUTPUT_PATH!r}"
    )
