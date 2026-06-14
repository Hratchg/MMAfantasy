"""Phase 65 Plan 65-04 (FEAT-V261-02) — unit tests for the REF substrate builder.

Targets ``scripts/build_ref_substrate_v261.py`` (Phase 65 D-02 + D-03a
deliverable).

Twelve tests cover the contract Plan 65-04 ``<success_criteria>`` requires.
Tests 1-10 mirror Phase 64's nine tests (with width 15→13 + col[0] name
swap); tests 11+12+13 are NEW for Phase 65 (coverage-gate + anti-overwrite
invariants):

  1. ``test_builder_writes_parquet_at_explicit_output_path``
     — mirrors Phase 64 #1: explicit ``--output PATH`` lands the parquet.
  2. ``test_output_roundtrips_through_substrate_loader``
     — mirrors Phase 64 #2: Phase 63 loader accepts the output + yields
     the three locked slice names mapped to populated ``EvalSlice`` objects.
  3. ``test_feature_vector_width_is_13_for_all_rows``
     — width is 13 (canonical META-V22), NOT 15 (Phase 64 D-03a pin).
  4. ``test_feature_column_order_matches_meta_v2_refv2_meta_json``
     — pins ``REF_FEATURE_COLUMNS`` to the Plan 65-03 META-V2-REFV2 layout.
  5. ``test_feature_col_0_is_xgb_v2_refv2_oof_not_canonical``
     — pins the substrate-drift design intent (col[0] swap).
  6. ``test_per_slice_substrate_sha_is_distinct_across_slices``
     — mirrors Phase 64 #5: Phase 63 D-03 R7 holds.
  7. ``test_within_slice_substrate_sha_is_consistent``
     — mirrors Phase 64 #6: Phase 63 D-03 R6 holds.
  8. ``test_builder_is_deterministic_across_reruns``
     — mirrors Phase 64 #7: byte-identical parquet across two builds.
  9. ``test_builder_is_deterministic_across_simulated_calendar_drift``
     — mirrors Phase 64 #8 (CR-03 regression): byte-identical across
     simulated different ``date.today()`` values.
  10. ``test_slice_outcomes_are_int8_in_zero_one``
      — mirrors Phase 64 #9: Phase 63 D-03 R3 holds.
  11. ``test_coverage_gate_fires_when_unknown_exceeds_threshold``
      — Phase 65 D-02: monkeypatched UNKNOWN dominance trips the gate.
  12. ``test_coverage_gate_bypass_via_allow_low_coverage``
      — Phase 65 D-02: ``--allow-low-coverage`` overrides the gate.
  13. ``test_anti_overwrite_guard_refuses_phase_64_substrate_path``
      — Phase 64 CR-01 inheritance: refuses to overwrite Phase 64 substrate.

All tests use the default synthetic source (DB-free) so they run in CI
without a PostgreSQL container. Real-data round-trip is exercised in the
Plan 65-04 end-to-end CLI integration test (``tests/cli/test_gate_verify_ref_e2e.py``).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

# ── Module-import helpers ─────────────────────────────────────────────────
#
# ``scripts/`` is not on ``sys.path`` by default for the tests/unit/ tree;
# we make a one-shot ``sys.path`` injection at module import time so the
# direct import below resolves. Mirrors the pattern in
# ``tests/unit/scripts/test_build_travel_substrate_v261.py``.

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# Direct import from scripts/ — Task 1 deliverable. Tests collect-fail with
# ImportError until the script exists (RED phase contract).
from build_ref_substrate_v261 import (  # noqa: E402
    PROTECTED_OUTPUTS,
    RANDOM_15PCT_SEED,
    REF_FEATURE_COLUMNS,
    REF_SUBSTRATE_REFERENCE_DATE,
    SLICE_NAMES,
    UNKNOWN_BUCKET_MAX_PROPORTION,
    build_substrate_parquet,
    main,
)

from ufc_prediction.ml.gate_verifier import EvalSlice  # noqa: E402
from ufc_prediction.ml.substrate_loader import load_substrate_snapshot  # noqa: E402

# ── Shared fixture ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def built_parquet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a fresh substrate-snapshot parquet under a module-scope tmp dir.

    WR-02 fix (Phase 64 review inheritance): module-scope so the heavy
    ``build_substrate_parquet`` call runs once per module run, not once per
    test that uses this fixture. Inputs are deterministic-by-construction
    (fixed seed + REF_SUBSTRATE_REFERENCE_DATE), so sharing across tests
    is safe — no test mutates the parquet.
    """
    tmp_dir = tmp_path_factory.mktemp("ref_substrate_module")
    out_path = tmp_dir / "ref_substrate_v261.parquet"
    written = build_substrate_parquet(out_path)
    assert written == out_path
    return written


# ── Tests ─────────────────────────────────────────────────────────────────


def test_builder_writes_parquet_at_explicit_output_path(tmp_path: Path) -> None:
    """Mirrors Phase 64 test #1: ``--output PATH`` lands the file on disk."""
    target = tmp_path / "explicit_output.parquet"
    assert not target.exists()
    out = build_substrate_parquet(target)
    assert out == target
    assert out.exists()
    assert out.stat().st_size > 0


def test_output_roundtrips_through_substrate_loader(built_parquet: Path) -> None:
    """Mirrors Phase 64 test #2: Phase 63 ``load_substrate_snapshot`` accepts
    the builder's output and returns ``dict[str, EvalSlice]`` keyed by the
    three locked slice names.
    """
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
    """Phase 65 D-03a pin: every row in every slice has a 13-element
    ``feature_vector`` (canonical META-V22 width — NOT 15 like Phase 64).
    Pins the design intent that ONLY col[0] differs from canonical.
    """
    slices = load_substrate_snapshot(built_parquet)
    for name, sl in slices.items():
        widths = {len(fv) for fv in sl.feature_vectors}
        assert widths == {13}, (
            f"slice {name!r} has heterogeneous or non-13 feature_vector widths: {widths}"
        )


def test_feature_column_order_matches_meta_v2_refv2_meta_json() -> None:
    """Phase 65 D-03a pin: ``REF_FEATURE_COLUMNS`` matches the locked order
    in ``models/meta/meta_v2_refv2_meta.json::meta_feature_columns``.

    Drift here would re-shape what ``meta_v2_refv2.joblib`` (Plan 65-03)
    sees at predict time and Plan 65-05's verdict would measure column-
    mapping noise, not REF v2 signal. Also: cols[1..12] MUST match the
    canonical ``meta_v2_meta.json`` byte-identical (only col[0] differs).
    """
    refv2_meta_path = REPO_ROOT / "models" / "meta" / "meta_v2_refv2_meta.json"
    refv2_meta: dict[str, Any] = json.loads(refv2_meta_path.read_text(encoding="utf-8"))
    refv2_locked_order: list[str] = refv2_meta["meta_feature_columns"]
    assert len(refv2_locked_order) == 13
    assert list(REF_FEATURE_COLUMNS) == refv2_locked_order, (
        f"REF_FEATURE_COLUMNS drifted from meta_v2_refv2_meta.json:\n"
        f"  builder: {list(REF_FEATURE_COLUMNS)}\n"
        f"  meta:    {refv2_locked_order}"
    )

    # Cols[1..12] of REF must equal cols[1..12] of canonical META-V22 — this
    # is the byte-identical substrate that lets the verifier compare candidate
    # vs canonical at predict time on the same meta-input shape, with only
    # col[0] (the OOF source) differing.
    canonical_meta_path = REPO_ROOT / "models" / "meta" / "meta_v2_meta.json"
    canonical_meta: dict[str, Any] = json.loads(
        canonical_meta_path.read_text(encoding="utf-8")
    )
    canonical_cols: list[str] = canonical_meta["meta_feature_columns"]
    assert list(REF_FEATURE_COLUMNS[1:]) == canonical_cols[1:], (
        f"REF cols[1..12] drifted from canonical META-V22 cols[1..12]:\n"
        f"  REF:       {list(REF_FEATURE_COLUMNS[1:])}\n"
        f"  canonical: {canonical_cols[1:]}"
    )


def test_feature_col_0_is_xgb_v2_refv2_oof_not_canonical() -> None:
    """Phase 65 D-03a pin: col[0] is the candidate-aligned OOF
    (``xgb_v2_refv2_oof``), NOT the canonical OOF (``xgb_oof_prob``).

    This pins the substrate-drift design intent — the col[0] swap IS the
    signal the GATE-V26-02 verifier's refit_baseline path detects (Phase 55
    + Phase 64 patterns). If a future maintainer renames col[0] without
    updating both REF_FEATURE_COLUMNS and the candidate-meta retrain, the
    verifier loses its drift signal silently.
    """
    assert REF_FEATURE_COLUMNS[0] == "xgb_v2_refv2_oof", (
        f"REF_FEATURE_COLUMNS[0] should be 'xgb_v2_refv2_oof' "
        f"(candidate-aligned), got {REF_FEATURE_COLUMNS[0]!r}"
    )
    assert REF_FEATURE_COLUMNS[0] != "xgb_oof_prob", (
        f"REF_FEATURE_COLUMNS[0] must NOT be the canonical name "
        f"'xgb_oof_prob' — col[0] swap is the substrate-drift signal"
    )


def test_per_slice_substrate_sha_is_distinct_across_slices(
    built_parquet: Path,
) -> None:
    """Mirrors Phase 64 test #5 (Phase 63 D-03 R7): the loader rejects any
    parquet whose slices share a ``substrate_sha``. The builder must produce
    three distinct per-slice SHAs.
    """
    slices = load_substrate_snapshot(built_parquet)
    shas = {name: sl.substrate_sha for name, sl in slices.items()}
    assert len(set(shas.values())) == 3, (
        f"Per-slice substrate_sha collision detected — Phase 63 R7 would "
        f"reject this parquet:\n{shas}"
    )


def test_within_slice_substrate_sha_is_consistent(built_parquet: Path) -> None:
    """Mirrors Phase 64 test #6 (Phase 63 D-03 R6): every row within a slice
    shares the same ``substrate_sha``. Read raw parquet via pandas (not the
    loader) so we exercise the on-disk dataframe shape directly.
    """
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
    """Mirrors Phase 64 test #7: two consecutive builds produce byte-identical
    parquet bytes (deterministic re-run contract per Phase 65 CONTEXT §D-02).

    The per-slice ``substrate_sha`` audit trail relies on the parquet being
    reproducible bit-exact for any future audit (operator re-runs the
    builder, expects same SHAs, same bytes).
    """
    p1 = build_substrate_parquet(tmp_path / "run1.parquet")
    p2 = build_substrate_parquet(tmp_path / "run2.parquet")
    sha1 = hashlib.sha256(p1.read_bytes()).hexdigest()
    sha2 = hashlib.sha256(p2.read_bytes()).hexdigest()
    assert sha1 == sha2, (
        f"Builder is non-deterministic across re-runs:\n"
        f"  run1: {sha1}\n"
        f"  run2: {sha2}\n"
        f"Check RANDOM_15PCT_SEED={RANDOM_15PCT_SEED} + "
        f"REF_SUBSTRATE_REFERENCE_DATE={REF_SUBSTRATE_REFERENCE_DATE} plumbing."
    )


def test_builder_is_deterministic_across_simulated_calendar_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors Phase 64 test #8 (CR-03 regression): builder stays byte-
    deterministic across calendar-day drift.

    The upstream ``compose_v25_travel._build_synthetic_v25`` helper generates
    the last ``n // 3`` synthetic fight dates as ``date.today() -
    random(1, 364) days``. Without the CR-03 ``_FixedDate`` freeze in
    ``build_eval_matrix``, that breaks parquet-byte determinism across
    re-runs on different calendar days. This test monkeypatches
    ``compose_v25_travel.date`` to two different ``today()`` return values
    between two builds; if CR-03 is in place, the inner ``_FixedDate`` patch
    wins and both builds match.
    """
    import compose_v25_travel as _cv  # type: ignore[import-not-found]
    from datetime import date as _date

    # First build: monkeypatch ``date`` so .today() returns 2025-01-15.
    class _Today2025_01_15(_date):
        @classmethod
        def today(cls):
            return cls(2025, 1, 15)

    monkeypatch.setattr(_cv, "date", _Today2025_01_15)
    p1 = build_substrate_parquet(tmp_path / "run_2025_01_15.parquet")
    sha1 = hashlib.sha256(p1.read_bytes()).hexdigest()

    # Second build: monkeypatch ``date`` so .today() returns 2027-11-22.
    class _Today2027_11_22(_date):
        @classmethod
        def today(cls):
            return cls(2027, 11, 22)

    monkeypatch.setattr(_cv, "date", _Today2027_11_22)
    p2 = build_substrate_parquet(tmp_path / "run_2027_11_22.parquet")
    sha2 = hashlib.sha256(p2.read_bytes()).hexdigest()

    assert sha1 == sha2, (
        f"CR-03 regression: builder is non-deterministic across simulated "
        f"calendar-day drift:\n"
        f"  run_2025_01_15: {sha1}\n"
        f"  run_2027_11_22: {sha2}\n"
        f"The build_eval_matrix() ``_FixedDate`` freeze guard must wrap "
        f"the _build_synthetic_v25 call so the last n//3 fight dates are "
        f"anchored to REF_SUBSTRATE_REFERENCE_DATE instead of "
        f"``date.today()``."
    )


def test_slice_outcomes_are_int8_in_zero_one(built_parquet: Path) -> None:
    """Mirrors Phase 64 test #9 (Phase 63 D-03 R3): outcome values lie in
    ``{0, 1}``. The loader raises R3 ``ValueError`` on any out-of-set value,
    so this catches a type-cast regression in the builder before Plan 65-05
    invokes it.
    """
    slices = load_substrate_snapshot(built_parquet)
    for name, sl in slices.items():
        for outcome in sl.outcomes:
            assert outcome in (0, 1), (
                f"slice {name!r} has outcome {outcome!r} outside {{0, 1}} "
                f"(Phase 63 R3 violation)"
            )
        assert all(isinstance(o, int) for o in sl.outcomes), (
            f"slice {name!r} has non-int outcome values"
        )


def test_coverage_gate_fires_when_unknown_exceeds_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 65 D-02 NEW: monkeypatch ``derive_event_country_bucket`` to
    return ``"UNKNOWN"`` for every event; the blocking RuntimeError must fire.

    The coverage gate protects Plan 65-05's verdict integrity: a substrate
    dominated by UNKNOWN cohorts would produce shrinkage-fallback-everywhere
    feature values, masking real REF v2 signal. The gate uses the threshold
    ``UNKNOWN_BUCKET_MAX_PROPORTION`` (= 0.20).
    """
    # Force every venue_country lookup to return UNKNOWN — proportion = 100%
    # which is well above the 20% threshold.
    monkeypatch.setattr(
        "build_ref_substrate_v261.derive_event_country_bucket",
        lambda _: "UNKNOWN",
    )
    # Sanity: the threshold constant exists and is the expected value.
    assert UNKNOWN_BUCKET_MAX_PROPORTION == pytest.approx(0.20)

    target = tmp_path / "should_block.parquet"
    with pytest.raises(RuntimeError, match="coverage gate"):
        build_substrate_parquet(target, allow_low_coverage=False)
    # The blocked build must NOT have written the output parquet (fail-fast
    # contract — no half-emitted side effects).
    assert not target.exists(), (
        "Coverage gate fired but output parquet was still written — "
        "the gate must fail BEFORE pq.write_table is called"
    )


def test_coverage_gate_bypass_via_allow_low_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 65 D-02 NEW: ``allow_low_coverage=True`` overrides the gate even
    when UNKNOWN proportion is 100%. The build proceeds and the parquet
    lands on disk.
    """
    monkeypatch.setattr(
        "build_ref_substrate_v261.derive_event_country_bucket",
        lambda _: "UNKNOWN",
    )
    target = tmp_path / "bypass.parquet"
    written = build_substrate_parquet(target, allow_low_coverage=True)
    assert written == target
    assert target.exists()
    assert target.stat().st_size > 0


def test_anti_overwrite_guard_refuses_phase_64_substrate_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Phase 64 CR-01 inheritance: pointing ``--output`` at the Phase 64
    committed substrate path is refused with a clean RuntimeError + exit 1.

    Tests via ``main([...])`` (the CLI entry) rather than calling
    ``build_substrate_parquet`` directly so the operator-facing error path
    (stderr write + exit code) is exercised end-to-end. ``PROTECTED_OUTPUTS``
    is asserted to contain the Phase 64 path so a future maintainer who
    drops the constant trips this test first.
    """
    # Sanity: the protected set still contains the Phase 64 path.
    assert Path("data/intermediate/travel_substrate_v261.parquet") in PROTECTED_OUTPUTS, (
        f"PROTECTED_OUTPUTS missing Phase 64 path; set is: {PROTECTED_OUTPUTS}"
    )

    rc = main([
        "--source", "synthetic",
        "--output", "data/intermediate/travel_substrate_v261.parquet",
    ])
    assert rc == 1, f"Expected exit code 1, got {rc}"
    captured = capsys.readouterr()
    assert "refusing to overwrite" in captured.err.lower(), (
        f"Expected 'refusing to overwrite' in stderr; got: {captured.err!r}"
    )
