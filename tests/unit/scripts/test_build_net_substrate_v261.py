"""Phase 66 Plan 66-03 (FEAT-V261-03) — unit tests for the NET substrate builder.

Targets ``scripts/build_net_substrate_v261.py`` (Phase 66 D-03a + D-04
deliverable).

Fifteen tests cover the contract Plan 66-03 ``<success_criteria>`` requires.
Tests 1-10 mirror Phase 65 Plan 65-04's twelve tests (with col[0] name swap
+ NET-specific gate semantics); tests 11-15 are NEW for Phase 66
(NaN-debutant gate + dual-path anti-overwrite + seed-distinctness):

  1. ``test_builder_writes_parquet_at_explicit_output_path``
     — mirrors Phase 65 #1: explicit ``--output PATH`` lands the parquet.
  2. ``test_output_roundtrips_through_substrate_loader``
     — mirrors Phase 65 #2: Phase 63 loader accepts the output + yields
     the three locked slice names mapped to populated ``EvalSlice`` objects.
  3. ``test_feature_vector_width_is_13_for_all_rows``
     — width is 13 (canonical META-V22), NOT 15 (Phase 66 D-03a pin).
  4. ``test_feature_column_order_matches_meta_v2_netd_meta_json``
     — pins ``NET_FEATURE_COLUMNS`` to the Plan 66-02 META-V2-NETD layout.
  5. ``test_feature_col_0_is_xgb_v2_netd_oof_not_canonical_nor_refv2``
     — pins the substrate-drift design intent (col[0] swap; T-66-18).
  6. ``test_per_slice_substrate_sha_is_distinct_across_slices``
     — mirrors Phase 65 #6: Phase 63 D-03 R7 holds.
  7. ``test_within_slice_substrate_sha_is_consistent``
     — mirrors Phase 65 #7: Phase 63 D-03 R6 holds.
  8. ``test_builder_is_deterministic_across_reruns``
     — mirrors Phase 65 #8: byte-identical parquet across two builds.
  9. ``test_builder_is_deterministic_across_simulated_calendar_drift``
     — mirrors Phase 65 #9 (CR-03 regression): byte-identical across
     simulated different ``date.today()`` values (T-66-19).
  10. ``test_slice_outcomes_are_int8_in_zero_one``
      — mirrors Phase 65 #10: Phase 63 D-03 R3 holds.
  11. ``test_nan_coverage_gate_fires_when_debutant_exceeds_threshold``
      — Phase 66 NaN-debutant gate: monkeypatched all-NaN OOF map trips the
      gate (T-66-17).
  12. ``test_nan_coverage_gate_bypass_via_allow_low_coverage``
      — Phase 66 NaN-debutant gate: ``--allow-low-coverage`` overrides
      the gate.
  13. ``test_anti_overwrite_guard_refuses_phase_64_substrate_path``
      — Phase 64 CR-01 inheritance: refuses to overwrite Phase 64 substrate
      (T-66-15).
  14. ``test_anti_overwrite_guard_refuses_phase_65_substrate_path``
      — Phase 65 carry-forward: refuses to overwrite Phase 65 substrate
      (T-66-15).
  15. ``test_random_15pct_seed_is_6606``
      — Phase 66 D-03a pin: seed distinct from Phase 64 (4202) + Phase 65
      (6505) so cross-phase random_15pct slices do NOT collide (T-66-21).

All tests use the default synthetic source (DB-free) so they run in CI
without a PostgreSQL container. Real-data round-trip is exercised in the
Plan 66-03 end-to-end CLI integration test (``tests/cli/test_gate_verify_net_e2e.py``).
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
# ``tests/unit/scripts/test_build_ref_substrate_v261.py``.

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# Direct import from scripts/ — Task 1 deliverable. Tests collect-fail with
# ImportError until the script exists (RED phase contract).
from build_net_substrate_v261 import (
    DEBUTANT_NAN_MAX_PROPORTION,
    NET_FEATURE_COLUMNS,
    NET_SUBSTRATE_REFERENCE_DATE,
    PROTECTED_OUTPUTS,
    RANDOM_15PCT_SEED,
    SLICE_NAMES,
    build_substrate_parquet,
    main,
)

from ufc_prediction.ml.gate_verifier import EvalSlice
from ufc_prediction.ml.substrate_loader import load_substrate_snapshot

# ── Shared fixture ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def built_parquet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a fresh substrate-snapshot parquet under a module-scope tmp dir.

    WR-02 fix (Phase 64 + Phase 65 review inheritance): module-scope so the
    heavy ``build_substrate_parquet`` call runs once per module run, not once
    per test that uses this fixture. Inputs are deterministic-by-construction
    (fixed seed + NET_SUBSTRATE_REFERENCE_DATE), so sharing across tests is
    safe — no test mutates the parquet.
    """
    tmp_dir = tmp_path_factory.mktemp("net_substrate_module")
    out_path = tmp_dir / "net_substrate_v261.parquet"
    written = build_substrate_parquet(out_path)
    assert written == out_path
    return written


# ── Tests ─────────────────────────────────────────────────────────────────


def test_builder_writes_parquet_at_explicit_output_path(tmp_path: Path) -> None:
    """Mirrors Phase 65 test #1: ``--output PATH`` lands the file on disk."""
    target = tmp_path / "explicit_output.parquet"
    assert not target.exists()
    out = build_substrate_parquet(target)
    assert out == target
    assert out.exists()
    assert out.stat().st_size > 0


def test_output_roundtrips_through_substrate_loader(built_parquet: Path) -> None:
    """Mirrors Phase 65 test #2: Phase 63 ``load_substrate_snapshot`` accepts
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
    """Phase 66 D-03a pin: every row in every slice has a 13-element
    ``feature_vector`` (canonical META-V22 width — NOT 15 like Phase 64).
    Pins the design intent that ONLY col[0] differs from canonical.
    """
    slices = load_substrate_snapshot(built_parquet)
    for name, sl in slices.items():
        widths = {len(fv) for fv in sl.feature_vectors}
        assert widths == {13}, (
            f"slice {name!r} has heterogeneous or non-13 feature_vector widths: {widths}"
        )


def test_feature_column_order_matches_meta_v2_netd_meta_json() -> None:
    """Phase 66 D-03a pin: ``NET_FEATURE_COLUMNS`` matches the locked order
    in ``models/meta/meta_v2_netd_meta.json::meta_feature_columns``.

    Drift here would re-shape what ``meta_v2_netd.joblib`` (Plan 66-02) sees
    at predict time and Plan 66-04's verdict would measure column-mapping
    noise, not NET v2 signal. Also: cols[1..12] MUST match the canonical
    ``meta_v2_meta.json`` byte-identical (only col[0] differs).
    """
    netd_meta_path = REPO_ROOT / "models" / "meta" / "meta_v2_netd_meta.json"
    netd_meta: dict[str, Any] = json.loads(netd_meta_path.read_text(encoding="utf-8"))
    netd_locked_order: list[str] = netd_meta["meta_feature_columns"]
    assert len(netd_locked_order) == 13
    assert list(NET_FEATURE_COLUMNS) == netd_locked_order, (
        f"NET_FEATURE_COLUMNS drifted from meta_v2_netd_meta.json:\n"
        f"  builder: {list(NET_FEATURE_COLUMNS)}\n"
        f"  meta:    {netd_locked_order}"
    )

    # Cols[1..12] of NET must equal cols[1..12] of canonical META-V22 — this
    # is the byte-identical substrate that lets the verifier compare candidate
    # vs canonical at predict time on the same meta-input shape, with only
    # col[0] (the OOF source) differing.
    canonical_meta_path = REPO_ROOT / "models" / "meta" / "meta_v2_meta.json"
    canonical_meta: dict[str, Any] = json.loads(canonical_meta_path.read_text(encoding="utf-8"))
    canonical_cols: list[str] = canonical_meta["meta_feature_columns"]
    assert list(NET_FEATURE_COLUMNS[1:]) == canonical_cols[1:], (
        f"NET cols[1..12] drifted from canonical META-V22 cols[1..12]:\n"
        f"  NET:       {list(NET_FEATURE_COLUMNS[1:])}\n"
        f"  canonical: {canonical_cols[1:]}"
    )


def test_feature_col_0_is_xgb_v2_netd_oof_not_canonical_nor_refv2() -> None:
    """Phase 66 D-03a pin (T-66-18): col[0] is the candidate-aligned NET OOF
    (``xgb_v2_netd_oof``), NOT the canonical OOF (``xgb_oof_prob``), AND NOT
    the Phase 65 REF OOF (``xgb_v2_refv2_oof``).

    This pins the substrate-drift design intent — the col[0] swap IS the
    signal the GATE-V26-02 verifier's refit_baseline path detects (Phase 55
    + Phase 64 + Phase 65 patterns). If a future maintainer renames col[0]
    without updating both NET_FEATURE_COLUMNS and the candidate-meta retrain,
    the verifier loses its drift signal silently. The triple-NOT assertion
    pins col[0] uniquely to Phase 66 (vs Phase 65 vs canonical).
    """
    assert NET_FEATURE_COLUMNS[0] == "xgb_v2_netd_oof", (
        f"NET_FEATURE_COLUMNS[0] should be 'xgb_v2_netd_oof' "
        f"(candidate-aligned), got {NET_FEATURE_COLUMNS[0]!r}"
    )
    assert NET_FEATURE_COLUMNS[0] != "xgb_oof_prob", (
        "NET_FEATURE_COLUMNS[0] must NOT be the canonical name "
        "'xgb_oof_prob' — col[0] swap is the substrate-drift signal"
    )
    assert NET_FEATURE_COLUMNS[0] != "xgb_v2_refv2_oof", (
        "NET_FEATURE_COLUMNS[0] must NOT be the Phase 65 REF name "
        "'xgb_v2_refv2_oof' — Phase 66 is NET, NOT REF"
    )


def test_per_slice_substrate_sha_is_distinct_across_slices(
    built_parquet: Path,
) -> None:
    """Mirrors Phase 65 test #6 (Phase 63 D-03 R7): the loader rejects any
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
    """Mirrors Phase 65 test #7 (Phase 63 D-03 R6): every row within a slice
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
    """Mirrors Phase 65 test #8: two consecutive builds produce byte-identical
    parquet bytes (deterministic re-run contract per Phase 66 CONTEXT §D-03a).

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
        f"NET_SUBSTRATE_REFERENCE_DATE={NET_SUBSTRATE_REFERENCE_DATE} plumbing."
    )


def test_builder_is_deterministic_across_simulated_calendar_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors Phase 65 test #9 (CR-03 regression; T-66-19): builder stays
    byte-deterministic across calendar-day drift.

    The upstream ``compose_v25_travel._build_synthetic_v25`` helper generates
    the last ``n // 3`` synthetic fight dates as ``date.today() -
    random(1, 364) days``. Without the CR-03 ``_FixedDate`` freeze in
    ``build_eval_matrix``, that breaks parquet-byte determinism across
    re-runs on different calendar days. This test monkeypatches
    ``compose_v25_travel.date`` to two different ``today()`` return values
    between two builds; if CR-03 is in place, the inner ``_FixedDate`` patch
    wins and both builds match.
    """
    from datetime import date as _date

    import compose_v25_travel as _cv  # type: ignore[import-not-found]

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
        f"anchored to NET_SUBSTRATE_REFERENCE_DATE instead of "
        f"``date.today()``."
    )


def test_slice_outcomes_are_int8_in_zero_one(built_parquet: Path) -> None:
    """Mirrors Phase 65 test #10 (Phase 63 D-03 R3): outcome values lie in
    ``{0, 1}``. The loader raises R3 ``ValueError`` on any out-of-set value,
    so this catches a type-cast regression in the builder before Plan 66-04
    invokes it.
    """
    slices = load_substrate_snapshot(built_parquet)
    for name, sl in slices.items():
        for outcome in sl.outcomes:
            assert outcome in (0, 1), (
                f"slice {name!r} has outcome {outcome!r} outside {{0, 1}} (Phase 63 R3 violation)"
            )
        assert all(isinstance(o, int) for o in sl.outcomes), (
            f"slice {name!r} has non-int outcome values"
        )


def test_nan_coverage_gate_fires_when_debutant_exceeds_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 66 NaN-debutant gate (T-66-17): monkeypatch the OOF-load helper
    so every fight_id returns NaN; the blocking RuntimeError must fire.

    The coverage gate protects Plan 66-04's verdict integrity: a substrate
    dominated by debutant-NaN rows would produce imputation-fallback-
    everywhere col[0] values, masking real NET v2 signal. The gate uses the
    threshold ``DEBUTANT_NAN_MAX_PROPORTION`` (= 0.20).

    Strategy: monkeypatch ``_load_xgb_v2_netd_oof_map`` to return a map where
    every key 0..N-1 maps to NaN. Then the synthetic fight_ids (0..n-1) all
    hit the OOF map → NaN values → debutant_indicator=True everywhere
    → proportion = 100% (well above 20%) → gate fires.
    """
    # Sanity: the threshold constant exists and is the expected value.
    assert pytest.approx(0.20) == DEBUTANT_NAN_MAX_PROPORTION

    # Force every fight_id in the OOF map to carry NaN — every synthetic
    # row will be flagged as debutant → 100% gate-triggering.
    monkeypatch.setattr(
        "build_net_substrate_v261._load_xgb_v2_netd_oof_map",
        lambda: {fid: float("nan") for fid in range(0, 10_000)},
    )

    target = tmp_path / "should_block.parquet"
    with pytest.raises(RuntimeError, match="coverage gate"):
        build_substrate_parquet(target, allow_low_coverage=False)
    # The blocked build must NOT have written the output parquet (fail-fast
    # contract — no half-emitted side effects).
    assert not target.exists(), (
        "Coverage gate fired but output parquet was still written — "
        "the gate must fail BEFORE pq.write_table is called"
    )


def test_nan_coverage_gate_bypass_via_allow_low_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 66 NaN-debutant gate (T-66-17): ``allow_low_coverage=True``
    overrides the gate even when debutant proportion is 100%. The build
    proceeds and the parquet lands on disk.
    """
    monkeypatch.setattr(
        "build_net_substrate_v261._load_xgb_v2_netd_oof_map",
        lambda: {fid: float("nan") for fid in range(0, 10_000)},
    )
    target = tmp_path / "bypass.parquet"
    written = build_substrate_parquet(target, allow_low_coverage=True)
    assert written == target
    assert target.exists()
    assert target.stat().st_size > 0


def test_anti_overwrite_guard_refuses_phase_64_substrate_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Phase 64 CR-01 inheritance (T-66-15): pointing ``--output`` at the
    Phase 64 committed substrate path is refused with a clean RuntimeError
    + exit 1.

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

    rc = main(
        [
            "--source",
            "synthetic",
            "--output",
            "data/intermediate/travel_substrate_v261.parquet",
        ]
    )
    assert rc == 1, f"Expected exit code 1, got {rc}"
    captured = capsys.readouterr()
    assert "refusing to overwrite" in captured.err.lower(), (
        f"Expected 'refusing to overwrite' in stderr; got: {captured.err!r}"
    )


def test_anti_overwrite_guard_refuses_phase_65_substrate_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Phase 65 carry-forward (T-66-15): pointing ``--output`` at the Phase 65
    committed substrate path is refused with a clean RuntimeError + exit 1.

    Plan 66-03 extends ``PROTECTED_OUTPUTS`` to include BOTH Phase 64 AND
    Phase 65 substrate paths so Phase 66 cannot accidentally clobber either
    upstream audit trail. ``PROTECTED_OUTPUTS`` is asserted to contain the
    Phase 65 path so a future maintainer who drops the constant trips this
    test first.
    """
    # Sanity: the protected set still contains the Phase 65 path.
    assert Path("data/intermediate/ref_substrate_v261.parquet") in PROTECTED_OUTPUTS, (
        f"PROTECTED_OUTPUTS missing Phase 65 path; set is: {PROTECTED_OUTPUTS}"
    )

    rc = main(
        [
            "--source",
            "synthetic",
            "--output",
            "data/intermediate/ref_substrate_v261.parquet",
        ]
    )
    assert rc == 1, f"Expected exit code 1, got {rc}"
    captured = capsys.readouterr()
    assert "refusing to overwrite" in captured.err.lower(), (
        f"Expected 'refusing to overwrite' in stderr; got: {captured.err!r}"
    )


def test_random_15pct_seed_is_6606() -> None:
    """Phase 66 D-03a pin (T-66-21): ``RANDOM_15PCT_SEED`` is the Phase-66-
    mnemonic ``6606``, distinct from Phase 64's ``4202`` and Phase 65's
    ``6505``.

    Cross-phase seed distinctness ensures the Phase 66 random_15pct slice
    membership does NOT collide with a Phase 64 or Phase 65 one — important
    when all three substrates end up side-by-side in a debug or comparison
    session. A future maintainer who copies the Phase 65 seed verbatim
    instead of bumping it would trip this test first.
    """
    assert RANDOM_15PCT_SEED == 6606, (
        f"RANDOM_15PCT_SEED should be 6606 (Phase 66 mnemonic); got {RANDOM_15PCT_SEED}"
    )
    # Anti-collision pins — must NOT be Phase 64's seed nor Phase 65's seed.
    assert RANDOM_15PCT_SEED != 4202, (
        "RANDOM_15PCT_SEED must NOT be Phase 64's 4202 — cross-phase collision"
    )
    assert RANDOM_15PCT_SEED != 6505, (
        "RANDOM_15PCT_SEED must NOT be Phase 65's 6505 — cross-phase collision"
    )
