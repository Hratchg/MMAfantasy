"""Phase 64 Plan 64-02 (FEAT-V261-01) — unit tests for the TRAVEL substrate builder.

Targets ``scripts/build_travel_substrate_v261.py`` (Phase 64 D-02 deliverable).

Nine tests cover the contract Plan 64-02 ``<success_criteria>`` requires
(plus the Phase 64 CR-03 regression test added during code-review fix):

  1. ``test_builder_writes_parquet_at_explicit_output_path``
     — explicit ``--output PATH`` lands the parquet on disk.
  2. ``test_output_roundtrips_through_substrate_loader``
     — Phase 63 ``load_substrate_snapshot`` accepts the output and yields
     the three locked slice names mapped to populated ``EvalSlice`` objects.
  3. ``test_feature_vector_width_is_15_for_all_rows``
     — every row in every slice has a 15-element ``feature_vector``.
  4. ``test_feature_column_order_matches_meta_v22_travel_meta_json``
     — the builder's ``TRAVEL_FEATURE_COLUMNS`` constant matches the locked
     order in ``models/meta/meta_v22_travel_meta.json::feature_columns``.
  5. ``test_per_slice_substrate_sha_is_distinct_across_slices``
     — Phase 63 D-03 R7 (no duplicate SHAs across slices) holds.
  6. ``test_within_slice_substrate_sha_is_consistent``
     — Phase 63 D-03 R6 (single SHA per slice) holds.
  7. ``test_builder_is_deterministic_across_reruns``
     — two consecutive builds produce byte-identical parquet (SHA256 of
     the file bytes matches across runs).
  8. ``test_builder_is_deterministic_across_simulated_calendar_drift``
     — Phase 64 CR-03 regression: two builds with simulated different
     ``date.today()`` values still produce byte-identical parquet (proves
     the ``_FixedDate`` freeze in ``build_eval_matrix`` overrides the
     upstream ``_build_synthetic_v25`` date-of-the-day code path).
  9. ``test_slice_outcomes_are_int8_in_zero_one``
     — Phase 63 D-03 R3 (outcome ∈ {0, 1}) holds.

All tests use the default synthetic source (DB-free), so they run in CI
without a PostgreSQL container. Real-data round-trip is exercised in the
Plan 64-04 end-to-end CLI integration test.
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
# ``tests/unit/scripts/test_ingest_pre_ufc_records_v25.py``.

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# Direct import from scripts/ — Task 1 deliverable. Tests collect-fail with
# ImportError until the script exists (RED phase contract).
from build_travel_substrate_v261 import (  # noqa: E402
    RANDOM_15PCT_SEED,
    SLICE_NAMES,
    TRAVEL_FEATURE_COLUMNS,
    build_substrate_parquet,
)

from ufc_prediction.ml.gate_verifier import EvalSlice  # noqa: E402
from ufc_prediction.ml.substrate_loader import load_substrate_snapshot  # noqa: E402

# ── Shared fixture ────────────────────────────────────────────────────────


@pytest.fixture
def built_parquet(tmp_path: Path) -> Path:
    """Build a fresh substrate-snapshot parquet under ``tmp_path``.

    Used by the round-trip / width / SHA-distinctness / outcomes tests that
    only need to inspect the produced parquet (not rebuild). Function-scope
    so each test gets a fresh build — keeps test ordering independent.
    """
    out_path = tmp_path / "travel_substrate_v261.parquet"
    written = build_substrate_parquet(out_path)
    assert written == out_path
    return written


# ── Tests ─────────────────────────────────────────────────────────────────


def test_builder_writes_parquet_at_explicit_output_path(tmp_path: Path) -> None:
    """Task 1 invariant: ``--output PATH`` lands the file on disk."""
    target = tmp_path / "explicit_output.parquet"
    assert not target.exists()
    out = build_substrate_parquet(target)
    assert out == target
    assert out.exists()
    assert out.stat().st_size > 0


def test_output_roundtrips_through_substrate_loader(built_parquet: Path) -> None:
    """Phase 63 ``load_substrate_snapshot`` accepts the builder's output and
    returns ``dict[str, EvalSlice]`` keyed by the three locked slice names.
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


def test_feature_vector_width_is_15_for_all_rows(built_parquet: Path) -> None:
    """Every row in every slice has a 15-element feature_vector (locked-width
    contract per ``models/meta/meta_v22_travel_meta.json::feature_columns``).
    """
    slices = load_substrate_snapshot(built_parquet)
    for name, sl in slices.items():
        widths = {len(fv) for fv in sl.feature_vectors}
        assert widths == {15}, f"slice {name!r} has heterogeneous feature_vector widths: {widths}"


def test_feature_column_order_matches_meta_v22_travel_meta_json() -> None:
    """Builder's ``TRAVEL_FEATURE_COLUMNS`` matches the locked order in
    ``models/meta/meta_v22_travel_meta.json::feature_columns`` byte-identical.

    Drift here would re-shape what ``meta_v22_travel.joblib`` sees at
    predict time and the Phase 64 verdict would measure column-mapping
    noise, not candidate-vs-canonical signal.
    """
    meta_path = REPO_ROOT / "models" / "meta" / "meta_v22_travel_meta.json"
    meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    locked_order: list[str] = meta["feature_columns"]
    assert len(locked_order) == 15
    assert list(TRAVEL_FEATURE_COLUMNS) == locked_order, (
        f"TRAVEL_FEATURE_COLUMNS drifted from meta_v22_travel_meta.json:\n"
        f"  builder: {list(TRAVEL_FEATURE_COLUMNS)}\n"
        f"  meta:    {locked_order}"
    )


def test_per_slice_substrate_sha_is_distinct_across_slices(
    built_parquet: Path,
) -> None:
    """Phase 63 D-03 R7: the loader rejects any parquet whose slices share a
    ``substrate_sha``. The builder must produce three distinct per-slice SHAs.
    """
    slices = load_substrate_snapshot(built_parquet)
    shas = {name: sl.substrate_sha for name, sl in slices.items()}
    assert len(set(shas.values())) == 3, (
        f"Per-slice substrate_sha collision detected — Phase 63 R7 would "
        f"reject this parquet:\n{shas}"
    )


def test_within_slice_substrate_sha_is_consistent(built_parquet: Path) -> None:
    """Phase 63 D-03 R6: every row within a slice shares the same
    ``substrate_sha`` (the loader raises R6 ``ValueError`` otherwise).

    Read the raw parquet via pandas (not the loader) so we exercise the
    on-disk dataframe shape directly — confirms the builder wrote the SHA
    column row-by-row, not slice-by-slice in a way that the loader would
    cover for.
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
    """Two consecutive builds (different output paths, identical inputs)
    produce byte-identical parquet bytes.

    This is the determinism contract Phase 64 CONTEXT §D-02 commits to —
    the per-slice ``substrate_sha`` audit trail relies on the parquet being
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
        f"TRAVEL_SUBSTRATE_REFERENCE_DATE plumbing."
    )


def test_builder_is_deterministic_across_simulated_calendar_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CR-03 regression: builder stays byte-deterministic across calendar days.

    The upstream ``compose_v25_travel._build_synthetic_v25`` helper
    generates the last ``n // 3`` synthetic fight dates as
    ``date.today() - random(1, 364) days``. Without the CR-03 freeze in
    ``build_eval_matrix``, that breaks parquet-byte determinism across
    re-runs on different calendar days (the underlying fight-date values
    shift, which shifts the 12mo / 24mo window membership AND the
    per-slice ``(feature_vector, outcome)`` rows the SHA hashes over).

    This test simulates that calendar-day drift by monkeypatching
    ``compose_v25_travel.date`` to two different ``today()`` return values
    between the two builds. If CR-03's ``_FixedDate`` patch inside
    ``build_eval_matrix`` is in place, both builds produce byte-identical
    parquet because the inner patch wins over the outer monkeypatch. If
    the CR-03 guard regresses (e.g., is removed or bypassed), the second
    build's bytes diverge and the assertion fires.
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

    # Second build: monkeypatch ``date`` so .today() returns 2027-11-22
    # (~2.85 years later). If CR-03 freeze is in place, the inner
    # _FixedDate patch overrides this and both builds match.
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
        f"anchored to TRAVEL_SUBSTRATE_REFERENCE_DATE instead of "
        f"``date.today()``."
    )


def test_slice_outcomes_are_int8_in_zero_one(built_parquet: Path) -> None:
    """Phase 63 D-03 R3: outcome values must lie in ``{0, 1}``. The loader
    raises R3 ``ValueError`` on any out-of-set value, so this catches a
    type-cast regression in the builder before Plan 64-04 invokes it."""
    slices = load_substrate_snapshot(built_parquet)
    for name, sl in slices.items():
        for outcome in sl.outcomes:
            assert outcome in (0, 1), (
                f"slice {name!r} has outcome {outcome!r} outside {{0, 1}} (Phase 63 R3 violation)"
            )
        # Belt-and-braces: every outcome should be a plain ``int`` per the
        # EvalSlice contract (the loader casts ``int(o)`` for each row).
        assert all(isinstance(o, int) for o in sl.outcomes), (
            f"slice {name!r} has non-int outcome values"
        )


# NB: an alternative to the sys.path-based direct import above is
# ``importlib.util.spec_from_file_location`` against
# ``SCRIPTS_DIR / "build_travel_substrate_v261.py"`` — kept as a comment
# (not a test) so the exact-8-test acceptance criterion holds.
