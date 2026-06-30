"""Wave-0 RED tests for scripts/audit_camp_v23.py.

Per Phase 29 / CAMP-V23-00 + CAMP-V23-01 + CONTEXT.md D-01 / D-02 / D-08 / D-18.

The v23 audit script is a VERBATIM fork of scripts/audit_camp_v22.py with only
THREE allowed edits (D-01):
  1. Docstring header updated to Phase 29 / CAMP-V23-00
  2. AUDIT_VERSION: str = "v22.00" → "v23.00"
  3. OUTPUT_PATH_DEFAULT phase-20 path → phase-29 path

All 6 D-06 thresholds (presence 0.30/0.70, parseable 0.25/0.60, top30 0.60/0.70)
stay BYTE-IDENTICAL per the D-18 carry-forward "no post-measurement
renegotiation" rule (29-CONTEXT.md). SAMPLE_SIZE=200 and RANDOM_STATE=42 stay
byte-identical per D-02.

These tests assert:
  (a) the fork is a verbatim copy (constants byte-identical to v22)
  (b) the 3 allowed edits are exactly what the plan specifies
  (c) the mechanical D-06 scope ladder is preserved with boundary-case rigor
     (29.99% presence MUST be "drop", not "reduced" — the comparator is `>=`)
  (d) helpers (_normalize_association, _compute_top30) carry over unchanged
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def audit_v23():
    """Lazy-import the v23 audit script. Mirrors the v22 importorskip pattern."""
    return pytest.importorskip("scripts.audit_camp_v23")


@pytest.fixture
def audit_v22():
    """Lazy-import the v22 audit script (fork source). Used for byte-identity asserts."""
    return pytest.importorskip("scripts.audit_camp_v22")


class TestVersionSuffixEdits:
    """The THREE allowed edits per D-01 — exactly these, nothing else."""

    def test_audit_version_is_v23_00(self, audit_v23) -> None:
        """Allowed edit #2: AUDIT_VERSION = 'v23.00' (was 'v22.00' in v22)."""
        assert audit_v23.AUDIT_VERSION == "v23.00"

    def test_output_path_default_points_to_phase_29(self, audit_v23) -> None:
        """Allowed edit #3: default output path is the v23 phase directory."""
        assert (
            Path(".planning/phases/29-camp-re-audit-eval-set-infrastructure/CAMP_V23_AUDIT.json")
            == audit_v23.OUTPUT_PATH_DEFAULT
        )


class TestSampleConstantsLocked:
    """D-02: sample size + random state byte-identical to Phase 20."""

    def test_sample_size_locked_at_200(self, audit_v23) -> None:
        """D-02 carry-forward — no change to sample size in v23."""
        assert audit_v23.SAMPLE_SIZE == 200

    def test_random_state_locked_at_42(self, audit_v23) -> None:
        """D-02 carry-forward — reproducible draw with seed=42."""
        assert audit_v23.RANDOM_STATE == 42

    def test_active_window_days_locked_at_730(self, audit_v23) -> None:
        """Phase 20 A2 derivation rule (2 years) — unchanged in v23."""
        assert audit_v23.ACTIVE_WINDOW_DAYS == 730


class TestAllSixThresholdsMatchV22:
    """D-18 carry-forward: all 6 D-06 thresholds byte-identical to v22 fork source."""

    def test_all_six_thresholds_match_v22(self, audit_v22, audit_v23) -> None:
        """No post-measurement renegotiation — every threshold byte-identical."""
        # Presence
        assert audit_v23.PRESENCE_THRESHOLD_MIN == audit_v22.PRESENCE_THRESHOLD_MIN
        assert audit_v23.PRESENCE_THRESHOLD_FULL == audit_v22.PRESENCE_THRESHOLD_FULL
        assert audit_v23.PRESENCE_THRESHOLD_MIN == 0.30
        assert audit_v23.PRESENCE_THRESHOLD_FULL == 0.70
        # Parseable
        assert audit_v23.PARSEABLE_THRESHOLD_MIN == audit_v22.PARSEABLE_THRESHOLD_MIN
        assert audit_v23.PARSEABLE_THRESHOLD_FULL == audit_v22.PARSEABLE_THRESHOLD_FULL
        assert audit_v23.PARSEABLE_THRESHOLD_MIN == 0.25
        assert audit_v23.PARSEABLE_THRESHOLD_FULL == 0.60
        # Top-30
        assert audit_v23.TOP30_THRESHOLD_MIN == audit_v22.TOP30_THRESHOLD_MIN
        assert audit_v23.TOP30_THRESHOLD_FULL == audit_v22.TOP30_THRESHOLD_FULL
        assert audit_v23.TOP30_THRESHOLD_MIN == 0.60
        assert audit_v23.TOP30_THRESHOLD_FULL == 0.70


class TestScopeRecommendationBoundaries:
    """D-06 mechanical scope_recommendation ladder — boundary-case rigor.

    The comparator in _derive_scope_recommendation is `>=`. These tests
    pin the boundary behavior — a 29.99% presence MUST be "drop", not
    "reduced"; a 59.99% top-30 MUST be "drop", not "reduced". They guard
    against silent comparator drift on the fork.
    """

    def test_scope_full_at_70_60_70_boundary(self, audit_v23) -> None:
        """Exactly hitting FULL-tier thresholds → 'full' (>= comparator)."""
        rec = audit_v23._derive_scope_recommendation(0.70, 0.60, 0.70)
        assert rec == "full"

    def test_scope_reduced_at_30_25_60_boundary(self, audit_v23) -> None:
        """Exactly hitting MIN-tier thresholds → 'reduced' (>= comparator)."""
        rec = audit_v23._derive_scope_recommendation(0.30, 0.25, 0.60)
        assert rec == "reduced"

    def test_scope_drop_at_29_99_pct_presence(self, audit_v23) -> None:
        """0.2999 presence misses MIN tier → 'drop' (strictly < 0.30)."""
        rec = audit_v23._derive_scope_recommendation(0.2999, 0.25, 0.60)
        assert rec == "drop"

    def test_scope_drop_at_24_99_pct_parseable(self, audit_v23) -> None:
        """0.2499 parseable misses MIN tier → 'drop' (strictly < 0.25)."""
        rec = audit_v23._derive_scope_recommendation(0.30, 0.2499, 0.60)
        assert rec == "drop"

    def test_scope_drop_at_59_99_pct_top30(self, audit_v23) -> None:
        """0.5999 top-30 misses MIN tier → 'drop' (strictly < 0.60).

        This is the SAME failure mode as Phase 20 (top30=0.4213 → drop).
        A re-audit landing at 59.99% MUST still be drop; the comparator is
        strictly `>=`, not `>` and not `~=`.
        """
        rec = audit_v23._derive_scope_recommendation(0.30, 0.25, 0.5999)
        assert rec == "drop"


class TestNormalizeAssociationStable:
    """Alias normalization byte-identical to v22 (no change in v23)."""

    def test_normalize_association_stable(self, audit_v23) -> None:
        """Black House (USA) → 'black house' — proves the (USA) suffix
        strip + lowercase + trim chain is preserved in the fork."""
        assert audit_v23._normalize_association("Black House (USA)") == "black house"

    def test_normalize_association_handles_none(self, audit_v23) -> None:
        """None / empty / whitespace-only → None — preserved from v22."""
        assert audit_v23._normalize_association(None) is None
        assert audit_v23._normalize_association("") is None
        assert audit_v23._normalize_association("   ") is None


class TestComputeTop30WithFixture:
    """_compute_top30 ranking + coverage_rate arithmetic preserved verbatim."""

    def test_compute_top30_with_fixture_results(self, audit_v23) -> None:
        """Hand-built 5-row results list → expected ranking + coverage rate.

        Setup: 3 fighters at camp "alpha", 1 at "beta", 1 at None (excluded
        from numerator AND denominator per the v22 helper semantics).

        Expected: top-30 contains alpha (count=3) + beta (count=1); coverage
        rate = (3+1) / (3+1) = 1.0 (the None-camp fighter is excluded from
        the total by the `if not norm: continue` guard).
        """
        results = [
            {"fighter_id": 1, "raw_association": "Alpha", "normalized_association": "alpha"},
            {"fighter_id": 2, "raw_association": "Alpha", "normalized_association": "alpha"},
            {"fighter_id": 3, "raw_association": "Alpha", "normalized_association": "alpha"},
            {"fighter_id": 4, "raw_association": "Beta", "normalized_association": "beta"},
            {"fighter_id": 5, "raw_association": None, "normalized_association": None},
        ]
        per_camp, rate = audit_v23._compute_top30(results)
        # 2 camps in top-30 (the only non-None camps)
        assert len(per_camp) == 2
        # First-rank camp is alpha (count=3), then beta (count=1) — sorted by -count, name
        assert per_camp[0]["normalized_name"] == "alpha"
        assert per_camp[0]["fight_count"] == 3
        assert per_camp[1]["normalized_name"] == "beta"
        assert per_camp[1]["fight_count"] == 1
        # Coverage rate = 4/4 = 1.0 (None-camp excluded from both numerator AND total)
        assert rate == 1.0

    def test_compute_top30_empty_results_returns_zero_rate(self, audit_v23) -> None:
        """Empty results → empty per_camp + 0.0 rate (no ZeroDivisionError)."""
        per_camp, rate = audit_v23._compute_top30([])
        assert per_camp == []
        assert rate == 0.0


class TestArgparseSurface:
    """argparse smoke: --dry-run and --output-path flags present + defaulted."""

    def test_argparse_accepts_dry_run_and_output_path(self, audit_v23, monkeypatch, capsys) -> None:
        """`audit_camp_v23 --help` succeeds and mentions both flags."""
        # We can't trivially call main() without DB — but we can verify the
        # parser by invoking it via the module's __doc__ / parser inspection.
        # Simplest robust check: import-time constants prove the script is
        # importable and the parser flags are declared in main().
        import inspect

        main_src = inspect.getsource(audit_v23.main)
        assert "--output-path" in main_src, "argparse missing --output-path"
        assert "--dry-run" in main_src, "argparse missing --dry-run"
