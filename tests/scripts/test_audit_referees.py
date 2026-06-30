"""Wave-0 RED tests for scripts/audit_referees.py + referee_normalize module.

Pattern: importorskip lazy-import (mirrors tests/scripts/test_audit_camp_v22.py:23-30)
so this file lands GREEN at Task 2 (referee_normalize tests pass) and remains GREEN
at Task 3 (audit script tests turn ON when scripts/audit_referees.py lands).
"""

from __future__ import annotations

import pytest

# ─── Module 1: referee_normalize (lands at Task 2; tests must PASS now) ──────
from ufc_prediction.scraper.referee_normalize import (
    REFEREE_ALIASES,
    normalize_referee_name,
)


class TestNormalizeRefereeName:
    def test_canonical_lowercase_hyphen(self) -> None:
        assert normalize_referee_name("Herb Dean") == "herb-dean"

    def test_alias_collapse(self) -> None:
        assert normalize_referee_name("Herbert Dean") == "herb-dean"

    def test_nfkd_ascii_fold(self) -> None:
        assert normalize_referee_name("Mike Beltrán") == "mike-beltran"

    def test_whitespace_collapse(self) -> None:
        assert normalize_referee_name("  Marc  Goddard  ") == "marc-goddard"

    def test_none_returns_none(self) -> None:
        assert normalize_referee_name(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert normalize_referee_name("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert normalize_referee_name("   ") is None

    def test_unknown_referee_passes_through(self) -> None:
        assert normalize_referee_name("UnknownRef X") == "unknownref-x"

    def test_aliases_dict_typed_and_seeded(self) -> None:
        assert isinstance(REFEREE_ALIASES, dict)
        assert REFEREE_ALIASES.get("herbert-dean") == "herb-dean"


# ─── Module 2: audit_referees (lands at Task 3; importorskip until then) ─────


@pytest.fixture
def audit():
    return pytest.importorskip("scripts.audit_referees")


class TestLockedConstants:
    def test_sample_size_locked(self, audit) -> None:
        assert audit.SAMPLE_SIZE == 200

    def test_random_state_locked(self, audit) -> None:
        assert audit.RANDOM_STATE == 42

    def test_top30_threshold_locked(self, audit) -> None:
        assert audit.REF_TOP30_COVERAGE_THRESHOLD == 0.60

    def test_audit_version_locked(self, audit) -> None:
        assert audit.AUDIT_VERSION == "v22.00"

    def test_output_path_in_phase_dir(self, audit) -> None:
        assert "22-ref-travel-camp-schema-scrapers-and-migrations" in str(audit.OUTPUT_PATH_DEFAULT)
        assert str(audit.OUTPUT_PATH_DEFAULT).endswith("REF_00_AUDIT.json")


class TestDeriveScopeRecommendation:
    def test_proceed_at_threshold(self, audit) -> None:
        # Binary per CONTEXT D-01: top30 >= 0.60 -> proceed
        assert audit._derive_scope_recommendation(top30_rate=0.60) == "proceed"

    def test_proceed_above_threshold(self, audit) -> None:
        assert audit._derive_scope_recommendation(top30_rate=0.95) == "proceed"

    def test_drop_below_threshold(self, audit) -> None:
        # 0.5999 < 0.60 -> drop (mirror Phase 20 CAMP outcome semantics)
        assert audit._derive_scope_recommendation(top30_rate=0.5999) == "drop"

    def test_drop_at_zero(self, audit) -> None:
        assert audit._derive_scope_recommendation(top30_rate=0.0) == "drop"


# ─── Module 3: top30 coverage helper (lands at Task 3; importorskip) ─────────


class TestComputeTop30Coverage:
    def test_empty_counter_returns_zero(self, audit) -> None:
        from collections import Counter

        assert audit._compute_top30_coverage(Counter()) == 0.0

    def test_single_referee_returns_one(self, audit) -> None:
        from collections import Counter

        assert audit._compute_top30_coverage(Counter({"herb-dean": 100})) == 1.0

    def test_top30_concentration(self, audit) -> None:
        from collections import Counter

        # 30 refs × 10 fights = 300; plus 70 long-tail × 1 = 70; total=370; top30/total=300/370≈0.811
        c = Counter({f"ref-{i}": 10 for i in range(30)})
        c.update({f"tail-{i}": 1 for i in range(70)})
        rate = audit._compute_top30_coverage(c)
        assert abs(rate - (300 / 370)) < 1e-6
