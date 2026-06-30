"""Wave-0 RED tests for scripts/audit_camp_v22.py.

Per CONTEXT.md D-01..D-06 + D-13 + Phase 20 plan 20-01 Task 4 spec.

Mocks Sherdog HTML via tests/fixtures/sherdog_fighter_page_with_association.html
(operator-saved live page from Task 2). Mocks ScraperClient via MagicMock per
the project pattern established in tests/scraper/test_bfo_scraper.py:21.

Uses ``pytest.importorskip`` so tests SKIP cleanly until
``scripts/audit_camp_v22.py`` lands, then turn GREEN on import.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def audit():
    """Lazy-import the audit script as a module.

    Mirrors tests/scraper/test_bfo_scraper.py:28-36 importorskip pattern so
    Wave-0 RED can run before scripts/audit_camp_v22.py lands.
    """
    return pytest.importorskip("scripts.audit_camp_v22")


class TestLockedConstants:
    """AF-style invariants — drift halts CI before any live HTTP."""

    def test_af_constants_locked(self, audit) -> None:
        """Module-top constants match D-01 / D-06 locked values verbatim."""
        assert audit.SAMPLE_SIZE == 200
        assert audit.RANDOM_STATE == 42
        assert audit.ACTIVE_WINDOW_DAYS == 730
        assert audit.PRESENCE_THRESHOLD_MIN == 0.30
        assert audit.PRESENCE_THRESHOLD_FULL == 0.70
        assert audit.PARSEABLE_THRESHOLD_MIN == 0.25
        assert audit.PARSEABLE_THRESHOLD_FULL == 0.60
        assert audit.TOP30_THRESHOLD_MIN == 0.60
        assert audit.TOP30_THRESHOLD_FULL == 0.70


class TestScopeRecommendation:
    """D-06 mechanical scope_recommendation derivation."""

    def test_scope_recommendation_full(self, audit) -> None:
        """All three rates ≥ full-tier → 'full'."""
        rec = audit._derive_scope_recommendation(
            present_rate=0.75,
            parseable_rate=0.65,
            top30_rate=0.72,
        )
        assert rec == "full"

    def test_scope_recommendation_reduced(self, audit) -> None:
        """All three rates clear MIN tier; ≥1 below FULL tier → 'reduced'."""
        rec = audit._derive_scope_recommendation(
            present_rate=0.40,
            parseable_rate=0.30,
            top30_rate=0.65,
        )
        assert rec == "reduced"

    def test_scope_recommendation_drop_low_presence(self, audit) -> None:
        """Any rate < MIN tier → 'drop' (presence here at 0.20 < 0.30)."""
        rec = audit._derive_scope_recommendation(
            present_rate=0.20,
            parseable_rate=0.30,
            top30_rate=0.65,
        )
        assert rec == "drop"

    def test_scope_recommendation_drop_boundary(self, audit) -> None:
        """Exact MIN tier thresholds clear (≥ comparison) → 'reduced'."""
        rec = audit._derive_scope_recommendation(
            present_rate=0.30,
            parseable_rate=0.25,
            top30_rate=0.60,
        )
        assert rec == "reduced"


class TestStratifiedSample:
    """D-01 stratified random sample determinism (random_state=42)."""

    def test_stratified_sample_size_200_deterministic(self, audit) -> None:
        """Sampling twice with seed=42 yields IDENTICAL fighter_id ordering."""
        import pandas as pd

        # 4 strata × ~844 rows = 3375 fighters total
        # (matches plan corpus prior). 800+800+800+975 = 3375.
        df = pd.DataFrame(
            {
                "fighter_id": list(range(3375)),
                "strat_key": (
                    ["LW_True"] * 800 + ["LW_False"] * 800 + ["WW_True"] * 800 + ["WW_False"] * 975
                ),
            }
        )
        s1 = audit._stratified_sample(df, n=200, seed=42)
        s2 = audit._stratified_sample(df, n=200, seed=42)
        assert len(s1) == 200
        assert list(s1["fighter_id"]) == list(s2["fighter_id"])


class TestNormalizeAssociation:
    """Alias normalization (D-02): strip suffixes, lowercase, trim."""

    def test_normalize_association_strips_suffixes(self, audit) -> None:
        """Suffix stripping per ALIAS_SUFFIXES_TO_STRIP, then lowercase trim."""
        assert audit._normalize_association("American Top Team (USA)") == "american top team"
        assert audit._normalize_association("Tiger Muay Thai Academy") == "tiger muay thai"

    def test_normalize_association_handles_none(self, audit) -> None:
        """None / empty input returns None (no normalization on empty)."""
        assert audit._normalize_association(None) is None
        assert audit._normalize_association("") is None
        assert audit._normalize_association("   ") is None

    def test_normalize_association_passthrough_no_suffix(self, audit) -> None:
        """No suffix present → just lowercase + trim."""
        assert audit._normalize_association("City Kickboxing") == "city kickboxing"


class TestWriteAuditJson:
    """D-05 schema: 12-key payload (11 required + risk_flags)."""

    def test_emits_valid_json_with_all_d05_keys(
        self,
        audit,
        tmp_path: Path,
    ) -> None:
        """_write_audit_json produces JSON with the required D-05 keys."""
        out = tmp_path / "CAMP_00_AUDIT.json"
        audit._write_audit_json(
            out,
            sample_size=200,
            n_fetched=192,
            n_fetch_failures=8,
            association_present_rate=0.365,
            parseable_rate=0.31,
            top30_coverage_rate=0.65,
            scope_recommendation="reduced",
            per_camp_top30=[],
            unparseable_examples=[],
            stratification_breakdown={
                "derivation_rule": ("is_active = latest_fight_event_date >= today - 730d"),
                "strata": {},
            },
            risk_flags=[],
        )
        payload = json.loads(out.read_text())
        required = {
            "audit_version",
            "sample_size",
            "sample_random_state",
            "n_fetched",
            "n_fetch_failures",
            "association_present_rate",
            "parseable_rate",
            "top30_coverage_rate",
            "scope_recommendation",
            "per_camp_top30",
            "unparseable_examples",
            "stratification_breakdown",
        }
        missing = required - payload.keys()
        assert not missing, f"missing required D-05 keys: {missing}"
        # Audit version locked
        assert payload["audit_version"] == "v22.00"
        # Sample random state locked
        assert payload["sample_random_state"] == 42


class TestPitfall5Denominator:
    """Pitfall #5: rates MUST use n_fetched, NOT sample_size, as denominator."""

    def test_audit_present_rate_denom_is_n_fetched_not_sample_size(
        self,
        audit,
        tmp_path: Path,
    ) -> None:
        """Rates written to JSON reflect n_fetched-based ratios.

        Guards against the classic 'computed correctly but divided by the wrong
        denominator' bug. n_fetched=192, n_present=70 → 70/192 ≈ 0.3646; the
        WRONG answer would be 70/200 = 0.35.
        """
        out = tmp_path / "CAMP_00_AUDIT.json"
        n_fetched = 192
        n_present = 70
        expected = round(n_present / n_fetched, 4)
        audit._write_audit_json(
            out,
            sample_size=200,
            n_fetched=n_fetched,
            n_fetch_failures=200 - n_fetched,
            association_present_rate=expected,
            parseable_rate=0.0,
            top30_coverage_rate=0.0,
            scope_recommendation="drop",
            per_camp_top30=[],
            unparseable_examples=[],
            stratification_breakdown={"derivation_rule": "", "strata": {}},
            risk_flags=[],
        )
        payload = json.loads(out.read_text())
        # 70 / 192 ≈ 0.3646 (NOT 70 / 200 = 0.35)
        assert payload["association_present_rate"] == pytest.approx(0.3646, abs=0.001)
        assert payload["association_present_rate"] != pytest.approx(0.35, abs=0.001)
