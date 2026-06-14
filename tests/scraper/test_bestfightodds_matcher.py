"""Unit tests for BFO fighter-name fuzzy matching (ODDS-03).

Pure-logic — no network, no DB. Covers normalize_name + match_bfo_name,
including the Saint-Preux/St-Preux regression from Pitfall 6 in
.planning/phases/13-betting-odds-integration/13-RESEARCH.md.
"""

from __future__ import annotations

import inspect

from rapidfuzz import fuzz

from ufc_prediction.scraper.bfo_matcher import match_bfo_name, normalize_name


class TestNormalizeName:
    """`normalize_name` — lowercase, strip punct, collapse Saint/St, Junior/Jr."""

    def test_normalize_lowercases(self) -> None:
        """Lowercasing is idempotent across word boundaries."""
        assert normalize_name("Jon JONES") == "jon jones"

    def test_normalize_strips_hyphens(self) -> None:
        """'Saint-Preux' and 'St-Preux' normalize to a form where fuzz.ratio >= 90.

        The contract is that Saint/St collapse produces near-identical strings,
        not a specific literal output.
        """
        a = normalize_name("Saint-Preux")
        b = normalize_name("St-Preux")
        assert fuzz.ratio(a, b) >= 90

    def test_normalize_collapses_saint_to_st(self) -> None:
        """'Ovince Saint Preux' normalizes toward 'ovince st preux'."""
        out = normalize_name("Ovince Saint Preux")
        assert "st preux" in out
        # Cross-check with the hyphenated form
        assert fuzz.ratio(out, normalize_name("Ovince St-Preux")) >= 90

    def test_normalize_collapses_junior_to_jr(self) -> None:
        """'Dan Henderson Junior' and 'Dan Henderson Jr' are near-equivalent."""
        a = normalize_name("Dan Henderson Junior")
        b = normalize_name("Dan Henderson Jr")
        assert fuzz.ratio(a, b) >= 95

    def test_normalize_preserves_multibyte(self) -> None:
        """Non-empty string, includes 'aldo' for 'José Aldo'."""
        out = normalize_name("José Aldo")
        assert out  # non-empty
        assert "aldo" in out


class TestMatchBfoName:
    """`match_bfo_name` — fuzzy match with reversed-name fallback."""

    def test_exact_match_returns_candidate(self) -> None:
        """Exact string match selects its candidate."""
        result = match_bfo_name(
            "Jon Jones",
            [(1, "Jon Jones"), (2, "Khabib Nurmagomedov")],
        )
        assert result == (1, "Jon Jones")

    def test_below_threshold_returns_none(self) -> None:
        """No candidate above threshold => None."""
        result = match_bfo_name("Jon Jones", [(1, "Conor McGregor")])
        assert result is None

    def test_reversed_name_last_first_matches(self) -> None:
        """'Jones, Jon' (BFO) against 'Jon Jones' (DB) matches via reversal."""
        result = match_bfo_name("Jones, Jon", [(1, "Jon Jones")])
        assert result == (1, "Jon Jones")

    def test_reversed_name_first_last_matches(self) -> None:
        """'Jon Jones' (BFO) against 'Jones, Jon' (DB) matches via reversal."""
        result = match_bfo_name("Jon Jones", [(1, "Jones, Jon")])
        assert result == (1, "Jones, Jon")

    def test_empty_candidates_returns_none(self) -> None:
        """Empty candidate list => None (no iteration)."""
        assert match_bfo_name("Jon Jones", []) is None

    def test_best_match_wins_over_close_second(self) -> None:
        """Exact match beats a similar-but-not-identical candidate."""
        result = match_bfo_name(
            "Jon Jones",
            [(1, "Jonathan Jones"), (2, "Jon Jones")],
        )
        assert result == (2, "Jon Jones")

    def test_default_threshold_is_80(self) -> None:
        """Signature default threshold is exactly 80 (per Pitfall 6)."""
        sig = inspect.signature(match_bfo_name)
        assert sig.parameters["threshold"].default == 80

    def test_saint_preux_matches_st_preux_with_normalization(self) -> None:
        """Pitfall 6 regression: 'Ovince Saint-Preux' matches 'Ovince St-Preux'.

        Without normalize_name, fuzz.ratio ~ 78 (below the 80 threshold).
        After Saint->st + hyphen collapse, score >= threshold.
        """
        result = match_bfo_name(
            "Ovince Saint-Preux",
            [(1, "Ovince St-Preux")],
        )
        assert result == (1, "Ovince St-Preux")

    def test_returns_tuple_of_int_and_str(self) -> None:
        """Matched return is (int, str)."""
        result = match_bfo_name("Jon Jones", [(1, "Jon Jones")])
        assert result is not None
        assert isinstance(result[0], int)
        assert isinstance(result[1], str)

    def test_threshold_override_excludes_weak_match(self) -> None:
        """An explicit high threshold correctly rejects a weaker match."""
        # "Jon Jones" vs "Jonathan Jones" scores in the 70s-80s — threshold 99 rejects.
        result = match_bfo_name(
            "Jon Jones",
            [(1, "Jonathan Jones")],
            threshold=99,
        )
        assert result is None
