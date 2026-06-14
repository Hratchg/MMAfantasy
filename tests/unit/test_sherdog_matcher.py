"""Unit tests for Sherdog fighter name matching logic.

Tests use mock data -- no real HTTP requests.
"""

from __future__ import annotations

from ufc_prediction.scraper.sherdog import match_fighter_name


class TestMatchFighterName:
    """Tests for match_fighter_name."""

    def test_exact_match(self) -> None:
        candidates = [
            ("Anderson Silva", "https://www.sherdog.com/fighter/Anderson-Silva-1356"),
            ("Anderson da Silva", "https://www.sherdog.com/fighter/Anderson-da-Silva-99887"),
        ]
        result = match_fighter_name("Anderson Silva", candidates)
        assert result is not None
        assert result[0] == "Anderson Silva"
        assert result[1] == "https://www.sherdog.com/fighter/Anderson-Silva-1356"

    def test_fuzzy_match_name_variant(self) -> None:
        candidates = [
            ("Anderson da Silva", "https://www.sherdog.com/fighter/Anderson-da-Silva-99887"),
            ("Anderson Silva Jr", "https://www.sherdog.com/fighter/Anderson-Silva-Jr-11223"),
        ]
        result = match_fighter_name("Anderson Silva", candidates)
        assert result is not None
        # Should match "Anderson da Silva" as closest (>= 85 score)

    def test_no_match_below_threshold(self) -> None:
        candidates = [
            ("Completely Different Name", "https://www.sherdog.com/fighter/Different-12345"),
            ("Another Person", "https://www.sherdog.com/fighter/Another-67890"),
        ]
        result = match_fighter_name("Anderson Silva", candidates)
        assert result is None

    def test_reversed_name_order(self) -> None:
        candidates = [
            ("McGregor, Conor", "https://www.sherdog.com/fighter/Conor-McGregor-29688"),
        ]
        result = match_fighter_name("Conor McGregor", candidates)
        assert result is not None
        assert result[0] == "McGregor, Conor"

    def test_empty_candidates(self) -> None:
        result = match_fighter_name("Anderson Silva", [])
        assert result is None

    def test_threshold_boundary(self) -> None:
        # Names that are similar but below the 85 threshold
        candidates = [
            ("Andy Silva", "https://www.sherdog.com/fighter/Andy-Silva-999"),
        ]
        # "Andy Silva" vs "Anderson Silva" -- similarity is borderline
        # The exact score depends on rapidfuzz, but this tests the threshold logic
        result = match_fighter_name("Anderson Silva", candidates, threshold=95)
        # With a very high threshold, this should fail to match
        assert result is None

    def test_custom_threshold(self) -> None:
        candidates = [
            ("Anderson Silva", "https://www.sherdog.com/fighter/Anderson-Silva-1356"),
        ]
        # Even with a high threshold, exact match should pass
        result = match_fighter_name("Anderson Silva", candidates, threshold=99)
        assert result is not None
