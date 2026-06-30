"""Dedicated tests for cardinality checks across all parsers.

Verifies SCRP-04: cardinality checks fail loudly on malformed HTML.
"""

from __future__ import annotations

import pytest

from ufc_prediction.scraper.parse_event_detail import parse_event_detail
from ufc_prediction.scraper.parse_event_list import parse_event_list
from ufc_prediction.scraper.parse_fight_detail import parse_fight_detail
from ufc_prediction.scraper.parse_fighter import parse_fighter_profile


class TestCardinalityChecks:
    """All parsers raise ValueError on structurally invalid HTML."""

    def test_event_list_empty_html_raises(self) -> None:
        """parse_event_list on empty HTML raises ValueError."""
        with pytest.raises(ValueError):
            parse_event_list("<html></html>", min_events=1)

    def test_event_detail_empty_html_raises(self) -> None:
        """parse_event_detail on empty HTML raises ValueError."""
        with pytest.raises(ValueError):
            parse_event_detail("<html></html>", "http://x")

    def test_fight_detail_empty_html_raises(self) -> None:
        """parse_fight_detail on empty HTML raises ValueError."""
        with pytest.raises(ValueError):
            parse_fight_detail("<html></html>")

    def test_fighter_profile_empty_html_raises(self) -> None:
        """parse_fighter_profile on empty HTML raises ValueError."""
        with pytest.raises(ValueError):
            parse_fighter_profile("<html></html>")

    def test_fight_detail_one_person_raises(self) -> None:
        """HTML with only 1 person div raises ValueError about cardinality."""
        html = """<html><body>
        <div class="b-fight-details__person">
          <i class="b-fight-details__person-status">W</i>
          <a class="b-link b-fight-details__person-link" href="http://x">Fighter</a>
        </div>
        </body></html>"""
        with pytest.raises(ValueError, match="cardinality"):
            parse_fight_detail(html)

    def test_event_list_no_table_raises(self) -> None:
        """HTML with body but no event table raises ValueError."""
        html = "<html><body><p>No table here</p></body></html>"
        with pytest.raises(ValueError):
            parse_event_list(html, min_events=1)
