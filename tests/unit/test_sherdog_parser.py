"""Unit tests for Sherdog HTML parsers and stats computation.

Tests use saved HTML fixtures -- no real HTTP requests.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from ufc_prediction.scraper.sherdog import (
    compute_pre_ufc_stats,
    filter_pre_ufc_fights,
    parse_association_from_html,
    parse_sherdog_fighter_page,
    parse_sherdog_search_results,
)
from ufc_prediction.scraper.sherdog_models import SherdogFight

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def fighter_page_html() -> str:
    return (FIXTURES_DIR / "sherdog_fighter_page.html").read_text(encoding="utf-8")


@pytest.fixture
def search_results_html() -> str:
    return (FIXTURES_DIR / "sherdog_search_results.html").read_text(encoding="utf-8")


class TestParseSherdogFighterPage:
    """Tests for parse_sherdog_fighter_page."""

    def test_extracts_fighter_name(self, fighter_page_html: str) -> None:
        profile = parse_sherdog_fighter_page(
            fighter_page_html,
            "https://www.sherdog.com/fighter/Anderson-Silva-1356",
        )
        assert profile.name == "Anderson Silva"

    def test_extracts_sherdog_url(self, fighter_page_html: str) -> None:
        url = "https://www.sherdog.com/fighter/Anderson-Silva-1356"
        profile = parse_sherdog_fighter_page(fighter_page_html, url)
        assert profile.sherdog_url == url

    def test_extracts_correct_number_of_fights(self, fighter_page_html: str) -> None:
        profile = parse_sherdog_fighter_page(
            fighter_page_html,
            "https://www.sherdog.com/fighter/Anderson-Silva-1356",
        )
        assert len(profile.fights) == 5

    def test_extracts_fight_results(self, fighter_page_html: str) -> None:
        profile = parse_sherdog_fighter_page(
            fighter_page_html,
            "https://www.sherdog.com/fighter/Anderson-Silva-1356",
        )
        results = [f.result for f in profile.fights]
        assert results == ["win", "loss", "win", "win", "loss"]

    def test_extracts_opponent_names(self, fighter_page_html: str) -> None:
        profile = parse_sherdog_fighter_page(
            fighter_page_html,
            "https://www.sherdog.com/fighter/Anderson-Silva-1356",
        )
        assert profile.fights[0].opponent_name == "Derek Brunson"
        assert profile.fights[4].opponent_name == "Daiju Takase"

    def test_extracts_methods(self, fighter_page_html: str) -> None:
        profile = parse_sherdog_fighter_page(
            fighter_page_html,
            "https://www.sherdog.com/fighter/Anderson-Silva-1356",
        )
        assert profile.fights[2].method == "KO/TKO"
        assert profile.fights[3].method == "Submission"

    def test_extracts_event_dates(self, fighter_page_html: str) -> None:
        profile = parse_sherdog_fighter_page(
            fighter_page_html,
            "https://www.sherdog.com/fighter/Anderson-Silva-1356",
        )
        assert profile.fights[0].event_date == date(2017, 2, 11)
        assert profile.fights[4].event_date == date(2003, 6, 8)

    def test_extracts_event_names(self, fighter_page_html: str) -> None:
        profile = parse_sherdog_fighter_page(
            fighter_page_html,
            "https://www.sherdog.com/fighter/Anderson-Silva-1356",
        )
        assert "UFC 208" in profile.fights[0].event_name
        assert "PRIDE FC 26" in profile.fights[4].event_name

    def test_extracts_round_finished(self, fighter_page_html: str) -> None:
        profile = parse_sherdog_fighter_page(
            fighter_page_html,
            "https://www.sherdog.com/fighter/Anderson-Silva-1356",
        )
        assert profile.fights[2].round_finished == 2
        assert profile.fights[0].round_finished == 3


class TestParseSherdogSearchResults:
    """Tests for parse_sherdog_search_results."""

    def test_extracts_fighter_list(self, search_results_html: str) -> None:
        results = parse_sherdog_search_results(search_results_html)
        assert len(results) == 3

    def test_extracts_names_and_urls(self, search_results_html: str) -> None:
        results = parse_sherdog_search_results(search_results_html)
        assert results[0] == (
            "Anderson Silva",
            "https://www.sherdog.com/fighter/Anderson-Silva-1356",
        )
        assert results[1][0] == "Anderson da Silva"

    def test_urls_are_absolute(self, search_results_html: str) -> None:
        results = parse_sherdog_search_results(search_results_html)
        for _name, url in results:
            assert url.startswith("https://www.sherdog.com/")


class TestFilterPreUFCFights:
    """Tests for filter_pre_ufc_fights."""

    def test_filters_fights_before_first_ufc_date(self) -> None:
        fights = [
            SherdogFight(
                result="win",
                opponent_name="A",
                method="KO/TKO",
                event_name="Local Show 1",
                event_date=date(2005, 1, 1),
                round_finished=1,
            ),
            SherdogFight(
                result="win",
                opponent_name="B",
                method="Decision",
                event_name="UFC 100",
                event_date=date(2009, 7, 11),
                round_finished=3,
            ),
            SherdogFight(
                result="loss",
                opponent_name="C",
                method="Submission",
                event_name="Cage Warriors 42",
                event_date=date(2004, 6, 15),
                round_finished=2,
            ),
        ]
        pre_ufc = filter_pre_ufc_fights(fights, first_ufc_date=date(2009, 7, 11))
        assert len(pre_ufc) == 2
        assert all(f.event_date < date(2009, 7, 11) for f in pre_ufc)

    def test_excludes_ufc_events_even_before_cutoff(self) -> None:
        fights = [
            SherdogFight(
                result="win",
                opponent_name="A",
                method="KO/TKO",
                event_name="UFC Fight Night 10",
                event_date=date(2005, 1, 1),
                round_finished=1,
            ),
            SherdogFight(
                result="win",
                opponent_name="B",
                method="Decision",
                event_name="Local Show",
                event_date=date(2004, 6, 15),
                round_finished=3,
            ),
        ]
        pre_ufc = filter_pre_ufc_fights(fights, first_ufc_date=date(2010, 1, 1))
        assert len(pre_ufc) == 1
        assert pre_ufc[0].opponent_name == "B"

    def test_handles_none_event_date(self) -> None:
        fights = [
            SherdogFight(
                result="win",
                opponent_name="A",
                method="KO/TKO",
                event_name="Local Show",
                event_date=None,
                round_finished=1,
            ),
        ]
        pre_ufc = filter_pre_ufc_fights(fights, first_ufc_date=date(2010, 1, 1))
        # Fights with no date are excluded (can't determine if pre-UFC)
        assert len(pre_ufc) == 0


class TestComputePreUFCStats:
    """Tests for compute_pre_ufc_stats."""

    def test_computes_correct_stats(self) -> None:
        fights = [
            SherdogFight(
                result="win",
                opponent_name="A",
                method="KO/TKO",
                event_name="Show 1",
                event_date=date(2002, 1, 1),
                round_finished=1,
            ),
            SherdogFight(
                result="win",
                opponent_name="B",
                method="Submission",
                event_name="Show 2",
                event_date=date(2003, 1, 1),
                round_finished=2,
            ),
            SherdogFight(
                result="win",
                opponent_name="C",
                method="Decision",
                event_name="Show 3",
                event_date=date(2004, 1, 1),
                round_finished=3,
            ),
            SherdogFight(
                result="loss",
                opponent_name="D",
                method="KO/TKO",
                event_name="Show 4",
                event_date=date(2005, 1, 1),
                round_finished=1,
            ),
        ]
        stats = compute_pre_ufc_stats(fights)
        assert stats.total_wins == 3
        assert stats.total_losses == 1
        assert stats.total_draws == 0
        assert stats.total_fights == 4
        assert stats.win_pct == pytest.approx(0.75)
        # KO wins / total wins = 1/3
        assert stats.ko_finish_rate == pytest.approx(1 / 3, abs=0.01)
        # Sub wins / total wins = 1/3
        assert stats.sub_finish_rate == pytest.approx(1 / 3, abs=0.01)
        # Decision wins / total wins = 1/3
        assert stats.decision_rate == pytest.approx(1 / 3, abs=0.01)
        # Career years: 2002 to 2005 = 3.0 years
        assert stats.career_years == pytest.approx(3.0, abs=0.1)
        assert len(stats.fights) == 4

    def test_handles_empty_fight_list(self) -> None:
        stats = compute_pre_ufc_stats([])
        assert stats.total_wins == 0
        assert stats.total_losses == 0
        assert stats.total_draws == 0
        assert stats.total_fights == 0
        assert stats.win_pct == 0.0
        assert stats.ko_finish_rate == 0.0
        assert stats.sub_finish_rate == 0.0
        assert stats.decision_rate == 0.0
        assert stats.career_years == 0.0
        assert len(stats.fights) == 0

    def test_handles_all_draws(self) -> None:
        fights = [
            SherdogFight(
                result="draw",
                opponent_name="A",
                method="Decision",
                event_name="Show 1",
                event_date=date(2003, 6, 1),
                round_finished=3,
            ),
            SherdogFight(
                result="draw",
                opponent_name="B",
                method="Decision",
                event_name="Show 2",
                event_date=date(2004, 6, 1),
                round_finished=3,
            ),
        ]
        stats = compute_pre_ufc_stats(fights)
        assert stats.total_draws == 2
        assert stats.total_wins == 0
        assert stats.win_pct == 0.0
        # With zero wins, finish rates should be 0.0
        assert stats.ko_finish_rate == 0.0
        assert stats.sub_finish_rate == 0.0
        assert stats.decision_rate == 0.0


class TestParseAssociationFromHtml:
    """Tests for parse_association_from_html sibling helper (Phase 20 D-02).

    Defensive layered selectors:
      1. <span itemprop="memberOf"> + nested <span itemprop="name">
      2. div.association-class text content (strip "ASSOCIATION" label)
      3. href ?association= query param from <a class="association">

    Returns the raw verbatim association string or None (audit script handles
    alias normalization downstream per CONTEXT.md D-02 / D-05).
    """

    def test_extracts_itemprop_memberOf(self) -> None:
        """Layer 1: Schema.org microdata selector returns the camp text."""
        html = '<html><body><span itemprop="memberOf">American Top Team</span></body></html>'
        assert parse_association_from_html(html) == "American Top Team"

    def test_extracts_from_saved_fixture(self) -> None:
        """Layer 1 against the operator-saved live Adesanya fixture.

        Sherdog wraps the camp in a nested
        <span itemprop="memberOf"><a><span itemprop="name">...</span></a></span>
        structure inside div.association-class. The parser should extract
        "City Kickboxing" verbatim (no suffix-stripping needed for this fixture).
        """
        html = (FIXTURES_DIR / "sherdog_fighter_page_with_association.html").read_text(
            encoding="utf-8",
        )
        result = parse_association_from_html(html)
        assert result is not None
        assert len(result.strip()) >= 2
        # Tighter operator-confirmed assertion: live Adesanya fixture
        # carries the verbatim string "City Kickboxing".
        assert result == "City Kickboxing"

    def test_label_fallback(self) -> None:
        """Layer 2: <strong>Association:</strong> label + sibling anchor."""
        html = (
            '<html><body><strong>Association:</strong> <a href="/team/atss">ATSS</a></body></html>'
        )
        assert parse_association_from_html(html) == "ATSS"

    def test_returns_none_when_absent(self) -> None:
        """No association markup → None.

        Uses the existing Anderson Silva fixture, verified to lack the
        association microdata (FY2017 era Sherdog markup).
        """
        html = (FIXTURES_DIR / "sherdog_fighter_page.html").read_text(
            encoding="utf-8",
        )
        assert parse_association_from_html(html) is None

    def test_returns_none_on_empty_html(self) -> None:
        """Empty / whitespace-only HTML returns None."""
        assert parse_association_from_html("") is None
        assert parse_association_from_html("<html></html>") is None

    def test_empty_itemprop_falls_through_to_none(self) -> None:
        """An empty itemprop element does not falsely succeed."""
        html = '<html><body><span itemprop="memberOf">   </span></body></html>'
        assert parse_association_from_html(html) is None
