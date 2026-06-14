"""Tests for fight detail page parser."""

from __future__ import annotations

import re

import pytest

from ufc_prediction.scraper.parse_fight_detail import parse_fight_detail
from ufc_prediction.scraper.models import FightDetailPage, RoundStatsRaw, SigStrikesRaw


# -- 1-round KO/TKO fixture tests -------------------------------------------


class TestParseFightDetail1Round:
    """Tests using the 1-round KO/TKO fight detail fixture."""

    def test_parse_1round_fight_returns_page(self, fight_detail_1round_html: str) -> None:
        """parse_fight_detail returns a FightDetailPage."""
        page = parse_fight_detail(fight_detail_1round_html)
        assert isinstance(page, FightDetailPage)

    def test_parse_1round_fighter_names(self, fight_detail_1round_html: str) -> None:
        """Fighter names are non-empty strings."""
        page = parse_fight_detail(fight_detail_1round_html)
        assert page.fighter_a_name == "Carlos Ulberg"
        assert page.fighter_b_name == "Jiri Prochazka"

    def test_parse_1round_fighter_urls(self, fight_detail_1round_html: str) -> None:
        """Both fighter URLs contain 'fighter-details/'."""
        page = parse_fight_detail(fight_detail_1round_html)
        assert "fighter-details/" in page.fighter_a_url
        assert "fighter-details/" in page.fighter_b_url

    def test_parse_1round_status(self, fight_detail_1round_html: str) -> None:
        """Winner has status W, loser has status L."""
        page = parse_fight_detail(fight_detail_1round_html)
        assert page.fighter_a_status == "W"
        assert page.fighter_b_status == "L"

    def test_parse_1round_method(self, fight_detail_1round_html: str) -> None:
        """Method is KO/TKO."""
        page = parse_fight_detail(fight_detail_1round_html)
        assert page.method == "KO/TKO"

    def test_parse_1round_method_detail(self, fight_detail_1round_html: str) -> None:
        """Method detail extracted from details section."""
        page = parse_fight_detail(fight_detail_1round_html)
        assert page.method_detail is not None
        assert "Punch" in page.method_detail

    def test_parse_1round_round(self, fight_detail_1round_html: str) -> None:
        """Round finished is 1."""
        page = parse_fight_detail(fight_detail_1round_html)
        assert page.round_finished == 1

    def test_parse_1round_time(self, fight_detail_1round_html: str) -> None:
        """Time finished matches M:SS format."""
        page = parse_fight_detail(fight_detail_1round_html)
        assert page.time_finished is not None
        assert re.match(r"\d+:\d{2}", page.time_finished)

    def test_parse_1round_time_format(self, fight_detail_1round_html: str) -> None:
        """Time format extracted correctly."""
        page = parse_fight_detail(fight_detail_1round_html)
        assert page.time_format is not None
        assert "5 Rnd" in page.time_format

    def test_parse_1round_referee(self, fight_detail_1round_html: str) -> None:
        """Referee extracted correctly."""
        page = parse_fight_detail(fight_detail_1round_html)
        assert page.referee == "Marc Goddard"

    def test_parse_1round_bout_type(self, fight_detail_1round_html: str) -> None:
        """Bout type extracted correctly."""
        page = parse_fight_detail(fight_detail_1round_html)
        assert "Light Heavyweight" in page.bout_type

    def test_parse_1round_totals(self, fight_detail_1round_html: str) -> None:
        """Totals is a 2-tuple of RoundStatsRaw with non-empty fields."""
        page = parse_fight_detail(fight_detail_1round_html)
        assert len(page.totals) == 2
        assert isinstance(page.totals[0], RoundStatsRaw)
        assert isinstance(page.totals[1], RoundStatsRaw)
        # Check some fields are populated
        assert page.totals[0].knockdowns == "1"
        assert page.totals[0].sig_str == "45 of 67"

    def test_parse_1round_sig_strikes(self, fight_detail_1round_html: str) -> None:
        """sig_strikes is a 2-tuple of SigStrikesRaw."""
        page = parse_fight_detail(fight_detail_1round_html)
        assert len(page.sig_strikes) == 2
        assert isinstance(page.sig_strikes[0], SigStrikesRaw)
        assert isinstance(page.sig_strikes[1], SigStrikesRaw)
        # Fight-level sig strikes should have head data
        assert "of" in page.sig_strikes[0].head

    def test_parse_1round_per_round(self, fight_detail_1round_html: str) -> None:
        """per_round_totals has length 1 for a 1-round fight."""
        page = parse_fight_detail(fight_detail_1round_html)
        assert len(page.per_round_totals) == 1
        assert len(page.per_round_sig_strikes) == 1

    def test_parse_1round_alignment(self, fight_detail_1round_html: str) -> None:
        """per_round_totals and per_round_sig_strikes have equal length."""
        page = parse_fight_detail(fight_detail_1round_html)
        assert len(page.per_round_totals) == len(page.per_round_sig_strikes)

    def test_parse_1round_per_round_fighter_names(self, fight_detail_1round_html: str) -> None:
        """per_round_totals[0] fighter names are non-empty."""
        page = parse_fight_detail(fight_detail_1round_html)
        assert page.per_round_totals[0][0].fighter_name != ""
        assert page.per_round_totals[0][1].fighter_name != ""


# -- 3-round Decision fixture tests -----------------------------------------


class TestParseFightDetail3Round:
    """Tests using the 3-round decision fight detail fixture."""

    def test_parse_3round_per_round(self, fight_detail_3round_html: str) -> None:
        """per_round_totals has length 3 for a 3-round fight."""
        page = parse_fight_detail(fight_detail_3round_html)
        assert len(page.per_round_totals) == 3
        assert len(page.per_round_sig_strikes) == 3

    def test_parse_3round_alignment(self, fight_detail_3round_html: str) -> None:
        """per_round_totals and per_round_sig_strikes have equal length."""
        page = parse_fight_detail(fight_detail_3round_html)
        assert len(page.per_round_totals) == len(page.per_round_sig_strikes)

    def test_parse_3round_per_round_different_data(self, fight_detail_3round_html: str) -> None:
        """Different rounds should have different stats (not all the same)."""
        page = parse_fight_detail(fight_detail_3round_html)
        # Round 1 sig_str = "28 of 48", Round 3 sig_str = "29 of 47" -- they differ
        assert page.per_round_totals[0][0].sig_str != page.per_round_totals[2][0].sig_str

    def test_parse_3round_method(self, fight_detail_3round_html: str) -> None:
        """Method is Decision."""
        page = parse_fight_detail(fight_detail_3round_html)
        assert page.method == "Decision"

    def test_parse_3round_round(self, fight_detail_3round_html: str) -> None:
        """Round finished is 3."""
        page = parse_fight_detail(fight_detail_3round_html)
        assert page.round_finished == 3

    def test_parse_3round_method_detail(self, fight_detail_3round_html: str) -> None:
        """Method detail contains Unanimous."""
        page = parse_fight_detail(fight_detail_3round_html)
        assert page.method_detail is not None
        assert "Unanimous" in page.method_detail

    def test_parse_3round_per_round_fighter_names(self, fight_detail_3round_html: str) -> None:
        """Each per_round_totals entry has non-empty fighter names."""
        page = parse_fight_detail(fight_detail_3round_html)
        for round_pair in page.per_round_totals:
            assert round_pair[0].fighter_name != ""
            assert round_pair[1].fighter_name != ""

    def test_parse_3round_sig_strikes_derived(self, fight_detail_3round_html: str) -> None:
        """Fight-level sig_strikes should be derived from per-round data."""
        page = parse_fight_detail(fight_detail_3round_html)
        # sig_strikes should reflect fight totals, not just Round 1
        assert isinstance(page.sig_strikes[0], SigStrikesRaw)
        assert isinstance(page.sig_strikes[1], SigStrikesRaw)
        # The fight-level sig_str should be the sum of all rounds
        # Round 1: 28of48, Round 2: 30of50, Round 3: 29of47 -> total: 87of145
        assert page.sig_strikes[0].sig_str == "87 of 145"


# -- Draw fixture tests ------------------------------------------------------


class TestParseFightDetailDraw:
    """Tests using the draw fight detail fixture."""

    def test_parse_draw_status(self, fight_detail_draw_html: str) -> None:
        """Both fighters have status D."""
        page = parse_fight_detail(fight_detail_draw_html)
        assert page.fighter_a_status == "D"
        assert page.fighter_b_status == "D"

    def test_parse_draw_method(self, fight_detail_draw_html: str) -> None:
        """Method is Decision."""
        page = parse_fight_detail(fight_detail_draw_html)
        assert page.method == "Decision"

    def test_parse_draw_method_detail(self, fight_detail_draw_html: str) -> None:
        """Method detail contains Majority Draw."""
        page = parse_fight_detail(fight_detail_draw_html)
        assert page.method_detail is not None
        assert "Majority Draw" in page.method_detail

    def test_parse_draw_alignment(self, fight_detail_draw_html: str) -> None:
        """per_round_totals and per_round_sig_strikes have equal length."""
        page = parse_fight_detail(fight_detail_draw_html)
        assert len(page.per_round_totals) == len(page.per_round_sig_strikes)

    def test_parse_draw_per_round_count(self, fight_detail_draw_html: str) -> None:
        """Draw fight (3 rounds) has 3 per-round entries."""
        page = parse_fight_detail(fight_detail_draw_html)
        assert len(page.per_round_totals) == 3
        assert len(page.per_round_sig_strikes) == 3
