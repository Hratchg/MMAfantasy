"""Tests for event listing and event detail parsers."""

from __future__ import annotations

import re

import pytest

from ufc_prediction.scraper.models import EventDetail, EventSummary
from ufc_prediction.scraper.parse_event_detail import parse_event_detail
from ufc_prediction.scraper.parse_event_list import parse_event_list

# ── Event Listing Tests ─────────────────────────────────────────────────────


class TestParseEventList:
    """Tests for parse_event_list."""

    def test_parse_event_list_returns_events(self, event_list_html: str) -> None:
        """parse_event_list returns a list of EventSummary."""
        events = parse_event_list(event_list_html, min_events=1)
        assert isinstance(events, list)
        assert all(isinstance(e, EventSummary) for e in events)

    def test_parse_event_list_identifies_upcoming(self, event_list_html: str) -> None:
        """Exactly 1 event has is_upcoming=True."""
        events = parse_event_list(event_list_html, min_events=1)
        upcoming = [e for e in events if e.is_upcoming]
        assert len(upcoming) == 1

    def test_parse_event_list_skips_spacer_rows(self, event_list_html: str) -> None:
        """Spacer rows with b-statistics__table-col_type_clear are skipped."""
        events = parse_event_list(event_list_html, min_events=1)
        # Fixture has 1 spacer + 4 real events
        assert len(events) == 4

    def test_parse_event_list_extracts_names(self, event_list_html: str) -> None:
        """Event names contain 'UFC'."""
        events = parse_event_list(event_list_html, min_events=1)
        for e in events:
            assert "UFC" in e.name

    def test_parse_event_list_extracts_dates(self, event_list_html: str) -> None:
        """date_str matches 'Month DD, YYYY' pattern."""
        events = parse_event_list(event_list_html, min_events=1)
        date_pattern = re.compile(r"[A-Z][a-z]+ \d{2}, \d{4}")
        for e in events:
            assert date_pattern.match(e.date_str), f"Bad date format: {e.date_str}"

    def test_parse_event_list_extracts_urls(self, event_list_html: str) -> None:
        """URLs contain 'event-details/'."""
        events = parse_event_list(event_list_html, min_events=1)
        for e in events:
            assert "event-details/" in e.url

    def test_parse_event_list_cardinality_passes(self, event_list_html: str) -> None:
        """Fixture parsed with min_events=1 succeeds."""
        events = parse_event_list(event_list_html, min_events=1)
        assert len(events) >= 1

    def test_parse_event_list_cardinality_fails_on_empty(self) -> None:
        """Empty HTML raises ValueError."""
        with pytest.raises(ValueError, match="Event listing table not found"):
            parse_event_list("<html></html>", min_events=1)

    def test_parse_event_list_extracts_locations(self, event_list_html: str) -> None:
        """Locations are non-empty strings."""
        events = parse_event_list(event_list_html, min_events=1)
        for e in events:
            assert len(e.location) > 0


# ── Event Detail Tests ───────────────────────────────────────────────────────


class TestParseEventDetail:
    """Tests for parse_event_detail."""

    def test_parse_event_detail_returns_event(self, event_detail_html: str) -> None:
        """parse_event_detail returns an EventDetail with correct name."""
        event = parse_event_detail(event_detail_html, "http://ufcstats.com/event-details/test")
        assert isinstance(event, EventDetail)
        assert "UFC 327" in event.name

    def test_parse_event_detail_has_fights(self, event_detail_html: str) -> None:
        """fights list contains 3 FightSummary items."""
        event = parse_event_detail(event_detail_html, "http://ufcstats.com/event-details/test")
        assert len(event.fights) == 3

    def test_parse_event_detail_fight_urls(self, event_detail_html: str) -> None:
        """Each fight has fight_url containing 'fight-details/'."""
        event = parse_event_detail(event_detail_html, "http://ufcstats.com/event-details/test")
        for fight in event.fights:
            assert "fight-details/" in fight.fight_url

    def test_parse_event_detail_fighter_names(self, event_detail_html: str) -> None:
        """Each fight has non-empty fighter_a_name and fighter_b_name."""
        event = parse_event_detail(event_detail_html, "http://ufcstats.com/event-details/test")
        for fight in event.fights:
            assert len(fight.fighter_a_name) > 0
            assert len(fight.fighter_b_name) > 0

    def test_parse_event_detail_fighter_urls(self, event_detail_html: str) -> None:
        """Each fight has fighter URLs containing 'fighter-details/'."""
        event = parse_event_detail(event_detail_html, "http://ufcstats.com/event-details/test")
        for fight in event.fights:
            assert "fighter-details/" in fight.fighter_a_url
            assert "fighter-details/" in fight.fighter_b_url

    def test_parse_event_detail_win_outcome(self, event_detail_html: str) -> None:
        """At least one fight has outcome='win'."""
        event = parse_event_detail(event_detail_html, "http://ufcstats.com/event-details/test")
        outcomes = [f.outcome for f in event.fights]
        assert "win" in outcomes

    def test_parse_event_detail_title_fight(self, event_detail_html: str) -> None:
        """At least one fight has is_title_fight=True."""
        event = parse_event_detail(event_detail_html, "http://ufcstats.com/event-details/test")
        title_fights = [f for f in event.fights if f.is_title_fight]
        assert len(title_fights) >= 1

    def test_parse_event_detail_method(self, event_detail_html: str) -> None:
        """Fights have method like 'KO/TKO', 'Decision', or 'Submission'."""
        event = parse_event_detail(event_detail_html, "http://ufcstats.com/event-details/test")
        valid_methods = {"KO/TKO", "Decision - Unanimous", "Submission"}
        for fight in event.fights:
            assert fight.method in valid_methods, f"Unexpected method: {fight.method}"

    def test_parse_event_detail_cardinality_fails(self) -> None:
        """Empty HTML raises ValueError."""
        with pytest.raises(ValueError, match="No fight rows found"):
            parse_event_detail("<html></html>", "http://x")

    def test_parse_event_detail_date_and_location(self, event_detail_html: str) -> None:
        """Event has correct date and location."""
        event = parse_event_detail(event_detail_html, "http://ufcstats.com/event-details/test")
        assert "April 11, 2026" in event.date_str
        assert "Miami" in event.location

    def test_parse_event_detail_round_and_time(self, event_detail_html: str) -> None:
        """Fights have round_finished and time_finished populated."""
        event = parse_event_detail(event_detail_html, "http://ufcstats.com/event-details/test")
        for fight in event.fights:
            assert fight.round_finished is not None
            assert fight.time_finished is not None
