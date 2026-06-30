"""Tests for fighter lookup query functions.

Tests cover search_fighters, get_fighter_detail, get_division_rankings,
and resolve_weight_class from the fighter_queries module.
"""

from __future__ import annotations

from datetime import date

import pytest

from ufc_prediction.models.elo_snapshot import EloSnapshot
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter, FighterAlias


@pytest.fixture()
def seed_fighters(session):
    """Seed the database with test fighters, events, fights, and Elo snapshots."""
    # Create fighters
    khabib = Fighter(name="Khabib Nurmagomedov", source="test")
    anderson = Fighter(name="Anderson Silva", source="test")
    antonio = Fighter(name="Antonio Silva", source="test")
    session.add_all([khabib, anderson, antonio])
    session.flush()

    # Create alias for Khabib
    alias = FighterAlias(fighter_id=khabib.id, alias_name="The Eagle", source="test")
    session.add(alias)
    session.flush()

    # Create events
    event1 = Event(name="UFC 229", date=date(2018, 10, 6), source="test")
    event2 = Event(name="UFC 242", date=date(2019, 9, 7), source="test")
    event3 = Event(name="UFC 168", date=date(2013, 12, 28), source="test")
    session.add_all([event1, event2, event3])
    session.flush()

    # Create fights for Khabib (2 in Lightweight, both wins)
    fight1 = Fight(
        event_id=event1.id,
        fighter_a_id=khabib.id,
        fighter_b_id=antonio.id,
        winner_id=khabib.id,
        weight_class="Lightweight",
        source="test",
    )
    fight2 = Fight(
        event_id=event2.id,
        fighter_a_id=khabib.id,
        fighter_b_id=antonio.id,
        winner_id=khabib.id,
        weight_class="Lightweight",
        source="test",
    )
    # Create fight for Anderson Silva (1 in Middleweight, win)
    fight3 = Fight(
        event_id=event3.id,
        fighter_a_id=anderson.id,
        fighter_b_id=antonio.id,
        winner_id=anderson.id,
        weight_class="Middleweight",
        source="test",
    )
    # Create a draw fight for Anderson Silva (no winner)
    fight4 = Fight(
        event_id=event1.id,
        fighter_a_id=anderson.id,
        fighter_b_id=khabib.id,
        winner_id=None,
        weight_class="Middleweight",
        method="Draw",
        source="test",
    )
    session.add_all([fight1, fight2, fight3, fight4])
    session.flush()

    # Create Elo snapshots
    # Khabib: 2 snapshots in Lightweight (different dates)
    snap1 = EloSnapshot(
        fighter_id=khabib.id,
        fight_id=fight1.id,
        division="Lightweight",
        elo_type="overall",
        elo_before=1500.0,
        elo_after=1530.0,
        elo_after_shrinkage=1525.0,
        k_factor_used=40.0,
        fight_date=date(2018, 10, 6),
    )
    snap2 = EloSnapshot(
        fighter_id=khabib.id,
        fight_id=fight2.id,
        division="Lightweight",
        elo_type="overall",
        elo_before=1530.0,
        elo_after=1560.0,
        elo_after_shrinkage=1555.0,
        k_factor_used=40.0,
        fight_date=date(2019, 9, 7),
    )
    # Anderson Silva: 1 snapshot in Middleweight
    snap3 = EloSnapshot(
        fighter_id=anderson.id,
        fight_id=fight3.id,
        division="Middleweight",
        elo_type="overall",
        elo_before=1500.0,
        elo_after=1520.0,
        elo_after_shrinkage=1515.0,
        k_factor_used=40.0,
        fight_date=date(2013, 12, 28),
    )
    session.add_all([snap1, snap2, snap3])
    session.flush()

    return {
        "khabib": khabib,
        "anderson": anderson,
        "antonio": antonio,
        "alias": alias,
        "event1": event1,
        "event2": event2,
        "event3": event3,
        "fight1": fight1,
        "fight2": fight2,
        "fight3": fight3,
        "fight4": fight4,
        "snap1": snap1,
        "snap2": snap2,
        "snap3": snap3,
    }


# ── search_fighters tests ──────────────────────────────────────────────────


class TestSearchFighters:
    def test_search_fighters_single_match(self, session, seed_fighters):
        from ufc_prediction.elo.fighter_queries import search_fighters

        results = search_fighters(session, "Khabib")
        assert len(results) == 1
        assert "Khabib" in results[0].name

    def test_search_fighters_by_alias(self, session, seed_fighters):
        from ufc_prediction.elo.fighter_queries import search_fighters

        results = search_fighters(session, "The Eagle")
        assert len(results) == 1
        assert results[0].name == "Khabib Nurmagomedov"

    def test_search_fighters_multiple_matches(self, session, seed_fighters):
        from ufc_prediction.elo.fighter_queries import search_fighters

        results = search_fighters(session, "Silva")
        assert len(results) == 2

    def test_search_fighters_no_match(self, session, seed_fighters):
        from ufc_prediction.elo.fighter_queries import search_fighters

        results = search_fighters(session, "ZZZZNOTANAME")
        assert results == []

    def test_search_fighters_case_insensitive(self, session, seed_fighters):
        from ufc_prediction.elo.fighter_queries import search_fighters

        results = search_fighters(session, "khabib")
        assert len(results) == 1
        assert results[0].name == "Khabib Nurmagomedov"


# ── get_fighter_detail tests ────────────────────────────────────────────────


class TestGetFighterDetail:
    def test_get_fighter_detail_returns_elo_and_record(self, session, seed_fighters):
        from ufc_prediction.elo.fighter_queries import get_fighter_detail

        khabib = seed_fighters["khabib"]
        detail = get_fighter_detail(session, khabib.id, "Lightweight")
        assert detail["elo"] == pytest.approx(1555.0)
        assert detail["division"] == "Lightweight"
        assert detail["wins"] == 2
        assert detail["losses"] == 0
        assert detail["total_fights"] == 2
        assert detail["last_fight_date"] == date(2019, 9, 7)

    def test_get_fighter_detail_draw_not_counted_as_loss(self, session, seed_fighters):
        from ufc_prediction.elo.fighter_queries import get_fighter_detail

        # Anderson has 1 win + 1 draw in Middleweight — draw should NOT be a loss
        anderson = seed_fighters["anderson"]
        detail = get_fighter_detail(session, anderson.id, "Middleweight")
        assert detail["wins"] == 1
        assert detail["losses"] == 0
        assert detail["total_fights"] == 2  # win + draw both count as fights

    def test_get_fighter_detail_no_snapshots(self, session, seed_fighters):
        from ufc_prediction.elo.fighter_queries import get_fighter_detail

        antonio = seed_fighters["antonio"]
        detail = get_fighter_detail(session, antonio.id, "Lightweight")
        assert detail["elo"] is None


# ── get_division_rankings tests ─────────────────────────────────────────────


class TestGetDivisionRankings:
    def test_get_division_rankings_ordered_by_elo(self, session, seed_fighters):
        from ufc_prediction.elo.fighter_queries import get_division_rankings

        rankings = get_division_rankings(session, "Lightweight", limit=5)
        assert len(rankings) >= 1
        # Khabib should be top with 1555 Elo
        assert rankings[0]["name"] == "Khabib Nurmagomedov"
        assert rankings[0]["elo"] == pytest.approx(1555.0)

    def test_get_division_rankings_uses_latest_snapshot(self, session, seed_fighters):
        from ufc_prediction.elo.fighter_queries import get_division_rankings

        rankings = get_division_rankings(session, "Lightweight", limit=10)
        # Khabib has 2 snapshots but should only appear once
        khabib_entries = [r for r in rankings if r["name"] == "Khabib Nurmagomedov"]
        assert len(khabib_entries) == 1
        # Should use the latest Elo (1555, not 1525)
        assert khabib_entries[0]["elo"] == pytest.approx(1555.0)

    def test_get_division_rankings_limit(self, session, seed_fighters):
        from ufc_prediction.elo.fighter_queries import get_division_rankings

        rankings = get_division_rankings(session, "Lightweight", limit=2)
        assert len(rankings) <= 2


# ── resolve_weight_class tests ──────────────────────────────────────────────


class TestResolveWeightClass:
    def test_resolve_weight_class_exact(self):
        from ufc_prediction.elo.fighter_queries import resolve_weight_class

        result = resolve_weight_class("Lightweight")
        assert result == ["Lightweight"]

    def test_resolve_weight_class_partial(self):
        from ufc_prediction.elo.fighter_queries import resolve_weight_class

        result = resolve_weight_class("light")
        assert "Lightweight" in result
        assert "Light Heavyweight" in result

    def test_resolve_weight_class_no_match(self):
        from ufc_prediction.elo.fighter_queries import resolve_weight_class

        result = resolve_weight_class("ZZZZZ")
        assert result == []

    def test_resolve_weight_class_excludes_catch_open(self):
        from ufc_prediction.elo.fighter_queries import resolve_weight_class

        result = resolve_weight_class("weight")
        assert "Catch Weight" not in result
        assert "Open Weight" not in result
