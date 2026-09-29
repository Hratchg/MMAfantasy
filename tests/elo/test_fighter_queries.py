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

        rankings = get_division_rankings(session, "Lightweight", limit=5, as_of=date(2019, 9, 8))
        assert len(rankings) >= 1
        # Khabib should be top with 1555 Elo
        assert rankings[0]["name"] == "Khabib Nurmagomedov"
        assert rankings[0]["elo"] == pytest.approx(1555.0)

    def test_get_division_rankings_uses_latest_snapshot(self, session, seed_fighters):
        from ufc_prediction.elo.fighter_queries import get_division_rankings

        rankings = get_division_rankings(session, "Lightweight", limit=10, as_of=date(2019, 9, 8))
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


# ── S09 regression tests (API fighter search / rankings review) ────────────


def _snap(fighter, fight, division, elo, fight_date, elo_type="overall"):
    return EloSnapshot(
        fighter_id=fighter.id,
        fight_id=fight.id,
        division=division,
        elo_type=elo_type,
        elo_before=1500.0,
        elo_after=elo,
        elo_after_shrinkage=elo,
        k_factor_used=20.0,
        fight_date=fight_date,
    )


@pytest.fixture()
def cross_source_fighters(session):
    """Same real-world fighters ingested under ufcstats AND a Kaggle source."""
    islam_ufc = Fighter(name="Islam Makhachev", source="ufcstats")
    islam_kag = Fighter(name="Islam Makhachev", source="kaggle-rajeevw")
    jones_ufc = Fighter(name="Jon Jones", source="ufcstats")
    jones_kag = Fighter(name="Jon Jones", source="kaggle-rajeevw")
    jones_jr = Fighter(name="Jon Jones Jr", source="ufcstats")
    session.add_all([islam_kag, islam_ufc, jones_kag, jones_ufc, jones_jr])
    session.flush()
    return {
        "islam_ufc": islam_ufc,
        "islam_kag": islam_kag,
        "jones_ufc": jones_ufc,
        "jones_kag": jones_kag,
        "jones_jr": jones_jr,
    }


class TestSearchFightersEscaping:
    """Finding 3: LIKE metacharacters in the user's input are literals."""

    def test_percent_is_literal(self, session, seed_fighters):
        from ufc_prediction.elo.fighter_queries import search_fighters

        assert search_fighters(session, "%") == []

    def test_underscore_is_literal(self, session, seed_fighters):
        from ufc_prediction.elo.fighter_queries import search_fighters

        assert search_fighters(session, "_") == []
        assert search_fighters(session, "Khab_b") == []

    def test_limit_caps_rows(self, session, seed_fighters):
        from ufc_prediction.elo.fighter_queries import search_fighters

        assert len(search_fighters(session, "a")) == 3
        assert len(search_fighters(session, "a", limit=2)) == 2


class TestResolveFighterCandidates:
    """Finding 1: cross-source duplicates collapse; exact name wins."""

    def test_cross_source_duplicate_collapses_to_ufcstats_row(self, session, cross_source_fighters):
        from ufc_prediction.elo.fighter_queries import resolve_fighter_candidates

        result = resolve_fighter_candidates(session, "Islam Makhachev")
        assert [f.id for f in result] == [cross_source_fighters["islam_ufc"].id]

    def test_substring_query_collapses_duplicates(self, session, cross_source_fighters):
        from ufc_prediction.elo.fighter_queries import resolve_fighter_candidates

        result = resolve_fighter_candidates(session, "makhachev")
        assert [f.id for f in result] == [cross_source_fighters["islam_ufc"].id]

    def test_exact_name_beats_substring_hits(self, session, cross_source_fighters):
        from ufc_prediction.elo.fighter_queries import resolve_fighter_candidates

        result = resolve_fighter_candidates(session, "jon jones")
        assert [f.id for f in result] == [cross_source_fighters["jones_ufc"].id]

    def test_distinct_people_stay_candidates(self, session, cross_source_fighters):
        from ufc_prediction.elo.fighter_queries import resolve_fighter_candidates

        result = resolve_fighter_candidates(session, "Jones")
        ids = {f.id for f in result}
        assert ids == {
            cross_source_fighters["jones_ufc"].id,
            cross_source_fighters["jones_jr"].id,
        }

    def test_wildcard_input_matches_nothing(self, session, cross_source_fighters):
        from ufc_prediction.elo.fighter_queries import resolve_fighter_candidates

        assert resolve_fighter_candidates(session, "%") == []
        assert resolve_fighter_candidates(session, "   ") == []

    def test_candidate_count_is_capped(self, session):
        from ufc_prediction.elo.fighter_queries import resolve_fighter_candidates

        session.add_all(
            [Fighter(name=f"Capped Fighter {i:02d}", source="ufcstats") for i in range(30)]
        )
        session.flush()
        result = resolve_fighter_candidates(session, "Capped", limit=25)
        assert len(result) == 25


class TestLatestOverallEloBulk:
    """Finding 3: candidate Elo is batch-fetched, not one query set per row."""

    def test_returns_latest_overall_snapshot_per_fighter(self, session, seed_fighters):
        from ufc_prediction.elo.fighter_queries import get_latest_overall_elo_bulk

        khabib = seed_fighters["khabib"]
        anderson = seed_fighters["anderson"]
        antonio = seed_fighters["antonio"]
        result = get_latest_overall_elo_bulk(session, [khabib.id, anderson.id, antonio.id])
        assert result[khabib.id] == (pytest.approx(1555.0), "Lightweight")
        assert result[anderson.id] == (pytest.approx(1515.0), "Middleweight")
        assert antonio.id not in result

    def test_empty_input(self, session):
        from ufc_prediction.elo.fighter_queries import get_latest_overall_elo_bulk

        assert get_latest_overall_elo_bulk(session, []) == {}


class TestResolveWeightClassExact:
    """Finding 2: an exact division name is never ambiguous."""

    @pytest.mark.parametrize(
        "name",
        ["Heavyweight", "Flyweight", "Bantamweight", "Featherweight"],
    )
    def test_mens_division_exact(self, name):
        from ufc_prediction.elo.fighter_queries import resolve_weight_class

        assert resolve_weight_class(name) == [name]
        assert resolve_weight_class(name.lower()) == [name]
        assert resolve_weight_class(f"  {name.upper()} ") == [name]

    def test_womens_division_exact(self):
        from ufc_prediction.elo.fighter_queries import resolve_weight_class

        assert resolve_weight_class("women's flyweight") == ["Women's Flyweight"]

    def test_partial_still_ambiguous(self):
        from ufc_prediction.elo.fighter_queries import resolve_weight_class

        assert set(resolve_weight_class("heavy")) == {"Heavyweight", "Light Heavyweight"}


def _ufc_fight(session, event, fighter, opponent, weight_class):
    fight = Fight(
        event_id=event.id,
        fighter_a_id=fighter.id,
        fighter_b_id=opponent.id,
        winner_id=fighter.id,
        weight_class=weight_class,
        source="ufcstats",
    )
    session.add(fight)
    session.flush()
    return fight


class TestDivisionRankingsInactivity:
    """Finding 4: rankings apply the engine's inactivity regression as of today."""

    def test_stale_fighter_regressed_below_active_fighter(self, session):
        from ufc_prediction.elo.fighter_queries import get_division_rankings

        stale = Fighter(name="Stale Veteran", source="ufcstats")
        active = Fighter(name="Active Contender", source="ufcstats")
        opp = Fighter(name="Opponent", source="ufcstats")
        session.add_all([stale, active, opp])
        session.flush()
        ev_old = Event(name="Old", date=date(2015, 1, 1), source="ufcstats")
        ev_new = Event(name="New", date=date(2024, 6, 1), source="ufcstats")
        session.add_all([ev_old, ev_new])
        session.flush()
        f_old = _ufc_fight(session, ev_old, stale, opp, "Lightweight")
        f_new = _ufc_fight(session, ev_new, active, opp, "Lightweight")
        session.add_all(
            [
                _snap(stale, f_old, "Lightweight", 1680.0, date(2015, 1, 1)),
                _snap(active, f_new, "Lightweight", 1620.0, date(2024, 6, 1)),
            ]
        )
        session.flush()

        rankings = get_division_rankings(session, "Lightweight", limit=5, as_of=date(2024, 12, 1))
        assert [r["name"] for r in rankings] == ["Active Contender", "Stale Veteran"]
        # Active: 183 days idle < 270-day threshold -> unchanged.
        assert rankings[0]["elo"] == pytest.approx(1620.0)
        # Stale: capped 50% regression toward 1500.
        assert rankings[1]["elo"] == pytest.approx(1590.0)
        assert rankings[1]["last_date"] == date(2015, 1, 1)

    def test_regression_mirrors_engine_across_divisions(self, session):
        """Mirrors the engine: every gap > 270 days regresses all of the
        fighter's division ratings, and inactivity to as_of is measured from
        the last fight in ANY division."""
        from ufc_prediction.elo.fighter_queries import get_division_rankings

        mover = Fighter(name="Division Mover", source="ufcstats")
        opp = Fighter(name="Opp", source="ufcstats")
        session.add_all([mover, opp])
        session.flush()
        ev1 = Event(name="E1", date=date(2023, 1, 1), source="ufcstats")
        ev2 = Event(name="E2", date=date(2024, 6, 1), source="ufcstats")
        session.add_all([ev1, ev2])
        session.flush()
        f1 = _ufc_fight(session, ev1, mover, opp, "Lightweight")
        f2 = _ufc_fight(session, ev2, mover, opp, "Welterweight")
        session.add_all(
            [
                _snap(mover, f1, "Lightweight", 1600.0, date(2023, 1, 1)),
                _snap(mover, f2, "Welterweight", 1560.0, date(2024, 6, 1)),
            ]
        )
        session.flush()

        rankings = get_division_rankings(session, "Lightweight", limit=5, as_of=date(2024, 12, 1))
        # 2023-01-01 -> 2024-06-01 is 517 days (247 past the threshold ->
        # 82% capped at 50%) -> 1550. 2024-06-01 -> as_of is 183 days: no
        # further regression.
        assert rankings[0]["elo"] == pytest.approx(1550.0)


class TestDivisionRankingsGhostRows:
    """Finding 5: a Kaggle twin inside the over-fetch window must not be
    returned when its ufcstats twin sits below the window."""

    def test_ufcstats_twin_wins_even_far_below_window(self, session):
        from ufc_prediction.elo.fighter_queries import get_division_rankings

        d = date(2024, 6, 1)
        ev = Event(name="Card", date=d, source="ufcstats")
        session.add(ev)
        session.flush()
        ghost = Fighter(name="Kenny Florian", source="kaggle-rajeevw")
        real = Fighter(name="Kenny Florian", source="ufcstats")
        fillers = [Fighter(name=f"Filler {i}", source="ufcstats") for i in range(4)]
        opp = Fighter(name="Opp", source="ufcstats")
        session.add_all([ghost, real, opp, *fillers])
        session.flush()

        session.add(
            _snap(
                ghost, _ufc_fight(session, ev, ghost, opp, "Lightweight"), "Lightweight", 1700.0, d
            )
        )
        for i, filler in enumerate(fillers):
            fight = _ufc_fight(session, ev, filler, opp, "Lightweight")
            session.add(_snap(filler, fight, "Lightweight", 1650.0 - i, d))
        session.add(
            _snap(real, _ufc_fight(session, ev, real, opp, "Lightweight"), "Lightweight", 1550.0, d)
        )
        session.flush()

        # limit=2 -> the old code fetched 4 rows: ghost + 3 fillers. The
        # ufcstats twin (6th) was outside the window, so the ghost survived.
        rankings = get_division_rankings(session, "Lightweight", limit=2, as_of=d)
        assert all(r["fighter_id"] != ghost.id for r in rankings)

        full = get_division_rankings(session, "Lightweight", limit=10, as_of=d)
        florian = [r for r in full if r["name"] == "Kenny Florian"]
        assert len(florian) == 1
        assert florian[0]["fighter_id"] == real.id
        assert florian[0]["elo"] == pytest.approx(1550.0)
