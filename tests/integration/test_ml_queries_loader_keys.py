"""``ml.queries`` loaders against a real Postgres: v2.2 REF/TRAVEL keys and
the fighter/date scoping the serve-time career replay depends on."""

from __future__ import annotations

from datetime import date

import pytest

from ufc_prediction.ml import queries
from ufc_prediction.models.elo_snapshot import EloSnapshot
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter
from ufc_prediction.models.referee import Referee
from ufc_prediction.models.round_stats import RoundStats
from ufc_prediction.models.venue import Venue

pytestmark = pytest.mark.integration


@pytest.fixture
def corpus(session):
    fighters = [Fighter(name=f"Loader Fighter {i}", source="ufcstats") for i in range(4)]
    session.add_all(fighters)
    venue = Venue(
        name="Tokyo, Japan",
        city="Tokyo",
        country="Japan",
        lat=35.6762,
        lon=139.6503,
        timezone_iana="Asia/Tokyo",
    )
    ref = Referee(name="Herb Dean", normalized_name="herb dean")
    session.add_all([venue, ref])
    session.flush()
    f0, f1, f2, f3 = fighters
    ev_with = Event(
        name="UFC Loader 1",
        date=date(2023, 3, 4),
        source="ufcstats",
        venue_id=venue.id,
        referee_id=ref.id,
    )
    ev_without = Event(name="UFC Loader 2", date=date(2023, 9, 9), source="ufcstats")
    ev_kaggle = Event(name="Kaggle Loader", date=date(2023, 3, 4), source="kaggle-mdabbert")
    ev_future = Event(name="UFC Loader 3", date=date(2024, 6, 1), source="ufcstats")
    session.add_all([ev_with, ev_without, ev_kaggle, ev_future])
    session.flush()
    fights = [
        Fight(
            event_id=ev_with.id,
            fighter_a_id=f0.id,
            fighter_b_id=f1.id,
            winner_id=f0.id,
            weight_class="Lightweight",
            method="Decision",
            source="ufcstats",
        ),
        Fight(
            event_id=ev_without.id,
            fighter_a_id=f2.id,
            fighter_b_id=f0.id,
            winner_id=f2.id,
            weight_class="Lightweight",
            method="KO/TKO",
            source="ufcstats",
        ),
        Fight(
            event_id=ev_without.id,
            fighter_a_id=f1.id,
            fighter_b_id=f3.id,
            winner_id=None,
            weight_class="Lightweight",
            method="No Contest",
            source="ufcstats",
        ),
        Fight(
            event_id=ev_kaggle.id,
            fighter_a_id=f0.id,
            fighter_b_id=f1.id,
            winner_id=f0.id,
            weight_class="Lightweight",
            method="Decision",
            source="kaggle-mdabbert",
        ),
        Fight(
            event_id=ev_future.id,
            fighter_a_id=f2.id,
            fighter_b_id=f3.id,
            winner_id=f3.id,
            weight_class="Welterweight",
            method="Submission",
            source="ufcstats",
        ),
    ]
    session.add_all(fights)
    session.flush()
    first, second, _nc, _kaggle, future = fights
    session.add_all(
        [
            EloSnapshot(
                fighter_id=f0.id,
                fight_id=first.id,
                division="Lightweight",
                elo_type="overall",
                elo_before=1500.0,
                elo_after=1520.0,
                elo_after_shrinkage=1504.0,
                k_factor_used=40.0,
                fight_date=ev_with.date,
            ),
            EloSnapshot(
                fighter_id=f0.id,
                fight_id=second.id,
                division="Lightweight",
                elo_type="overall",
                elo_before=1520.0,
                elo_after=1505.0,
                elo_after_shrinkage=1502.0,
                k_factor_used=40.0,
                fight_date=ev_without.date,
            ),
            EloSnapshot(
                fighter_id=f2.id,
                fight_id=future.id,
                division="Welterweight",
                elo_type="overall",
                elo_before=1500.0,
                elo_after=1480.0,
                elo_after_shrinkage=1496.0,
                k_factor_used=40.0,
                fight_date=ev_future.date,
            ),
            RoundStats(
                fight_id=first.id,
                fighter_id=f0.id,
                round_number=1,
                sig_strikes_landed=10,
                takedowns_landed=1,
            ),
            RoundStats(
                fight_id=first.id,
                fighter_id=f0.id,
                round_number=2,
                sig_strikes_landed=12,
                takedowns_landed=0,
            ),
            RoundStats(
                fight_id=future.id,
                fighter_id=f2.id,
                round_number=1,
                sig_strikes_landed=3,
                takedowns_landed=0,
            ),
        ]
    )
    session.flush()
    return {
        "fighters": fighters,
        "fights": fights,
        "events": [ev_with, ev_without, ev_kaggle, ev_future],
        "venue": venue,
        "ref": ref,
    }


def test_fight_records_carry_referee_and_venue_keys(session, corpus):
    first, second, _nc, _kaggle, future = corpus["fights"]
    records = {r["fight_id"]: r for r in queries.load_fight_records(session)}
    mine = {fid: records[fid] for fid in (first.id, second.id, future.id) if fid in records}
    assert set(mine) == {first.id, second.id, future.id}, (
        "winner-less and kaggle fights must be excluded"
    )
    with_venue = mine[first.id]
    assert with_venue["event_id"] == corpus["events"][0].id
    assert with_venue["referee_id"] == corpus["ref"].id
    assert with_venue["venue_lat"] == pytest.approx(35.6762)
    assert with_venue["venue_lon"] == pytest.approx(139.6503)
    assert with_venue["venue_timezone_iana"] == "Asia/Tokyo"
    without = mine[second.id]
    assert without["event_id"] == corpus["events"][1].id
    assert without["referee_id"] is None
    assert without["venue_lat"] is None and without["venue_lon"] is None
    assert without["venue_timezone_iana"] is None


def test_fight_records_scoped_by_fighter_and_date(session, corpus):
    f0, f1, f2, f3 = corpus["fighters"]
    first, second, _nc, _kaggle, future = corpus["fights"]
    scoped = queries.load_fight_records(
        session, fighter_ids=(f0.id, f3.id), before_date=date(2024, 1, 1)
    )
    assert [r["fight_id"] for r in scoped] == [first.id, second.id]
    later = queries.load_fight_records(
        session, fighter_ids=(f2.id,), before_date=date(2024, 12, 31)
    )
    assert {r["fight_id"] for r in later} == {second.id, future.id}
    assert (
        queries.load_fight_records(session, fighter_ids=(f3.id,), before_date=date(2024, 1, 1))
        == []
    )


def test_scoped_elo_computed_and_round_loaders(session, corpus):
    f0, _f1, f2, _f3 = corpus["fighters"]
    first, second, _nc, _kaggle, future = corpus["fights"]
    elo = queries.load_elo_features(session, fight_ids=[first.id, second.id])
    assert set(elo) == {(f0.id, first.id), (f0.id, second.id)}
    assert elo[(f0.id, second.id)]["elo_overall"] == 1520.0  # elo_before, not shrunk after
    rounds = queries.load_round_stats_for_ml(session, fight_ids=[first.id])
    assert set(rounds) == {(f0.id, first.id)}
    assert [r["round_number"] for r in rounds[(f0.id, first.id)]] == [1, 2]
    assert queries.load_round_stats_for_ml(session, fight_ids=[]) == {}
    assert queries.load_computed_features(session, fight_ids=[first.id]) == {}
    assert queries.load_pre_ufc_records(session, fighter_ids=(f0.id, f2.id)) == {}
