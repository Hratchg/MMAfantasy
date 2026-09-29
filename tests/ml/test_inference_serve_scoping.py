"""Serve-side DB helpers in ``inference_features`` must scope like training.

Training (``ml.queries.load_fight_records``) only ever sees ufcstats fights
that have a winner. The v2.2 serve helpers used to read every source (so
kaggle duplicates counted ~1.95x) and NC / scheduled rows, and averaged
division reach per fight appearance instead of per fighter. These tests seed a
disposable Postgres with a ufcstats corpus plus a kaggle duplicate and a No
Contest, and check each helper against the training definition.

Also covers the scheduled-bout lookup and the division-median helper, which
need real SQL.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ufc_prediction.ml import inference_features as inf
from ufc_prediction.ml.feature_matrix import (
    _build_division_mean_reaches,
    compute_division_medians,
)
from ufc_prediction.ml.features_v22.ref import classify_outcome
from ufc_prediction.models.elo_snapshot import EloSnapshot
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter
from ufc_prediction.models.referee import Referee
from ufc_prediction.models.venue import Venue

WC = "Welterweight"


def _venue(session: Session, name: str, lat: float, lon: float, tz: str) -> Venue:
    # Explicit id: other suites commit venues with explicit ids into the shared
    # test container, leaving the pk sequence behind the table.
    next_id = (session.scalar(select(func.max(Venue.id))) or 0) + 1
    v = Venue(id=next_id, name=name, country="USA", lat=lat, lon=lon, timezone_iana=tz)
    session.add(v)
    session.flush()
    return v


def _event(session, day, source="ufcstats", venue=None, referee=None) -> Event:
    ev = Event(
        name=f"{source} {day}",
        date=day,
        source=source,
        venue_id=venue.id if venue else None,
        referee_id=referee.id if referee else None,
    )
    session.add(ev)
    session.flush()
    return ev


def _fight(session, ev, a, b, winner, *, wc=WC, method="Decision", source=None, **kw) -> Fight:
    f = Fight(
        event_id=ev.id,
        fighter_a_id=a.id,
        fighter_b_id=b.id,
        winner_id=winner.id if winner is not None else None,
        weight_class=wc,
        method=method,
        source=source or ev.source,
        **kw,
    )
    session.add(f)
    session.flush()
    return f


def _snap(session, fighter, fight, day, elo):
    for et in ("overall", "striking", "grappling"):
        session.add(
            EloSnapshot(
                fighter_id=fighter.id,
                fight_id=fight.id,
                division=fight.weight_class,
                elo_type=et,
                elo_before=elo,
                elo_after=elo,
                elo_after_shrinkage=elo,
                k_factor_used=32.0,
                fight_date=day,
            )
        )
    session.flush()


@pytest.fixture
def corpus(session: Session):
    """Fighter ``a`` (reach 80) has three winner fights, then a No Contest,
    then a kaggle duplicate of an older fight re-dated later. ``b`` (reach 70)
    has one fight. ``c`` (reach None) pads the division."""
    a = Fighter(name="Serve A", source="ufcstats", reach_inches=80.0, height_inches=74.0)
    b = Fighter(name="Serve B", source="ufcstats", reach_inches=70.0, height_inches=70.0)
    c = Fighter(name="Serve C", source="ufcstats", reach_inches=None, height_inches=72.0)
    session.add_all([a, b, c])
    session.flush()
    ref = Referee(name="Ref One", normalized_name="ref one")
    session.add(ref)
    session.flush()
    v_old = _venue(session, "Old Arena", 36.1, -115.2, "America/Los_Angeles")
    v_nc = _venue(session, "NC Arena", 40.7, -74.0, "America/New_York")
    v_kag = _venue(session, "Kaggle Arena", 51.5, -0.1, "Europe/London")

    e1 = _event(session, date(2022, 1, 8), venue=v_old, referee=ref)
    e2 = _event(session, date(2022, 6, 4), venue=v_old, referee=ref)
    e3 = _event(session, date(2023, 3, 4), venue=v_old, referee=ref)
    e_nc = _event(session, date(2024, 2, 10), venue=v_nc, referee=ref)
    e_kag = _event(session, date(2024, 5, 11), source="kaggle-mdabbert", venue=v_kag, referee=ref)

    f1 = _fight(session, e1, a, c, a, method="KO/TKO")
    f2 = _fight(session, e2, a, c, a, method="Decision")
    f3 = _fight(session, e3, a, b, a, method="Submission")
    f_nc = _fight(session, e_nc, a, c, None, method="Overturned")
    f_kag = _fight(session, e_kag, a, b, a, method="KO/TKO")

    _snap(session, a, f1, e1.date, 1500.0)
    _snap(session, a, f2, e2.date, 1520.0)
    _snap(session, a, f3, e3.date, 1540.0)
    _snap(session, a, f_nc, e_nc.date, 1999.0)
    _snap(session, a, f_kag, e_kag.date, 1888.0)
    return {
        "a": a,
        "b": b,
        "c": c,
        "ref": ref,
        "v_old": v_old,
        "training_fights": [f1, f2, f3],
        "training_events": [e1, e2, e3],
    }


AS_OF = date(2025, 1, 1)


class TestV22ServeScoping:
    def test_prior_fight_date_skips_nc_and_other_sources(self, session, corpus):
        assert inf._query_fighter_prior_fight_date(session, corpus["a"].id, AS_OF) == date(
            2023, 3, 4
        )

    def test_prior_venue_skips_nc_and_other_sources(self, session, corpus):
        prior = inf._query_fighter_prior_venue(session, corpus["a"].id, AS_OF)
        assert prior is not None
        assert (prior["lat"], prior["lon"]) == (corpus["v_old"].lat, corpus["v_old"].lon)
        assert prior["event_date"] == date(2023, 3, 4)

    def test_elo_history_skips_nc_and_other_sources(self, session, corpus):
        hist = inf._query_elo_history(session, corpus["a"].id, AS_OF, limit=6)
        assert [h["elo_overall"] for h in hist] == [1500.0, 1520.0, 1540.0]

    def test_division_state_counts_training_fights_only(self, session, corpus):
        div_hist, global_rate = inf._query_division_state(session, WC, AS_OF)
        assert len(div_hist[WC]) == 3
        finishes = sum(classify_outcome(f.method) == "finish" for f in corpus["training_fights"])
        assert global_rate == pytest.approx(finishes / 3)

    def test_ref_state_counts_training_fights_only(self, session, corpus):
        ref_id = corpus["ref"].id
        history, global_rates = inf._query_ref_state(session, ref_id, AS_OF)
        assert len(history[ref_id]) == 3
        assert global_rates["finish"] == pytest.approx(2 / 3)
        assert global_rates["decision"] == pytest.approx(1 / 3)
        assert global_rates["no_action"] == 0.0

    def test_division_mean_reach_is_per_unique_fighter(self, session, corpus):
        physicals = {
            f.id: {"reach_inches": f.reach_inches} for f in (corpus["a"], corpus["b"], corpus["c"])
        }
        records = [
            {"weight_class": WC, "fighter_a_id": f.fighter_a_id, "fighter_b_id": f.fighter_b_id}
            for f in corpus["training_fights"]
        ]
        expected = _build_division_mean_reaches(physicals, records)[WC]
        assert expected == pytest.approx(75.0)
        assert inf._query_division_mean_reach(session, WC) == pytest.approx(expected)

    def test_fighter_division_skips_catch_weight(self, session, corpus):
        ev = _event(session, date(2024, 8, 3))
        _fight(session, ev, corpus["a"], corpus["b"], corpus["a"], wc="Catch Weight")
        assert inf._query_fighter_division(session, corpus["a"].id) == WC


class TestScheduledBout:
    def test_reads_stored_fight_row_either_orientation(self, session, corpus):
        day = date(2026, 10, 3)
        ev = _event(session, day)
        _fight(
            session,
            ev,
            corpus["a"],
            corpus["b"],
            None,
            method=None,
            wc="Middleweight",
            num_rounds=5,
            is_title_fight=True,
        )
        expected = ("Middleweight", 5, True)
        assert inf._query_scheduled_bout(session, corpus["a"].id, corpus["b"].id, day) == expected
        assert inf._query_scheduled_bout(session, corpus["b"].id, corpus["a"].id, day) == expected

    def test_prefers_ufcstats_row_over_kaggle_duplicate(self, session, corpus):
        day = date(2026, 11, 7)
        kag = _event(session, day, source="kaggle-rajeevw")
        _fight(session, kag, corpus["a"], corpus["b"], None, method=None, wc="Catch Weight")
        ufc = _event(session, day)
        _fight(session, ufc, corpus["a"], corpus["b"], None, method=None, num_rounds=5)
        assert inf._query_scheduled_bout(session, corpus["a"].id, corpus["b"].id, day) == (
            WC,
            5,
            False,
        )

    def test_none_without_a_stored_row(self, session, corpus):
        assert (
            inf._query_scheduled_bout(session, corpus["a"].id, corpus["b"].id, date(2027, 1, 1))
            is None
        )


class TestDivisionPhysicalMedians:
    def test_matches_training_medians(self, session, corpus):
        cutoff = date(2023, 1, 1)
        fighters = (corpus["a"], corpus["b"], corpus["c"])
        physicals = {
            f.id: {
                "height_inches": f.height_inches,
                "reach_inches": f.reach_inches,
                "leg_reach_inches": f.leg_reach_inches,
            }
            for f in fighters
        }
        records = [
            {
                "weight_class": WC,
                "event_date": e.date,
                "fighter_a_id": f.fighter_a_id,
                "fighter_b_id": f.fighter_b_id,
            }
            for f, e in zip(corpus["training_fights"], corpus["training_events"], strict=True)
        ]
        expected = compute_division_medians(physicals, records, cutoff)[WC]
        # Only a and c fought before the cutoff: reach median is a's 80 (c has
        # none), height median averages a and c. b (2023) is excluded.
        assert expected == {"height_inches": 73.0, "reach_inches": 80.0, "leg_reach_inches": 0.0}
        assert inf._query_division_physical_medians(session, WC, cutoff) == expected

    def test_unknown_division_is_empty(self, session, corpus):
        assert inf._query_division_physical_medians(session, "Flyweight", AS_OF) == {}
