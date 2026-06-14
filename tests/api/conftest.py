"""Shared fixtures for API integration tests."""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from ufc_prediction.api.app import create_app
from ufc_prediction.api.deps import get_db
from ufc_prediction.models.computed_feature import ComputedFeature
from ufc_prediction.models.elo_snapshot import EloSnapshot
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter, FighterAlias
from ufc_prediction.models.round_stats import RoundStats


@pytest.fixture()
def seed_api_data(session):
    """Seed database with fighters, events, fights, and Elo snapshots for API tests.

    Creates data across Lightweight and Middleweight divisions for testing
    both lookup (single/multi match) and rankings (ordering, limiting).
    Same data as seed_cli_data in tests/cli/conftest.py.
    """
    # Fighters
    khabib = Fighter(
        name="Khabib Nurmagomedov",
        source="test",
        height_inches=70.0,
        reach_inches=70.0,
        leg_reach_inches=40.0,
        date_of_birth=date(1988, 9, 20),
    )
    anderson = Fighter(name="Anderson Silva", source="test")
    antonio = Fighter(name="Antonio Silva", source="test")
    conor = Fighter(
        name="Conor McGregor",
        source="test",
        height_inches=69.0,
        reach_inches=74.0,
        leg_reach_inches=40.0,
        date_of_birth=date(1988, 7, 14),
    )
    dustin = Fighter(name="Dustin Poirier", source="test")
    session.add_all([khabib, anderson, antonio, conor, dustin])
    session.flush()

    # Alias
    alias = FighterAlias(
        fighter_id=khabib.id, alias_name="The Eagle", source="test"
    )
    session.add(alias)
    session.flush()

    # Events
    event1 = Event(name="UFC 229", date=date(2018, 10, 6), source="test")
    event2 = Event(name="UFC 242", date=date(2019, 9, 7), source="test")
    event3 = Event(name="UFC 168", date=date(2013, 12, 28), source="test")
    event4 = Event(name="UFC 264", date=date(2021, 7, 10), source="test")
    session.add_all([event1, event2, event3, event4])
    session.flush()

    # Fights
    fight1 = Fight(
        event_id=event1.id,
        fighter_a_id=khabib.id,
        fighter_b_id=conor.id,
        winner_id=khabib.id,
        weight_class="Lightweight",
        source="test",
    )
    fight2 = Fight(
        event_id=event2.id,
        fighter_a_id=khabib.id,
        fighter_b_id=dustin.id,
        winner_id=khabib.id,
        weight_class="Lightweight",
        source="test",
    )
    fight3 = Fight(
        event_id=event3.id,
        fighter_a_id=anderson.id,
        fighter_b_id=antonio.id,
        winner_id=anderson.id,
        weight_class="Middleweight",
        source="test",
    )
    fight4 = Fight(
        event_id=event4.id,
        fighter_a_id=conor.id,
        fighter_b_id=dustin.id,
        winner_id=dustin.id,
        weight_class="Lightweight",
        source="test",
    )
    session.add_all([fight1, fight2, fight3, fight4])
    session.flush()

    # Elo snapshots -- Lightweight (overall)
    snap_khabib1 = EloSnapshot(
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
    snap_khabib2 = EloSnapshot(
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
    snap_conor1 = EloSnapshot(
        fighter_id=conor.id,
        fight_id=fight1.id,
        division="Lightweight",
        elo_type="overall",
        elo_before=1500.0,
        elo_after=1470.0,
        elo_after_shrinkage=1475.0,
        k_factor_used=40.0,
        fight_date=date(2018, 10, 6),
    )
    snap_conor2 = EloSnapshot(
        fighter_id=conor.id,
        fight_id=fight4.id,
        division="Lightweight",
        elo_type="overall",
        elo_before=1470.0,
        elo_after=1445.0,
        elo_after_shrinkage=1448.0,
        k_factor_used=40.0,
        fight_date=date(2021, 7, 10),
    )
    snap_dustin1 = EloSnapshot(
        fighter_id=dustin.id,
        fight_id=fight2.id,
        division="Lightweight",
        elo_type="overall",
        elo_before=1500.0,
        elo_after=1470.0,
        elo_after_shrinkage=1475.0,
        k_factor_used=40.0,
        fight_date=date(2019, 9, 7),
    )
    snap_dustin2 = EloSnapshot(
        fighter_id=dustin.id,
        fight_id=fight4.id,
        division="Lightweight",
        elo_type="overall",
        elo_before=1470.0,
        elo_after=1500.0,
        elo_after_shrinkage=1498.0,
        k_factor_used=40.0,
        fight_date=date(2021, 7, 10),
    )
    snap_anderson = EloSnapshot(
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

    session.add_all([
        snap_khabib1, snap_khabib2,
        snap_conor1, snap_conor2,
        snap_dustin1, snap_dustin2,
        snap_anderson,
    ])
    session.flush()

    # Domain Elo snapshots -- Khabib (striking + grappling)
    snap_khabib_str = EloSnapshot(
        fighter_id=khabib.id,
        fight_id=fight2.id,
        division="Lightweight",
        elo_type="striking",
        elo_before=1500.0,
        elo_after=1520.0,
        elo_after_shrinkage=1515.0,
        k_factor_used=40.0,
        fight_date=date(2019, 9, 7),
    )
    snap_khabib_grap = EloSnapshot(
        fighter_id=khabib.id,
        fight_id=fight2.id,
        division="Lightweight",
        elo_type="grappling",
        elo_before=1500.0,
        elo_after=1540.0,
        elo_after_shrinkage=1535.0,
        k_factor_used=40.0,
        fight_date=date(2019, 9, 7),
    )
    snap_conor_str = EloSnapshot(
        fighter_id=conor.id,
        fight_id=fight1.id,
        division="Lightweight",
        elo_type="striking",
        elo_before=1500.0,
        elo_after=1530.0,
        elo_after_shrinkage=1525.0,
        k_factor_used=40.0,
        fight_date=date(2018, 10, 6),
    )
    snap_conor_grap = EloSnapshot(
        fighter_id=conor.id,
        fight_id=fight1.id,
        division="Lightweight",
        elo_type="grappling",
        elo_before=1500.0,
        elo_after=1470.0,
        elo_after_shrinkage=1475.0,
        k_factor_used=40.0,
        fight_date=date(2018, 10, 6),
    )
    session.add_all([
        snap_khabib_str, snap_khabib_grap,
        snap_conor_str, snap_conor_grap,
    ])
    session.flush()

    # RoundStats for Khabib and Conor in fight1
    rs_khabib_f1_r1 = RoundStats(
        fight_id=fight1.id,
        fighter_id=khabib.id,
        round_number=1,
        sig_strikes_landed=30,
        sig_strikes_attempted=60,
    )
    rs_khabib_f1_r2 = RoundStats(
        fight_id=fight1.id,
        fighter_id=khabib.id,
        round_number=2,
        sig_strikes_landed=20,
        sig_strikes_attempted=40,
    )
    rs_conor_f1_r1 = RoundStats(
        fight_id=fight1.id,
        fighter_id=conor.id,
        round_number=1,
        sig_strikes_landed=25,
        sig_strikes_attempted=50,
    )
    rs_conor_f1_r2 = RoundStats(
        fight_id=fight1.id,
        fighter_id=conor.id,
        round_number=2,
        sig_strikes_landed=15,
        sig_strikes_attempted=30,
    )
    session.add_all([rs_khabib_f1_r1, rs_khabib_f1_r2, rs_conor_f1_r1, rs_conor_f1_r2])
    session.flush()

    # ComputedFeature for Khabib (grappler style)
    cf_khabib = ComputedFeature(
        fighter_id=khabib.id,
        fight_id=fight1.id,
        as_of_date=date(2018, 10, 6),
        feature_set_version="v1",
        features={
            "sig_str_per_minute": 4.1,
            "sig_str_per_minute_ewma": 4.2,
            "strike_defense": 0.55,
            "strike_defense_ewma": 0.56,
            "td_rate": 5.3,
            "td_rate_ewma": 5.4,
            "td_accuracy": 0.48,
            "td_accuracy_ewma": 0.49,
            "td_defense": 0.84,
            "td_defense_ewma": 0.85,
            "ctrl_time_per_fight": 320.0,
            "ctrl_time_per_fight_ewma": 330.0,
            "style_tag": "grappler",
        },
    )
    cf_conor = ComputedFeature(
        fighter_id=conor.id,
        fight_id=fight1.id,
        as_of_date=date(2018, 10, 6),
        feature_set_version="v1",
        features={
            "sig_str_per_minute": 5.5,
            "sig_str_per_minute_ewma": 5.6,
            "strike_defense": 0.51,
            "strike_defense_ewma": 0.52,
            "td_rate": 0.8,
            "td_rate_ewma": 0.9,
            "td_accuracy": 0.40,
            "td_accuracy_ewma": 0.41,
            "td_defense": 0.57,
            "td_defense_ewma": 0.58,
            "ctrl_time_per_fight": 45.0,
            "ctrl_time_per_fight_ewma": 50.0,
            "style_tag": "striker",
        },
    )
    session.add_all([cf_khabib, cf_conor])
    session.flush()

    return {
        "khabib": khabib,
        "anderson": anderson,
        "antonio": antonio,
        "conor": conor,
        "dustin": dustin,
        "fights": [fight1, fight2, fight3, fight4],
        "events": [event1, event2, event3, event4],
    }


@pytest.fixture()
def seed_matchup_api_data(session, seed_api_data):
    """Extended seed data with extra fights for style matchup win rate testing.

    Adds 10 striker-vs-grappler fights so at least one style matchup type
    passes the D-09 minimum sample size of 10 fights.
    """
    data = seed_api_data

    strikers = []
    grapplers = []
    events = []
    fights = []

    for i in range(10):
        striker = Fighter(name=f"Test Striker {i}", source="test")
        grappler = Fighter(name=f"Test Grappler {i}", source="test")
        strikers.append(striker)
        grapplers.append(grappler)
    session.add_all(strikers + grapplers)
    session.flush()

    for i in range(10):
        ev = Event(
            name=f"UFC Style Test {i}",
            date=date(2020, 1, 1 + i),
            source="test",
        )
        events.append(ev)
    session.add_all(events)
    session.flush()

    for i in range(10):
        winner = strikers[i] if i < 6 else grapplers[i]
        fight = Fight(
            event_id=events[i].id,
            fighter_a_id=strikers[i].id,
            fighter_b_id=grapplers[i].id,
            winner_id=winner.id,
            weight_class="Lightweight",
            source="test",
        )
        fights.append(fight)
    session.add_all(fights)
    session.flush()

    for i, fight in enumerate(fights):
        session.add(EloSnapshot(
            fighter_id=strikers[i].id,
            fight_id=fight.id,
            division="Lightweight",
            elo_type="overall",
            elo_before=1500.0,
            elo_after=1510.0,
            elo_after_shrinkage=1508.0,
            k_factor_used=40.0,
            fight_date=events[i].date,
        ))
        session.add(EloSnapshot(
            fighter_id=grapplers[i].id,
            fight_id=fight.id,
            division="Lightweight",
            elo_type="overall",
            elo_before=1500.0,
            elo_after=1490.0,
            elo_after_shrinkage=1492.0,
            k_factor_used=40.0,
            fight_date=events[i].date,
        ))
        session.add(ComputedFeature(
            fighter_id=strikers[i].id,
            fight_id=fight.id,
            as_of_date=events[i].date,
            feature_set_version="v1",
            features={
                "sig_str_per_minute": 5.0,
                "td_rate": 1.0,
                "style_tag": "striker",
            },
        ))
        session.add(ComputedFeature(
            fighter_id=grapplers[i].id,
            fight_id=fight.id,
            as_of_date=events[i].date,
            feature_set_version="v1",
            features={
                "sig_str_per_minute": 2.0,
                "td_rate": 6.0,
                "style_tag": "grappler",
            },
        ))
    session.flush()

    data["style_fights"] = fights
    return data


@pytest.fixture()
def client(session: Session, seed_api_data) -> TestClient:
    """TestClient with DB session overridden to use test transaction."""
    app = create_app()

    def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def client_matchup(session: Session, seed_matchup_api_data) -> TestClient:
    """TestClient with extended matchup seed data."""
    app = create_app()

    def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
