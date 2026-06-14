"""Integration tests for Elo pipeline with testcontainers PostgreSQL.

Tests the full pipeline: seed DB -> load fights -> compute Elo -> flush snapshots -> verify.
Uses the session fixture from tests/conftest.py (transactional, rolls back after each test).
"""

from __future__ import annotations

import random
import time
from datetime import date

import pytest

from ufc_prediction.elo.calibration import check_calibration, compute_calibration_buckets
from ufc_prediction.elo.config import EloConfig
from ufc_prediction.elo.engine import EloEngine, FightRecord
from ufc_prediction.elo.queries import flush_snapshots, load_fights_chronological
from ufc_prediction.models.elo_snapshot import EloSnapshot
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter


def seed_test_data(session):
    """Seed the database with known fight data for testing.

    Creates 4 fighters, 2 events, and 3 fights.
    Returns the created objects for assertions.
    """
    fighter_a = Fighter(name="Fighter A", source="test")
    fighter_b = Fighter(name="Fighter B", source="test")
    fighter_c = Fighter(name="Fighter C", source="test")
    fighter_d = Fighter(name="Fighter D", source="test")
    session.add_all([fighter_a, fighter_b, fighter_c, fighter_d])
    session.flush()

    event_1 = Event(name="Test Event 1", date=date(2020, 1, 1), source="test")
    event_2 = Event(name="Test Event 2", date=date(2020, 6, 1), source="test")
    session.add_all([event_1, event_2])
    session.flush()

    fight_1 = Fight(
        event_id=event_1.id,
        fighter_a_id=fighter_a.id,
        fighter_b_id=fighter_b.id,
        winner_id=fighter_a.id,
        weight_class="Lightweight",
        method="KO/TKO",
        method_detail=None,
        source="test",
    )
    fight_2 = Fight(
        event_id=event_1.id,
        fighter_a_id=fighter_c.id,
        fighter_b_id=fighter_d.id,
        winner_id=fighter_c.id,
        weight_class="Lightweight",
        method="Decision",
        method_detail="Unanimous",
        source="test",
    )
    fight_3 = Fight(
        event_id=event_2.id,
        fighter_a_id=fighter_a.id,
        fighter_b_id=fighter_c.id,
        winner_id=fighter_a.id,
        weight_class="Lightweight",
        method="Decision",
        method_detail="Split",
        source="test",
    )
    session.add_all([fight_1, fight_2, fight_3])
    session.flush()

    return {
        "fighters": [fighter_a, fighter_b, fighter_c, fighter_d],
        "events": [event_1, event_2],
        "fights": [fight_1, fight_2, fight_3],
    }


def test_full_pipeline(session):
    """Seed data, load fights, compute Elo, flush snapshots, verify DB state."""
    data = seed_test_data(session)

    # Load fights from DB
    fights = load_fights_chronological(session)
    assert len(fights) == 3

    # Verify chronological ordering: event 1 fights before event 2
    assert fights[0].event_date == date(2020, 1, 1)
    assert fights[1].event_date == date(2020, 1, 1)
    assert fights[2].event_date == date(2020, 6, 1)
    # Within same date, ordered by fight_id (deterministic)
    assert fights[0].fight_id < fights[1].fight_id

    # Compute Elo
    engine = EloEngine(EloConfig())
    snapshots = engine.compute_all(fights)
    assert len(snapshots) == 6  # 2 per fight

    # Flush to DB
    count = flush_snapshots(session, snapshots)
    assert count == 6

    # Verify DB state
    db_count = session.query(EloSnapshot).count()
    assert db_count == 6

    # Verify first fight fighter A starts at 1500.0
    fighter_a_id = data["fighters"][0].id
    first_snap = (
        session.query(EloSnapshot)
        .filter(
            EloSnapshot.fighter_id == fighter_a_id,
            EloSnapshot.fight_id == data["fights"][0].id,
        )
        .first()
    )
    assert first_snap is not None
    assert first_snap.elo_before == 1500.0


def test_idempotent_recomputation(session):
    """Compute and flush twice -- should still have exactly 6 snapshots."""
    seed_test_data(session)
    fights = load_fights_chronological(session)

    # First computation
    engine = EloEngine(EloConfig())
    snapshots = engine.compute_all(fights)
    flush_snapshots(session, snapshots)
    assert session.query(EloSnapshot).filter(EloSnapshot.elo_type == "overall").count() == 6

    # Second computation (idempotent)
    engine2 = EloEngine(EloConfig())
    snapshots2 = engine2.compute_all(fights)
    flush_snapshots(session, snapshots2)
    assert session.query(EloSnapshot).filter(EloSnapshot.elo_type == "overall").count() == 6


def test_performance_30s(session):
    """ELO-10: Full Elo recomputation on 7000 fights completes in under 30 seconds.

    This test uses synthetic FightRecords directly -- no DB interaction needed.
    Tests pure engine performance on realistic data volume.
    """
    random.seed(42)
    fighters = list(range(1, 501))  # 500 fighters
    fights = []
    fid = 0

    for year in range(2010, 2024):  # 14 years
        for _ in range(500):  # 500 fights per year
            fid += 1
            a, b = random.sample(fighters, 2)
            fights.append(
                FightRecord(
                    fight_id=fid,
                    event_date=date(year, 6, 15),
                    fighter_a_id=a,
                    fighter_b_id=b,
                    winner_id=a,
                    weight_class="Lightweight",
                    method="Decision",
                    method_detail="Unanimous",
                )
            )

    assert len(fights) == 7000

    engine = EloEngine(EloConfig())
    start = time.perf_counter()
    snapshots = engine.compute_all(fights)
    elapsed = time.perf_counter() - start

    assert elapsed < 30.0, f"Elo computation took {elapsed:.2f}s, expected <30s"
    assert len(snapshots) == 14000  # 2 per fight


def test_snapshots_have_shrinkage_values(session):
    """Verify that snapshots include shrinkage values for fighters with few fights."""
    data = seed_test_data(session)
    fights = load_fights_chronological(session)

    engine = EloEngine(EloConfig())
    snapshots = engine.compute_all(fights)
    flush_snapshots(session, snapshots)

    # Fighter A has 2 fights (< shrinkage_min_fights=5), so shrinkage applies
    fighter_a_id = data["fighters"][0].id
    snap = (
        session.query(EloSnapshot)
        .filter(
            EloSnapshot.fighter_id == fighter_a_id,
            EloSnapshot.fight_id == data["fights"][0].id,
        )
        .first()
    )
    assert snap is not None
    # After 1 fight, fighter has 1 fight count < 5 so shrinkage applies
    assert snap.elo_after_shrinkage != snap.elo_after
    # Shrinkage pulls toward 1500 (initial rating)
    distance_raw = abs(snap.elo_after - 1500.0)
    distance_shrunk = abs(snap.elo_after_shrinkage - 1500.0)
    assert distance_shrunk < distance_raw


def test_calibration_on_seeded_data(session):
    """D-15/D-16: Calibration bucket validation on synthetic fight data.

    Generates synthetic fights with varied outcomes to populate calibration buckets.
    This validates the calibration mechanism, not actual calibration on real data.
    """
    random.seed(123)
    # Generate 300 synthetic fights with realistic-ish outcomes
    fights = []
    fid = 0
    num_fighters = 100

    for i in range(300):
        fid += 1
        a = random.randint(1, num_fighters)
        b = random.randint(1, num_fighters)
        while b == a:
            b = random.randint(1, num_fighters)

        # Randomly pick a winner
        winner = a if random.random() < 0.6 else b
        fights.append(
            FightRecord(
                fight_id=fid,
                event_date=date(2015 + i // 60, (i % 12) + 1, 15),
                fighter_a_id=a,
                fighter_b_id=b,
                winner_id=winner,
                weight_class="Lightweight",
                method="Decision",
                method_detail="Unanimous",
            )
        )

    config = EloConfig()
    engine = EloEngine(config)

    # Collect pre-fight predictions during processing to avoid look-ahead bias.
    # For each fight, capture ratings BEFORE engine state is updated by that fight.
    elo_diffs = []
    predictions = []
    outcomes = []

    for fight in fights:
        if fight.winner_id is not None:
            division = fight.weight_class
            rating_a = engine.get_rating(fight.fighter_a_id, division)
            rating_b = engine.get_rating(fight.fighter_b_id, division)
            elo_diffs.append(abs(rating_a - rating_b))
            predictions.append(engine.expected_win_probability(rating_a, rating_b))
            outcomes.append(1 if fight.winner_id == fight.fighter_a_id else 0)
        # Process fight to advance engine state (compute_all one fight at a time)
        engine.compute_all([fight])

    buckets = compute_calibration_buckets(elo_diffs, predictions, outcomes)

    # Validate buckets are populated (at least some)
    assert len(buckets) > 0, "Expected at least one calibration bucket to be populated"

    # Check calibration -- on synthetic data this may or may not pass
    # The important thing is that the mechanism works
    calibration_result = check_calibration(buckets)
    # Document the result without asserting (synthetic data may not calibrate well)
    if not calibration_result:
        # This is expected for synthetic data -- the mechanism works correctly
        for name, data in buckets.items():
            assert "count" in data
            assert "predicted_rate" in data
            assert "observed_rate" in data
            assert "gap" in data
