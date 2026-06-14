"""Integration tests for domain Elo pipeline with testcontainers PostgreSQL.

Tests the full domain Elo pipeline: seed DB -> load fights -> compute overall ->
compute domain -> flush domain snapshots -> verify.

Uses the session fixture from tests/conftest.py (transactional, rolls back after each test).
"""

from __future__ import annotations

from datetime import date

import pytest

from ufc_prediction.elo.config import EloConfig
from ufc_prediction.elo.domain import DomainEloComputer
from ufc_prediction.elo.engine import EloEngine, SnapshotRecord
from ufc_prediction.elo.queries import (
    flush_domain_snapshots,
    flush_snapshots,
    load_fights_chronological,
    load_round_stats_by_fight,
)
from ufc_prediction.models.elo_snapshot import EloSnapshot
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter
from ufc_prediction.models.round_stats import RoundStats


# ── Seed helpers ────────────────────────────────────────────────────────────


def _make_opponent(session, name: str) -> Fighter:
    """Create and flush a named opponent fighter."""
    fighter = Fighter(name=name, source="test")
    session.add(fighter)
    session.flush()
    return fighter


def seed_striker_archetype(session):
    """Seed a fighter mimicking Adesanya's pattern: KO/TKO heavy, high striking stats.

    6 fights across 3 events (dates spread over 2 years):
    - 3x KO/TKO wins
    - 2x Decision (Unanimous) wins
    - 1x Decision loss

    Decision fights have HIGH sig_strikes_landed (40-60), LOW takedowns (0-1),
    LOW control time (0-20 seconds).
    """
    striker = Fighter(name="Test Striker", source="test")
    session.add(striker)
    session.flush()

    events = []
    for i, (name, d) in enumerate([
        ("Striker Event 1", date(2019, 1, 15)),
        ("Striker Event 2", date(2019, 7, 20)),
        ("Striker Event 3", date(2020, 3, 10)),
    ]):
        ev = Event(name=name, date=d, source="test")
        session.add(ev)
        events.append(ev)
    session.flush()

    opponents = [_make_opponent(session, f"Striker Opp {i}") for i in range(6)]

    fight_specs = [
        # (event_idx, opponent_idx, winner_is_striker, method, method_detail)
        (0, 0, True, "KO/TKO", None),
        (0, 1, True, "KO/TKO", None),
        (1, 2, True, "Decision", "Unanimous"),
        (1, 3, True, "Decision", "Unanimous"),
        (2, 4, True, "KO/TKO", None),
        (2, 5, False, "Decision", "Unanimous"),  # loss
    ]

    fights = []
    for event_idx, opp_idx, striker_wins, method, detail in fight_specs:
        opp = opponents[opp_idx]
        f = Fight(
            event_id=events[event_idx].id,
            fighter_a_id=striker.id,
            fighter_b_id=opp.id,
            winner_id=striker.id if striker_wins else opp.id,
            weight_class="Middleweight",
            method=method,
            method_detail=detail,
            source="test",
        )
        session.add(f)
        fights.append(f)
    session.flush()

    # Add round stats for ALL fights (so domain Elo is computed for all)
    for i, (fight, opp_idx) in enumerate(zip(fights, range(6))):
        opp = opponents[opp_idx]
        method = fight_specs[i][3]

        if method == "KO/TKO":
            # KO/TKO fights: high strikes, fixed 80/20 ratio applies anyway
            for round_num in range(1, 3):
                session.add(RoundStats(
                    fight_id=fight.id, fighter_id=striker.id,
                    round_number=round_num,
                    sig_strikes_landed=50, takedowns_landed=0,
                    control_time_seconds=5,
                ))
                session.add(RoundStats(
                    fight_id=fight.id, fighter_id=opp.id,
                    round_number=round_num,
                    sig_strikes_landed=30, takedowns_landed=0,
                    control_time_seconds=5,
                ))
        else:
            # Decision fights: HIGH sig_strikes, LOW grappling
            for round_num in range(1, 4):
                session.add(RoundStats(
                    fight_id=fight.id, fighter_id=striker.id,
                    round_number=round_num,
                    sig_strikes_landed=55, takedowns_landed=0,
                    control_time_seconds=10,
                ))
                session.add(RoundStats(
                    fight_id=fight.id, fighter_id=opp.id,
                    round_number=round_num,
                    sig_strikes_landed=45, takedowns_landed=1,
                    control_time_seconds=15,
                ))
    session.flush()

    return {"fighter": striker, "fights": fights, "opponents": opponents}


def seed_grappler_archetype(session):
    """Seed a fighter mimicking Khabib's pattern: Submission heavy, high TD + control.

    6 fights across 3 events:
    - 3x Submission wins
    - 2x Decision (Unanimous) wins
    - 1x Decision loss

    Decision fights have LOW sig_strikes (10-20), HIGH takedowns (3-5),
    HIGH control time (120-240 seconds).
    """
    grappler = Fighter(name="Test Grappler", source="test")
    session.add(grappler)
    session.flush()

    events = []
    for name, d in [
        ("Grappler Event 1", date(2019, 2, 10)),
        ("Grappler Event 2", date(2019, 8, 15)),
        ("Grappler Event 3", date(2020, 4, 20)),
    ]:
        ev = Event(name=name, date=d, source="test")
        session.add(ev)
        events.append(ev)
    session.flush()

    opponents = [_make_opponent(session, f"Grappler Opp {i}") for i in range(6)]

    fight_specs = [
        (0, 0, True, "Submission", None),
        (0, 1, True, "Submission", None),
        (1, 2, True, "Decision", "Unanimous"),
        (1, 3, True, "Decision", "Unanimous"),
        (2, 4, True, "Submission", None),
        (2, 5, False, "Decision", "Unanimous"),  # loss
    ]

    fights = []
    for event_idx, opp_idx, grappler_wins, method, detail in fight_specs:
        opp = opponents[opp_idx]
        f = Fight(
            event_id=events[event_idx].id,
            fighter_a_id=grappler.id,
            fighter_b_id=opp.id,
            winner_id=grappler.id if grappler_wins else opp.id,
            weight_class="Lightweight",
            method=method,
            method_detail=detail,
            source="test",
        )
        session.add(f)
        fights.append(f)
    session.flush()

    # Add round stats for ALL fights
    for i, (fight, opp_idx) in enumerate(zip(fights, range(6))):
        opp = opponents[opp_idx]
        method = fight_specs[i][3]

        if method == "Submission":
            # Submission fights: moderate stats, 20/80 fixed ratio applies
            for round_num in range(1, 3):
                session.add(RoundStats(
                    fight_id=fight.id, fighter_id=grappler.id,
                    round_number=round_num,
                    sig_strikes_landed=15, takedowns_landed=4,
                    control_time_seconds=180,
                ))
                session.add(RoundStats(
                    fight_id=fight.id, fighter_id=opp.id,
                    round_number=round_num,
                    sig_strikes_landed=20, takedowns_landed=0,
                    control_time_seconds=10,
                ))
        else:
            # Decision fights: LOW strikes, HIGH grappling
            for round_num in range(1, 4):
                session.add(RoundStats(
                    fight_id=fight.id, fighter_id=grappler.id,
                    round_number=round_num,
                    sig_strikes_landed=15, takedowns_landed=4,
                    control_time_seconds=200,
                ))
                session.add(RoundStats(
                    fight_id=fight.id, fighter_id=opp.id,
                    round_number=round_num,
                    sig_strikes_landed=12, takedowns_landed=0,
                    control_time_seconds=20,
                ))
    session.flush()

    return {"fighter": grappler, "fights": fights, "opponents": opponents}


def seed_kaggle_only_fighter(session):
    """Seed a fighter with fights but NO round stats (Kaggle-only data, D-05)."""
    fighter = Fighter(name="Kaggle Only Fighter", source="test")
    session.add(fighter)
    session.flush()

    events = []
    for name, d in [
        ("Kaggle Event 1", date(2018, 3, 1)),
        ("Kaggle Event 2", date(2018, 9, 1)),
    ]:
        ev = Event(name=name, date=d, source="test")
        session.add(ev)
        events.append(ev)
    session.flush()

    opponents = [_make_opponent(session, f"Kaggle Opp {i}") for i in range(3)]

    fights = []
    for event_idx, opp_idx, method in [
        (0, 0, "KO/TKO"),
        (0, 1, "Decision"),
        (1, 2, "Submission"),
    ]:
        opp = opponents[opp_idx]
        f = Fight(
            event_id=events[event_idx].id,
            fighter_a_id=fighter.id,
            fighter_b_id=opp.id,
            winner_id=fighter.id,
            weight_class="Welterweight",
            method=method,
            source="test",
        )
        session.add(f)
        fights.append(f)
    session.flush()

    # No RoundStats rows -- this is the point
    return {"fighter": fighter, "fights": fights}


def run_full_pipeline(session):
    """Run overall + domain Elo computation pipeline."""
    fights = load_fights_chronological(session)
    engine = EloEngine(EloConfig())
    overall_snaps = engine.compute_all(fights)
    flush_snapshots(session, overall_snaps)

    round_stats = load_round_stats_by_fight(session)
    domain_computer = DomainEloComputer(EloConfig())
    domain_snaps = domain_computer.compute_all(fights, overall_snaps, round_stats)
    flush_domain_snapshots(session, domain_snaps)
    session.flush()

    return overall_snaps, domain_snaps


# ── Tests ───────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_flush_preserves_overall(session):
    """flush_domain_snapshots writes domain snapshots and preserves overall snapshots."""
    # Seed basic test data (reuse pattern from test_integration.py)
    fighters = [Fighter(name=f"Flush Fighter {i}", source="test") for i in range(4)]
    session.add_all(fighters)
    session.flush()

    event = Event(name="Flush Event", date=date(2020, 1, 1), source="test")
    session.add(event)
    session.flush()

    fights = []
    for i in range(0, 4, 2):
        f = Fight(
            event_id=event.id,
            fighter_a_id=fighters[i].id,
            fighter_b_id=fighters[i + 1].id,
            winner_id=fighters[i].id,
            weight_class="Lightweight",
            method="KO/TKO",
            source="test",
        )
        session.add(f)
        fights.append(f)
    session.flush()

    # Run overall computation
    db_fights = load_fights_chronological(session)
    engine = EloEngine(EloConfig())
    overall_snaps = engine.compute_all(db_fights)
    flush_snapshots(session, overall_snaps)
    session.flush()

    overall_count = (
        session.query(EloSnapshot)
        .filter(EloSnapshot.elo_type == "overall")
        .count()
    )
    assert overall_count > 0

    # Create fake domain snapshots manually
    fake_domain = [
        SnapshotRecord(
            fighter_id=fighters[0].id,
            fight_id=fights[0].id,
            division="Lightweight",
            elo_type="striking",
            elo_before=1500.0,
            elo_after=1510.0,
            elo_after_shrinkage=1505.0,
            k_factor_used=32.0,
            fight_date=date(2020, 1, 1),
        ),
    ]
    flush_domain_snapshots(session, fake_domain)
    session.flush()

    # Verify overall snapshots are unchanged
    new_overall_count = (
        session.query(EloSnapshot)
        .filter(EloSnapshot.elo_type == "overall")
        .count()
    )
    assert new_overall_count == overall_count, "Domain flush destroyed overall snapshots!"

    # Verify domain snapshots were written
    domain_count = (
        session.query(EloSnapshot)
        .filter(EloSnapshot.elo_type.in_(["striking", "grappling"]))
        .count()
    )
    assert domain_count == 1


@pytest.mark.integration
def test_idempotent_domain(session):
    """Running domain compute twice produces identical snapshot counts."""
    seed_striker_archetype(session)

    # First pass
    fights = load_fights_chronological(session)
    engine = EloEngine(EloConfig())
    overall_snaps = engine.compute_all(fights)
    flush_snapshots(session, overall_snaps)

    round_stats = load_round_stats_by_fight(session)
    domain_computer = DomainEloComputer(EloConfig())
    domain_snaps = domain_computer.compute_all(fights, overall_snaps, round_stats)
    flush_domain_snapshots(session, domain_snaps)
    session.flush()

    first_count = (
        session.query(EloSnapshot)
        .filter(EloSnapshot.elo_type.in_(["striking", "grappling"]))
        .count()
    )
    assert first_count > 0

    # Second pass (idempotent)
    domain_computer2 = DomainEloComputer(EloConfig())
    domain_snaps2 = domain_computer2.compute_all(fights, overall_snaps, round_stats)
    flush_domain_snapshots(session, domain_snaps2)
    session.flush()

    second_count = (
        session.query(EloSnapshot)
        .filter(EloSnapshot.elo_type.in_(["striking", "grappling"]))
        .count()
    )
    assert second_count == first_count, (
        f"Domain flush not idempotent: {first_count} vs {second_count}"
    )


@pytest.mark.integration
def test_full_pipeline(session):
    """Full pipeline: seed -> overall compute -> domain compute -> verify DB snapshots."""
    seed_striker_archetype(session)
    overall_snaps, domain_snaps = run_full_pipeline(session)

    # Both striking and grappling snapshots should exist
    striking_count = (
        session.query(EloSnapshot)
        .filter(EloSnapshot.elo_type == "striking")
        .count()
    )
    grappling_count = (
        session.query(EloSnapshot)
        .filter(EloSnapshot.elo_type == "grappling")
        .count()
    )
    assert striking_count > 0, "No striking snapshots in DB"
    assert grappling_count > 0, "No grappling snapshots in DB"


@pytest.mark.integration
def test_adesanya_archetype(session):
    """Striker archetype (D-10): striking_elo > grappling_elo."""
    data = seed_striker_archetype(session)
    striker = data["fighter"]
    run_full_pipeline(session)

    # Get latest striking snapshot for the striker
    striking_snap = (
        session.query(EloSnapshot)
        .filter(
            EloSnapshot.fighter_id == striker.id,
            EloSnapshot.elo_type == "striking",
        )
        .order_by(EloSnapshot.fight_date.desc())
        .first()
    )
    grappling_snap = (
        session.query(EloSnapshot)
        .filter(
            EloSnapshot.fighter_id == striker.id,
            EloSnapshot.elo_type == "grappling",
        )
        .order_by(EloSnapshot.fight_date.desc())
        .first()
    )

    assert striking_snap is not None, "No striking snapshot for striker archetype"
    assert grappling_snap is not None, "No grappling snapshot for striker archetype"

    striking_elo = striking_snap.elo_after_shrinkage
    grappling_elo = grappling_snap.elo_after_shrinkage

    assert striking_elo > grappling_elo, (
        f"Striker archetype violation: striking_elo ({striking_elo:.1f}) "
        f"should be > grappling_elo ({grappling_elo:.1f})"
    )


@pytest.mark.integration
def test_khabib_archetype(session):
    """Grappler archetype (D-10): grappling_elo > striking_elo."""
    data = seed_grappler_archetype(session)
    grappler = data["fighter"]
    run_full_pipeline(session)

    striking_snap = (
        session.query(EloSnapshot)
        .filter(
            EloSnapshot.fighter_id == grappler.id,
            EloSnapshot.elo_type == "striking",
        )
        .order_by(EloSnapshot.fight_date.desc())
        .first()
    )
    grappling_snap = (
        session.query(EloSnapshot)
        .filter(
            EloSnapshot.fighter_id == grappler.id,
            EloSnapshot.elo_type == "grappling",
        )
        .order_by(EloSnapshot.fight_date.desc())
        .first()
    )

    assert striking_snap is not None, "No striking snapshot for grappler archetype"
    assert grappling_snap is not None, "No grappling snapshot for grappler archetype"

    striking_elo = striking_snap.elo_after_shrinkage
    grappling_elo = grappling_snap.elo_after_shrinkage

    assert grappling_elo > striking_elo, (
        f"Grappler archetype violation: grappling_elo ({grappling_elo:.1f}) "
        f"should be > striking_elo ({striking_elo:.1f})"
    )


@pytest.mark.integration
def test_no_round_data_no_domain_snapshots(session):
    """Fighter with only Kaggle data (no round stats) has no domain snapshots (D-05)."""
    data = seed_kaggle_only_fighter(session)
    fighter = data["fighter"]
    run_full_pipeline(session)

    domain_count = (
        session.query(EloSnapshot)
        .filter(
            EloSnapshot.fighter_id == fighter.id,
            EloSnapshot.elo_type.in_(["striking", "grappling"]),
        )
        .count()
    )
    assert domain_count == 0, (
        f"Expected no domain snapshots for fighter without round stats, got {domain_count}"
    )


@pytest.mark.integration
def test_both_fighters_stats_summed(session):
    """Both fighters' round stats are summed for attribution ratio."""
    fighter_a = Fighter(name="Stats Sum A", source="test")
    fighter_b = Fighter(name="Stats Sum B", source="test")
    session.add_all([fighter_a, fighter_b])
    session.flush()

    event = Event(name="Stats Sum Event", date=date(2020, 6, 1), source="test")
    session.add(event)
    session.flush()

    fight = Fight(
        event_id=event.id,
        fighter_a_id=fighter_a.id,
        fighter_b_id=fighter_b.id,
        winner_id=fighter_a.id,
        weight_class="Lightweight",
        method="Decision",
        method_detail="Unanimous",
        source="test",
    )
    session.add(fight)
    session.flush()

    # Fighter A: sig_str=40, td=1, ctrl=10
    session.add(RoundStats(
        fight_id=fight.id, fighter_id=fighter_a.id,
        round_number=1,
        sig_strikes_landed=40, takedowns_landed=1, control_time_seconds=10,
    ))
    # Fighter B: sig_str=30, td=3, ctrl=100
    session.add(RoundStats(
        fight_id=fight.id, fighter_id=fighter_b.id,
        round_number=1,
        sig_strikes_landed=30, takedowns_landed=3, control_time_seconds=100,
    ))
    session.flush()

    # Total: striking=70, grappling=1+10+3+100=114, total=184
    # Expected striking_ratio = 70/184 ~ 0.38 (grappling-heavy fight)

    overall_snaps, domain_snaps = run_full_pipeline(session)

    # Both fighters should have domain snapshots
    domain_for_a = [
        s for s in domain_snaps
        if s.fighter_id == fighter_a.id and s.elo_type == "grappling"
    ]
    domain_for_b = [
        s for s in domain_snaps
        if s.fighter_id == fighter_b.id and s.elo_type == "grappling"
    ]
    assert len(domain_for_a) > 0, "Fighter A should have grappling domain snapshots"
    assert len(domain_for_b) > 0, "Fighter B should have grappling domain snapshots"

    # The grappling domain delta should be larger than striking delta
    # because grappling_ratio > striking_ratio (~0.62 vs ~0.38)
    striking_a = [
        s for s in domain_snaps
        if s.fighter_id == fighter_a.id and s.elo_type == "striking"
    ]
    grappling_a = [
        s for s in domain_snaps
        if s.fighter_id == fighter_a.id and s.elo_type == "grappling"
    ]

    striking_delta = abs(striking_a[0].elo_after - striking_a[0].elo_before)
    grappling_delta = abs(grappling_a[0].elo_after - grappling_a[0].elo_before)
    assert grappling_delta > striking_delta, (
        f"Grappling-heavy fight should have larger grappling delta "
        f"({grappling_delta:.2f}) than striking delta ({striking_delta:.2f})"
    )
