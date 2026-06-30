"""Integration tests for domain Elo display in CLI fighter lookup."""

from __future__ import annotations

from datetime import date

import pytest
from typer.testing import CliRunner

from ufc_prediction.cli.main import app
from ufc_prediction.elo.config import EloConfig
from ufc_prediction.elo.domain import DomainEloComputer
from ufc_prediction.elo.engine import EloEngine
from ufc_prediction.elo.queries import (
    flush_domain_snapshots,
    flush_snapshots,
    load_fights_chronological,
    load_round_stats_by_fight,
)
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter
from ufc_prediction.models.round_stats import RoundStats

runner = CliRunner()


def _seed_fighter_with_domain_elo(session):
    """Seed a fighter with fights and round stats, then run full pipeline.

    Creates a striker-like fighter with KO/TKO wins (high striking stats)
    so domain Elo snapshots are generated.
    """
    fighter = Fighter(name="Domain Test Fighter", source="test")
    opp1 = Fighter(name="Domain Opp 1", source="test")
    opp2 = Fighter(name="Domain Opp 2", source="test")
    session.add_all([fighter, opp1, opp2])
    session.flush()

    event1 = Event(name="Domain CLI Event 1", date=date(2020, 3, 1), source="test")
    event2 = Event(name="Domain CLI Event 2", date=date(2020, 9, 1), source="test")
    session.add_all([event1, event2])
    session.flush()

    fight1 = Fight(
        event_id=event1.id,
        fighter_a_id=fighter.id,
        fighter_b_id=opp1.id,
        winner_id=fighter.id,
        weight_class="Middleweight",
        method="KO/TKO",
        source="test",
    )
    fight2 = Fight(
        event_id=event2.id,
        fighter_a_id=fighter.id,
        fighter_b_id=opp2.id,
        winner_id=fighter.id,
        weight_class="Middleweight",
        method="Decision",
        method_detail="Unanimous",
        source="test",
    )
    session.add_all([fight1, fight2])
    session.flush()

    # Round stats for both fights
    for round_num in range(1, 3):
        session.add(
            RoundStats(
                fight_id=fight1.id,
                fighter_id=fighter.id,
                round_number=round_num,
                sig_strikes_landed=45,
                takedowns_landed=0,
                control_time_seconds=5,
            )
        )
        session.add(
            RoundStats(
                fight_id=fight1.id,
                fighter_id=opp1.id,
                round_number=round_num,
                sig_strikes_landed=20,
                takedowns_landed=0,
                control_time_seconds=5,
            )
        )
    for round_num in range(1, 4):
        session.add(
            RoundStats(
                fight_id=fight2.id,
                fighter_id=fighter.id,
                round_number=round_num,
                sig_strikes_landed=50,
                takedowns_landed=1,
                control_time_seconds=10,
            )
        )
        session.add(
            RoundStats(
                fight_id=fight2.id,
                fighter_id=opp2.id,
                round_number=round_num,
                sig_strikes_landed=35,
                takedowns_landed=0,
                control_time_seconds=5,
            )
        )
    session.flush()

    # Run overall + domain pipeline
    fights = load_fights_chronological(session)
    engine = EloEngine(EloConfig())
    overall_snaps = engine.compute_all(fights)
    flush_snapshots(session, overall_snaps)

    round_stats = load_round_stats_by_fight(session)
    domain_computer = DomainEloComputer(EloConfig())
    domain_snaps = domain_computer.compute_all(fights, overall_snaps, round_stats)
    flush_domain_snapshots(session, domain_snaps)
    session.flush()

    return fighter


def _seed_fighter_without_round_stats(session):
    """Seed a fighter with fights but NO round stats (no domain Elo)."""
    fighter = Fighter(name="No Domain Fighter", source="test")
    opp = Fighter(name="No Domain Opp", source="test")
    session.add_all([fighter, opp])
    session.flush()

    event = Event(name="No Domain Event", date=date(2020, 5, 1), source="test")
    session.add(event)
    session.flush()

    fight = Fight(
        event_id=event.id,
        fighter_a_id=fighter.id,
        fighter_b_id=opp.id,
        winner_id=fighter.id,
        weight_class="Lightweight",
        method="KO/TKO",
        source="test",
    )
    session.add(fight)
    session.flush()

    # Run only overall pipeline (no round stats -> no domain snapshots)
    fights = load_fights_chronological(session)
    engine = EloEngine(EloConfig())
    overall_snaps = engine.compute_all(fights)
    flush_snapshots(session, overall_snaps)

    round_stats = load_round_stats_by_fight(session)
    domain_computer = DomainEloComputer(EloConfig())
    domain_snaps = domain_computer.compute_all(fights, overall_snaps, round_stats)
    flush_domain_snapshots(session, domain_snaps)
    session.flush()

    return fighter


class TestDomainCLI:
    def test_lookup_shows_domain_elo(self, session, monkeypatch):
        """Fighter lookup displays Striking Elo and Grappling Elo when available."""
        _seed_fighter_with_domain_elo(session)
        monkeypatch.setattr("ufc_prediction.cli.main.SessionLocal", lambda: session)

        result = runner.invoke(app, ["fighter", "lookup", "Domain Test Fighter"])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Striking Elo" in result.output
        assert "Grappling Elo" in result.output
        assert "Elo Rating" in result.output

    def test_lookup_no_domain_elo_when_no_round_stats(self, session, monkeypatch):
        """Fighter without round stats shows no Striking/Grappling Elo rows."""
        _seed_fighter_without_round_stats(session)
        monkeypatch.setattr("ufc_prediction.cli.main.SessionLocal", lambda: session)

        result = runner.invoke(app, ["fighter", "lookup", "No Domain Fighter"])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Elo Rating" in result.output
        assert "Striking Elo" not in result.output
        assert "Grappling Elo" not in result.output
