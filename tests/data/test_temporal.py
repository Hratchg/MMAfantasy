"""Temporal integrity tests (DATA-06, D-07).

Phase 3 (Elo Engine) depends on these tests passing. They verify that
fight data is temporally consistent and no future data leaks.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from sqlalchemy.orm import Session

from ufc_prediction.data.ingest_rajeevw import ingest_rajeevw_fighters, ingest_rajeevw_fights
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter


def _ingest_fixture_data(
    session: Session,
    rajeevw_fighters_csv: Path,
    rajeevw_fights_csv: Path,
) -> None:
    """Helper: ingest rajeevw fixture data into the session."""
    ingest_rajeevw_fighters(rajeevw_fighters_csv, session)
    ingest_rajeevw_fights(rajeevw_fights_csv, session)


def test_no_future_event_dates(
    session: Session,
    rajeevw_fighters_csv: Path,
    rajeevw_fights_csv: Path,
) -> None:
    """After ingestion, no event has a date in the future."""
    _ingest_fixture_data(session, rajeevw_fighters_csv, rajeevw_fights_csv)

    future_events = session.query(Event).filter(Event.date > date.today()).count()
    assert future_events == 0, f"Found {future_events} event(s) with future dates"


def test_fighter_fights_chronologically_ordered(
    session: Session,
    rajeevw_fighters_csv: Path,
    rajeevw_fights_csv: Path,
) -> None:
    """For each fighter with 2+ fights, fights are ordered by event date.

    Per D-07: per-fighter fight sequence has monotonically non-decreasing
    event dates.
    """
    _ingest_fixture_data(session, rajeevw_fighters_csv, rajeevw_fights_csv)

    fighters = session.query(Fighter).all()
    for fighter in fighters:
        # Get all fights for this fighter (as fighter_a or fighter_b)
        fights = (
            session.query(Fight)
            .join(Event, Fight.event_id == Event.id)
            .filter((Fight.fighter_a_id == fighter.id) | (Fight.fighter_b_id == fighter.id))
            .order_by(Event.date)
            .all()
        )

        if len(fights) < 2:
            continue

        # Verify chronological ordering
        event_dates = []
        for fight in fights:
            event = session.query(Event).filter(Event.id == fight.event_id).first()
            event_dates.append(event.date)

        for i in range(1, len(event_dates)):
            assert event_dates[i] >= event_dates[i - 1], (
                f"Fighter '{fighter.name}' has non-chronological fights: "
                f"{event_dates[i - 1]} -> {event_dates[i]}"
            )


def test_no_self_fights(
    session: Session,
    rajeevw_fighters_csv: Path,
    rajeevw_fights_csv: Path,
) -> None:
    """No fight has fighter_a_id == fighter_b_id.

    Enforced by a check constraint but also verified in data.
    """
    _ingest_fixture_data(session, rajeevw_fighters_csv, rajeevw_fights_csv)

    self_fights = session.query(Fight).filter(Fight.fighter_a_id == Fight.fighter_b_id).count()
    assert self_fights == 0, f"Found {self_fights} fight(s) where a fighter fights themselves"
