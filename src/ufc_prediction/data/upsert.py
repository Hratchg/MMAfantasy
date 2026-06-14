"""Shared upsert helpers for database ingestion.

Extracted from ingest_rajeevw.py to be reused by Kaggle ingestors and
the UFCStats scraper. All helpers follow the same pattern: lookup by
natural key, update if exists, insert if not.

Banned imports per Pitfall #1 / Finding 11: nothing under ``ufc_prediction.ml.*``.
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter
from ufc_prediction.models.round_stats import RoundStats
from ufc_prediction.scraper.referee_normalize import normalize_referee_name


def upsert_fighter(
    session: Session,
    name: str,
    source: str,
    **profile_fields: object,
) -> Fighter:
    """Lookup fighter by name+source; update if exists, insert if not.

    Only overwrites existing fields with non-None new values.
    Returns the Fighter instance (flushed, has id).
    """
    fighter = (
        session.query(Fighter)
        .filter(Fighter.name == name, Fighter.source == source)
        .first()
    )
    if fighter is not None:
        for field_name, value in profile_fields.items():
            if value is not None:
                setattr(fighter, field_name, value)
        return fighter

    fighter = Fighter(
        name=name,
        source=source,
        **{k: v for k, v in profile_fields.items() if v is not None},
    )
    session.add(fighter)
    session.flush()
    return fighter


def upsert_event(
    session: Session,
    name: str,
    event_date: date,
    location: str | None,
    source: str,
    *,
    source_url: str | None = None,
) -> Event:
    """Lookup event by name+date+source; update if exists, insert if not.

    Returns the Event instance (flushed, has id).
    """
    event = (
        session.query(Event)
        .filter(Event.name == name, Event.date == event_date, Event.source == source)
        .first()
    )
    if event is not None:
        if location is not None:
            event.location = location
        if source_url is not None:
            event.source_url = source_url
        return event

    event = Event(
        name=name,
        date=event_date,
        location=location,
        source=source,
        source_url=source_url,
    )
    session.add(event)
    session.flush()
    return event


def upsert_fight(
    session: Session,
    event_id: int,
    fighter_a_id: int,
    fighter_b_id: int,
    source: str,
    **fight_fields: object,
) -> Fight:
    """Lookup fight by (event_id, fighter_a_id, fighter_b_id); update if exists, insert if not.

    Also checks reversed fighter order (event_id, fighter_b_id, fighter_a_id) per the
    Phase 2 decision about reversed fighter order dedup.

    Returns the Fight instance (flushed, has id).
    """
    # Check normal order
    fight = (
        session.query(Fight)
        .filter(
            Fight.event_id == event_id,
            Fight.fighter_a_id == fighter_a_id,
            Fight.fighter_b_id == fighter_b_id,
        )
        .first()
    )

    # Check reversed order
    if fight is None:
        fight = (
            session.query(Fight)
            .filter(
                Fight.event_id == event_id,
                Fight.fighter_a_id == fighter_b_id,
                Fight.fighter_b_id == fighter_a_id,
            )
            .first()
        )

    if fight is not None:
        for field_name, value in fight_fields.items():
            if value is not None:
                setattr(fight, field_name, value)
        return fight

    fight = Fight(
        event_id=event_id,
        fighter_a_id=fighter_a_id,
        fighter_b_id=fighter_b_id,
        source=source,
        **{k: v for k, v in fight_fields.items() if v is not None},
    )
    session.add(fight)
    session.flush()
    return fight


def upsert_referee(session: Session, raw_name: str | None) -> int | None:
    """Upsert a Referee row by normalized_name; append raw to alias_list on hit.

    Per CONTEXT D-10 (REVISION-02): consolidates the existing
    parse_fight_detail.py:80-90 referee extraction. ``normalized_name`` is the
    canonical key (lowercase-hyphen via referee_normalize.normalize_referee_name).
    Returns the referee id, or None if raw_name is None / empty / whitespace-only.

    On duplicate detection: appends raw_name to alias_list if not already there
    (set semantics, no duplicates). Does NOT mutate the canonical ``name`` field.

    The Referee model is lazy-imported inside the function to avoid circular-import
    issues with ufc_prediction.models.__init__ -> upsert paths.
    """
    # Lazy import to avoid circular dependency (models/__init__ imports Fighter etc.;
    # upsert.py is imported by ingest paths that touch models on load).
    from ufc_prediction.models.referee import Referee

    normalized = normalize_referee_name(raw_name)
    if normalized is None:
        return None
    # normalize_referee_name returns None for None / "" / whitespace-only,
    # so raw_name is now guaranteed non-None and non-empty post-strip.
    assert raw_name is not None
    raw_stripped = raw_name.strip()

    existing = session.execute(
        select(Referee).where(Referee.normalized_name == normalized)
    ).scalar_one_or_none()

    if existing is not None:
        aliases = list(existing.alias_list or [])
        if raw_stripped not in aliases:
            aliases.append(raw_stripped)
            existing.alias_list = aliases
            session.flush()
        return existing.id

    new_ref = Referee(
        name=raw_stripped,
        normalized_name=normalized,
        alias_list=[raw_stripped],
    )
    session.add(new_ref)
    session.flush()
    return new_ref.id


def upsert_round_stats(
    session: Session,
    fight_id: int,
    fighter_id: int,
    round_number: int,
    **stat_fields: object,
) -> RoundStats:
    """Lookup round stats by (fight_id, fighter_id, round_number); update or insert.

    Returns the RoundStats instance.
    """
    existing = (
        session.query(RoundStats)
        .filter(
            RoundStats.fight_id == fight_id,
            RoundStats.fighter_id == fighter_id,
            RoundStats.round_number == round_number,
        )
        .first()
    )

    if existing is not None:
        for field_name, value in stat_fields.items():
            setattr(existing, field_name, value)
        return existing

    rs = RoundStats(
        fight_id=fight_id,
        fighter_id=fighter_id,
        round_number=round_number,
        **stat_fields,
    )
    session.add(rs)
    return rs
