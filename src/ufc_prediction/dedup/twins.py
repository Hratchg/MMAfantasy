"""Cross-source "twin" fight matching: the same bout ingested from two sources.

The corpus holds each pre-2021 UFC bout up to three times (ufcstats,
kaggle-rajeevw, kaggle-mdabbert). ``fighters.id`` is per-source (the same
person has one row per ingest source), so a twin is identified by NAME:

    same ``events.date``  AND  the same two fighters by canonical name
    (order-insensitive)

The canonical name is :func:`ufc_prediction.scraper.bfo_matcher.normalize_name`
(lowercase, punctuation/hyphens stripped, Saint->St, Junior->Jr), a superset of
the ``lower(trim(name))`` key ``scripts/recon_dedup.py`` measured. On the live
corpus every ufcstats<->kaggle twin is found on the exact date (a +/-1 day
window adds none).

This module only MATCHES; callers decide what to do with zero / several twins
(the repair scripts refuse on any ambiguity).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session, aliased

from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter
from ufc_prediction.scraper.bfo_matcher import normalize_name

KAGGLE_SOURCES: tuple[str, ...] = ("kaggle-rajeevw", "kaggle-mdabbert")

PairKey = tuple[date, frozenset[str]]


@dataclass(frozen=True)
class FightSides:
    """One fight row with its event date/source and both fighters' names."""

    fight_id: int
    source: str  # events.source
    event_date: date
    fighter_a_id: int
    fighter_a_name: str
    fighter_b_id: int
    fighter_b_name: str


@dataclass
class TwinMatch:
    """Result of :func:`find_twins`.

    ``twins`` has an entry for EVERY target (``[]`` when it has none, or when
    it is unkeyable / a duplicate target).
    """

    twins: dict[int, list[FightSides]] = field(default_factory=dict)
    unkeyable: set[int] = field(default_factory=set)
    duplicate_targets: set[int] = field(default_factory=set)


def canonical_name(name: str) -> str:
    return normalize_name(name or "")


def pair_key(fight: FightSides) -> PairKey | None:
    """``(date, {canonical a, canonical b})`` or ``None`` if not uniquely keyable."""
    a, b = canonical_name(fight.fighter_a_name), canonical_name(fight.fighter_b_name)
    if not a or not b or a == b:
        return None
    return (fight.event_date, frozenset((a, b)))


def find_twins(targets: Iterable[FightSides], candidates: Iterable[FightSides]) -> TwinMatch:
    """Match each target fight to every candidate with the same pair key.

    Targets that share a key with another target (two same-day bouts between
    the same canonical names) are flagged ``duplicate_targets`` and given no
    twins: a candidate cannot be attributed to one of them.
    """
    by_key: dict[PairKey, list[FightSides]] = defaultdict(list)
    for c in candidates:
        key = pair_key(c)
        if key is not None:
            by_key[key].append(c)

    target_keys: dict[int, PairKey | None] = {}
    key_count: dict[PairKey, int] = defaultdict(int)
    for t in targets:
        key = pair_key(t)
        target_keys[t.fight_id] = key
        if key is not None:
            key_count[key] += 1

    match = TwinMatch()
    for fight_id, key in target_keys.items():
        if key is None:
            match.unkeyable.add(fight_id)
            match.twins[fight_id] = []
        elif key_count[key] > 1:
            match.duplicate_targets.add(fight_id)
            match.twins[fight_id] = []
        else:
            match.twins[fight_id] = list(by_key.get(key, []))
    return match


def map_fighters_by_name(target: FightSides, twin: FightSides) -> dict[int, int] | None:
    """``{twin fighter_id: target fighter_id}`` by canonical name, or ``None``.

    ``None`` unless the twin's two canonical names are distinct and each equals
    exactly one of the target's two (distinct) canonical names.
    """
    t_names = {
        canonical_name(target.fighter_a_name): target.fighter_a_id,
        canonical_name(target.fighter_b_name): target.fighter_b_id,
    }
    tw = (
        (twin.fighter_a_id, canonical_name(twin.fighter_a_name)),
        (twin.fighter_b_id, canonical_name(twin.fighter_b_name)),
    )
    if len(t_names) != 2 or tw[0][1] == tw[1][1] or twin.fighter_a_id == twin.fighter_b_id:
        return None
    out: dict[int, int] = {}
    for fid, name in tw:
        if name not in t_names:
            return None
        out[fid] = t_names[name]
    return out


def load_fight_sides(session: Session, sources: Sequence[str]) -> list[FightSides]:
    """Every fight whose EVENT source is in ``sources``, ordered by fight id."""
    fa = aliased(Fighter)
    fb = aliased(Fighter)
    stmt = (
        select(
            Fight.id,
            Event.source,
            Event.date,
            Fight.fighter_a_id,
            fa.name,
            Fight.fighter_b_id,
            fb.name,
        )
        .join(Event, Event.id == Fight.event_id)
        .join(fa, fa.id == Fight.fighter_a_id)
        .join(fb, fb.id == Fight.fighter_b_id)
        .where(Event.source.in_(list(sources)))
        .order_by(Fight.id)
    )
    return [
        FightSides(
            fight_id=r[0],
            source=r[1],
            event_date=r[2],
            fighter_a_id=r[3],
            fighter_a_name=r[4],
            fighter_b_id=r[5],
            fighter_b_name=r[6],
        )
        for r in session.execute(stmt).all()
    ]
