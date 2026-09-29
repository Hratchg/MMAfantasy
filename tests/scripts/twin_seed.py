"""Seed helpers: a ufcstats fight plus its kaggle "twin" rows, for the
orientation-repair and odds-backfill script tests.

Kaggle ingest convention (data/ingest_rajeevw.py, data/ingest_mdabbert.py):
``fighter_a`` = RED corner, ``fighter_b`` = BLUE corner. Only rajeevw rows
carry round-0 stats. The ufcstats convention: ``fighter_a`` = the winner (the
event page lists the winner first).
"""

from __future__ import annotations

from datetime import date
from itertools import count
from typing import Any

from sqlalchemy.orm import Session

from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fight_odds import FightOdds
from ufc_prediction.models.fighter import Fighter
from ufc_prediction.models.round_stats import RoundStats

_seq = count(1)

# (sig_landed, sig_att, td_landed, td_att, sub_att, rev, ctrl, kd, head, body, leg,
#  distance, clinch, ground) — order of the repair script's STAT_FIELDS.
STAT_COLUMNS: tuple[str, ...] = (
    "sig_strikes_landed",
    "sig_strikes_attempted",
    "takedowns_landed",
    "takedowns_attempted",
    "submission_attempts",
    "reversals",
    "control_time_seconds",
    "knockdowns",
    "head_strikes_landed",
    "body_strikes_landed",
    "leg_strikes_landed",
    "distance_strikes_landed",
    "clinch_strikes_landed",
    "ground_strikes_landed",
)

Stats = tuple[int | None, ...]

RED_STATS: Stats = (30, 60, 2, 4, 1, 0, 120, 1, 20, 5, 5, 25, 3, 2)
BLUE_STATS: Stats = (12, 40, 0, 3, 0, 0, 15, 0, 8, 2, 2, 11, 1, 0)
NO_STATS: Stats = (None,) * 14


def _event(session: Session, source: str, d: date) -> Event:
    ev = Event(name=f"{source} {d} #{next(_seq)}", date=d, source=source)
    session.add(ev)
    session.flush()
    return ev


def _fighter(session: Session, name: str, source: str) -> Fighter:
    f = Fighter(name=name, source=source, source_id=f"{next(_seq):016x}")
    session.add(f)
    session.flush()
    return f


def add_fight(
    session: Session,
    source: str,
    d: date,
    a_name: str,
    b_name: str,
    *,
    stats_a: Stats | None = None,
    stats_b: Stats | None = None,
    winner_is_a: bool = True,
    extra_rounds: int = 0,
) -> Fight:
    """One fight (own event + fighters) with optional round-0 stats.

    ``extra_rounds`` adds per-round rows (round 1..n) mirroring round 0, as the
    ufcstats scraper writes them.
    """
    ev = _event(session, source, d)
    fa, fb = _fighter(session, a_name, source), _fighter(session, b_name, source)
    fight = Fight(
        event_id=ev.id,
        fighter_a_id=fa.id,
        fighter_b_id=fb.id,
        winner_id=fa.id if winner_is_a else None,
        weight_class="Lightweight",
        source=source,
        source_url=(
            f"http://ufcstats.com/fight-details/{next(_seq):016x}" if source == "ufcstats" else None
        ),
    )
    session.add(fight)
    session.flush()
    for fighter, stats in ((fa, stats_a), (fb, stats_b)):
        if stats is None:
            continue
        for rnd in range(extra_rounds + 1):
            session.add(
                RoundStats(
                    fight_id=fight.id,
                    fighter_id=fighter.id,
                    round_number=rnd,
                    **dict(zip(STAT_COLUMNS, stats, strict=True)),
                )
            )
    session.flush()
    return fight


def add_odds(session: Session, fight: Fight, fighter_id: int, **values: Any) -> FightOdds:
    row = FightOdds(fight_id=fight.id, fighter_id=fighter_id, **values)
    session.add(row)
    session.flush()
    return row


def round0(session: Session, fight_id: int, fighter_id: int) -> Stats | None:
    rs = (
        session.query(RoundStats)
        .filter(
            RoundStats.fight_id == fight_id,
            RoundStats.fighter_id == fighter_id,
            RoundStats.round_number == 0,
        )
        .one_or_none()
    )
    return None if rs is None else tuple(getattr(rs, c) for c in STAT_COLUMNS)
