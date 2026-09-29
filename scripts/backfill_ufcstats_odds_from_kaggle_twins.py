#!/usr/bin/env python
"""Copy orphaned BFO odds from kaggle twin fights onto their ufcstats fight.

Background (PR #18, fix/bfo-odds-source-scoping): ``BFOOddsIngester`` matched
BFO names against fighters from EVERY source and resolved fights without a
source filter, so for ~245 bouts the odds landed on the kaggle-rajeevw /
kaggle-mdabbert duplicate ("twin") of the fight instead of the ufcstats fight
— the only lineage training (``load_fight_records``) and serving read. Those
ufcstats fights have no ``fight_odds`` at all while their twin has a pair.

This copies the twin's rows onto the ufcstats fight WITHOUT re-scraping
(it never touches BFO; in particular not the BFO event-URL name search, which
fuzzy-matches wrong older events):

  1. Orphans: ufcstats fights with no ``fight_odds`` row.
  2. Twins: kaggle fights on the same event date with the same two fighters
     by canonical name (``ufc_prediction.dedup.twins``).
  3. Each odds row's kaggle ``fighter_id`` is re-keyed to the ufcstats fighter
     with the same canonical name WITHIN THAT FIGHT.

Refusals (nothing written for the fight, listed in the report):
  ambiguous_twin            two twins from the same kaggle source, a target
                            that is not uniquely keyable, or names that do not
                            pair up one-to-one
  odds_fighter_not_in_twin  an odds row whose fighter is not one of the twin
                            fight's two fighters
  conflicting_twin_odds     rajeevw and mdabbert twins both carry odds and the
                            re-keyed rows differ

Report-only by default; ``--apply`` inserts with ON CONFLICT (fight_id,
fighter_id) DO NOTHING, so re-runs are no-ops (a backfilled fight is no longer
an orphan). Copied rows keep the twin row's values, ``source`` and
``scraped_at``.

After ``--apply``: the odds features read ``fight_odds`` → ``features
compute`` → retrain gate → re-anchor pinned baselines → regenerate the seed
dump (CLAUDE.md substrate order).

Banned imports: nothing under ``ufc_prediction.ml.*`` (AUDIT-01).

Usage:
    uv run python scripts/backfill_ufcstats_odds_from_kaggle_twins.py --report out.json
    uv run python scripts/backfill_ufcstats_odds_from_kaggle_twins.py --apply --report out.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ufc_prediction.dedup.source_priority import SOURCE_PRIORITY
from ufc_prediction.dedup.twins import (
    KAGGLE_SOURCES,
    FightSides,
    find_twins,
    load_fight_sides,
    map_fighters_by_name,
)
from ufc_prediction.models.fight_odds import FightOdds

logger = logging.getLogger(__name__)

SOURCE = "ufcstats"

# Copied verbatim from the twin row (fight_id / fighter_id are re-keyed).
VALUE_COLUMNS: tuple[str, ...] = (
    "opening_ml",
    "closing_range_min_ml",
    "closing_range_max_ml",
    "opening_implied_prob",
    "closing_implied_prob",
    "source",
    "scraped_at",
)
# What must agree when two twins both carry odds (provenance may differ).
_COMPARE_COLUMNS: tuple[str, ...] = VALUE_COLUMNS[:5]


@dataclass
class BackfillReport:
    counts: Counter[str] = field(default_factory=Counter)
    backfilled: list[dict[str, Any]] = field(default_factory=list)
    refusals: list[dict[str, Any]] = field(default_factory=list)

    def refuse(self, fight_id: int, reason: str) -> None:
        self.counts[f"refused:{reason}"] += 1
        self.refusals.append({"fight_id": fight_id, "reason": reason})

    def as_dict(self, *, applied: bool) -> dict[str, Any]:
        return {
            "applied": applied,
            "counts": dict(self.counts),
            "backfilled": self.backfilled,
            "refusals": self.refusals,
        }


class _RefusalError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _row_values(row: FightOdds) -> dict[str, Any]:
    return {c: getattr(row, c) for c in VALUE_COLUMNS}


def rekey_twin_odds(
    target: FightSides, twin: FightSides, rows: list[FightOdds]
) -> dict[int, dict[str, Any]]:
    """``{ufcstats fighter_id: values}`` for one twin's odds rows, or raise ``_RefusalError``."""
    mapping = map_fighters_by_name(target, twin)
    if mapping is None:
        raise _RefusalError("ambiguous_twin")
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.fighter_id not in mapping:
            raise _RefusalError("odds_fighter_not_in_twin")
        out[mapping[row.fighter_id]] = _row_values(row)
    return out


def plan_fight(
    target: FightSides, twins_with_odds: list[FightSides], odds: dict[int, list[FightOdds]]
) -> tuple[FightSides, list[dict[str, Any]]]:
    """The twin used and the rows to insert for one orphan, or raise ``_RefusalError``."""
    per_source = Counter(t.source for t in twins_with_odds)
    if any(n > 1 for n in per_source.values()):
        raise _RefusalError("ambiguous_twin")
    ranked = sorted(twins_with_odds, key=lambda t: SOURCE_PRIORITY.get(t.source, 99))
    rekeyed = [(t, rekey_twin_odds(target, t, odds[t.fight_id])) for t in ranked]
    chosen, values = rekeyed[0]
    for _, other in rekeyed[1:]:
        if {k: [v[c] for c in _COMPARE_COLUMNS] for k, v in other.items()} != {
            k: [v[c] for c in _COMPARE_COLUMNS] for k, v in values.items()
        }:
            raise _RefusalError("conflicting_twin_odds")
    rows = [
        {"fight_id": target.fight_id, "fighter_id": fid, **vals}
        for fid, vals in sorted(values.items())
    ]
    return chosen, rows


def insert_odds(session: Session, rows: list[dict[str, Any]]) -> int:
    """Insert ``rows``; existing (fight_id, fighter_id) keys are left untouched."""
    if not rows:
        return 0
    stmt = (
        insert(FightOdds)
        .values(rows)
        .on_conflict_do_nothing(index_elements=["fight_id", "fighter_id"])
        .returning(FightOdds.fight_id)
    )
    # RETURNING yields only the rows actually inserted (rowcount is -1 for a
    # multi-VALUES insert under psycopg).
    return len(session.execute(stmt).all())


def backfill(session: Session, *, apply: bool) -> BackfillReport:
    """Plan (and with ``apply``, write) odds for every orphaned ufcstats fight."""
    report = BackfillReport()
    ufc_fights = load_fight_sides(session, [SOURCE])
    kaggle = load_fight_sides(session, KAGGLE_SOURCES)
    # Match against ALL ufcstats fights so a key shared with a non-orphan
    # ufcstats fight is still flagged as a duplicate target.
    match = find_twins(ufc_fights, kaggle)

    fights_with_odds = set(session.execute(select(FightOdds.fight_id).distinct()).scalars())
    kaggle_ids = [k.fight_id for k in kaggle if k.fight_id in fights_with_odds]
    odds: dict[int, list[FightOdds]] = defaultdict(list)
    if kaggle_ids:
        for row in session.execute(
            select(FightOdds).where(FightOdds.fight_id.in_(kaggle_ids))
        ).scalars():
            odds[row.fight_id].append(row)

    planned: list[dict[str, Any]] = []
    for target in ufc_fights:
        if target.fight_id in fights_with_odds:
            continue
        report.counts["orphans"] += 1
        if target.fight_id in match.unkeyable or target.fight_id in match.duplicate_targets:
            report.refuse(target.fight_id, "ambiguous_twin")
            continue
        twins = match.twins[target.fight_id]
        if not twins:
            report.counts["no_twin"] += 1
            continue
        with_odds = [t for t in twins if t.fight_id in odds]
        if not with_odds:
            report.counts["twin_without_odds"] += 1
            continue
        try:
            chosen, rows = plan_fight(target, with_odds, odds)
        except _RefusalError as exc:
            report.refuse(target.fight_id, exc.reason)
            continue
        report.counts["backfillable_fights"] += 1
        report.counts["rows_planned"] += len(rows)
        report.backfilled.append(
            {
                "fight_id": target.fight_id,
                "kaggle_fight_id": chosen.fight_id,
                "kaggle_source": chosen.source,
            }
        )
        planned.extend(rows)

    if apply and planned:
        report.counts["rows_inserted"] = insert_odds(session, planned)
        session.commit()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--apply", action="store_true", help="Insert the rows. Without it, report only."
    )
    parser.add_argument("--report", type=Path, default=None, help="Write a JSON report here.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[odds-backfill] %(message)s")

    from ufc_prediction.db.session import SessionLocal

    session = SessionLocal()
    try:
        report = backfill(session, apply=args.apply)
        summary = report.as_dict(applied=args.apply)
        print(f"[odds-backfill] counts: {summary['counts']} (applied={args.apply})")
        for r in report.refusals:
            print(f"[odds-backfill] REFUSED fight {r['fight_id']}: {r['reason']}")
        if args.report is not None:
            args.report.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
            print(f"[odds-backfill] report written to {args.report}")
        return 0
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
