#!/usr/bin/env python
"""Split another UFCStats fighter's bouts off a merged ufcstats ``fighters`` row.

Background (S23, 2026-10-01): before ``data/upsert.py::upsert_fighter`` keyed
ufcstats fighters by their UFCStats hex id (``fighters.source_id``), a name
match could attach a different fighter with the same display name to an
existing row. Live case: ``fighters.id = 6121`` "Bruno Silva" (flyweight,
UFCStats ``294aa73dbf37d281``) also carries the 11 Middleweight bouts of the
other UFC "Bruno Silva" (UFCStats ``12ebd7d157e91701``), who has no row; their
fight-detail pages list ``12ebd7d157e91701``. The orientation repair reported
them as ``page_mismatch``.

What it does (report-only unless ``--apply``; ``--apply`` writes everything in
ONE transaction and checks post-conditions before committing):

  1. Target row: reuses the ufcstats row whose ``source_id`` is
     ``--to-source-id``, or creates it (name defaults to the source row's).
  2. For each ``--fight-ids`` fight (a ufcstats fight with ``--from-fighter``
     on one side): re-points ``fighter_a_id`` / ``fighter_b_id`` /
     ``winner_id``, the fight's ``round_stats`` rows and its ``fight_odds``
     rows (composite PK ``fight_id, fighter_id``) from the source fighter to
     the target.
  3. ``--move-sherdog``: moves the source fighter's ``debutant_seed_inputs``
     row and its ``fighters.sherdog_url`` / ``pre_ufc_record`` to the target.
     Use it only when the Sherdog record is the OTHER fighter's; without it
     they stay on the source row and the report lists them.
  4. Profile: fills the target's NULL ``nickname`` / ``height_inches`` /
     ``reach_inches`` / ``leg_reach_inches`` / ``stance`` / ``date_of_birth``
     from ``--profile-from-fighter`` (read only), with ``--set FIELD=VALUE``
     overrides on top. Non-NULL target values are never overwritten.

Never written:
  kaggle rows        a kaggle fight id or a non-ufcstats ``--from-fighter`` is
                     refused; ``--profile-from-fighter`` is only read
  elo_snapshots,     derived; rows of the moved fights still on the source
  computed_features  fighter are counted as stale. Re-run ``ufc elo compute``
                     then ``ufc features compute`` after ``--apply``.
  fighter_aliases    listed for the source fighter, not moved

Refusals (nothing is written, exit 1): unknown / non-ufcstats fight, a fight
without the source fighter on it, the target already on the other side of a
fight, more than one ufcstats row with ``--to-source-id``, an existing
``round_stats`` / ``fight_odds`` / seed / Sherdog value on the target that
would collide, and (with ``--verify-pages``) a cached fight-detail page that
is missing, does not list ``--to-source-id``, or lists the source fighter's id.

Idempotent: a fight already on the target counts as ``already_moved``, moved
rows are no longer on the source fighter, and only NULL profile fields are
filled, so a re-run plans (and writes) nothing.

The ``round_stats`` rows move as stored. If the pre-fix scraper had crossed a
moved fight's rows (S23: 5 of the 11), they are still crossed afterwards. With
the target's ``source_id`` now matching the page, the fight is no longer a
``page_mismatch``, so run ``repair_ufcstats_round_stats_orientation.py`` on
the moved fights next.

After ``--apply``: orientation repair on the moved fights → point the seed's
row in ``data/sherdog/pre_ufc_records.csv`` at the new id (``ufc db
backfill-pre-ufc-seeds`` replays it by ``fighter_id``) → ``ufc elo compute``
(domain Elo reads ``round_stats``) → ``ufc features compute`` → retrain gate →
re-anchor pinned baselines → regenerate the seed dump (CLAUDE.md order).

Banned imports: nothing under ``ufc_prediction.ml.*`` (AUDIT-01).

Usage (S23):
    uv run python scripts/split_merged_fighter.py \\
        --from-fighter 6121 --to-source-id 12ebd7d157e91701 \\
        --fight-ids 14890,15053,15143,15251,15495,15823,15919,16120,16320,16632,16902 \\
        --profile-from-fighter 2997 --set reach_inches=74 --set nickname=Blindado \\
        --move-sherdog --verify-pages data/ufcstats_event_detail_cache/fights \\
        --report split.json
    (same with --apply)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy import func, null, select, text, update
from sqlalchemy.orm import Session

from ufc_prediction.models.computed_feature import ComputedFeature
from ufc_prediction.models.debutant_seed_input import DebutantSeedInput
from ufc_prediction.models.elo_snapshot import EloSnapshot
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fight_odds import FightOdds
from ufc_prediction.models.fighter import Fighter, FighterAlias
from ufc_prediction.models.round_stats import RoundStats

logger = logging.getLogger(__name__)

SOURCE = "ufcstats"

FIGHT_COLUMNS: tuple[str, ...] = ("fighter_a_id", "fighter_b_id", "winner_id")

# Profile fields the target may receive, with the --set parser for each.
PROFILE_FIELDS: dict[str, Callable[[str], Any]] = {
    "nickname": str,
    "height_inches": float,
    "reach_inches": float,
    "leg_reach_inches": float,
    "stance": str,
    "date_of_birth": date.fromisoformat,
}

SHERDOG_COLUMNS: tuple[str, ...] = ("sherdog_url", "pre_ufc_record")

_FIGHTER_LINK = re.compile(r"fighter-details/([0-9a-f]+)")


@dataclass(frozen=True)
class SplitSpec:
    from_fighter_id: int
    to_source_id: str
    fight_ids: tuple[int, ...]
    profile_from_fighter_id: int | None = None
    profile_overrides: dict[str, Any] = field(default_factory=dict)
    move_sherdog: bool = False
    page_cache_dir: Path | None = None
    to_name: str | None = None


@dataclass
class SplitReport:
    applied: bool = False
    from_fighter: dict[str, Any] = field(default_factory=dict)
    to_fighter: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    fights: list[dict[str, Any]] = field(default_factory=list)
    already_moved: list[int] = field(default_factory=list)
    sherdog: dict[str, Any] = field(default_factory=dict)
    derived_stale: dict[str, int] = field(default_factory=dict)
    aliases_on_source: list[str] = field(default_factory=list)
    pages_checked: int = 0
    counts: Counter[str] = field(default_factory=Counter)
    written: Counter[str] = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "from_fighter": self.from_fighter,
            "to_fighter": self.to_fighter,
            "profile": self.profile,
            "fights": self.fights,
            "already_moved": self.already_moved,
            "sherdog": self.sherdog,
            "derived_stale": self.derived_stale,
            "aliases_on_source": self.aliases_on_source,
            "pages_checked": self.pages_checked,
            "counts": dict(self.counts),
            "written": dict(self.written),
            "errors": self.errors,
        }


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """``["reach_inches=74", ...]`` → ``{"reach_inches": 74.0, ...}`` (typed)."""
    out: dict[str, Any] = {}
    for item in items:
        name, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects FIELD=VALUE, got {item!r}")
        name = name.strip()
        if name not in PROFILE_FIELDS:
            raise ValueError(f"--set: {name!r} is not one of {sorted(PROFILE_FIELDS)}")
        out[name] = PROFILE_FIELDS[name](raw.strip())
    return out


def _fighter_summary(f: Fighter) -> dict[str, Any]:
    return {"id": f.id, "name": f.name, "source": f.source, "source_id": f.source_id}


def _page_error(cache_dir: Path, fight: Fight, to_sid: str, from_sid: str | None) -> str | None:
    """Why the cached fight-detail page does not confirm the move, or None."""
    slug = (fight.source_url or "").rstrip("/").rsplit("/", 1)[-1]
    if not slug:
        return f"fight {fight.id}: no source_url to locate its cached page"
    path = cache_dir / f"{slug}.html"
    if not path.is_file():
        return f"fight {fight.id}: no cached page at {path}"
    ids = set(_FIGHTER_LINK.findall(path.read_text(encoding="utf-8", errors="replace")))
    if to_sid not in ids:
        return f"fight {fight.id}: page {path.name} does not list fighter-details/{to_sid}"
    if from_sid is not None and from_sid in ids:
        return f"fight {fight.id}: page {path.name} lists the source fighter {from_sid}"
    return None


def _plan_profile(
    session: Session, spec: SplitSpec, target: Fighter | None, report: SplitReport
) -> dict[str, Any]:
    """Fields to fill on the target (only its NULL fields)."""
    candidate: dict[str, Any] = {}
    source: str | None = None
    if spec.profile_from_fighter_id is not None:
        donor = session.get(Fighter, spec.profile_from_fighter_id)
        if donor is None:
            report.errors.append(
                f"profile source fighters.id={spec.profile_from_fighter_id} not found"
            )
        else:
            source = f"fighters.id={donor.id} ({donor.source})"
            candidate = {k: getattr(donor, k) for k in PROFILE_FIELDS}
    candidate.update(spec.profile_overrides)
    fill: dict[str, Any] = {}
    kept: dict[str, Any] = {}
    for name, value in candidate.items():
        if value is None:
            continue
        current = getattr(target, name) if target is not None else None
        if current is None:
            fill[name] = value
        elif current != value:
            kept[name] = current
    report.profile = {
        "source": source,
        "overrides": dict(spec.profile_overrides),
        "fill": fill,
        "kept_existing": kept,
    }
    report.counts["profile_fields"] = len(fill)
    return fill


def _plan_sherdog(
    session: Session, spec: SplitSpec, src: Fighter, target: Fighter | None, report: SplitReport
) -> tuple[bool, dict[str, Any]]:
    """(move the seed row?, fighters columns to move)."""
    seed = session.get(DebutantSeedInput, src.id)
    target_seed = session.get(DebutantSeedInput, target.id) if target is not None else None
    info: dict[str, Any] = {
        "seed_sherdog_url": seed.sherdog_url if seed is not None else None,
        "fighter_sherdog_url": src.sherdog_url,
        "target_seed_sherdog_url": target_seed.sherdog_url if target_seed is not None else None,
    }
    move_seed = False
    columns: dict[str, Any] = {}
    if not spec.move_sherdog:
        on_source = (
            seed is not None or src.sherdog_url is not None or src.pre_ufc_record is not None
        )
        info["action"] = "left_on_source" if on_source else "none"
        report.sherdog = info
        return move_seed, columns

    if seed is not None:
        if target_seed is not None:
            report.errors.append(
                f"debutant_seed_inputs conflict: both fighters.id={src.id} and the target "
                f"(id={target.id if target else None}) have a seed row"
            )
        else:
            move_seed = True
    for col in SHERDOG_COLUMNS:
        value = getattr(src, col)
        if value is None:
            continue
        current = getattr(target, col) if target is not None else None
        if current is not None and current != value:
            report.errors.append(
                f"fighters.{col} conflict: target already has a different value ({current!r:.80})"
            )
            continue
        columns[col] = value
    if move_seed or columns:
        info["action"] = "move"
    elif target_seed is not None or (target is not None and target.sherdog_url is not None):
        info["action"] = "already_moved"
    else:
        info["action"] = "none"
    info["columns"] = sorted(columns)
    report.sherdog = info
    report.counts["debutant_seed_inputs"] = int(move_seed)
    report.counts["fighter_sherdog_columns"] = len(columns)
    return move_seed, columns


def split(session: Session, spec: SplitSpec, *, apply: bool) -> SplitReport:
    """Plan the split; with ``apply`` (and no errors) write it in one transaction."""
    report = SplitReport()
    src = session.get(Fighter, spec.from_fighter_id)
    if src is None:
        report.errors.append(f"fighters.id={spec.from_fighter_id} not found")
        return report
    report.from_fighter = _fighter_summary(src)
    if src.source != SOURCE:
        report.errors.append(
            f"fighters.id={src.id} is not a ufcstats fighter (source={src.source})"
        )
    if src.source_id == spec.to_source_id:
        report.errors.append(f"fighters.id={src.id} already has source_id {spec.to_source_id}")

    targets = list(
        session.execute(
            select(Fighter).where(Fighter.source == SOURCE, Fighter.source_id == spec.to_source_id)
        ).scalars()
    )
    if len(targets) > 1:
        report.errors.append(
            f"{len(targets)} ufcstats rows have source_id {spec.to_source_id}: "
            f"{sorted(t.id for t in targets)}"
        )
    target = targets[0] if len(targets) == 1 else None
    target_id = target.id if target is not None else None
    report.to_fighter = {
        "id": target_id,
        "action": "reuse" if target is not None else "create",
        "name": target.name if target is not None else (spec.to_name or src.name),
        "source_id": spec.to_source_id,
    }

    # ── fights ───────────────────────────────────────────────────────────────
    wanted = list(dict.fromkeys(spec.fight_ids))
    rows = {
        f.id: (f, d)
        for f, d in session.execute(
            select(Fight, Event.date)
            .join(Event, Event.id == Fight.event_id)
            .where(Fight.id.in_(wanted))
        ).all()
    }
    to_move: list[Fight] = []
    checkable: list[Fight] = []
    for fid in wanted:
        if fid not in rows:
            report.errors.append(f"fight {fid}: not found")
            continue
        fight, event_date = rows[fid]
        if fight.source != SOURCE:
            report.errors.append(f"fight {fid}: not a ufcstats fight (source={fight.source})")
            continue
        sides = (fight.fighter_a_id, fight.fighter_b_id)
        cols = [c for c in FIGHT_COLUMNS if getattr(fight, c) == src.id]
        if not cols:
            if target_id is not None and target_id in sides:
                report.already_moved.append(fid)
                checkable.append(fight)
            else:
                report.errors.append(f"fight {fid}: neither fighter is fighters.id={src.id}")
            continue
        if target_id is not None and target_id in sides:
            report.errors.append(f"fight {fid}: target fighters.id={target_id} is the opponent")
            continue
        to_move.append(fight)
        checkable.append(fight)
        report.fights.append(
            {
                "fight_id": fid,
                "event_date": event_date.isoformat(),
                "weight_class": fight.weight_class,
                "columns": cols,
            }
        )
        report.counts["fight_columns"] += len(cols)
    report.counts["fights"] = len(to_move)
    report.counts["already_moved_fights"] = len(report.already_moved)
    move_ids = [f.id for f in to_move]

    if spec.page_cache_dir is not None:
        for fight in checkable:
            err = _page_error(spec.page_cache_dir, fight, spec.to_source_id, src.source_id)
            if err is not None:
                report.errors.append(err)
        report.pages_checked = len(checkable)

    # ── round_stats / fight_odds ─────────────────────────────────────────────
    report.counts["round_stats"] = _count(session, RoundStats, move_ids, src.id)
    report.counts["fight_odds"] = _count(session, FightOdds, move_ids, src.id)
    if target_id is not None:
        for model, label in ((RoundStats, "round_stats"), (FightOdds, "fight_odds")):
            clash = session.execute(
                select(model.fight_id)
                .where(model.fight_id.in_(move_ids), model.fighter_id == target_id)
                .distinct()
            ).scalars()
            for fid in sorted(clash):
                report.errors.append(
                    f"{label} conflict: fight {fid} already has rows for target fighters.id={target_id}"
                )

    # ── Sherdog seed / profile / derived / aliases ───────────────────────────
    move_seed, sherdog_cols = _plan_sherdog(session, spec, src, target, report)
    fill = _plan_profile(session, spec, target, report)
    done_ids = move_ids + report.already_moved
    report.derived_stale = {
        "elo_snapshots": _count(session, EloSnapshot, done_ids, src.id),
        "computed_features": _count(session, ComputedFeature, done_ids, src.id),
    }
    report.aliases_on_source = list(
        session.execute(
            select(FighterAlias.alias_name).where(FighterAlias.fighter_id == src.id)
        ).scalars()
    )

    if not apply or report.errors:
        return report
    try:
        _write(session, spec, report, src, target, to_move, move_seed, sherdog_cols, fill)
    except BaseException:
        session.rollback()
        raise
    report.applied = True
    return report


def _count(session: Session, model: Any, fight_ids: list[int], fighter_id: int) -> int:
    if not fight_ids:
        return 0
    return int(
        session.execute(
            select(func.count())
            .select_from(model)
            .where(model.fight_id.in_(fight_ids), model.fighter_id == fighter_id)
        ).scalar_one()
    )


def _write(
    session: Session,
    spec: SplitSpec,
    report: SplitReport,
    src: Fighter,
    target: Fighter | None,
    to_move: list[Fight],
    move_seed: bool,
    sherdog_cols: dict[str, Any],
    fill: dict[str, Any],
) -> None:
    written = report.written
    if target is None:
        target = Fighter(
            name=report.to_fighter["name"], source=SOURCE, source_id=spec.to_source_id, **fill
        )
        session.add(target)
        session.flush()
        written["fighters_created"] = 1
        report.to_fighter["id"] = target.id
    else:
        for name, value in fill.items():
            setattr(target, name, value)
        session.flush()
        written["fighters_created"] = 0
    written["profile_fields"] = len(fill)
    tid, sid = target.id, src.id
    move_ids = [f.id for f in to_move]

    written["fight_columns"] = 0
    if move_ids:
        for col in FIGHT_COLUMNS:
            res = session.execute(
                update(Fight)
                .where(Fight.id.in_(move_ids), getattr(Fight, col) == sid)
                .values({col: tid})
                .execution_options(synchronize_session=False)
            )
            written["fight_columns"] += int(res.rowcount)  # type: ignore[attr-defined]
    written["fights"] = len(move_ids)
    for model, key in ((RoundStats, "round_stats"), (FightOdds, "fight_odds")):
        written[key] = 0
        if move_ids:
            res = session.execute(
                update(model)
                .where(model.fight_id.in_(move_ids), model.fighter_id == sid)
                .values(fighter_id=tid)
                .execution_options(synchronize_session=False)
            )
            written[key] = int(res.rowcount)  # type: ignore[attr-defined]
    written["debutant_seed_inputs"] = 0
    if move_seed:
        res = session.execute(
            update(DebutantSeedInput)
            .where(DebutantSeedInput.fighter_id == sid)
            .values(fighter_id=tid)
            .execution_options(synchronize_session=False)
        )
        written["debutant_seed_inputs"] = int(res.rowcount)  # type: ignore[attr-defined]
    written["fighter_sherdog_columns"] = len(sherdog_cols)
    if sherdog_cols:
        # null() writes SQL NULL; a bare None would store JSON 'null' in pre_ufc_record.
        session.execute(
            update(Fighter)
            .where(Fighter.id == sid)
            .values({c: null() for c in sherdog_cols})
            .execution_options(synchronize_session=False)
        )
        session.execute(
            update(Fighter)
            .where(Fighter.id == tid)
            .values(sherdog_cols)
            .execution_options(synchronize_session=False)
        )

    for key in ("fight_columns", "round_stats", "fight_odds", "debutant_seed_inputs"):
        if written[key] != report.counts[key]:
            raise RuntimeError(f"{key}: wrote {written[key]} rows, planned {report.counts[key]}")
    left = {
        "fights": session.execute(
            select(func.count())
            .select_from(Fight)
            .where(
                Fight.id.in_(move_ids or [-1]),
                (Fight.fighter_a_id == sid)
                | (Fight.fighter_b_id == sid)
                | (Fight.winner_id == sid),
            )
        ).scalar_one(),
        "round_stats": _count(session, RoundStats, move_ids, sid),
        "fight_odds": _count(session, FightOdds, move_ids, sid),
    }
    if any(left.values()):
        raise RuntimeError(f"rows still on fighters.id={sid} after the move: {left}")
    session.commit()
    session.expire_all()


def _print_report(report: SplitReport) -> None:
    p = "[split]"
    verb = "APPLIED" if report.applied else "planned (dry run, nothing written)"
    print(f"{p} {verb}")
    fr, to = report.from_fighter, report.to_fighter
    if fr:
        print(
            f"{p} source: fighters.id={fr['id']} {fr['name']!r} ({fr['source']} {fr['source_id']})"
        )
    if to:
        print(
            f"{p} fighters: {to['action'].upper()} ufcstats {to['name']!r} "
            f"source_id={to['source_id']} (id={to['id']})"
        )
    if report.profile:
        print(
            f"{p}   profile source={report.profile['source']} "
            f"overrides={report.profile['overrides']} fill={report.profile['fill']}"
        )
        if report.profile["kept_existing"]:
            print(f"{p}   kept existing target values: {report.profile['kept_existing']}")
    c = report.counts
    print(
        f"{p} fights: {c['fights']} to move ({c['fight_columns']} column updates), "
        f"{c['already_moved_fights']} already moved"
    )
    for f in report.fights:
        print(
            f"{p}   fight {f['fight_id']} {f['event_date']} {f['weight_class']}: "
            f"{', '.join(f['columns'])}"
        )
    print(f"{p} round_stats: {c['round_stats']} rows")
    print(f"{p} fight_odds: {c['fight_odds']} rows")
    if report.sherdog:
        s = report.sherdog
        print(
            f"{p} debutant_seed_inputs + fighters.sherdog_url/pre_ufc_record: {s['action']} "
            f"(seed rows {c['debutant_seed_inputs']}, fighter columns {s.get('columns', [])}; "
            f"seed url={s['seed_sherdog_url']})"
        )
    print(
        f"{p} derived, NOT edited (re-run `ufc elo compute` then `ufc features compute`): "
        f"{report.derived_stale}"
    )
    print(f"{p} fighter_aliases on source (not moved): {report.aliases_on_source or 'none'}")
    if report.pages_checked:
        print(f"{p} cached fight pages checked: {report.pages_checked}")
    if report.applied:
        print(f"{p} written: {dict(report.written)}")
    for e in report.errors:
        print(f"{p} ERROR {e}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--from-fighter", type=int, required=True, help="Merged fighters.id.")
    parser.add_argument(
        "--to-source-id", required=True, help="UFCStats hex id of the fighter to split off."
    )
    parser.add_argument(
        "--fight-ids", required=True, help="Comma-separated fights.id list to move (ufcstats only)."
    )
    parser.add_argument("--to-name", default=None, help="Name for a new target row.")
    parser.add_argument(
        "--profile-from-fighter",
        type=int,
        default=None,
        help="Read profile fields from this fighters.id (never written).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help=f"Profile override; FIELD in {sorted(PROFILE_FIELDS)}. Repeatable.",
    )
    parser.add_argument(
        "--move-sherdog",
        action="store_true",
        help="Move the seed row and fighters.sherdog_url/pre_ufc_record to the target.",
    )
    parser.add_argument(
        "--verify-pages",
        type=Path,
        default=None,
        help="Cached fight-detail page dir; each page must list --to-source-id.",
    )
    parser.add_argument(
        "--apply", action="store_true", help="Write (one transaction). Without it, report only."
    )
    parser.add_argument("--report", type=Path, default=None, help="Write a JSON report here.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[split] %(message)s")
    spec = SplitSpec(
        from_fighter_id=args.from_fighter,
        to_source_id=args.to_source_id.strip(),
        fight_ids=tuple(int(x) for x in args.fight_ids.split(",") if x.strip()),
        profile_from_fighter_id=args.profile_from_fighter,
        profile_overrides=parse_overrides(args.overrides),
        move_sherdog=args.move_sherdog,
        page_cache_dir=args.verify_pages,
        to_name=args.to_name,
    )

    from ufc_prediction.db.session import SessionLocal

    session = SessionLocal()
    try:
        if not args.apply:
            # Belt and braces: a dry run cannot write even by accident.
            session.execute(text("SET TRANSACTION READ ONLY"))
        report = split(session, spec, apply=args.apply)
        _print_report(report)
        if args.report is not None:
            args.report.write_text(
                json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8"
            )
            print(f"[split] report written to {args.report}")
        return 1 if report.errors else 0
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
