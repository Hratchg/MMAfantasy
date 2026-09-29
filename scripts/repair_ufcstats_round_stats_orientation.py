#!/usr/bin/env python
"""One-off repair: swap mis-oriented ufcstats ``round_stats`` rows in place.

Background (code review 2026-09-28, fix/scraper-fighter-order): before the
fix, ``scraper/ingest.py::_scrape_event`` assigned fight-detail stats by
POSITION — ``totals[0]`` / ``per_round_*[i][0]`` went to the EVENT page's
fighter_a. The event page lists the winner first; the fight-detail page uses
its own (corner) order. For roughly a third of ufcstats fights every
``round_stats`` row (round 0 and each round) was therefore stored under the
OPPONENT's ``fighter_id``. The fights table itself (fighter_a/b, winner) is
correct; only ``round_stats.fighter_id`` is crossed.

What this does, per ufcstats fight:
  1. Cache-first fetch of the fight-detail page (``fights.source_url``); the
     cache is ``data/ufcstats_event_detail_cache/fights/<slug>.html``, shared
     with ``scrape_referees_full.py``, so an interrupted run resumes cheaply.
  2. Parse it and orient it to the DB fighters by UFCStats hex id
     (``fighters.source_id``) with the same ``_orient_fight_detail`` the fixed
     scraper uses, giving the true per-fighter stats.
  3. Compare to the stored rows (every round, every stat column):
       correct    — stored rows match the page for each fighter
       swapped    — stored rows match the page with the fighters crossed
                    → with --apply, swap ``fighter_id`` between the two
                    fighters on every ``round_stats`` row of the fight
       unmatched  — neither (ufcstats corrected the page since the scrape,
                    or the stored rows are partial); left untouched and
                    listed in the report for a targeted re-scrape
     plus fetch_failed / parse_failed / page_mismatch / no_stats.

Classification is content-based, so the repair is idempotent: once swapped, a
fight classifies as ``correct`` on any re-run.

After ``--apply``, regenerate the substrate (CLAUDE.md order): ``features
compute`` → retrain gate → re-anchor pinned baselines → regenerate the seed
dump. (Elo reads only results, which this does not touch.)

Banned imports: nothing under ``ufc_prediction.ml.*`` (AUDIT-01).

Usage:
    uv run python scripts/repair_ufcstats_round_stats_orientation.py --dry-run
    uv run python scripts/repair_ufcstats_round_stats_orientation.py --report out.json
    uv run python scripts/repair_ufcstats_round_stats_orientation.py --apply --report out.json
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import case, select, update
from sqlalchemy.orm import Session, aliased

from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter
from ufc_prediction.models.round_stats import RoundStats
from ufc_prediction.scraper.ingest import _build_fight_stats, _orient_fight_detail, _safe_fetch
from ufc_prediction.scraper.models import FightDetailPage
from ufc_prediction.scraper.parse_fight_detail import parse_fight_detail

logger = logging.getLogger(__name__)

SOURCE = "ufcstats"
FIGHT_CACHE_DIR: Path = Path("data/ufcstats_event_detail_cache/fights")
HTTP_DELAY_SECONDS_DEFAULT: float = 1.2
HTTP_TIMEOUT_SECONDS: float = 10.0
HTTP_MAX_RETRIES: int = 2
WORKERS_DEFAULT: int = 4
BATCH_SIZE: int = 50

_CACHE_KEY_RE = re.compile(r"[A-Za-z0-9_-]+")

# Every stat column the scraper writes (ingest._upsert_stats_for_fighter).
STAT_FIELDS: tuple[str, ...] = (
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

StatsByRound = dict[int, tuple[Any, ...]]


@dataclass(frozen=True)
class FightRef:
    fight_id: int
    source_url: str
    fighter_a_id: int
    fighter_b_id: int
    fighter_a_hex: str
    fighter_b_hex: str


@dataclass
class RepairReport:
    counts: Counter[str] = field(default_factory=Counter)
    swapped_fight_ids: list[int] = field(default_factory=list)
    unmatched: list[dict[str, Any]] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self, *, applied: bool) -> dict[str, Any]:
        return {
            "applied": applied,
            "counts": dict(self.counts),
            "swapped_fight_ids": self.swapped_fight_ids,
            "unmatched": self.unmatched,
            "failed": self.failed,
        }


# ── Pure classification ────────────────────────────────────────────────────


def _stat_tuple(obj: object) -> tuple[Any, ...]:
    return tuple(getattr(obj, f) for f in STAT_FIELDS)


def expected_stats(detail: FightDetailPage) -> tuple[StatsByRound, StatsByRound]:
    """Per-round stat tuples for (fighter_a, fighter_b) of an ORIENTED page.

    Mirrors what the fixed scraper writes: round 0 from totals + summed sig
    strikes, rounds 1..n from the per-round tables.
    """
    a: StatsByRound = {0: _stat_tuple(_build_fight_stats(detail.totals[0], detail.sig_strikes[0]))}
    b: StatsByRound = {0: _stat_tuple(_build_fight_stats(detail.totals[1], detail.sig_strikes[1]))}
    num_rounds = min(len(detail.per_round_totals), len(detail.per_round_sig_strikes))
    for idx in range(num_rounds):
        tot_a, tot_b = detail.per_round_totals[idx]
        sig_a, sig_b = detail.per_round_sig_strikes[idx]
        a[idx + 1] = _stat_tuple(_build_fight_stats(tot_a, sig_a))
        b[idx + 1] = _stat_tuple(_build_fight_stats(tot_b, sig_b))
    return a, b


def classify(
    stored_a: StatsByRound,
    stored_b: StatsByRound,
    page_a: StatsByRound,
    page_b: StatsByRound,
) -> str:
    """Return ``correct`` / ``swapped`` / ``unmatched`` / ``no_stats``."""
    if not stored_a and not stored_b:
        return "no_stats"
    if stored_a == page_a and stored_b == page_b:
        return "correct"
    if stored_a == page_b and stored_b == page_a:
        return "swapped"
    return "unmatched"


# ── DB access ──────────────────────────────────────────────────────────────


def load_fights(session: Session, limit: int | None = None) -> list[FightRef]:
    fa = aliased(Fighter)
    fb = aliased(Fighter)
    stmt = (
        select(
            Fight.id,
            Fight.source_url,
            Fight.fighter_a_id,
            Fight.fighter_b_id,
            fa.source_id,
            fb.source_id,
        )
        .join(fa, fa.id == Fight.fighter_a_id)
        .join(fb, fb.id == Fight.fighter_b_id)
        .where(Fight.source == SOURCE, Fight.source_url.isnot(None))
        .order_by(Fight.id)
    )
    if limit is not None:
        stmt = stmt.limit(limit)
    return [
        FightRef(
            fight_id=row[0],
            source_url=row[1],
            fighter_a_id=row[2],
            fighter_b_id=row[3],
            fighter_a_hex=row[4] or "",
            fighter_b_hex=row[5] or "",
        )
        for row in session.execute(stmt).all()
    ]


def load_stored_stats(
    session: Session, fight_ids: list[int]
) -> dict[tuple[int, int], StatsByRound]:
    """``{(fight_id, fighter_id): {round_number: stat_tuple}}`` for ``fight_ids``."""
    out: dict[tuple[int, int], StatsByRound] = defaultdict(dict)
    rows = session.execute(select(RoundStats).where(RoundStats.fight_id.in_(fight_ids))).scalars()
    for rs in rows:
        out[(rs.fight_id, rs.fighter_id)][rs.round_number] = _stat_tuple(rs)
    return out


def swap_round_stats(session: Session, fight: FightRef) -> int:
    """Swap ``fighter_id`` between the two fighters on every row of ``fight``."""
    a, b = fight.fighter_a_id, fight.fighter_b_id
    result = session.execute(
        update(RoundStats)
        .where(RoundStats.fight_id == fight.fight_id, RoundStats.fighter_id.in_((a, b)))
        .values(fighter_id=case((RoundStats.fighter_id == a, b), else_=a))
        .execution_options(synchronize_session=False)
    )
    return int(result.rowcount or 0)  # type: ignore[attr-defined]


# ── Fetch (cache-first) ────────────────────────────────────────────────────


def _cache_key(source_url: str) -> str | None:
    key = source_url.rstrip("/").rsplit("/", 1)[-1]
    return key if _CACHE_KEY_RE.fullmatch(key) else None


def fetch_pages(client: Any, fights: list[FightRef], cache_dir: Path) -> dict[int, str | None]:
    """Cache-first fetch of each fight's detail page; misses go through ``client.map``."""
    pages: dict[int, str | None] = {}
    misses: list[FightRef] = []
    for f in fights:
        key = _cache_key(f.source_url)
        path = cache_dir / f"{key}.html" if key else None
        if path is not None and path.exists():
            pages[f.fight_id] = path.read_text(encoding="utf-8")
        else:
            misses.append(f)
    if misses:
        # Per-URL error isolation (same helper the scraper uses): one dead page
        # must not fail the whole batch.
        fetched = client.map(functools.partial(_safe_fetch, client), [f.source_url for f in misses])
        for f, (url, html, err) in zip(misses, fetched, strict=True):
            if html is None:
                logger.warning("fetch failed for %s: %s", url, err)
            pages[f.fight_id] = html or None
            key = _cache_key(f.source_url)
            if html and key:
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    (cache_dir / f"{key}.html").write_text(html, encoding="utf-8")
                except OSError:
                    logger.warning("cache write failed for %s", key)
    return pages


# ── Driver ─────────────────────────────────────────────────────────────────


def repair(
    session: Session,
    client: Any,
    *,
    apply: bool,
    cache_dir: Path = FIGHT_CACHE_DIR,
    limit: int | None = None,
    batch_size: int = BATCH_SIZE,
) -> RepairReport:
    """Classify every ufcstats fight and (with ``apply``) swap the crossed ones."""
    report = RepairReport()
    fights = load_fights(session, limit=limit)
    for start in range(0, len(fights), batch_size):
        batch = fights[start : start + batch_size]
        pages = fetch_pages(client, batch, cache_dir)
        stored = load_stored_stats(session, [f.fight_id for f in batch])
        for f in batch:
            html = pages.get(f.fight_id)
            if html is None:
                report.counts["fetch_failed"] += 1
                report.failed.append({"fight_id": f.fight_id, "reason": "fetch_failed"})
                continue
            try:
                detail = parse_fight_detail(html)
            except (ValueError, RuntimeError) as exc:
                report.counts["parse_failed"] += 1
                report.failed.append(
                    {"fight_id": f.fight_id, "reason": "parse_failed", "error": str(exc)}
                )
                continue
            oriented = _orient_fight_detail(detail, f.fighter_a_hex, f.fighter_b_hex)
            if oriented is None:
                report.counts["page_mismatch"] += 1
                report.failed.append({"fight_id": f.fight_id, "reason": "page_mismatch"})
                continue
            page_a, page_b = expected_stats(oriented)
            status = classify(
                stored.get((f.fight_id, f.fighter_a_id), {}),
                stored.get((f.fight_id, f.fighter_b_id), {}),
                page_a,
                page_b,
            )
            report.counts[status] += 1
            if status == "swapped":
                report.swapped_fight_ids.append(f.fight_id)
                if apply:
                    swap_round_stats(session, f)
            elif status == "unmatched":
                report.unmatched.append({"fight_id": f.fight_id, "source_url": f.source_url})
        if apply:
            session.commit()  # per-batch commit: a crash mid-run keeps finished batches
        logger.info(
            "progress %d/%d fights: %s",
            min(start + batch_size, len(fights)),
            len(fights),
            dict(report.counts),
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No network, no writes: print how many fights would be checked.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Swap the crossed rows. Without it, classify and report only.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=HTTP_DELAY_SECONDS_DEFAULT)
    parser.add_argument("--workers", type=int, default=WORKERS_DEFAULT)
    parser.add_argument("--report", type=Path, default=None, help="Write a JSON report here.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[repair-orientation] %(message)s")

    from ufc_prediction.db.session import SessionLocal

    session = SessionLocal()
    try:
        if args.dry_run:
            fights = load_fights(session, limit=args.limit)
            cached = sum(
                1
                for f in fights
                if (key := _cache_key(f.source_url)) and (FIGHT_CACHE_DIR / f"{key}.html").exists()
            )
            print(f"[repair-orientation] DRY-RUN: {len(fights)} ufcstats fights to check")
            print(f"[repair-orientation] cached pages: {cached}; to fetch: {len(fights) - cached}")
            return 0

        from ufc_prediction.scraper.client import ScraperClient

        with ScraperClient(
            delay=args.delay,
            max_retries=HTTP_MAX_RETRIES,
            timeout=HTTP_TIMEOUT_SECONDS,
            workers=args.workers,
        ) as client:
            report = repair(session, client, apply=args.apply, limit=args.limit)

        summary = report.as_dict(applied=args.apply)
        print(f"[repair-orientation] counts: {summary['counts']} (applied={args.apply})")
        if args.report is not None:
            args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[repair-orientation] report written to {args.report}")
        return 0
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
