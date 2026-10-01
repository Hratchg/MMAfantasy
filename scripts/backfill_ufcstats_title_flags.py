#!/usr/bin/env python
"""Backfill ``fights.is_title_fight`` for existing ufcstats fights.

Background (S22, 2026-09-29): ``scraper/ingest.py::_convert_fight`` inferred
the title flag from the event page's weight cell, whose text is only the
division ("Light Heavyweight"); UFCStats marks a title bout with a
``belt.png`` image instead. Every stored ufcstats fight is therefore a
non-title fight (0 of 8,736 on the live DB), and since training reads only
ufcstats fights (Plan 28-04 dedup) the ``is_title_fight`` feature is constant
0. The scraper is fixed; ``ufc scrape latest`` never revisits stored events,
so this sets the flag on the rows already in the DB.

Evidence for each ufcstats fight (``fights.source`` and ``events.source`` both
ufcstats):

  fight_page  The cached fight-detail page
              ``<fight-cache-dir>/<fight hex>.html`` — the last path segment of
              ``fights.source_url``, the cache key ``scrape_referees_full.py``
              and ``repair_ufcstats_round_stats_orientation.py`` write. Skipped
              (not evidence): anti-bot challenge pages, unparseable pages, and
              pages whose two fighters are not the fight's (UFCStats hex ids).
              Title = ``belt.png`` in the fight-title cell or "Title" in its
              text (``ingest.title_flag_from_fight_page``).
  event_page  The ufcstats event page ``<event-cache-dir>/<event hex>.html``
              (from ``events.source_url``). One page flags every bout on the
              card (``ingest.title_flag_from_event_row``: belt.png in the weight
              cell, the exact signal the fixed scraper stores). Read from the
              cache; with ``--backend browser`` fetched through the headless
              ``BrowserFetcher`` for every event that still has a fight with no
              page and no usable twin (``--recheck-twins``: no page at all),
              and cached (real pages only).
  twin        The same bout in kaggle-rajeevw / kaggle-mdabbert: the same two
              fighters by canonical name (``dedup/twins.py``) on the same date,
              else within +/-1 day. Usable when every twin agrees; two twins
              from one source (or an unkeyable / duplicate target) is
              ``ambiguous``, twins that disagree are ``conflict``.

Decision: a page (fight or event) is authoritative and may set True or False;
if both pages exist and disagree nothing is written (``page_conflict``).
Without a page, a twin True sets True and a twin False writes nothing (only a
page is authoritative for False). Anything else stays as stored (``unknown``).
Page and twin are cross-checked wherever both exist (``cross_check``).

Only rows whose stored value differs are updated, and the UPDATE is guarded by
``fights.source = 'ufcstats'``: kaggle rows are never written. Re-runs are
no-ops. Dry run (the default) opens a READ ONLY transaction and only reports;
``--apply`` writes. The browser backend writes fetched event pages to the event
cache even in a dry run, so the follow-up ``--apply`` needs no network.

After ``--apply``: training reads ``fights.is_title_fight`` directly
(``ml/queries.py::load_fight_records``) and the serve path reads the stored
row too, so no Elo / ``features compute`` recompute is needed for this column;
re-run the retrain gate and re-anchor the pinned baselines, then regenerate the
seed dump (CLAUDE.md substrate order).

Banned imports: nothing under ``ufc_prediction.ml.*`` (AUDIT-01).

Usage:
    uv run python scripts/backfill_ufcstats_title_flags.py --report dry.json
    uv run python scripts/backfill_ufcstats_title_flags.py --backend browser --report browser.json
    uv run python scripts/backfill_ufcstats_title_flags.py --apply --report apply.json
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select, text, update
from sqlalchemy.orm import Session, aliased

from ufc_prediction.dedup.twins import (
    KAGGLE_SOURCES,
    FightSides,
    TwinMatch,
    find_twins,
    load_fight_sides,
)
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter
from ufc_prediction.scraper.antibot import detect_antibot
from ufc_prediction.scraper.ingest import (
    _extract_hex_id,
    _orient_fight_detail,
    _safe_fetch,
    title_flag_from_event_row,
    title_flag_from_fight_page,
)
from ufc_prediction.scraper.models import EventDetail
from ufc_prediction.scraper.parse_event_detail import parse_event_detail
from ufc_prediction.scraper.parse_fight_detail import parse_fight_detail

logger = logging.getLogger(__name__)

SOURCE = "ufcstats"
CACHE_ROOT: Path = Path("data/ufcstats_event_detail_cache")
FIGHT_CACHE_DIR: Path = CACHE_ROOT / "fights"
EVENT_CACHE_DIR: Path = CACHE_ROOT / "events"
BACKENDS: tuple[str, ...] = ("none", "browser")
BROWSER_DELAY_SECONDS_DEFAULT: float = 1.5
FETCH_BATCH_SIZE: int = 25
TWIN_MAX_DAYS: int = 1

_CACHE_KEY_RE = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class UfcFight:
    fight_id: int
    event_id: int
    fight_key: str | None
    fighter_hexes: frozenset[str]
    fighter_a_hex: str
    fighter_b_hex: str
    is_title_fight: bool


@dataclass(frozen=True)
class UfcEvent:
    event_id: int
    event_key: str | None
    source_url: str | None


@dataclass
class TitleBackfillReport:
    counts: Counter[str] = field(default_factory=Counter)
    coverage: dict[str, Any] = field(default_factory=dict)
    cross_check: dict[str, Any] = field(default_factory=dict)
    changes: list[dict[str, Any]] = field(default_factory=list)
    page_conflicts: list[dict[str, Any]] = field(default_factory=list)
    page_problems: list[dict[str, Any]] = field(default_factory=list)
    unknown_events: list[dict[str, Any]] = field(default_factory=list)
    fight_page_title_bouts: Counter[str] = field(default_factory=Counter)

    def as_dict(self, *, applied: bool) -> dict[str, Any]:
        return {
            "applied": applied,
            "counts": dict(sorted(self.counts.items())),
            "coverage": self.coverage,
            "cross_check": self.cross_check,
            "changes": self.changes,
            "page_conflicts": self.page_conflicts,
            "page_problems": self.page_problems,
            "unknown_events": self.unknown_events,
            "fight_page_title_bouts": dict(self.fight_page_title_bouts.most_common()),
        }


# ── Keys and pages ─────────────────────────────────────────────────────────


def cache_key(url: str | None) -> str | None:
    """The last path segment of a UFCStats URL, if it is a safe file name."""
    if not url:
        return None
    key = url.rstrip("/").rsplit("/", 1)[-1]
    return key if _CACHE_KEY_RE.fullmatch(key) else None


def read_cached(cache_dir: Path, key: str | None) -> str | None:
    if key is None:
        return None
    path = cache_dir / f"{key}.html"
    return path.read_text(encoding="utf-8") if path.exists() else None


def parse_event_page(html: str, url: str) -> EventDetail | None:
    """The parsed event page, or ``None`` for a challenge / unparseable page."""
    if detect_antibot(html, 200):
        return None
    try:
        return parse_event_detail(html, url)
    except (ValueError, RuntimeError):
        return None


def fight_page_evidence(fight: UfcFight, html: str | None) -> tuple[bool | None, str, str]:
    """``(title flag, status, bout text)`` from a cached fight-detail page.

    Status: ``ok`` / ``missing`` / ``challenge`` / ``unparseable`` /
    ``mismatch`` (the page's fighters are not this fight's).
    """
    if html is None:
        return None, "missing", ""
    if detect_antibot(html, 200):
        return None, "challenge", ""
    try:
        detail = parse_fight_detail(html)
    except (ValueError, RuntimeError):
        return None, "unparseable", ""
    if _orient_fight_detail(detail, fight.fighter_a_hex, fight.fighter_b_hex) is None:
        return None, "mismatch", ""
    return title_flag_from_fight_page(detail), "ok", detail.bout_type


def event_page_evidence(
    fights: Iterable[UfcFight], page: EventDetail
) -> dict[int, tuple[bool | None, str]]:
    """``{fight_id: (title flag, status)}`` for the card's fights.

    Rows are matched by fight hex id (the row's ``data-link``); status is
    ``ok``, ``not_on_page`` or ``mismatch`` (the row lists other fighters).
    """
    rows = {cache_key(s.fight_url): s for s in page.fights}
    out: dict[int, tuple[bool | None, str]] = {}
    for fight in fights:
        row = rows.get(fight.fight_key) if fight.fight_key else None
        if row is None:
            out[fight.fight_id] = (None, "not_on_page")
            continue
        hexes = frozenset((_extract_hex_id(row.fighter_a_url), _extract_hex_id(row.fighter_b_url)))
        if hexes != fight.fighter_hexes:
            out[fight.fight_id] = (None, "mismatch")
            continue
        out[fight.fight_id] = (title_flag_from_event_row(row), "ok")
    return out


# ── Kaggle twins ───────────────────────────────────────────────────────────


def find_twins_near(
    targets: Sequence[FightSides], candidates: Sequence[FightSides], max_days: int = TWIN_MAX_DAYS
) -> tuple[TwinMatch, set[int]]:
    """``find_twins`` on the exact date, falling back to +/-``max_days``.

    Returns the match and the target ids whose twins came from the fallback.
    """
    match = find_twins(targets, candidates)
    shifted = [
        find_twins(
            targets, [replace(c, event_date=c.event_date + timedelta(days=d)) for c in candidates]
        )
        for d in range(-max_days, max_days + 1)
        if d != 0
    ]
    near: set[int] = set()
    for fight_id, twins in match.twins.items():
        if twins or fight_id in match.unkeyable or fight_id in match.duplicate_targets:
            continue
        found = {t.fight_id: t for m in shifted for t in m.twins.get(fight_id, [])}
        if found:
            match.twins[fight_id] = list(found.values())
            near.add(fight_id)
    return match, near


def twin_evidence(
    twins: list[FightSides], flags: dict[int, bool]
) -> tuple[bool | None, str, list[str]]:
    """``(title flag, status, twin sources)``; status ``ok`` / ``none`` /
    ``ambiguous`` (two twins from one source) / ``conflict`` (twins disagree)."""
    sources = sorted(t.source for t in twins)
    if not twins:
        return None, "none", sources
    if any(n > 1 for n in Counter(sources).values()):
        return None, "ambiguous", sources
    values = {flags[t.fight_id] for t in twins}
    if len(values) > 1:
        return None, "conflict", sources
    return values.pop(), "ok", sources


# ── DB access ──────────────────────────────────────────────────────────────


def load_ufc_fights(session: Session) -> list[UfcFight]:
    fa = aliased(Fighter)
    fb = aliased(Fighter)
    stmt = (
        select(
            Fight.id,
            Fight.event_id,
            Fight.source_url,
            fa.source_id,
            fb.source_id,
            Fight.is_title_fight,
        )
        .join(Event, Event.id == Fight.event_id)
        .join(fa, fa.id == Fight.fighter_a_id)
        .join(fb, fb.id == Fight.fighter_b_id)
        .where(Fight.source == SOURCE, Event.source == SOURCE)
        .order_by(Fight.id)
    )
    return [
        UfcFight(
            fight_id=r[0],
            event_id=r[1],
            fight_key=cache_key(r[2]),
            fighter_hexes=frozenset((r[3] or "", r[4] or "")),
            fighter_a_hex=r[3] or "",
            fighter_b_hex=r[4] or "",
            is_title_fight=bool(r[5]),
        )
        for r in session.execute(stmt).all()
    ]


def load_ufc_events(session: Session) -> dict[int, UfcEvent]:
    stmt = select(Event.id, Event.source_url).where(Event.source == SOURCE)
    return {
        r[0]: UfcEvent(event_id=r[0], event_key=cache_key(r[1]), source_url=r[1])
        for r in session.execute(stmt).all()
    }


def load_kaggle_flags(session: Session) -> dict[int, bool]:
    stmt = (
        select(Fight.id, Fight.is_title_fight)
        .join(Event, Event.id == Fight.event_id)
        .where(Event.source.in_(KAGGLE_SOURCES))
    )
    return {r[0]: bool(r[1]) for r in session.execute(stmt).all()}


def write_flags(session: Session, fight_ids: list[int], value: bool) -> int:
    """Set ``is_title_fight = value`` on these ufcstats fights; returns rows changed."""
    if not fight_ids:
        return 0
    result = session.execute(
        update(Fight)
        .where(
            Fight.id.in_(fight_ids),
            Fight.source == SOURCE,
            Fight.is_title_fight.is_not(value),
        )
        .values(is_title_fight=value)
        .execution_options(synchronize_session=False)
    )
    return int(result.rowcount or 0)  # type: ignore[attr-defined]


# ── Event-page fetch (browser backend) ─────────────────────────────────────


def fetch_event_pages(
    client: Any,
    events: list[UfcEvent],
    cache_dir: Path,
    report: TitleBackfillReport,
    batch_size: int = FETCH_BATCH_SIZE,
) -> dict[int, EventDetail]:
    """Fetch these events' pages, caching every real one as it arrives.

    Per-URL isolation via ``ingest._safe_fetch`` (a challenge page is a fetch
    failure, never content). Batches are cached as they complete, so a run
    halted by ``AntiBotChallengeError`` resumes from the cache.
    """
    pages: dict[int, EventDetail] = {}
    for start in range(0, len(events), batch_size):
        batch = events[start : start + batch_size]
        urls = [e.source_url or "" for e in batch]
        fetched = client.map(functools.partial(_safe_fetch, client), urls)
        for event, (url, html, err) in zip(batch, fetched, strict=True):
            if html is None:
                logger.warning("event fetch failed for %s: %s", url, err)
                report.counts["events:fetch_failed"] += 1
                continue
            page = parse_event_page(html, url)
            if page is None:
                report.counts["events:unparseable"] += 1
                continue
            report.counts["events:fetched"] += 1
            pages[event.event_id] = page
            if event.event_key is not None:
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    (cache_dir / f"{event.event_key}.html").write_text(html, encoding="utf-8")
                except OSError:
                    logger.warning("event cache write failed for %s", event.event_key)
        logger.info(
            "event pages %d/%d: %s",
            min(start + batch_size, len(events)),
            len(events),
            {k: v for k, v in report.counts.items() if k.startswith("events:")},
        )
    return pages


# ── Driver ─────────────────────────────────────────────────────────────────


def _bump(coverage: dict[str, Any], source: str, value: bool) -> None:
    entry = coverage[source]
    entry["settled"] += 1
    entry["true" if value else "false"] += 1


def backfill(
    session: Session,
    *,
    apply: bool,
    fight_cache_dir: Path = FIGHT_CACHE_DIR,
    event_cache_dir: Path = EVENT_CACHE_DIR,
    client: Any = None,
    max_events: int | None = None,
    recheck_twins: bool = False,
) -> TitleBackfillReport:
    """Decide every ufcstats fight's title flag and (with ``apply``) write it.

    With a ``client``, event pages are fetched for every event that has a fight
    with no page and no usable twin; ``recheck_twins`` widens that to every
    fight with no page, so twin-only fights get an authoritative page too.
    """
    report = TitleBackfillReport()
    fights = load_ufc_fights(session)
    events = load_ufc_events(session)
    by_event: dict[int, list[UfcFight]] = {}
    for f in fights:
        by_event.setdefault(f.event_id, []).append(f)
    report.counts["fights"] = len(fights)
    report.counts["stored_true"] = sum(f.is_title_fight for f in fights)

    # (b) kaggle twins. Every ufcstats fight is a target, so a pair key shared
    # by two ufcstats fights is flagged as a duplicate target.
    match, near = find_twins_near(
        load_fight_sides(session, [SOURCE]), load_fight_sides(session, KAGGLE_SOURCES)
    )
    kaggle_flags = load_kaggle_flags(session)
    twin: dict[int, tuple[bool | None, str, list[str]]] = {}
    for f in fights:
        if f.fight_id in match.unkeyable or f.fight_id in match.duplicate_targets:
            twin[f.fight_id] = (None, "ambiguous", [])
        else:
            twin[f.fight_id] = twin_evidence(match.twins.get(f.fight_id, []), kaggle_flags)
        report.counts[f"twin:{twin[f.fight_id][1]}"] += 1
        if twin[f.fight_id][1] == "ok" and f.fight_id in near:
            report.counts["twin:near_date"] += 1

    # (a) cached fight-detail pages
    fight_page: dict[int, bool | None] = {}
    for f in fights:
        value, status, bout = fight_page_evidence(f, read_cached(fight_cache_dir, f.fight_key))
        fight_page[f.fight_id] = value
        report.counts[f"fight_page:{status}"] += 1
        if status not in ("ok", "missing"):
            report.page_problems.append({"fight_id": f.fight_id, "fight_page": status})
        if value:
            report.fight_page_title_bouts[bout] += 1

    # (c) event pages: cached ones, then (browser) fetch the unsettled events
    event_pages: dict[int, EventDetail] = {}
    for event in events.values():
        html = read_cached(event_cache_dir, event.event_key)
        if html is not None and event.event_id in by_event:
            page = parse_event_page(html, event.source_url or "")
            if page is not None:
                event_pages[event.event_id] = page
                report.counts["events:cached"] += 1

    def unsettled(f: UfcFight) -> bool:
        if fight_page[f.fight_id] is not None:
            return False
        return recheck_twins or twin[f.fight_id][1] != "ok"

    needed = [
        events[event_id]
        for event_id in sorted(by_event)
        if event_id in events
        and event_id not in event_pages
        and events[event_id].event_key is not None
        and any(unsettled(f) for f in by_event[event_id])
    ]
    report.counts["events:needed"] = len(needed)
    if client is not None:
        to_fetch = needed if max_events is None else needed[:max_events]
        event_pages.update(fetch_event_pages(client, to_fetch, event_cache_dir, report))

    event_page: dict[int, bool | None] = {}
    for event_id, page in event_pages.items():
        for fight_id, (value, status) in event_page_evidence(by_event[event_id], page).items():
            event_page[fight_id] = value
            report.counts[f"event_page:{status}"] += 1

    # Decide, cross-check, plan.
    coverage: dict[str, Any] = {
        src: {"settled": 0, "true": 0, "false": 0} for src in ("fight_page", "event_page", "twin")
    }
    coverage["unknown"] = 0
    page_vs_twin: Counter[str] = Counter(
        {"agree_true": 0, "agree_false": 0, "page_true_twin_false": 0, "page_false_twin_true": 0}
    )
    pages_agree: Counter[str] = Counter({"agree": 0, "disagree": 0})
    disagreements: list[dict[str, Any]] = []
    to_true: list[int] = []
    to_false: list[int] = []
    unknown_by_event: Counter[int] = Counter()

    for f in fights:
        fp, ep = fight_page[f.fight_id], event_page.get(f.fight_id)
        tw, _tw_status, tw_sources = twin[f.fight_id]
        if fp is not None and ep is not None:
            pages_agree["agree" if fp == ep else "disagree"] += 1
            if fp != ep:  # two authoritative pages disagree: write nothing
                report.counts["page_conflict"] += 1
                coverage["unknown"] += 1
                report.page_conflicts.append(
                    {"fight_id": f.fight_id, "fight_page": fp, "event_page": ep}
                )
                continue
        page = fp if fp is not None else ep
        if page is not None and tw is not None:
            if page == tw:
                page_vs_twin["agree_true" if page else "agree_false"] += 1
            else:
                page_vs_twin["page_true_twin_false" if page else "page_false_twin_true"] += 1
                disagreements.append(
                    {"fight_id": f.fight_id, "page": page, "twin": tw, "twin_sources": tw_sources}
                )

        target: bool | None
        if page is not None:
            evidence = "fight_page" if fp is not None else "event_page"
            target = page
        elif tw is not None:
            evidence = "twin"
            target = True if tw else None  # a twin is not authoritative for False
        else:
            coverage["unknown"] += 1
            report.counts["unknown"] += 1
            unknown_by_event[f.event_id] += 1
            continue
        _bump(coverage, evidence, page if page is not None else bool(tw))
        report.counts[f"settled:{evidence}"] += 1

        if target is None or target == f.is_title_fight:
            report.counts["unchanged"] += 1
            continue
        (to_true if target else to_false).append(f.fight_id)
        report.counts["to_set_true" if target else "to_set_false"] += 1
        report.changes.append({"fight_id": f.fight_id, "to": target, "evidence": evidence})

    report.coverage = coverage
    report.cross_check = {
        "page_vs_twin": dict(page_vs_twin),
        "fight_page_vs_event_page": dict(pages_agree),
        "disagreements": disagreements,
    }
    report.unknown_events = [
        {
            "event_id": event_id,
            "source_url": events[event_id].source_url if event_id in events else None,
            "unknown_fights": n,
        }
        for event_id, n in sorted(unknown_by_event.items())
    ]

    if apply:
        report.counts["rows_updated"] = write_flags(session, to_true, True) + write_flags(
            session, to_false, False
        )
        session.commit()
    return report


# ── CLI ────────────────────────────────────────────────────────────────────


def open_session(*, apply: bool) -> Session:
    """A session on the configured DB; without ``apply`` its transaction is READ ONLY."""
    from ufc_prediction.db import session as db_session

    session = db_session.SessionLocal()
    if not apply:
        session.execute(text("SET TRANSACTION READ ONLY"))
    return session


def build_client(backend: str, *, delay: float) -> Any:
    """``None`` for the offline backend, else the headless-browser fetcher."""
    if backend == "none":
        return None
    if backend == "browser":
        from ufc_prediction.scraper.browser_fetch import BrowserFetcher

        return BrowserFetcher(delay=delay)
    msg = f"unknown backend {backend!r} (expected one of {BACKENDS})"
    raise ValueError(msg)


def _print_summary(report: TitleBackfillReport, *, applied: bool) -> None:
    cov = report.coverage
    print(f"[title-backfill] ufcstats fights: {report.counts['fights']} (applied={applied})")
    print(f"[title-backfill] {'source':<11} {'settled':>8} {'true':>6} {'false':>6}")
    for src in ("fight_page", "event_page", "twin"):
        c = cov[src]
        print(f"[title-backfill] {src:<11} {c['settled']:>8} {c['true']:>6} {c['false']:>6}")
    print(f"[title-backfill] {'unknown':<11} {cov['unknown']:>8}")
    print(f"[title-backfill] cross-check: {report.cross_check['page_vs_twin']}")
    print(f"[title-backfill] counts: {dict(sorted(report.counts.items()))}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--apply", action="store_true", help="Write the flags. Without it, report only."
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="none",
        help="none: cached pages + kaggle twins only (no network). browser: also fetch "
        "the ufcstats event page of every event that is still unsettled.",
    )
    parser.add_argument(
        "--recheck-twins",
        action="store_true",
        help="(browser) Also fetch the event page of every event whose fights are "
        "settled only by kaggle twins, so the page (authoritative) decides them.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="(browser) Fetch at most this many event pages.",
    )
    parser.add_argument("--delay", type=float, default=BROWSER_DELAY_SECONDS_DEFAULT)
    parser.add_argument("--fight-cache-dir", type=Path, default=FIGHT_CACHE_DIR)
    parser.add_argument("--event-cache-dir", type=Path, default=EVENT_CACHE_DIR)
    parser.add_argument("--report", type=Path, default=None, help="Write a JSON report here.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[title-backfill] %(message)s")

    session = open_session(apply=args.apply)
    try:
        client = build_client(args.backend, delay=args.delay)
        try:
            report = backfill(
                session,
                apply=args.apply,
                fight_cache_dir=args.fight_cache_dir,
                event_cache_dir=args.event_cache_dir,
                client=client,
                max_events=args.max_events,
                recheck_twins=args.recheck_twins,
            )
        finally:
            if client is not None:
                client.close()
        if not args.apply:
            session.rollback()
        summary = report.as_dict(applied=args.apply)
        _print_summary(report, applied=args.apply)
        if args.report is not None:
            args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[title-backfill] report written to {args.report}")
        return 0
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
