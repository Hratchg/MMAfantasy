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

Two modes (both report-only unless ``--apply``):

``--mode offline`` (no network) — decide from kaggle twins
  The kaggle sources record corners (``fighter_a`` = RED, ``fighter_b`` =
  BLUE) and the fight-detail page lists the red corner first, so the stored
  stats are crossed iff the ufcstats fighter_a (the winner) is the kaggle BLUE
  corner. A twin is the same bout in kaggle-rajeevw / kaggle-mdabbert (same
  event date, same two fighters by canonical name — ``dedup/twins.py``).
  kaggle-rajeevw also stores round-0 stats (scraped from the same ufcstats
  pages), which gives an independent check: the stored stats are crossed iff
  the ufcstats fighter_a's round-0 row equals the kaggle row of the OPPONENT.
  A decision is trusted offline only when BOTH agree:
    offline_crossed           corner BLUE + stats crossed  -> swapped by --apply
    offline_correct           corner RED  + stats self     -> left alone
    offline_already_repaired  corner BLUE + stats self     -> the state a fight
                              is in after --apply (no-op; keeps re-runs quiet)
  Everything else goes to the needs-live list (``needs_live`` in the report,
  consumable by ``--mode live --fight-ids-from``): no_twin, ambiguous_twin,
  corner_conflict, corner_only (mdabbert twin, which has no stats — its
  corner-rule projection is reported but never acted on), stats_inconclusive,
  stats_corner_disagree. Fights with no stored stats count as ``no_stats``.
  The report's ``premise`` matrix ("<corner>|<stats verdict>") measures how
  well the corner rule agrees with the stats evidence on the current DB.

``--mode live`` (default) — decide from the fight-detail page
  1. Cache-first fetch of the fight-detail page (``fights.source_url``); the
     cache is ``data/ufcstats_event_detail_cache/fights/<slug>.html``, shared
     with ``scrape_referees_full.py``, so an interrupted run resumes cheaply.
     Only a real, parseable page is ever cached; a cached anti-bot challenge
     or unparseable page (a pre-fix run cached 200 "Checking your browser"
     pages) is treated as a miss and refetched.
     ``--backend browser`` fetches through the headless-Chromium
     ``BrowserFetcher`` (single worker, its own polite rate + backoff), which
     passes the ufcstats JS proof-of-work challenge that plain httpx cannot.
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
     plus fetch_failed (incl. challenge pages) / parse_failed /
     page_mismatch / no_stats.

Classification is content-based in both modes, so the repair is idempotent:
once swapped, a fight classifies as correct / already-repaired on any re-run.

After ``--apply``, regenerate the substrate (CLAUDE.md order): ``features
compute`` → retrain gate → re-anchor pinned baselines → regenerate the seed
dump. (Elo reads only results, which this does not touch.)

Banned imports: nothing under ``ufc_prediction.ml.*`` (AUDIT-01).

Usage:
    uv run python scripts/repair_ufcstats_round_stats_orientation.py --mode offline \\
        --report offline.json
    uv run python scripts/repair_ufcstats_round_stats_orientation.py --mode offline --apply
    uv run python scripts/repair_ufcstats_round_stats_orientation.py --dry-run \\
        --fight-ids-from offline.json
    uv run python scripts/repair_ufcstats_round_stats_orientation.py --backend browser \\
        --fight-ids-from offline.json --apply --report live.json
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

from ufc_prediction.dedup.twins import (
    KAGGLE_SOURCES,
    FightSides,
    find_twins,
    load_fight_sides,
    map_fighters_by_name,
)
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter
from ufc_prediction.models.round_stats import RoundStats
from ufc_prediction.scraper.antibot import detect_antibot
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
BACKENDS: tuple[str, ...] = ("httpx", "browser")

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
# The offline stats check ignores control time: ufcstats has since corrected
# CTRL by a few seconds on some fights, so the kaggle snapshot differs there
# while every count column still matches.
_OFFLINE_COMPARE_IDX: tuple[int, ...] = tuple(
    i for i, f in enumerate(STAT_FIELDS) if f != "control_time_seconds"
)

StatsByRound = dict[int, tuple[Any, ...]]
StatTuple = tuple[Any, ...]


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
            "mode": "live",
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


def side_equal(stored: StatTuple | None, kaggle: StatTuple | None) -> bool | None:
    """Do two round-0 stat rows describe the same fighter-side? ``None`` = no signal.

    No signal when either row is missing or all-NULL (rajeevw writes an
    all-NULL row when every key stat is 0, D-04), or both are all-zero.
    Control time is ignored (see ``_OFFLINE_COMPARE_IDX``).
    """
    if stored is None or kaggle is None:
        return None
    x = tuple(stored[i] for i in _OFFLINE_COMPARE_IDX)
    y = tuple(kaggle[i] for i in _OFFLINE_COMPARE_IDX)
    if all(v is None for v in x) or all(v is None for v in y):
        return None
    if not any(x) and not any(y):
        return None
    return x == y


def stats_verdict(
    stored_a: StatTuple | None,
    stored_b: StatTuple | None,
    kaggle_of_a: StatTuple | None,
    kaggle_of_b: StatTuple | None,
) -> str:
    """``self`` / ``cross`` / ``inconclusive`` for one fight vs its kaggle twin.

    ``kaggle_of_a`` is the twin's row for the fighter with fighter_a's
    canonical name. ``self``: at least one side matches its own kaggle row and
    no side contradicts, and the crossed reading is not also supported (and
    symmetrically for ``cross``).
    """
    own = (side_equal(stored_a, kaggle_of_a), side_equal(stored_b, kaggle_of_b))
    crossed = (side_equal(stored_a, kaggle_of_b), side_equal(stored_b, kaggle_of_a))
    own_ok = True in own and False not in own
    cross_ok = True in crossed and False not in crossed
    if own_ok and not cross_ok:
        return "self"
    if cross_ok and not own_ok:
        return "cross"
    return "inconclusive"


# ── DB access ──────────────────────────────────────────────────────────────


def load_fights(
    session: Session, limit: int | None = None, fight_ids: set[int] | None = None
) -> list[FightRef]:
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
    if fight_ids is not None:
        stmt = stmt.where(Fight.id.in_(sorted(fight_ids)))
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


def load_round0_stats(session: Session) -> dict[tuple[int, int], StatTuple]:
    """``{(fight_id, fighter_id): round-0 stat tuple}`` for every fight."""
    cols = [getattr(RoundStats, f) for f in STAT_FIELDS]
    stmt = select(RoundStats.fight_id, RoundStats.fighter_id, *cols).where(
        RoundStats.round_number == 0
    )
    return {(r[0], r[1]): tuple(r[2:]) for r in session.execute(stmt).all()}


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


def _is_real_fight_page(html: str) -> bool:
    """A page worth caching: not an anti-bot challenge, and it parses."""
    if detect_antibot(html, 200):
        return False
    try:
        parse_fight_detail(html)
    except (ValueError, RuntimeError):
        return False
    return True


def fetch_pages(client: Any, fights: list[FightRef], cache_dir: Path) -> dict[int, str | None]:
    """Cache-first fetch of each fight's detail page; misses go through ``client.map``.

    Only real, parseable pages are cached. A cached challenge / unparseable
    page is ignored (and overwritten if the refetch is good).
    """
    pages: dict[int, str | None] = {}
    misses: list[FightRef] = []
    for f in fights:
        key = _cache_key(f.source_url)
        path = cache_dir / f"{key}.html" if key else None
        if path is not None and path.exists():
            cached = path.read_text(encoding="utf-8")
            if _is_real_fight_page(cached):
                pages[f.fight_id] = cached
                continue
            logger.warning("ignoring bad cached page %s (challenge/unparseable)", path)
        misses.append(f)
    if misses:
        # Per-URL error isolation (same helper the scraper uses): one dead page
        # must not fail the whole batch. A challenge page comes back as a fetch
        # failure (html None), never as content.
        fetched = client.map(functools.partial(_safe_fetch, client), [f.source_url for f in misses])
        for f, (url, html, err) in zip(misses, fetched, strict=True):
            if html is None:
                logger.warning("fetch failed for %s: %s", url, err)
            pages[f.fight_id] = html or None
            key = _cache_key(f.source_url)
            if html and key and _is_real_fight_page(html):
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    (cache_dir / f"{key}.html").write_text(html, encoding="utf-8")
                except OSError:
                    logger.warning("cache write failed for %s", key)
    return pages


# ── Live driver ────────────────────────────────────────────────────────────


def repair(
    session: Session,
    client: Any,
    *,
    apply: bool,
    cache_dir: Path = FIGHT_CACHE_DIR,
    limit: int | None = None,
    batch_size: int = BATCH_SIZE,
    fight_ids: set[int] | None = None,
) -> RepairReport:
    """Classify every ufcstats fight (or just ``fight_ids``) and (with ``apply``)
    swap the crossed ones."""
    report = RepairReport()
    fights = load_fights(session, limit=limit, fight_ids=fight_ids)
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


# ── Offline driver (kaggle twins) ──────────────────────────────────────────


@dataclass
class OfflineReport:
    counts: Counter[str] = field(default_factory=Counter)
    offline_crossed: list[int] = field(default_factory=list)
    offline_correct: list[int] = field(default_factory=list)
    needs_live: list[dict[str, Any]] = field(default_factory=list)
    swapped_fight_ids: list[int] = field(default_factory=list)
    premise: dict[str, int] = field(default_factory=dict)
    corner_only_projection: dict[str, int] = field(default_factory=dict)

    def as_dict(self, *, applied: bool) -> dict[str, Any]:
        agree = self.premise.get("blue|cross", 0) + self.premise.get("red|self", 0)
        total = sum(self.premise.values())
        return {
            "mode": "offline",
            "applied": applied,
            "counts": dict(self.counts),
            "premise": {
                "matrix": self.premise,
                "agree": agree,
                "conclusive": total,
                "agreement_rate": (agree / total) if total else None,
            },
            "corner_only_projection": self.corner_only_projection,
            "offline_crossed": self.offline_crossed,
            "offline_correct_count": len(self.offline_correct),
            "swapped_fight_ids": self.swapped_fight_ids,
            "needs_live": self.needs_live,
        }


def _offline_decision(
    ufc: FightSides,
    twins: list[FightSides],
    stats: dict[tuple[int, int], StatTuple],
    premise: Counter[str],
    corner_projection: Counter[str],
) -> tuple[str, str | None]:
    """``(bucket, needs_live_reason)`` for one ufcstats fight with stored stats."""
    if not twins:
        return "needs_live", "no_twin"
    per_source = Counter(t.source for t in twins)
    if any(n > 1 for n in per_source.values()):
        return "needs_live", "ambiguous_twin"

    corners: set[str] = set()
    verdicts: set[str] = set()
    for twin in twins:
        mapping = map_fighters_by_name(ufc, twin)
        if mapping is None:
            return "needs_live", "ambiguous_twin"
        # kaggle fighter_a = RED corner.
        corners.add("red" if mapping[twin.fighter_a_id] == ufc.fighter_a_id else "blue")
        k_by_ufc = {mapping[twin.fighter_a_id]: twin.fighter_a_id}
        k_by_ufc[mapping[twin.fighter_b_id]] = twin.fighter_b_id
        k_a = stats.get((twin.fight_id, k_by_ufc[ufc.fighter_a_id]))
        k_b = stats.get((twin.fight_id, k_by_ufc[ufc.fighter_b_id]))
        if k_a is None and k_b is None:
            continue  # no stats on this twin (mdabbert): corner evidence only
        verdicts.add(
            stats_verdict(
                stats.get((ufc.fight_id, ufc.fighter_a_id)),
                stats.get((ufc.fight_id, ufc.fighter_b_id)),
                k_a,
                k_b,
            )
        )

    if len(corners) != 1:
        return "needs_live", "corner_conflict"
    corner = corners.pop()
    if not verdicts:
        corner_projection["crossed" if corner == "blue" else "correct"] += 1
        return "needs_live", "corner_only"
    verdict = verdicts.pop() if len(verdicts) == 1 else "inconclusive"
    if verdict == "inconclusive":
        return "needs_live", "stats_inconclusive"
    premise[f"{corner}|{verdict}"] += 1
    if corner == "blue" and verdict == "cross":
        return "offline_crossed", None
    if corner == "red" and verdict == "self":
        return "offline_correct", None
    if corner == "blue" and verdict == "self":
        return "offline_already_repaired", None
    return "needs_live", "stats_corner_disagree"


def repair_offline(session: Session, *, apply: bool) -> OfflineReport:
    """Classify every ufcstats fight from its kaggle twins; with ``apply``,
    swap the ``offline_crossed`` ones (and only those)."""
    report = OfflineReport()
    ufc_fights = load_fight_sides(session, [SOURCE])
    kaggle = load_fight_sides(session, KAGGLE_SOURCES)
    match = find_twins(ufc_fights, kaggle)
    stats = load_round0_stats(session)
    urls = dict(
        session.execute(select(Fight.id, Fight.source_url).where(Fight.source == SOURCE)).all()
    )
    premise: Counter[str] = Counter()
    projection: Counter[str] = Counter()

    for ufc in ufc_fights:
        if (ufc.fight_id, ufc.fighter_a_id) not in stats and (
            ufc.fight_id,
            ufc.fighter_b_id,
        ) not in stats:
            report.counts["no_stats"] += 1
            continue
        if ufc.fight_id in match.unkeyable or ufc.fight_id in match.duplicate_targets:
            bucket, reason = "needs_live", "ambiguous_twin"
        else:
            bucket, reason = _offline_decision(
                ufc, match.twins[ufc.fight_id], stats, premise, projection
            )
        if reason is not None:
            report.counts[f"needs_live:{reason}"] += 1
            report.needs_live.append(
                {"fight_id": ufc.fight_id, "source_url": urls.get(ufc.fight_id), "reason": reason}
            )
            continue
        report.counts[bucket] += 1
        if bucket == "offline_crossed":
            report.offline_crossed.append(ufc.fight_id)
        elif bucket == "offline_correct":
            report.offline_correct.append(ufc.fight_id)

    report.counts["needs_live"] = len(report.needs_live)
    report.premise = dict(premise)
    report.corner_only_projection = dict(projection)

    if apply and report.offline_crossed:
        by_id = {f.fight_id: f for f in ufc_fights}
        for fight_id in report.offline_crossed:
            ufc = by_id[fight_id]
            swap_round_stats(
                session,
                FightRef(
                    fight_id, urls.get(fight_id) or "", ufc.fighter_a_id, ufc.fighter_b_id, "", ""
                ),
            )
            report.swapped_fight_ids.append(fight_id)
        session.commit()
    return report


# ── CLI ────────────────────────────────────────────────────────────────────


def build_client(backend: str, *, delay: float, workers: int) -> Any:
    """The fetcher for ``--backend``: plain httpx, or the headless browser.

    The browser fetcher is single-worker/serial with its own polite rate
    (``delay``) and challenge backoff; ``workers`` applies to httpx only.
    """
    if backend == "httpx":
        from ufc_prediction.scraper.client import ScraperClient

        return ScraperClient(
            delay=delay,
            max_retries=HTTP_MAX_RETRIES,
            timeout=HTTP_TIMEOUT_SECONDS,
            workers=workers,
        )
    if backend == "browser":
        from ufc_prediction.scraper.browser_fetch import BrowserFetcher

        return BrowserFetcher(delay=delay)
    msg = f"unknown backend {backend!r} (expected one of {BACKENDS})"
    raise ValueError(msg)


def read_fight_ids(path: Path) -> set[int]:
    """Fight ids from an offline report JSON (``needs_live``) or a one-per-line list."""
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {int(line) for line in text.split() if line.strip()}
    if isinstance(data, dict):
        return {int(e["fight_id"]) for e in data.get("needs_live", [])}
    return {int(x) for x in data}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--mode",
        choices=("live", "offline"),
        default="live",
        help="live: compare to the fight-detail page; offline: decide from kaggle twins.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="(live) No network, no writes: print how many fights would be checked.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Swap the crossed rows. Without it, classify and report only.",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="httpx",
        help="(live) Fetcher: plain httpx, or the headless browser that passes "
        "the ufcstats JS challenge.",
    )
    parser.add_argument(
        "--fight-ids-from",
        type=Path,
        default=None,
        help="(live) Only check these fights: an offline report JSON (its needs_live "
        "list) or a file of fight ids, one per line.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=HTTP_DELAY_SECONDS_DEFAULT)
    parser.add_argument("--workers", type=int, default=WORKERS_DEFAULT)
    parser.add_argument("--report", type=Path, default=None, help="Write a JSON report here.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[repair-orientation] %(message)s")

    from ufc_prediction.db.session import SessionLocal

    session = SessionLocal()
    try:
        summary: dict[str, Any]
        if args.mode == "offline":
            offline = repair_offline(session, apply=args.apply)
            summary = offline.as_dict(applied=args.apply)
            print(f"[repair-orientation] OFFLINE counts: {summary['counts']}")
            print(f"[repair-orientation] premise: {summary['premise']}")
            print(f"[repair-orientation] corner-only projection: {offline.corner_only_projection}")
        else:
            fight_ids = read_fight_ids(args.fight_ids_from) if args.fight_ids_from else None
            if args.dry_run:
                fights = load_fights(session, limit=args.limit, fight_ids=fight_ids)
                cached = sum(
                    1
                    for f in fights
                    if (key := _cache_key(f.source_url))
                    and (p := FIGHT_CACHE_DIR / f"{key}.html").exists()
                    and _is_real_fight_page(p.read_text(encoding="utf-8"))
                )
                print(f"[repair-orientation] DRY-RUN: {len(fights)} ufcstats fights to check")
                print(
                    f"[repair-orientation] cached pages: {cached}; to fetch: {len(fights) - cached}"
                )
                return 0
            with build_client(args.backend, delay=args.delay, workers=args.workers) as client:
                report = repair(
                    session, client, apply=args.apply, limit=args.limit, fight_ids=fight_ids
                )
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
