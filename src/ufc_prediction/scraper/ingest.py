"""Scraping orchestrator: composes HTTP client, parsers, and DB upserts.

Provides two public entry points:
- scrape_all_events: Full historical scrape of all completed events
- scrape_latest_events: Incremental scrape of events not yet in DB

All scraped data passes through the same Pydantic schemas used by Kaggle
ingestion (FightRow, FighterRow, FightStats) per D-12.
"""

from __future__ import annotations

import functools
import logging
import re
from collections import Counter
from collections.abc import Callable
from datetime import date

from pydantic import ValidationError
from sqlalchemy.orm import Session

from ufc_prediction.data.parsers import (
    parse_control_time,
    parse_dob,
    parse_fight_date,
    parse_height_to_inches,
    parse_num_rounds,
    parse_reach_to_inches,
    parse_weight_class,
    parse_x_of_y,
)
from ufc_prediction.data.schemas import FighterRow, FightRow, FightStats, IngestResult
from ufc_prediction.data.upsert import (
    upsert_event,
    upsert_fight,
    upsert_fighter,
    upsert_referee,
    upsert_round_stats,
)
from ufc_prediction.models.event import Event
from ufc_prediction.scraper.client import ScraperClient
from ufc_prediction.scraper.models import (
    EventSummary,
    FightDetailPage,
    FighterProfile,
    FightSummary,
    RoundStatsRaw,
    SigStrikesRaw,
)
from ufc_prediction.scraper.parse_event_detail import parse_event_detail
from ufc_prediction.scraper.parse_event_list import parse_event_list
from ufc_prediction.scraper.parse_fight_detail import parse_fight_detail
from ufc_prediction.scraper.parse_fighter import parse_fighter_profile

logger = logging.getLogger(__name__)

SOURCE = "ufcstats"
EVENT_LIST_URL = "http://ufcstats.com/statistics/events/completed?page=all"

_HEX_ID_RE = re.compile(r"[0-9a-f]{16}")


# ── Conversion helpers ──────────────────────────────────────────────────────


def _extract_hex_id(url: str) -> str:
    """Extract the 16-character hex ID from a UFCStats URL.

    Example:
        "http://ufcstats.com/fighter-details/9014c02eff8b3d62"
        -> "9014c02eff8b3d62"
    """
    match = _HEX_ID_RE.search(url)
    if match:
        return match.group(0)
    return url.rsplit("/", 1)[-1]


def _safe_fetch(
    client: ScraperClient | object,
    url: str,
) -> tuple[str, str | None, Exception | None]:
    """Fetch ``url`` via ``client.get`` with per-URL error isolation.

    Returns ``(url, html, None)`` on success or ``(url, None, exc)`` on any
    fetch failure. Used inside ``client.map(...)`` so one bad URL does NOT
    abort the whole batch (preserves the D-11 per-page error contract while
    still getting the parallel-throughput win from ``client.map``).
    """
    try:
        return (url, client.get(url), None)
    except (RuntimeError, ValueError) as exc:
        return (url, None, exc)


def _convert_fighter_profile(profile: FighterProfile) -> FighterRow:
    """Convert scraper FighterProfile to the shared FighterRow schema.

    Uses parsers from data/parsers.py for type conversion.
    """
    return FighterRow(
        name=profile.name,
        height_inches=parse_height_to_inches(profile.height_str),
        reach_inches=parse_reach_to_inches(profile.reach_str),
        leg_reach_inches=None,  # Not available on UFCStats per RESEARCH.md
        stance=profile.stance,
        date_of_birth=parse_dob(profile.dob_str),
    )


def _parse_int_safe(value: str) -> int | None:
    """Safely parse an integer from a raw string."""
    try:
        return int(value.strip())
    except (ValueError, AttributeError):
        return None


def _convert_round_stats(raw: RoundStatsRaw) -> FightStats:
    """Convert RoundStatsRaw to FightStats via parsers."""
    sig_landed, sig_attempted = parse_x_of_y(raw.sig_str)
    td_landed, td_attempted = parse_x_of_y(raw.td)
    _, _total_attempted = parse_x_of_y(raw.total_str)

    return FightStats(
        sig_strikes_landed=sig_landed,
        sig_strikes_attempted=sig_attempted,
        takedowns_landed=td_landed,
        takedowns_attempted=td_attempted,
        submission_attempts=_parse_int_safe(raw.sub_att),
        reversals=_parse_int_safe(raw.rev),
        control_time_seconds=parse_control_time(raw.ctrl),
        knockdowns=_parse_int_safe(raw.knockdowns),
    )


def _convert_sig_strikes(raw: SigStrikesRaw) -> dict:
    """Convert SigStrikesRaw to additional fields dict for merging into FightStats."""
    head_landed, _ = parse_x_of_y(raw.head)
    body_landed, _ = parse_x_of_y(raw.body)
    leg_landed, _ = parse_x_of_y(raw.leg)
    distance_landed, _ = parse_x_of_y(raw.distance)
    clinch_landed, _ = parse_x_of_y(raw.clinch)
    ground_landed, _ = parse_x_of_y(raw.ground)

    return {
        "head_strikes_landed": head_landed,
        "body_strikes_landed": body_landed,
        "leg_strikes_landed": leg_landed,
        "distance_strikes_landed": distance_landed,
        "clinch_strikes_landed": clinch_landed,
        "ground_strikes_landed": ground_landed,
    }


def _build_fight_stats(
    totals_raw: RoundStatsRaw,
    sig_raw: SigStrikesRaw,
) -> FightStats:
    """Build a complete FightStats from totals + sig strikes raw data."""
    base = _convert_round_stats(totals_raw)
    sig_fields = _convert_sig_strikes(sig_raw)

    # Merge sig strike fields into the base stats
    return FightStats(
        sig_strikes_landed=base.sig_strikes_landed,
        sig_strikes_attempted=base.sig_strikes_attempted,
        takedowns_landed=base.takedowns_landed,
        takedowns_attempted=base.takedowns_attempted,
        submission_attempts=base.submission_attempts,
        reversals=base.reversals,
        control_time_seconds=base.control_time_seconds,
        knockdowns=base.knockdowns,
        **sig_fields,
    )


def _convert_fight(
    event_name: str,
    event_date_str: str,
    location: str,
    fight_summary: FightSummary,
    fight_detail: FightDetailPage,
) -> FightRow:
    """Convert scraper fight models to the shared FightRow schema.

    Validates through Pydantic FightRow per D-12.
    """
    event_date = parse_fight_date(event_date_str)
    if event_date is None:
        msg = f"Could not parse event date: {event_date_str}"
        raise ValueError(msg)

    weight_class, is_title = parse_weight_class(fight_summary.weight_class_raw)
    num_rounds = parse_num_rounds(fight_detail.time_format)

    # Determine winner name
    winner_name: str | None = None
    if fight_summary.winner == "fighter_a":
        winner_name = fight_summary.fighter_a_name
    elif fight_summary.winner == "fighter_b":
        winner_name = fight_summary.fighter_b_name

    # Build fight-level stats from totals + sig strikes
    fighter_a_stats = _build_fight_stats(fight_detail.totals[0], fight_detail.sig_strikes[0])
    fighter_b_stats = _build_fight_stats(fight_detail.totals[1], fight_detail.sig_strikes[1])

    return FightRow(
        event_name=event_name,
        event_date=event_date,
        location=location,
        fighter_a_name=fight_summary.fighter_a_name,
        fighter_b_name=fight_summary.fighter_b_name,
        winner_name=winner_name,
        weight_class=weight_class,
        method=fight_summary.method,
        method_detail=fight_summary.method_detail,
        round_finished=fight_summary.round_finished,
        time_finished=fight_summary.time_finished,
        is_title_fight=is_title,
        num_rounds=num_rounds,
        fighter_a_stats=fighter_a_stats,
        fighter_b_stats=fighter_b_stats,
    )


# ── Per-round stats upserting ───────────────────────────────────────────────


def _upsert_per_round_stats(
    session: Session,
    fight_id: int,
    fighter_a_id: int,
    fighter_b_id: int,
    per_round_totals: list[tuple[RoundStatsRaw, RoundStatsRaw]],
    per_round_sig_strikes: list[tuple[SigStrikesRaw, SigStrikesRaw]],
) -> None:
    """Upsert per-round stats for both fighters."""
    num_rounds = min(len(per_round_totals), len(per_round_sig_strikes))
    for round_idx in range(num_rounds):
        round_number = round_idx + 1
        totals_a, totals_b = per_round_totals[round_idx]
        sig_a, sig_b = per_round_sig_strikes[round_idx]

        stats_a = _build_fight_stats(totals_a, sig_a)
        stats_b = _build_fight_stats(totals_b, sig_b)

        _upsert_stats_for_fighter(session, fight_id, fighter_a_id, round_number, stats_a)
        _upsert_stats_for_fighter(session, fight_id, fighter_b_id, round_number, stats_b)


def _upsert_stats_for_fighter(
    session: Session,
    fight_id: int,
    fighter_id: int,
    round_number: int,
    stats: FightStats,
) -> None:
    """Upsert a single round stats record from FightStats object."""
    upsert_round_stats(
        session,
        fight_id=fight_id,
        fighter_id=fighter_id,
        round_number=round_number,
        sig_strikes_landed=stats.sig_strikes_landed,
        sig_strikes_attempted=stats.sig_strikes_attempted,
        takedowns_landed=stats.takedowns_landed,
        takedowns_attempted=stats.takedowns_attempted,
        submission_attempts=stats.submission_attempts,
        reversals=stats.reversals,
        control_time_seconds=stats.control_time_seconds,
        knockdowns=stats.knockdowns,
        head_strikes_landed=stats.head_strikes_landed,
        body_strikes_landed=stats.body_strikes_landed,
        leg_strikes_landed=stats.leg_strikes_landed,
        distance_strikes_landed=stats.distance_strikes_landed,
        clinch_strikes_landed=stats.clinch_strikes_landed,
        ground_strikes_landed=stats.ground_strikes_landed,
    )


# ── Single-event scraping ──────────────────────────────────────────────────


def _scrape_event(
    client: ScraperClient | object,
    session: Session,
    event_summary: EventSummary,
    event_html: str | None,
    fighter_cache: dict[str, int],
    result: IngestResult,
) -> None:
    """Process a single event: parse pre-fetched detail HTML and upsert to DB.

    Per D-11: parse failures on a page abort that page but continue to the next.

    The caller is responsible for batch-fetching event_html via
    ``client.map(_safe_fetch, ...)``. If fetching failed upstream, pass
    ``event_html=None`` — we'll count this event as rejected and return.

    Args:
        client: ScraperClient (or mock) for HTTP fetches.
        session: SQLAlchemy session.
        event_summary: Summary from event listing page.
        event_html: Pre-fetched event-detail HTML, or None if upstream fetch failed.
        fighter_cache: URL -> fighter_id cache (mutated in place).
        result: IngestResult to track accepted/rejected counts.
    """
    if event_html is None:
        # Upstream batch fetch already logged the failure; just tally it.
        result.rejected += 1
        return

    try:
        event_detail = parse_event_detail(event_html, event_summary.url)
    except (ValueError, RuntimeError) as exc:
        logger.error("Failed to parse event %s: %s", event_summary.name, exc)
        result.rejected += 1
        return

    # Parse event date
    event_date = parse_fight_date(event_detail.date_str)
    if event_date is None:
        logger.error(
            "Could not parse event date for %s: %s",
            event_detail.name,
            event_detail.date_str,
        )
        result.rejected += 1
        return

    # Upsert event
    db_event = upsert_event(
        session,
        name=event_detail.name,
        event_date=event_date,
        location=event_detail.location,
        source=SOURCE,
        source_url=event_summary.url,
    )

    # Batch-fetch all fight-detail HTMLs for this event in parallel. Using
    # _safe_fetch so one bad URL doesn't abort the whole batch.
    fight_urls = [f.fight_url for f in event_detail.fights]
    fight_fetch_results = client.map(functools.partial(_safe_fetch, client), fight_urls)

    # Process each fight
    fights_accepted = 0
    # Phase 22 REF-V22-01 — collect per-fight referee raw names for per-event
    # most-frequent aggregation (CONTEXT D-10 lossy aggregation; per-fight
    # placement deferred to v2.3+).
    event_referee_raw_names: list[str] = []
    for fight_summary, (_url, fight_html, fetch_err) in zip(
        event_detail.fights, fight_fetch_results, strict=False
    ):
        if fight_html is None:
            logger.warning("Failed to fetch fight %s: %s", fight_summary.fight_url, fetch_err)
            continue

        try:
            fight_detail = parse_fight_detail(fight_html)
        except (ValueError, RuntimeError) as exc:
            logger.warning("Failed to parse fight %s: %s", fight_summary.fight_url, exc)
            continue

        # Phase 22 REF-V22-01 — accumulate raw referee names for per-event
        # most-frequent aggregation post-loop.
        if fight_detail.referee:
            event_referee_raw_names.append(fight_detail.referee)

        # Fetch/cache fighter profiles
        fighter_a_id = _ensure_fighter(
            client,
            session,
            fight_summary.fighter_a_name,
            fight_summary.fighter_a_url,
            fighter_cache,
        )
        fighter_b_id = _ensure_fighter(
            client,
            session,
            fight_summary.fighter_b_name,
            fight_summary.fighter_b_url,
            fighter_cache,
        )

        # Convert to FightRow for validation per D-12
        try:
            fight_row = _convert_fight(
                event_name=event_detail.name,
                event_date_str=event_detail.date_str,
                location=event_detail.location,
                fight_summary=fight_summary,
                fight_detail=fight_detail,
            )
        except (ValidationError, ValueError) as exc:
            logger.warning("Rejected fight %s: %s", fight_summary.fight_url, exc)
            continue

        # Determine winner_id
        winner_id = None
        if fight_row.winner_name == fight_row.fighter_a_name:
            winner_id = fighter_a_id
        elif fight_row.winner_name == fight_row.fighter_b_name:
            winner_id = fighter_b_id

        # Upsert fight
        db_fight = upsert_fight(
            session,
            event_id=db_event.id,
            fighter_a_id=fighter_a_id,
            fighter_b_id=fighter_b_id,
            source=SOURCE,
            winner_id=winner_id,
            weight_class=fight_row.weight_class,
            method=fight_row.method,
            method_detail=fight_row.method_detail,
            round_finished=fight_row.round_finished,
            time_finished=fight_row.time_finished,
            is_title_fight=fight_row.is_title_fight,
            num_rounds=fight_row.num_rounds,
            source_url=fight_summary.fight_url,
        )

        # Upsert fight-level stats (round_number=0) for both fighters
        if fight_row.fighter_a_stats is not None:
            _upsert_stats_for_fighter(
                session, db_fight.id, fighter_a_id, 0, fight_row.fighter_a_stats
            )
        if fight_row.fighter_b_stats is not None:
            _upsert_stats_for_fighter(
                session, db_fight.id, fighter_b_id, 0, fight_row.fighter_b_stats
            )

        # Upsert per-round stats
        _upsert_per_round_stats(
            session,
            db_fight.id,
            fighter_a_id,
            fighter_b_id,
            fight_detail.per_round_totals,
            fight_detail.per_round_sig_strikes,
        )

        fights_accepted += 1

    # Phase 22 REF-V22-01 — per-event referee resolution + upsert. Per
    # CONTEXT D-10 lossy aggregation, pick the most-frequent referee across
    # the event's fights (ties broken by insertion order, Counter.most_common
    # default). If all fights had referee=None, leaves events.referee_id NULL.
    #
    # Idempotency (CR-01 fix): only set referee_id on first resolution. This
    # matches the "non-None-only" overwrite policy used by upsert_event /
    # upsert_fight / upsert_fighter elsewhere in the pipeline. Without this
    # guard, a partial re-ingest (subset of fight pages) could shift the
    # most-frequent winner and silently replace a previously-correct
    # referee_id with a less-well-supported value. HTML drift on a single
    # fight could likewise flip the event referee on rerun with no audit
    # trail (the raw alias is still appended to referees.alias_list via
    # upsert_referee, but the canonical FK overwrite would be lossy).
    if event_referee_raw_names:
        most_frequent_referee = Counter(event_referee_raw_names).most_common(1)[0][0]
        new_ref_id = upsert_referee(session, raw_name=most_frequent_referee)
        if db_event.referee_id is None and new_ref_id is not None:
            db_event.referee_id = new_ref_id

    # Atomic per event
    session.commit()
    result.accepted += fights_accepted
    logger.info("Event %s: %d fights accepted", event_detail.name, fights_accepted)


def _ensure_fighter(
    client: ScraperClient | object,
    session: Session,
    name: str,
    url: str,
    fighter_cache: dict[str, int],
) -> int:
    """Ensure fighter exists in DB, fetching profile if not cached.

    Uses the name from the event detail page (not the profile page) as the
    upsert key, because the event detail has the authoritative per-fight name.
    Profile data is only used for physical attributes.

    Returns fighter_id.
    """
    if url in fighter_cache:
        # Already fetched this session; just look up by name for the id
        fighter = upsert_fighter(session, name, SOURCE)
        return fighter.id

    # Fetch and parse profile for physical attributes
    source_id = _extract_hex_id(url)
    try:
        profile_html = client.get(url)
        profile = parse_fighter_profile(profile_html)
        fighter_row = _convert_fighter_profile(profile)

        # Use name from event detail (not profile) as upsert key
        fighter = upsert_fighter(
            session,
            name=name,
            source=SOURCE,
            source_id=source_id,
            nickname=profile.nickname,
            height_inches=fighter_row.height_inches,
            reach_inches=fighter_row.reach_inches,
            stance=fighter_row.stance,
            date_of_birth=fighter_row.date_of_birth,
        )
    except (ValueError, RuntimeError) as exc:
        logger.warning(
            "Failed to fetch/parse fighter profile %s, using name-only fallback: %s",
            url,
            exc,
        )
        fighter = upsert_fighter(session, name, SOURCE, source_id=source_id)

    fighter_cache[url] = fighter.id
    return fighter.id


# ── Public entry points ─────────────────────────────────────────────────────


def scrape_all_events(
    session: Session,
    client: ScraperClient | object | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    after_date: date | None = None,
    workers: int = 4,
) -> IngestResult:
    """Scrape all completed events from UFCStats.com.

    Per D-06 (ufc scrape all):
    1. Fetch event listing from EVENT_LIST_URL
    2. Filter out upcoming events
    3. Sort chronologically (oldest first) for Elo computation dependency
    4. Batch-fetch event-detail HTML in parallel (workers=4 by default)
    5. Process each event via _scrape_event (per-event DB commit preserved)

    Args:
        session: SQLAlchemy session.
        client: ScraperClient instance (created if not provided).
        progress_callback: Optional callback(current_index, total_events).
        after_date: If provided, skip events on or before this date (for resuming).
        workers: Concurrent worker threads for the batch fetches when ``client``
            is not provided (default 4). Ignored when ``client`` is supplied
            (the caller already chose the concurrency).

    Returns:
        IngestResult with accepted/rejected/updated counts.
    """
    own_client = False
    if client is None:
        client = ScraperClient(workers=workers)
        own_client = True

    try:
        result = IngestResult()

        # Fetch and parse event listing
        event_list_html = client.get(EVENT_LIST_URL)
        events = parse_event_list(event_list_html, min_events=1)

        # Filter out upcoming events
        completed = [e for e in events if not e.is_upcoming]
        logger.info(
            "Found %d completed events (filtered %d upcoming)",
            len(completed),
            len(events) - len(completed),
        )

        # Sort oldest-first by date for chronological processing
        completed.sort(key=lambda e: parse_fight_date(e.date_str) or date.min)

        # Skip events on or before after_date (for resuming)
        if after_date is not None:
            before_count = len(completed)
            completed = [
                e for e in completed if (parse_fight_date(e.date_str) or date.min) > after_date
            ]
            logger.info(
                "Skipped %d events on or before %s, %d remaining",
                before_count - len(completed),
                after_date,
                len(completed),
            )

        # Batch-fetch all event-detail HTMLs in parallel. Preserves input order.
        event_urls = [e.url for e in completed]
        event_fetch_results = client.map(functools.partial(_safe_fetch, client), event_urls)

        fighter_cache: dict[str, int] = {}

        for idx, (event_summary, (_url, event_html, fetch_err)) in enumerate(
            zip(completed, event_fetch_results, strict=False)
        ):
            if event_html is None:
                logger.error(
                    "Failed to fetch event %s: %s",
                    event_summary.name,
                    fetch_err,
                )
            _scrape_event(
                client,
                session,
                event_summary,
                event_html,
                fighter_cache,
                result,
            )
            if progress_callback is not None:
                progress_callback(idx, len(completed))

        return result

    finally:
        if own_client:
            client.close()


def scrape_latest_events(
    session: Session,
    client: ScraperClient | object | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    workers: int = 4,
) -> IngestResult:
    """Scrape only new events from UFCStats.com (incremental update).

    Per D-06, D-07 (ufc scrape latest):
    1. Fetch event listing
    2. Filter out upcoming events
    3. Compare against existing events in DB (by source_url and name+date)
    4. Batch-fetch event-detail HTML in parallel (workers=4 by default)
    5. Scrape only events not yet in DB

    Args:
        session: SQLAlchemy session.
        client: ScraperClient instance (created if not provided).
        progress_callback: Optional callback(current_index, total_events).
        workers: Concurrent worker threads for the batch fetches when ``client``
            is not provided (default 4). Ignored when ``client`` is supplied.

    Returns:
        IngestResult with accepted/rejected/updated counts.
    """
    own_client = False
    if client is None:
        client = ScraperClient(workers=workers)
        own_client = True

    try:
        result = IngestResult()

        # Fetch and parse event listing
        event_list_html = client.get(EVENT_LIST_URL)
        events = parse_event_list(event_list_html, min_events=1)

        # Filter out upcoming events
        completed = [e for e in events if not e.is_upcoming]

        # Query existing events in DB -- match by source_url (unique per event)
        # and also by name+date as fallback for events without source_url
        existing_events = session.query(Event).filter(Event.source == SOURCE).all()
        existing_urls = {e.source_url for e in existing_events if e.source_url}
        existing_name_dates = {(e.name, e.date) for e in existing_events}

        # Filter to new events only
        new_events = [
            e
            for e in completed
            if (
                e.url not in existing_urls
                and (e.name, parse_fight_date(e.date_str)) not in existing_name_dates
            )
        ]

        logger.info(
            "Found %d new events out of %d completed (%d already in DB)",
            len(new_events),
            len(completed),
            len(completed) - len(new_events),
        )

        if not new_events:
            return result

        # Sort oldest-first
        new_events.sort(key=lambda e: parse_fight_date(e.date_str) or date.min)

        # Batch-fetch all event-detail HTMLs in parallel.
        event_urls = [e.url for e in new_events]
        event_fetch_results = client.map(functools.partial(_safe_fetch, client), event_urls)

        fighter_cache: dict[str, int] = {}

        for idx, (event_summary, (_url, event_html, fetch_err)) in enumerate(
            zip(new_events, event_fetch_results, strict=False)
        ):
            if event_html is None:
                logger.error(
                    "Failed to fetch event %s: %s",
                    event_summary.name,
                    fetch_err,
                )
            _scrape_event(
                client,
                session,
                event_summary,
                event_html,
                fighter_cache,
                result,
            )
            if progress_callback is not None:
                progress_callback(idx, len(new_events))

        return result

    finally:
        if own_client:
            client.close()
