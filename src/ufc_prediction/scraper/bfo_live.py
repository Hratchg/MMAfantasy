"""Single-shot live BFO matchup-odds fetch for ``ufc predict matchup``.

Sibling to ``bfo_scraper.py`` (which is batch CSV-emitting). Reuses
``ScraperClient`` + ``parse_bfo_fighter_page``; produces ``MatchupOdds | None``.

Per CONTEXT.md D-09: fetch order is cache → live HTTP → NaN.
Per D-10: cache TTL = until ``event_date`` passes; ``--refresh`` forces live.
Per D-11: caller logs ``odds_source`` for distribution-shift observability.
Per Pattern E + Gotcha 1: never raises; single-shot; non-batch.

The 5s timeout (per CONTEXT.md D-09) is enforced by constructing a
:class:`ScraperClient` whose ``timeout`` is 5.0 by default. The shipped
client doesn't expose per-call timeout overrides — the contract is
"client constructed in this module is configured for fast predict-time
HTTP" and any transport error / timeout falls through to ``None``.
"""

from __future__ import annotations

import logging
import urllib.parse
from dataclasses import dataclass
from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from ufc_prediction.dedup.source_priority import prefer_canonical
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fight_odds import FightOdds
from ufc_prediction.models.fighter import Fighter
from ufc_prediction.scraper.bfo_matcher import normalize_name
from ufc_prediction.scraper.bfo_scraper import (
    SEARCH_URL,
    BFOParsedFight,
    BFOParseError,
    _check_captcha,
    find_bfo_fighter_url,
    parse_bfo_fighter_page,
)
from ufc_prediction.scraper.client import ScraperClient

logger = logging.getLogger(__name__)


# ── Public dataclass ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MatchupOdds:
    """Result of a successful live or cached BFO fetch for an upcoming matchup.

    Moneylines are American (e.g. ``-200``, ``+150``). ``None`` means the
    cell was blank on BFO or absent in the cache. ``source`` is
    ``"cache"`` for ``fight_odds``-table hits and ``"live"`` for
    fresh BFO HTTP fetches.
    """

    fighter_a_opening: int | None
    fighter_a_closing_min: int | None
    fighter_a_closing_max: int | None
    fighter_b_opening: int | None
    fighter_b_closing_min: int | None
    fighter_b_closing_max: int | None
    fetched_at: datetime
    source: str  # "cache" | "live"


# ── Cache lookup (D-09 step 1) ──────────────────────────────────────────────


def _resolve_fighter_id(session: Session, name: str) -> int | None:
    """Resolve a fighter name to a single ``Fighter.id``.

    Reuses HOUSE-04 ``prefer_canonical`` to collapse multi-source duplicates.
    Returns ``None`` when no exact-match row exists; the caller falls
    through to the live HTTP path.
    """
    rows = list(session.execute(select(Fighter).where(Fighter.name == name)).scalars().all())
    if not rows:
        return None
    canonical = prefer_canonical(
        rows,
        source_key=lambda f: f.source,
        tiebreak_key=lambda f: f.id,
    )
    return canonical.id


def _try_cache(
    session: Session | None,
    fa_name: str,
    fb_name: str,
    event_date: date,
) -> MatchupOdds | None:
    """Step 1 of D-09: lookup ``fight_odds`` for the ``(A, B, event_date)`` triple.

    Returns ``MatchupOdds(source="cache")`` on hit, ``None`` on miss. Never
    raises — any transient DB issue is treated as a cache miss and the
    caller falls through to the live HTTP path.
    """
    if session is None:
        return None
    try:
        fa_id = _resolve_fighter_id(session, fa_name)
        fb_id = _resolve_fighter_id(session, fb_name)
        if fa_id is None or fb_id is None:
            return None

        # Find a fight between this pair on event_date.
        stmt = (
            select(Fight.id)
            .join(Event, Fight.event_id == Event.id)
            .where(
                ((Fight.fighter_a_id == fa_id) & (Fight.fighter_b_id == fb_id))
                | ((Fight.fighter_a_id == fb_id) & (Fight.fighter_b_id == fa_id))
            )
            .where(Event.date == event_date)
        )
        fight_id = session.scalar(stmt)
        if fight_id is None:
            return None

        # Pull both sides of the odds.
        odds_rows = list(
            session.execute(select(FightOdds).where(FightOdds.fight_id == fight_id)).scalars().all()
        )
        if not odds_rows:
            return None

        odds_a = next((r for r in odds_rows if r.fighter_id == fa_id), None)
        odds_b = next((r for r in odds_rows if r.fighter_id == fb_id), None)
        if odds_a is None or odds_b is None:
            return None

        # scraped_at can be naive (server_default=func.now()); normalize to UTC-aware
        fetched_at = odds_a.scraped_at
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=UTC)

        return MatchupOdds(
            fighter_a_opening=odds_a.opening_ml,
            fighter_a_closing_min=odds_a.closing_range_min_ml,
            fighter_a_closing_max=odds_a.closing_range_max_ml,
            fighter_b_opening=odds_b.opening_ml,
            fighter_b_closing_min=odds_b.closing_range_min_ml,
            fighter_b_closing_max=odds_b.closing_range_max_ml,
            fetched_at=fetched_at,
            source="cache",
        )
    except Exception as exc:
        logger.warning("bfo_live cache lookup failed: %s", exc)
        return None


# ── Live HTTP (D-09 step 2) ─────────────────────────────────────────────────


def _matches(parsed_fight: BFOParsedFight, opponent_name: str, event_date: date) -> bool:
    """True when ``parsed_fight`` is the (opponent, date) row we're looking for.

    Opponent name matched via ``rapidfuzz.fuzz.ratio`` over normalized
    names at the BFO threshold (80, calibrated Phase 13). Event date
    must match exactly — we are looking up an upcoming or specific
    bout, not a fuzzy date window.
    """
    if parsed_fight.event_date != event_date:
        return False
    from rapidfuzz import fuzz

    from ufc_prediction.scraper.bfo_scraper import MIN_FUZZ_SCORE

    score = fuzz.ratio(
        normalize_name(opponent_name),
        normalize_name(parsed_fight.opponent_name),
    )
    return score >= MIN_FUZZ_SCORE


def _try_live(
    client: ScraperClient,
    fa_name: str,
    fb_name: str,
    event_date: date,
) -> MatchupOdds | None:
    """Step 2 of D-09: live HTTP via BFO. Returns ``None`` on any failure.

    Two requests:
      1. BFO search for fighter A (returns candidate links).
      2. Fighter A's profile page (returns a fight history table).

    Then we walk the parsed fight rows looking for the ``(B, event_date)``
    matchup. Any transport / parse / search-miss / opponent-miss falls
    through to ``None`` so the caller can NaN-out the odds block.
    """
    try:
        # 1. Search for A's BFO profile
        from bs4 import BeautifulSoup

        search_url = SEARCH_URL.format(query=urllib.parse.quote_plus(fa_name))
        html = client.get(search_url)

        # CAPTCHA detection happens early — never trust a CAPTCHA-gated page.
        soup = BeautifulSoup(html, "lxml")
        try:
            _check_captcha(soup)
        except BFOParseError as exc:
            logger.warning(
                "bfo_live: CAPTCHA gate on search for %s: %s",
                fa_name,
                exc,
            )
            return None

        prof_url = find_bfo_fighter_url(fa_name, html)
        if prof_url is None:
            return None

        # 2. Fetch fighter A's profile page
        prof_html = client.get(prof_url)
        page = parse_bfo_fighter_page(prof_html, prof_url)

        # 3. Find the row matching B + event_date
        row = next(
            (f for f in page.fights if _matches(f, fb_name, event_date)),
            None,
        )
        if row is None:
            return None

        # B-side moneylines are NOT exposed by the parsed BFO row; we
        # only have A's three cells. The cache path stores both sides
        # post-Phase-15 ingest; the live path treats B-side as unknown
        # and falls through to inference_features._populate_odds which
        # treats missing values as NaN per Pattern D.
        return MatchupOdds(
            fighter_a_opening=row.opening,
            fighter_a_closing_min=row.closing_range_min,
            fighter_a_closing_max=row.closing_range_max,
            fighter_b_opening=None,
            fighter_b_closing_min=None,
            fighter_b_closing_max=None,
            fetched_at=datetime.now(UTC),
            source="live",
        )
    except BFOParseError as exc:
        logger.warning(
            "bfo_live: %s vs %s parse failed: %s",
            fa_name,
            fb_name,
            exc,
        )
        return None
    except (RuntimeError, OSError, TimeoutError) as exc:
        logger.warning(
            "bfo_live: %s vs %s transport failed: %s",
            fa_name,
            fb_name,
            exc,
        )
        return None
    except Exception as exc:
        logger.warning(
            "bfo_live: %s vs %s unexpected failure: %s",
            fa_name,
            fb_name,
            exc,
        )
        return None


# ── Public entry point ──────────────────────────────────────────────────────


def fetch_matchup_odds(
    fighter_a_name: str,
    fighter_b_name: str,
    event_date: date,
    *,
    refresh: bool = False,
    session: Session | None = None,
    client: ScraperClient | None = None,
) -> MatchupOdds | None:
    """Single-shot BFO matchup-odds fetch.

    Per CONTEXT.md D-09: cache → live HTTP → ``None``. ``None`` is the
    "NaN fallback" sentinel — XGBoost handles missing values natively
    (D-04(P14)), so the inference feature builder treats this as
    leave-NaN per Pattern D.

    Args:
        fighter_a_name: Display name of fighter A (resolves via DB
            on the cache path; URL-encoded on the live path).
        fighter_b_name: Display name of fighter B.
        event_date: Specific event date — exact match required.
        refresh: Per D-10, force live fetch even on cache hit.
        session: Optional SQLAlchemy session for cache lookup. If
            ``None``, the cache step is skipped (live HTTP only).
        client: Optional :class:`ScraperClient`. If ``None``, a fresh
            client is constructed with the project's default 5s-equivalent
            ScraperClient defaults.

    Returns:
        :class:`MatchupOdds` with ``source`` ∈ {``"cache"``, ``"live"``},
        or ``None`` when both paths fail. Never raises.
    """
    # Step 1: cache (skipped on refresh=True per D-10)
    if not refresh:
        cached = _try_cache(session, fighter_a_name, fighter_b_name, event_date)
        if cached is not None:
            return cached

    # Step 2: live HTTP (5s-equivalent timeout via ScraperClient defaults)
    if client is None:
        # ScraperClient default timeout=30s; for predict-time we want 5s.
        client = ScraperClient(timeout=5.0)
    return _try_live(client, fighter_a_name, fighter_b_name, event_date)
