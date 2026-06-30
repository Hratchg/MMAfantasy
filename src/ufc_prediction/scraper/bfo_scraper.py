"""BestFightOdds.com scraper for historical UFC fight odds.

Provides HTML parsers for BFO fighter profiles and search results, a
:class:`BFOScraper` orchestrator that ties the pipeline together, and
typed result dataclasses (:class:`BFOParsedFight`, :class:`BFOFighterPage`,
:class:`ScrapeSummary`) plus :class:`BFOParseError`.

Why this module exists (Phase 15, ODDS-01):
    The upstream ``ufcscraper.odds_scraper.BestFightOddsScraper`` crashes
    on Python 3.14 because it passes a closure to ``multiprocessing.Process``
    (spawn-mode default on macOS since 3.8 cannot pickle closures). This
    in-house scraper uses our :class:`ScraperClient` thread pool instead,
    which has no pickling step.

Design decisions (CONTEXT.md):
    D-01: Do NOT use ufcscraper's BestFightOddsScraper.
    D-02: Read the fighter list directly from the DB (no UFCStats CSV prereq).
    D-03: Use the existing ScraperClient threading model (workers=4 default).
    D-04: Emit CSVs in the exact format the existing ``BFOOddsIngester``
        already consumes; ingester is unchanged.

Threat mitigations (15-01-PLAN.md threat_model):
    T-15-01-01 Tampering: defensive selectors raise BFOParseError on
        malformed HTML (Pitfall 5).
    T-15-01-02 DoS / CAPTCHA: ``id="hfmr8"`` element (BFO's Cloudflare
        gate) detected up-front and raised as a typed error so the caller
        can pause / back off (Pitfall 1).
    T-15-01-04 Spoofing: search-result fuzzy match through
        ``bfo_matcher.match_bfo_name`` at threshold 80 (Phase 13 calibrated);
        unmatched DB names go to ``ScrapeSummary.unmatched_db_names`` for
        human review (Pitfall 6).

Notes:
    - ``parse_bfo_fighter_page`` consumes server-side-rendered HTML; BFO
      does NOT need Selenium (verified via WebFetch 2026-04-29 in
      15-RESEARCH.md).
    - All CSV cells for ``None`` are written as the empty string. The
      Pydantic ``BFOOddsRow._coerce_blank_ml`` validator on the ingester
      side converts blanks back to ``None``.
    - ``functools.partial(_safe_fetch_bfo, self._client)`` is the exact
      callable handed to ``ScraperClient.map`` so per-URL HTTP failures
      are surfaced as sentinel tuples instead of aborting the batch.
"""

from __future__ import annotations

import csv
import functools
import logging
import re
import urllib.parse
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from rapidfuzz import fuzz
from sqlalchemy import select

from ufc_prediction.models.fighter import Fighter
from ufc_prediction.scraper.bfo_matcher import normalize_name  # noqa: F401  (kept for downstream re-use)
from ufc_prediction.scraper.bfo_models import BFOFighterName, BFOOddsRow

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

BFO_BASE = "https://www.bestfightodds.com"
SEARCH_URL = f"{BFO_BASE}/search?query={{query}}"
FIGHTER_URL_TEMPLATE = f"{BFO_BASE}/fighters/{{slug}}"
MIN_FUZZ_SCORE = 80  # mirrors bfo_matcher (Phase 13 calibration)
_ODDS_CSV = "BestFightOdds_odds.csv"
_NAMES_CSV = "fighters_names.csv"
_CAPTCHA_ID = "hfmr8"  # BFO's Cloudflare-style gate element

# BFO uses ordinal date suffixes ("Mar 4th 2023", "Jan 1st 2026"). We
# strip the suffix before strptime; otherwise %d would not match "4th".
_ORDINAL_RE = re.compile(r"(\d+)(?:st|nd|rd|th)\b", flags=re.IGNORECASE)
_ID_FROM_HREF_RE = re.compile(r"-(\d+)$")


# ── Public exception type ────────────────────────────────────────────────────


class BFOParseError(ValueError):
    """Raised when BFO HTML structure is malformed or CAPTCHA-gated.

    Two distinct failure modes share this type, distinguished by the
    message body:

        - Structural: "No team-stats-table on <url>" (T-15-01-01)
        - CAPTCHA: "BFO CAPTCHA gate hit (id=hfmr8): ..." (T-15-01-02)
    """


# ── Public dataclasses ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class BFOParsedFight:
    """One parsed (fighter, opponent, event_date) row from a BFO profile.

    Moneylines are American (e.g. ``-200``, ``+150``). ``None`` means the
    cell was blank on the BFO page (upcoming or absent line).
    """

    event_date: date
    opponent_name: str
    opponent_bfo_id: str | None
    opening: int | None
    closing_range_min: int | None
    closing_range_max: int | None


@dataclass(frozen=True)
class BFOFighterPage:
    """Result of parsing one BFO fighter profile page.

    ``fights`` is in BFO's display order (most-recent first by virtue of
    BFO's HTML; we preserve order without sorting).
    """

    name: str
    url: str
    fights: list[BFOParsedFight]


@dataclass
class ScrapeSummary:
    """Aggregate counts + diagnostics for one ``BFOScraper.scrape_all`` run.

    ``unmatched_db_names`` is a list of DB fighter names whose BFO search
    returned no candidate >= MIN_FUZZ_SCORE (Pitfall 6 — surface for
    manual review rather than silently dropping).
    """

    fighters_processed: int = 0
    fighters_matched: int = 0
    fighters_unmatched: int = 0
    fights_emitted: int = 0
    parse_errors: int = 0
    captcha_hits: int = 0
    unmatched_db_names: list[str] = field(default_factory=list)


# ── Module-level helpers ─────────────────────────────────────────────────────


def _safe_fetch_bfo(
    client: object,
    url: str,
) -> tuple[str, str | None, Exception | None]:
    """Fetch ``url`` via ``client.get``, returning a sentinel tuple.

    Mirrors ``_safe_fetch_sherdog`` in :mod:`scraper.sherdog`: lets
    ``client.map(...)`` batches survive individual URL failures instead
    of aborting on the first exception.

    Returns ``(url, html_or_None, exc_or_None)``.
    """
    try:
        return (url, client.get(url), None)  # type: ignore[attr-defined]
    except (RuntimeError, ValueError) as exc:
        return (url, None, exc)


def _parse_bfo_date(date_str: str) -> date:
    """Parse a BFO date string like 'Mar 4th 2023' into a ``date``.

    Strips the ordinal suffix (st/nd/rd/th) and uses ``strptime``.
    Raises ``ValueError`` on a non-parseable string.
    """
    cleaned = _ORDINAL_RE.sub(r"\1", date_str.strip())
    return datetime.strptime(cleaned, "%b %d %Y").date()


def _id_from_href(href: str) -> str | None:
    """Extract the trailing BFO ID from a fighter URL.

    ``/fighters/Jon-Jones-819`` -> ``"819"``. Returns ``None`` if the
    pattern does not match (defensive against malformed URLs).
    """
    if not href:
        return None
    m = _ID_FROM_HREF_RE.search(href.rstrip("/"))
    return m.group(1) if m else None


def _check_captcha(soup: BeautifulSoup) -> None:
    """Raise BFOParseError if the BFO CAPTCHA gate element is present."""
    el = soup.find(id=_CAPTCHA_ID)
    if el is not None:
        snippet = el.get_text(strip=True)[:80] if hasattr(el, "get_text") else ""
        raise BFOParseError(f"BFO CAPTCHA gate hit (id={_CAPTCHA_ID}): {snippet}")


def _parse_moneyline(cell_text: str) -> int | None:
    """Return ``int`` of an American moneyline string, or ``None`` if blank.

    BFO renders signed integers (``-200``, ``+150``) — both parse via
    ``int(...)``. An empty string maps to ``None`` (upcoming or absent).
    """
    text = cell_text.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        # Defensive: BFO may inject "..." or other non-numeric content.
        return None


def _moneyline_cells(row: Tag) -> list[Tag]:
    """Return the three ``<td.moneyline>`` cells in document order.

    BFO renders one ``dash-cell`` between closing_min and closing_max in
    the row, so a naive ``find_all`` already returns exactly 3 cells in
    [opening, closing_min, closing_max] order. We still slice to be
    defensive against future BFO HTML drift.
    """
    cells = row.find_all("td", class_="moneyline")
    return cells[:3]


# ── Public parsers ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BFOEventMatchup:
    """One parsed (fighter A, fighter B, odds-range) row from a BFO event page.

    Phase 21 (Plan 21-01 Task 3): emitted by ``parse_bfo_event_page`` when
    walking the per-event ``<table class="odds-table">`` matchup rows. The
    ``opening`` field is set to the leftmost-bookie cell (BFO does not
    expose distinct opening/closing on event pages — the per-bookie cells
    ARE the closing range; ``opening`` is recorded as the first non-empty
    cell as a best-effort approximation, matching ufcscraper's CSV shape).

    Each matchup yields TWO ``BFOOddsRow`` rows on CSV emit (one per
    fighter side); both rows share the same composite ``fight_id`` so the
    existing ``BFOOddsIngester._load_odds_rows`` 2-rows-per-fight invariant
    holds.
    """

    matchup_id: str  # BFO ``mu-{N}`` numeric id (kept as string for CSV)
    fighter_a_name: str
    fighter_a_slug: str  # ``Justin-Burlinson-7808`` (BFO URL slug)
    fighter_b_name: str
    fighter_b_slug: str
    a_opening: int | None
    a_closing_min: int | None
    a_closing_max: int | None
    b_opening: int | None
    b_closing_min: int | None
    b_closing_max: int | None


@dataclass(frozen=True)
class BFOEventPage:
    """Parsed BFO event page — header + list of matchups.

    Phase 21 Task 3: produced by ``parse_bfo_event_page`` for use by
    ``BFOScraper.scrape_event_urls``. ``event_date`` is parsed from the
    page's ``<meta name="description">`` (which embeds ``"on Month D,
    YYYY"`` text); ``event_url`` is the URL the page was fetched from.
    """

    event_name: str
    event_date: date
    event_url: str
    matchups: list[BFOEventMatchup]


# ── Phase 21 Plan 21-01 Task 3: BFO event-page parser ────────────────────────

# Pattern: ``data-li="[bookieId,side,matchupId]"`` where side ∈ {1,2}.
_DATA_LI_RE = re.compile(r"\[(\d+),(\d+),(\d+)\]")
# Pattern: extract ``YYYY`` from a meta description like
# "...for Cage Warriors 205 on April 26, 2026. Find the best..."
_DESC_DATE_RE = re.compile(r"on\s+([A-Z][a-z]+)\s+(\d{1,2}),\s*(\d{4})", re.IGNORECASE)
# Pattern: extract a fighter slug from /fighters/<Slug-Name-NNNN>.
_FIGHTER_SLUG_RE = re.compile(r"^/fighters/([A-Za-z0-9-]+)$")
# Single source of truth for the BFO event slug+id grammar.
#
# HYG-V26-02 (Phase 49): consolidated from the duplicated copy in
# `scripts/bfo_backfill.py:_EVENT_HREF_RE`. Both extractors now compose
# regexes from this shared constant — the AF-startup assert in
# `bfo_backfill._run_af_startup_asserts()` pins it as the v2.6 canonical
# pattern so any drift between the two call sites halts at startup.
#
# CORPUS-V25-02 lineage (Phase 40 Plan 40-01 Task 3) URL-drift fix:
# Tightened the slug character class to forbid a trailing dash before the
# `-<numeric_id>` separator. The previous permissive `(.+-\d+)` greedy
# pattern matched malformed BFO hrefs such as `/events/foo--123` as
# slug=`foo-`, id=`123`, causing canonical URLs emitted by
# `_extract_event_name_and_url` to inherit the drift.
EVENT_SLUG_ID_PATTERN: str = r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?-\d+"
_EVENT_HREF_RE = re.compile(rf"^/events/({EVENT_SLUG_ID_PATTERN})$")


def _extract_event_date_from_meta(soup: BeautifulSoup) -> date | None:
    """Pull ``date(YYYY, M, D)`` from BFO's ``<meta name="description">``.

    BFO embeds "on Month D, YYYY" (e.g. "on April 26, 2026") in the page
    description. Returns ``None`` if no parseable date is found — the
    caller is responsible for skipping the event in that case.
    """
    meta = soup.find("meta", attrs={"name": "description"})
    if meta is None:
        return None
    content = meta.get("content", "") or ""
    match = _DESC_DATE_RE.search(content)
    if match is None:
        return None
    month_name, day_str, year_str = match.groups()
    try:
        return datetime.strptime(f"{month_name} {day_str} {year_str}", "%B %d %Y").date()
    except ValueError:
        return None


def _extract_event_name_and_url(
    soup: BeautifulSoup,
    event_url: str,
) -> tuple[str, str]:
    """Return (event_name, canonical_event_url) from the event-header div.

    Falls back to the input ``event_url`` when the header is missing —
    the parser still raises ``BFOParseError`` upstream if the table-div
    is absent, so this fallback is only used on partial-DOM edge cases.

    WR-03 fix (REVIEW.md): log a structural warning when the header div
    or the inner ``<h1>`` is missing. The function still returns ``""``
    for the event_name (to preserve the partial-DOM tolerance that
    ``parse_bfo_event_page`` relies on), but the silent fallback that
    rest-of-parser-observes "fail loudly" was inconsistent.
    """
    header = soup.select_one("div.table-header")
    if header is None:
        logger.warning(
            "BFO event page %s has no div.table-header (returning empty event_name)",
            event_url,
        )
        return ("", event_url)
    h1 = header.find("h1")
    if h1 is None:
        logger.warning(
            "BFO event header on %s has no <h1> (returning empty event_name)",
            event_url,
        )
        name = ""
    else:
        name = h1.get_text(strip=True)
    link = header.find("a", href=True)
    canonical = event_url
    if link is not None:
        href = link.get("href", "")
        canonical = canonicalize_event_url(href, fallback=event_url)
    return name, canonical


def canonicalize_event_url(href: str, *, fallback: str) -> str:
    """Return the absolute BFO canonical URL for an ``/events/<slug>-<id>`` href.

    CORPUS-V25-02 (Phase 40 Plan 40-01 Task 3) URL-drift fix point. The
    href is validated against :data:`_EVENT_HREF_RE` — which forbids
    trailing-dash slugs such as ``foo--123`` — before being concatenated
    with :data:`BFO_BASE`. Hrefs that fail validation fall back to the
    supplied ``fallback`` URL (typically the input ``event_url`` passed
    through ``_extract_event_name_and_url``), preventing drifted URLs
    from leaking into downstream rows.

    Pure function (no I/O, no side effects beyond a single warning log
    when an href is rejected) — pinned by the
    ``tests/scraper/test_scrape_event_urls.py`` golden-file regression.

    Args:
        href: Relative path emitted by the BFO event-header link
            (e.g. ``"/events/ufc-300-9999"``).
        fallback: URL to return when ``href`` does not match the
            canonical pattern (the originating event URL preserves
            partial-DOM tolerance for ``_extract_event_name_and_url``).

    Returns:
        Absolute URL (``BFO_BASE`` + validated href) on match; otherwise
        ``fallback``.
    """
    if not href:
        return fallback
    match = _EVENT_HREF_RE.match(href)
    if match is None:
        logger.warning(
            "BFO event href %r does not match canonical /events/<slug>-<id> "
            "pattern; falling back to %r (CORPUS-V25-02 drift guard).",
            href,
            fallback,
        )
        return fallback
    return f"{BFO_BASE}{href}"


def _slug_from_fighter_href(href: str) -> str | None:
    """Return the BFO fighter slug from an ``/fighters/<Slug-Name-NNNN>`` href.

    Returns ``None`` if the href does not match the expected pattern
    (defensive against link drift / non-fighter rows).
    """
    if not href:
        return None
    match = _FIGHTER_SLUG_RE.match(href)
    return match.group(1) if match is not None else None


def _matchup_id_from_tr(tr: Tag) -> str | None:
    """Return ``mu-{N}`` row's numeric id (without the ``mu-`` prefix).

    Returns ``None`` for rows that are not matchup-anchor rows (i.e.,
    proposition or fighter-B rows).
    """
    row_id = tr.get("id") or ""
    if not row_id.startswith("mu-"):
        return None
    return row_id[3:]


def _bookie_lines_for_matchup(
    odds_table: Tag,
    matchup_id: str,
    side: int,
) -> list[int]:
    """Return all integer bookie lines for ``(matchup_id, side)``.

    Walks the right-hand ``<table class="odds-table">`` and pulls every
    ``<td.but-sg data-li="[*,side,matchupId]">`` cell's first ``<span>``
    text, parsing it as an American moneyline. Skips blanks + non-numeric
    cells (e.g. the prop popup column).
    """
    lines: list[int] = []
    target = f"[{{}},{side},{matchup_id}]"  # data-li format: [bookie,side,mu]
    for cell in odds_table.find_all("td", attrs={"data-li": True}):
        data_li = cell.get("data-li", "")
        match = _DATA_LI_RE.match(data_li)
        if match is None:
            continue
        _bookie, cell_side, cell_mu = match.groups()
        if cell_side != str(side) or cell_mu != matchup_id:
            continue
        # First non-empty span text inside the cell.
        span = cell.find("span")
        if span is None:
            continue
        line = _parse_moneyline(span.get_text(strip=True))
        if line is not None:
            lines.append(line)
    return lines


def parse_bfo_event_page(html: str, event_url: str) -> BFOEventPage:
    """Parse a BFO event page (e.g. ``/events/<slug>-<id>``) into matchups.

    Phase 21 (Plan 21-01 Task 3): the inverse of ``parse_bfo_fighter_page``.
    Walks the matchup-label table on the left and the per-bookie odds
    table on the right; pairs each ``<tr id="mu-N">`` with the next
    sibling ``<tr>`` (opponent), then aggregates per-bookie cells via
    ``data-li="[bookie,side,muId]"`` into per-fighter min/max moneyline
    ranges.

    Sentinel + safety:
      - CAPTCHA gate (``id="hfmr8"``) raises ``BFOParseError`` upstream
        of any selector work (Pitfall #4).
      - Missing ``<div class="table-div">`` on the page raises
        ``BFOParseError`` (event page absent / malformed).
      - Date parsing falls back to ``date.max`` sentinel when the
        ``<meta name="description">`` lacks a parseable date — caller
        ``scrape_event_urls`` filters those out (mirrors the
        ``parse_bfo_fighter_page`` upcoming-fight pattern).

    Args:
        html: Raw HTML body of the event page.
        event_url: The URL the page was fetched from (for error context
            and as a canonical-URL fallback).

    Returns:
        ``BFOEventPage`` with the event name, parsed date, and a list of
        ``BFOEventMatchup`` rows in DOM order.

    Raises:
        BFOParseError: On CAPTCHA gate or missing table-div container.
    """
    soup = BeautifulSoup(html, "lxml")

    # 1. CAPTCHA detection FIRST (T-15-01-02 carry-forward).
    _check_captcha(soup)

    # 2. Locate the event container; BFO uses div.table-div with
    # id="event{numeric_id}". Missing it = not an event page (likely a
    # homepage_fallback or 404 body that returned 200).
    table_div = soup.select_one("div.table-div[id^='event']")
    if table_div is None:
        raise BFOParseError(f"No event table-div on {event_url}")

    event_name, canonical_url = _extract_event_name_and_url(soup, event_url)
    parsed_date = _extract_event_date_from_meta(soup)
    if parsed_date is None:
        # Use date.max sentinel (mirrors parse_bfo_fighter_page upcoming
        # path); scrape_event_urls drops these before CSV emit.
        parsed_date = date.max

    # 3. Walk both tables. The LEFT table (odds-table-responsive-header)
    # carries matchup labels in tr id="mu-N" → next tr (opponent) form.
    # The RIGHT table (table.odds-table inside table-scroller) carries the
    # per-bookie odds cells with data-li="[bookie,side,muId]".
    label_tables = table_div.select("table.odds-table-responsive-header")
    odds_tables = [
        t
        for t in table_div.select("table.odds-table")
        if "odds-table-responsive-header" not in (t.get("class") or [])
    ]
    label_table = label_tables[0] if label_tables else None
    odds_table = odds_tables[0] if odds_tables else None
    if label_table is None or odds_table is None:
        raise BFOParseError(f"Missing odds-table or label-table on {event_url}")

    matchups: list[BFOEventMatchup] = []
    rows = [r for r in label_table.find_all("tr") if isinstance(r, Tag)]
    i = 0
    while i < len(rows):
        mu_id = _matchup_id_from_tr(rows[i])
        if mu_id is None:
            i += 1
            continue
        # Find the next row that is the opponent: must be a <tr> WITHOUT
        # id="mu-..." AND WITHOUT class="pr" (proposition-row).
        opponent_row: Tag | None = None
        for j in range(i + 1, len(rows)):
            cand = rows[j]
            cand_id = cand.get("id") or ""
            cand_classes = cand.get("class") or []
            if cand_id.startswith("mu-"):
                # Hit next matchup before opponent — malformed pair.
                break
            if "pr" in cand_classes:
                # Proposition row (Over/Under) — skip and keep walking.
                continue
            opponent_row = cand
            break
        if opponent_row is None:
            # No opponent row found before next matchup — skip this mu.
            i += 1
            continue

        # Extract fighter slugs + names from the two rows.
        a_link = rows[i].find("a", href=lambda h: h and h.startswith("/fighters/"))
        b_link = opponent_row.find("a", href=lambda h: h and h.startswith("/fighters/"))
        if a_link is None or b_link is None:
            i += 1
            continue
        a_slug = _slug_from_fighter_href(a_link.get("href", ""))
        b_slug = _slug_from_fighter_href(b_link.get("href", ""))
        if a_slug is None or b_slug is None:
            i += 1
            continue
        a_name = a_link.get_text(strip=True)
        b_name = b_link.get_text(strip=True)

        # Aggregate per-bookie lines for each side.
        a_lines = _bookie_lines_for_matchup(odds_table, mu_id, side=1)
        b_lines = _bookie_lines_for_matchup(odds_table, mu_id, side=2)

        a_opening = a_lines[0] if a_lines else None
        a_min = min(a_lines) if a_lines else None
        a_max = max(a_lines) if a_lines else None
        b_opening = b_lines[0] if b_lines else None
        b_min = min(b_lines) if b_lines else None
        b_max = max(b_lines) if b_lines else None

        matchups.append(
            BFOEventMatchup(
                matchup_id=mu_id,
                fighter_a_name=a_name,
                fighter_a_slug=a_slug,
                fighter_b_name=b_name,
                fighter_b_slug=b_slug,
                a_opening=a_opening,
                a_closing_min=a_min,
                a_closing_max=a_max,
                b_opening=b_opening,
                b_closing_min=b_min,
                b_closing_max=b_max,
            )
        )
        i += 1

    return BFOEventPage(
        event_name=event_name,
        event_date=parsed_date,
        event_url=canonical_url,
        matchups=matchups,
    )


def parse_bfo_fighter_page(html: str, fighter_url: str) -> BFOFighterPage:
    """Parse a BFO fighter profile page into structured fight rows.

    BFO renders the fight history as a single ``table.team-stats-table``
    in groups of three rows per fight:

        1. ``<tr class="event-header item-mobile-only-row">`` — event
           name + date (mobile-only display, but always present in DOM).
        2. ``<tr class="main-row">`` — fighter side: oppcell name link
           + 3 ``<td.moneyline>`` (opening, closing_min, closing_max).
        3. ``<tr>`` (no class) — opponent side: oppcell name link + 3
           ``<td.moneyline>`` (opponent's odds — we ignore these) +
           ``<td.item-non-mobile>`` containing the date string
           ("Mar 4th 2023") for past fights, blank for upcoming.

    We pair each ``main-row`` with the next sibling ``<tr>`` (the detail
    row). Upcoming fights with empty dates are emitted with all-None
    moneylines; the caller (BFOScraper) drops those before CSV write.

    Args:
        html: Raw HTML string of the fighter profile page.
        fighter_url: The URL the page was fetched from (used in error
            messages and as the canonical URL stored on the result).

    Returns:
        :class:`BFOFighterPage` containing the fighter name and a list
        of :class:`BFOParsedFight` rows in display order.

    Raises:
        BFOParseError: If a CAPTCHA gate element is present (T-15-01-02)
            or no ``table.team-stats-table`` is found (T-15-01-01).
    """
    soup = BeautifulSoup(html, "lxml")

    # 1. CAPTCHA detection runs first — never trust any selector on a
    # CAPTCHA-gated page (Pitfall 1).
    _check_captcha(soup)

    table = soup.select_one("table.team-stats-table")
    if table is None:
        raise BFOParseError(f"No team-stats-table on {fighter_url}")

    # Walk all <tr> children; pair main-row with next plain <tr>.
    rows = [r for r in table.find_all("tr") if isinstance(r, Tag)]
    fighter_name = ""
    fights: list[BFOParsedFight] = []

    for i, row in enumerate(rows):
        classes = row.get("class") or []
        if "main-row" not in classes:
            continue

        # Fighter name: first oppcell link inside this main row.
        if not fighter_name:
            name_link = row.select_one("th.oppcell a")
            if name_link is not None:
                fighter_name = name_link.get_text(strip=True)

        # Locate the detail row: the next <tr> that is NOT itself a
        # main-row or event-header.
        detail: Tag | None = None
        for j in range(i + 1, len(rows)):
            cand = rows[j]
            cand_classes = cand.get("class") or []
            if "main-row" in cand_classes or "event-header" in cand_classes:
                # Encountered another fight before finding the detail —
                # the matchup row had no detail (BFO HTML drift).
                break
            detail = cand
            break

        if detail is None:
            continue  # malformed pair — skip rather than crash (T-15-01-01)

        # Date — past fights have a non-empty date string in
        # <td.item-non-mobile>; upcoming/no-date fights have it blank.
        date_el = detail.find("td", class_="item-non-mobile")
        date_str = date_el.get_text(strip=True) if date_el is not None else ""

        # Opponent — link in the detail row's oppcell.
        opp_link = detail.select_one("th.oppcell a")
        opponent_name = opp_link.get_text(strip=True) if opp_link is not None else ""
        opp_href = opp_link.get("href", "") if opp_link is not None else ""
        opponent_bfo_id = _id_from_href(opp_href)

        # Moneylines: 3 fighter-side cells in the main row.
        ml_cells = _moneyline_cells(row)
        opening = closing_min = closing_max = None
        if len(ml_cells) >= 3:
            opening = _parse_moneyline(ml_cells[0].get_text(strip=True))
            closing_min = _parse_moneyline(ml_cells[1].get_text(strip=True))
            closing_max = _parse_moneyline(ml_cells[2].get_text(strip=True))

        # Date parsing — if the date is missing the fight is upcoming /
        # has no committed event date. We synthesize date.max as a
        # sentinel so callers can filter; the BFOScraper layer drops
        # all-None-moneyline rows BEFORE emitting CSV, so upcoming
        # fights with no odds never make it past Stage D.
        if date_str:
            try:
                parsed_date = _parse_bfo_date(date_str)
            except ValueError:
                logger.warning(
                    "Could not parse BFO date %r on %s — skipping fight",
                    date_str,
                    fighter_url,
                )
                continue
        else:
            # Use date.max as a sentinel; downstream filters this out
            # via the all-None moneyline check (no odds posted yet).
            parsed_date = date.max

        fights.append(
            BFOParsedFight(
                event_date=parsed_date,
                opponent_name=opponent_name,
                opponent_bfo_id=opponent_bfo_id,
                opening=opening,
                closing_range_min=closing_min,
                closing_range_max=closing_max,
            )
        )

    return BFOFighterPage(name=fighter_name, url=fighter_url, fights=fights)


def find_bfo_fighter_url(
    db_name: str,
    search_html: str,
    threshold: int = MIN_FUZZ_SCORE,
) -> str | None:
    """Pick the best-matching BFO fighter URL from a search-results page.

    BFO's ``/search?query=...`` page renders a table of links to
    ``/fighters/<Slug>-<id>``. We score each candidate by
    ``rapidfuzz.fuzz.ratio`` over the normalized fighter names
    (``bfo_matcher.normalize_name``) and return the highest match >=
    ``threshold``, or ``None``.

    Args:
        db_name: The DB-side fighter name to look up.
        search_html: Raw HTML body of the BFO search page.
        threshold: Minimum fuzz ratio (0-100). Default
            :data:`MIN_FUZZ_SCORE` (80, calibrated in Phase 13).

    Returns:
        Absolute URL of the best-matching fighter (e.g.
        ``"https://www.bestfightodds.com/fighters/Jon-Jones-819"``) or
        ``None`` if no candidate clears the threshold.
    """
    soup = BeautifulSoup(search_html, "lxml")
    candidates: list[tuple[str, str]] = []
    for link in soup.select("a[href^='/fighters/']"):
        text = link.get_text(strip=True)
        href = link.get("href")
        if text and href:
            candidates.append((text, href))

    if not candidates:
        return None

    norm_db = normalize_name(db_name)
    best_score = 0.0
    best_href: str | None = None
    for name, href in candidates:
        score = fuzz.ratio(norm_db, normalize_name(name))
        if score > best_score:
            best_score = score
            best_href = href

    if best_href is not None and best_score >= threshold:
        return f"{BFO_BASE}{best_href}"
    return None


def _canonical_fight_id(event_date: date, self_id: str, opp_id: str) -> str:
    """Synthetic fight_id proxy that is shared across both sides of a fight.

    The downstream :class:`BFOOddsIngester` expects 2 rows per ``fight_id``
    (one per fighter side); without canonical ordering, scraping fighter A
    emits ``date|A|B`` and scraping fighter B emits ``date|B|A``, so every
    fight gets skipped as "expected 2 rows, got 1". Sorting the pair makes
    both sides emit the same key. The ingester still resolves the real DB
    ``Fight.id`` via the ``(fighter_a, fighter_b, event_date)`` triple match.
    """
    pair = sorted([str(self_id), str(opp_id)])
    return f"{event_date.isoformat()}|{pair[0]}|{pair[1]}"


# ── Orchestrator ─────────────────────────────────────────────────────────────


class BFOScraper:
    """Scrape BestFightOdds for every UFC fighter in the DB.

    Pipeline (D-02 through D-04):
        1. Read fighter list from DB. Prefer ``Fighter.source ==
           "ufcstats"`` post Phase 14 canonicalization; fall back to the
           full set if no UFCStats fighters exist.
        2. Stage A — parallel search-page fetch via
           :meth:`ScraperClient.map` (thread pool, NO multiprocessing).
        3. Stage B — parse + fuzzy-match (CPU-only, serial).
        4. Stage C — parallel profile-page fetch.
        5. Stage D — parse profile pages, validate every emit row
           through :class:`BFOOddsRow` / :class:`BFOFighterName`,
           deduplicate on ``(fight_id_proxy, fighter_id)``.
        6. Write ``BestFightOdds_odds.csv`` and ``fighters_names.csv``
           in the column order the existing :class:`BFOOddsIngester`
           reads (D-04).

    Args:
        client: A :class:`ScraperClient` (or duck-typed mock with
            ``.get`` and ``.map``). NOT a multiprocessing pool —
            the literal point of this class.
        session: A SQLAlchemy session bound to the target DB.
        data_folder: Directory where the two output CSVs are written.
            Created if missing.
        workers: Reserved for future per-stage tuning. Must be >= 1.
            (Concurrency is configured on the ``client`` itself; this
            argument is recorded for symmetry with other scrapers.)

    Raises:
        ValueError: If ``workers < 1``.
    """

    def __init__(
        self,
        client: object,
        session: object,
        data_folder: Path,
        workers: int = 4,
    ) -> None:
        if workers < 1:
            msg = f"workers must be >= 1, got {workers}"
            raise ValueError(msg)
        self._client = client
        self._session = session
        self._data_folder = Path(data_folder)
        self._workers = workers

    # ── Public API ──────────────────────────────────────────────────────

    def scrape_all(
        self,
        min_date: date | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ScrapeSummary:
        """Run the full discovery + scrape + CSV-emit pipeline.

        Args:
            min_date: If provided, drop any fight whose BFO event_date
                is strictly less than this. ufcscraper's default is
                ``date(2008, 8, 1)`` (BFO launched 2007).
            progress_callback: Optional ``callable(processed, emitted)``
                invoked once per parsed profile page.

        Returns:
            :class:`ScrapeSummary` aggregate counts and unmatched names.
        """
        summary = ScrapeSummary()

        # 1. Read fighter list (D-02 + Phase 14 canonical-source filter).
        db_fighters: list[tuple[int, str]] = list(
            self._session.execute(
                select(Fighter.id, Fighter.name).where(Fighter.source == "ufcstats")
            ).all()
        )
        if not db_fighters:
            # Fallback: no fighters tagged ufcstats — read the whole table.
            logger.warning("No fighters with source='ufcstats'; falling back to full table.")
            db_fighters = list(self._session.execute(select(Fighter.id, Fighter.name)).all())

        summary.fighters_processed = len(db_fighters)
        if not db_fighters:
            logger.info("No fighters in DB — scrape is a no-op.")
            self._write_odds_csv([])
            self._write_names_csv([])
            return summary

        # 2. Stage A — parallel search-page fetch.
        search_urls = [
            SEARCH_URL.format(query=urllib.parse.quote_plus(name)) for _id, name in db_fighters
        ]
        search_results = self._client.map(  # type: ignore[attr-defined]
            functools.partial(_safe_fetch_bfo, self._client),
            search_urls,
        )

        # 3. Stage B — parse + match.
        profile_targets: list[tuple[int, str, str]] = []  # (db_id, db_name, profile_url)
        for (db_id, db_name), result in zip(db_fighters, search_results, strict=False):
            # _safe_fetch_bfo always returns a 3-tuple; defensive unpacking
            # for cases where map was mocked to return something else.
            try:
                _u, html, err = result
            except (TypeError, ValueError):
                summary.fighters_unmatched += 1
                summary.unmatched_db_names.append(db_name)
                continue
            if err is not None or html is None:
                logger.warning(
                    "BFO search failed for %s: %s",
                    db_name,
                    err,
                )
                summary.fighters_unmatched += 1
                summary.unmatched_db_names.append(db_name)
                continue
            url = find_bfo_fighter_url(db_name, html, threshold=MIN_FUZZ_SCORE)
            if url is None:
                summary.fighters_unmatched += 1
                summary.unmatched_db_names.append(db_name)
                continue
            profile_targets.append((db_id, db_name, url))
            summary.fighters_matched += 1

        # 4. Stage C — parallel profile fetch.
        profile_urls = [t[2] for t in profile_targets]
        profile_results = self._client.map(  # type: ignore[attr-defined]
            functools.partial(_safe_fetch_bfo, self._client),
            profile_urls,
        )

        # 5. Stage D — parse + validate + dedupe.
        odds_rows: list[BFOOddsRow] = []
        name_rows: list[BFOFighterName] = []
        seen: set[tuple[str, str]] = set()
        for (db_id, db_name, prof_url), result in zip(
            profile_targets, profile_results, strict=False
        ):
            try:
                _u, html, err = result
            except (TypeError, ValueError):
                summary.parse_errors += 1
                continue
            if err is not None or html is None:
                summary.parse_errors += 1
                logger.warning("BFO profile fetch failed for %s: %s", db_name, err)
                continue

            try:
                parsed = parse_bfo_fighter_page(html, prof_url)
            except BFOParseError as exc:
                if "captcha" in str(exc).lower():
                    summary.captcha_hits += 1
                    logger.warning("CAPTCHA on %s: %s", prof_url, exc)
                else:
                    summary.parse_errors += 1
                    logger.warning("Parse error on %s: %s", prof_url, exc)
                continue
            except Exception:  # pragma: no cover — defensive
                summary.parse_errors += 1
                logger.exception("Unexpected parse failure on %s", prof_url)
                continue

            bfo_id = _id_from_href(prof_url) or str(db_id)
            # Always emit one fighters_names.csv row per matched DB fighter
            # so the ingester's name lookup can resolve our BFO IDs.
            name_rows.append(
                BFOFighterName(
                    fighter_id=bfo_id,
                    database="ufcstats",
                    name=parsed.name or db_name,
                    database_id=str(db_id),
                )
            )

            for fight in parsed.fights:
                # All-None moneylines mean upcoming / no odds posted —
                # skip (Pitfall 3). The ingester would devig nothing
                # useful out of these anyway.
                if (
                    fight.opening is None
                    and fight.closing_range_min is None
                    and fight.closing_range_max is None
                ):
                    continue
                if min_date is not None and fight.event_date < min_date:
                    continue
                # Drop date.max sentinel rows — these are upcoming fights
                # (no committed event date) which have no resolved outcome
                # and cannot be matched to our DB. Carrying them through to
                # the ingester would only inflate the CSV with unmatchable
                # rows. (Pitfall 3 / SUMMARY note about date.max.)
                if fight.event_date == date.max:
                    continue
                # Synthetic fight_id proxy SHARED across both sides of a
                # fight. The ingester expects 2 rows per fight_id (one per
                # fighter side); see _canonical_fight_id docstring.
                opp_id = fight.opponent_bfo_id or fight.opponent_name
                fight_id_proxy = _canonical_fight_id(fight.event_date, bfo_id, opp_id)
                key = (fight_id_proxy, bfo_id)
                if key in seen:
                    continue
                seen.add(key)
                odds_rows.append(
                    BFOOddsRow(
                        fight_id=fight_id_proxy,
                        fighter_id=bfo_id,
                        opening=fight.opening,
                        closing_range_min=fight.closing_range_min,
                        closing_range_max=fight.closing_range_max,
                    )
                )
                summary.fights_emitted += 1

            if progress_callback is not None:
                progress_callback(summary.fighters_processed, summary.fights_emitted)

        # 6. Write CSVs in BFOOddsIngester column order (D-04).
        self._write_odds_csv(odds_rows)
        self._write_names_csv(name_rows)
        return summary

    # ── Phase 21 Plan 21-01 Task 3: URL-list scrape mode ────────────────

    def scrape_event_urls(
        self,
        event_urls: list[str],
        output_path: Path,
    ) -> Path:
        """Scrape a curated list of BFO event URLs into a ufcscraper CSV.

        Phase 21 backfill driver entry point — the inverse of
        ``scrape_all``. Caller (``scripts/bfo_backfill.py``) supplies a
        list of ``/events/<slug>-<id>`` URLs identified via the gap query;
        this method fetches each, parses per-matchup odds, and writes one
        CSV row per fighter per matchup at ``output_path``.

        Output CSV columns (ufcscraper-compatible — required so the
        existing ``BFOOddsIngester._load_odds_rows`` reads it without
        modification):

            fight_id,fighter_id,opening,closing_range_min,closing_range_max

        Composite ``fight_id`` per matchup:

            "{event.date_iso}|{a_slug}|{b_slug}"

        with the slug pair ``(a_slug, b_slug)`` SORTED before joining so
        the same matchup yields the same key regardless of which side
        BFO renders first (mirrors ``_canonical_fight_id``; preserves
        the 999.4 rematch-overwrite fix). Each matchup emits TWO rows
        (one per fighter side), satisfying the ingester's
        2-rows-per-fight invariant.

        Skips:
          - URLs whose page is captcha-gated (``BFOParseError`` from
            ``_check_captcha``) — logged via ``self`` logger; URL
            contributes 0 rows.
          - URLs whose page is the homepage fallback (no
            ``div.table-div``) — same: logged + 0 rows.
          - Matchups whose parsed event_date is the ``date.max`` sentinel
            (no parseable date in the page meta) — dropped before CSV
            emit, mirroring the upcoming-fight pattern in ``scrape_all``.

        Args:
            event_urls: Absolute BFO event URLs to scrape.
            output_path: Target CSV path (parents created if missing).

        Returns:
            ``output_path`` (for chaining; the file is also written
            in place).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        odds_rows: list[BFOOddsRow] = []
        for url in event_urls:
            try:
                html = self._client.get(url)  # type: ignore[attr-defined]
            except (RuntimeError, ValueError) as exc:
                logger.warning(
                    "BFO event fetch failed for %s: %s",
                    url,
                    exc,
                )
                continue
            if html is None:
                logger.warning("BFO event fetch returned None for %s", url)
                continue

            try:
                page = parse_bfo_event_page(html, url)
            except BFOParseError as exc:
                # captcha vs structural — both surface as warnings; the
                # caller (Plan 21-01 backfill driver) decides whether to
                # back off or continue.
                logger.warning(
                    "BFO event parse failed for %s: %s",
                    url,
                    exc,
                )
                continue

            if page.event_date == date.max:
                # No parseable date in meta — cannot emit a usable
                # composite fight_id without it. Mirrors scrape_all's
                # date.max guard (Pitfall 3 carry-forward).
                logger.warning(
                    "BFO event %s has no parseable date; skipping (date.max sentinel).",
                    url,
                )
                continue

            for mu in page.matchups:
                # Pair-symmetric fight_id (mirrors _canonical_fight_id):
                # sort the slug pair so the same matchup yields the same
                # key regardless of side. Composite shape preserves the
                # 999.4 regression guard (date prefix needed for
                # _resolve_fight closest-date matching).
                pair = sorted([mu.fighter_a_slug, mu.fighter_b_slug])
                fight_id = f"{page.event_date.isoformat()}|{pair[0]}|{pair[1]}"
                # Side A row.
                odds_rows.append(
                    BFOOddsRow(
                        fight_id=fight_id,
                        fighter_id=mu.fighter_a_slug,
                        opening=mu.a_opening,
                        closing_range_min=mu.a_closing_min,
                        closing_range_max=mu.a_closing_max,
                    )
                )
                # Side B row.
                odds_rows.append(
                    BFOOddsRow(
                        fight_id=fight_id,
                        fighter_id=mu.fighter_b_slug,
                        opening=mu.b_opening,
                        closing_range_min=mu.b_closing_min,
                        closing_range_max=mu.b_closing_max,
                    )
                )

        # Write the CSV at the caller-specified path (NOT under the
        # data_folder convention — the caller controls layout for batch
        # staging in data/bfo/v22-backfill/<batch_id>/).
        with output_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "fight_id",
                    "fighter_id",
                    "opening",
                    "closing_range_min",
                    "closing_range_max",
                ]
            )
            for r in odds_rows:
                writer.writerow(
                    [
                        r.fight_id,
                        r.fighter_id,
                        "" if r.opening is None else r.opening,
                        "" if r.closing_range_min is None else r.closing_range_min,
                        "" if r.closing_range_max is None else r.closing_range_max,
                    ]
                )
        return output_path

    # ── CSV writers ─────────────────────────────────────────────────────

    def _write_odds_csv(self, rows: Iterable[BFOOddsRow]) -> None:
        """Emit ``BestFightOdds_odds.csv`` in ingester column order.

        Blank cells (``""``) are used for ``None`` moneylines —
        ``BFOOddsRow._coerce_blank_ml`` on the read side reverses this.
        """
        self._data_folder.mkdir(parents=True, exist_ok=True)
        out = self._data_folder / _ODDS_CSV
        with out.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "fight_id",
                    "fighter_id",
                    "opening",
                    "closing_range_min",
                    "closing_range_max",
                ]
            )
            for r in rows:
                writer.writerow(
                    [
                        r.fight_id,
                        r.fighter_id,
                        "" if r.opening is None else r.opening,
                        "" if r.closing_range_min is None else r.closing_range_min,
                        "" if r.closing_range_max is None else r.closing_range_max,
                    ]
                )

    def _write_names_csv(self, rows: Iterable[BFOFighterName]) -> None:
        """Emit ``fighters_names.csv`` in ingester column order."""
        self._data_folder.mkdir(parents=True, exist_ok=True)
        out = self._data_folder / _NAMES_CSV
        with out.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["fighter_id", "database", "name", "database_id"])
            for n in rows:
                writer.writerow(
                    [
                        n.fighter_id,
                        n.database,
                        n.name,
                        "" if n.database_id is None else n.database_id,
                    ]
                )
