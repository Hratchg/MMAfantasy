"""Sherdog.com scraper for pre-UFC fighter records.

Provides HTML parsers for Sherdog fighter profiles and search results,
a name matcher using rapidfuzz for fuzzy matching, and an orchestrator
class that ties the pipeline together.

Decisions:
- D-01: ScraperClient for HTTP with rate limiting
- D-03: BeautifulSoup for HTML parsing
- D-04: Rate limit at 2+ seconds between requests
- D-05/D-06: Fuzzy name matching via rapidfuzz
- D-07: Store Sherdog URL on Fighter model
- D-08: Log unmatched fighters at WARNING level
- D-09/D-10: Extract and compute pre-UFC fight stats
- D-11: Store PreUFCRecord as JSON on Fighter model

Threat mitigations:
- T-13-01: All parsed data validated through Pydantic models
- T-13-02: Uses ScraperClient which has generic User-Agent
- T-13-03: Enforces minimum 2.0s delay
- T-13-05: Pydantic serialization produces safe JSON
"""

from __future__ import annotations

import contextlib
import functools
import logging
import re
from datetime import date

from bs4 import BeautifulSoup, Tag
from rapidfuzz import fuzz

from ufc_prediction.scraper.sherdog_models import (
    PreUFCRecord,
    SherdogFight,
    SherdogFighterProfile,
)

logger = logging.getLogger(__name__)

_SHERDOG_BASE = "https://www.sherdog.com"
_SHERDOG_SEARCH_URL = "https://www.sherdog.com/stats/fightfinder?SearchTxt={name}"


def _parse_date(date_str: str) -> date | None:
    """Parse a Sherdog date string like 'Feb / 11 / 2017' into a date."""
    date_str = date_str.strip()
    if not date_str:
        return None
    # Sherdog format: "Mon / DD / YYYY"
    cleaned = re.sub(r"\s*/\s*", "/", date_str)
    try:
        from datetime import datetime

        return datetime.strptime(cleaned, "%b/%d/%Y").date()
    except ValueError:
        logger.warning("Could not parse date: %r", date_str)
        return None


def _normalize_method(method_text: str) -> str:
    """Normalize fight method text from Sherdog.

    Maps variations to canonical forms: KO/TKO, Submission, Decision, etc.
    """
    method_lower = method_text.strip().lower()
    if "ko" in method_lower or "tko" in method_lower:
        return "KO/TKO"
    if "submission" in method_lower or "sub" in method_lower:
        return "Submission"
    if "decision" in method_lower:
        return "Decision"
    if "draw" in method_lower:
        return "Draw"
    if "no contest" in method_lower or "nc" in method_lower:
        return "No Contest"
    return method_text.strip()


def _normalize_result(result_text: str) -> str:
    """Normalize result text (Win/Loss/Draw/NC) to lowercase."""
    result_lower = result_text.strip().lower()
    if result_lower in ("win", "w"):
        return "win"
    if result_lower in ("loss", "l"):
        return "loss"
    if result_lower in ("draw", "d"):
        return "draw"
    if result_lower in ("nc", "no contest"):
        return "nc"
    return result_lower


def parse_sherdog_fighter_page(html: str, url: str) -> SherdogFighterProfile:
    """Parse a Sherdog fighter profile page.

    Extracts fighter name and full fight history from the HTML.
    All parsed data is validated through SherdogFight Pydantic models (T-13-01).

    Args:
        html: Raw HTML of the fighter profile page.
        url: The URL the page was fetched from.

    Returns:
        SherdogFighterProfile with parsed fights.
    """
    soup = BeautifulSoup(html, "lxml")

    # Extract fighter name
    name_el = soup.select_one("h1.hero-fighter-name .fn")
    if name_el is None:
        # Fallback: try other known patterns
        name_el = soup.select_one(".bio_fighter .fn")
    name = name_el.get_text(strip=True) if name_el else "Unknown"

    # Extract fight history — first table.new_table in div.fight_history = pro fights
    fights: list[SherdogFight] = []
    fight_tables = soup.select("div.module.fight_history table.new_table")
    if not fight_tables:
        fight_tables = soup.select(".fight_history table")

    # Only use the first table (pro fights); second is amateur
    fight_history = fight_tables[0] if fight_tables else None

    if fight_history:
        rows = fight_history.select("tr")
        for row in rows:
            if not isinstance(row, Tag):
                continue
            cells = row.find_all("td")
            if len(cells) < 4:
                continue

            # Result — live HTML uses plain text like "win", "loss", "draw", "nc"
            result_text = cells[0].get_text(strip=True).lower()
            if result_text not in ("win", "loss", "draw", "nc", "no contest"):
                continue
            result = _normalize_result(result_text)

            # Opponent — name is in the second cell, may have an <a> tag
            opponent_el = cells[1].select_one("a")
            opponent_name = (
                opponent_el.get_text(strip=True) if opponent_el else cells[1].get_text(strip=True)
            )

            # Event info — third cell contains event name and possibly date
            event_cell = cells[2]
            # Live HTML: event name is in an <a> or plain text, date is inline
            event_links = event_cell.select("a")
            event_name = event_links[0].get_text(strip=True) if event_links else ""
            # Date is often appended after the event name
            event_text = event_cell.get_text(separator=" / ", strip=True)
            event_date = None
            # Try to extract date from end of text (format: Mon / DD / YYYY)
            import re as _re

            date_match = _re.search(
                r"([A-Z][a-z]{2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})",
                event_text,
            )
            if date_match:
                date_str = f"{date_match.group(1)}/{date_match.group(2)}/{date_match.group(3)}"
                event_date = _parse_date(date_str)

            # Method — fourth cell, may include referee name
            method_raw = cells[3].get_text(strip=True)
            # Strip referee name and "VIEW PLAY-BY-PLAY" suffix
            method_clean = _re.split(r"(?:VIEW|[A-Z][a-z]+ [A-Z][a-z]+$)", method_raw)[0].strip()
            method = _normalize_method(method_clean)

            # Round
            round_finished = None
            if len(cells) > 4:
                round_text = cells[4].get_text(strip=True)
                with contextlib.suppress(ValueError, TypeError):
                    round_finished = int(round_text)

            fights.append(
                SherdogFight(
                    result=result,
                    opponent_name=opponent_name,
                    method=method,
                    event_name=event_name,
                    event_date=event_date,
                    round_finished=round_finished,
                )
            )

    return SherdogFighterProfile(name=name, sherdog_url=url, fights=fights)


def parse_association_from_html(html: str) -> str | None:
    """Extract the Association/camp value from a Sherdog fighter page sidebar.

    Returns the raw association text (e.g., ``"American Top Team"``) or
    ``None`` if the field is absent. Caller is responsible for normalization
    (strip suffixes, lowercase, etc.) — this helper returns the verbatim
    value so ``unparseable_examples`` in ``CAMP_00_AUDIT.json`` can store
    the raw string for diagnostics (CONTEXT.md D-02 / D-05).

    Defensive layered selectors (mirrors ``parse_sherdog_fighter_page``'s
    name selector pattern at lines 112-115). Live Sherdog (verified
    2026-05-12 against the Israel Adesanya fixture) renders the field as::

        <div class="association-class">
            ASSOCIATION<br />
            <span itemprop="memberOf" itemscope ...>
                <a class="association" itemprop="url"
                   href="/stats/fightfinder?association=City+Kickboxing">
                    <span itemprop="name">City Kickboxing</span>
                </a>
            </span>
        </div>

    Selectors are tried in priority order:
      1. Microdata: ``div.association-class span[itemprop="memberOf"] span[itemprop="name"]``
         (most specific — the inner schema.org name span)
      2. Microdata fallback: ``span[itemprop="memberOf"]`` (older markup
         without the nested name span)
      3. Label-based fallback: ``<strong>Association:</strong>`` adjacent
         sibling / parent text (legacy non-microdata layout)
      4. URL-decoded fallback: ``div.association-class a.association`` href
         ``?association=`` query param (last-resort if all text nodes
         scrubbed but the link survives)

    Args:
        html: Raw HTML of the fighter profile page.

    Returns:
        The verbatim association text, or ``None`` if no association field
        is detected.
    """
    soup = BeautifulSoup(html, "lxml")

    # Layer 1: nested schema.org name span (modern Sherdog — most specific).
    el = soup.select_one('div.association-class span[itemprop="memberOf"] span[itemprop="name"]')
    if el is not None:
        text = el.get_text(strip=True)
        if text:
            return text

    # Layer 2: outer memberOf span fallback (older microdata layouts).
    el = soup.select_one('span[itemprop="memberOf"]')
    if el is not None:
        text = el.get_text(strip=True)
        if text:
            return text

    # Layer 3: label-based fallback (<strong>Association:</strong> ...).
    for strong in soup.find_all("strong"):
        if not isinstance(strong, Tag):
            continue
        if strong.get_text(strip=True).lower().startswith("association"):
            sibling = strong.find_next_sibling()
            if sibling is not None:
                text = sibling.get_text(strip=True)
                if text:
                    return text
            parent_text = strong.parent.get_text(strip=True) if strong.parent else ""
            cleaned = parent_text.removeprefix("Association:").strip()
            if cleaned:
                return cleaned

    # Layer 4: URL-decoded href fallback (?association=City+Kickboxing).
    anchor = soup.select_one("div.association-class a.association")
    if anchor is not None and isinstance(anchor, Tag):
        href = anchor.get("href", "")
        if isinstance(href, str) and "?association=" in href:
            from urllib.parse import unquote

            raw = href.split("?association=", 1)[1].split("&", 1)[0]
            decoded = unquote(raw.replace("+", " ")).strip()
            if decoded:
                return decoded

    return None


def parse_sherdog_search_results(html: str) -> list[tuple[str, str]]:
    """Parse Sherdog search results page.

    Args:
        html: Raw HTML of the search results page.

    Returns:
        List of (fighter_name, absolute_url) tuples.
    """
    soup = BeautifulSoup(html, "lxml")
    results: list[tuple[str, str]] = []

    # Find fighter result rows — Sherdog uses table.fightfinder_result
    fighter_table = soup.select_one("table.fightfinder_result")
    if fighter_table is None:
        fighter_table = soup.select_one(".fighter_results table")
    if fighter_table is None:
        fighter_table = soup.select_one("div.module.fighter_results table")

    if fighter_table:
        # Live Sherdog uses <tr onclick="..."> for result rows
        rows = fighter_table.select("tr[onclick]")
        if not rows:
            rows = fighter_table.select("tbody tr")
        for row in rows:
            if not isinstance(row, Tag):
                continue
            # Live HTML: name is in the second <td> inside an <a>
            name_cell = row.select_one("td a[href*='/fighter/']")
            if name_cell is None:
                name_cell = row.select_one(".col_name a")
            if name_cell is None:
                cells = row.find_all("td")
                if len(cells) >= 2:
                    name_cell = cells[1].select_one("a")
            if name_cell is None:
                continue

            name = name_cell.get_text(strip=True)
            href = name_cell.get("href", "")
            # Ensure absolute URL
            if href.startswith("/"):
                href = f"{_SHERDOG_BASE}{href}"
            elif not href.startswith("http"):
                href = f"{_SHERDOG_BASE}/{href}"

            results.append((name, href))

    return results


def filter_pre_ufc_fights(fights: list[SherdogFight], first_ufc_date: date) -> list[SherdogFight]:
    """Filter fights to only those before the first UFC fight date.

    Excludes:
    - Fights on or after first_ufc_date
    - Fights with "UFC" in the event name (even if before cutoff date)
    - Fights with no event_date (can't verify if pre-UFC)

    Args:
        fights: Full fight list from Sherdog profile.
        first_ufc_date: The fighter's first UFC fight date (cutoff).

    Returns:
        List of pre-UFC fights only.
    """
    pre_ufc: list[SherdogFight] = []
    for fight in fights:
        if fight.event_date is None:
            continue
        if fight.event_date >= first_ufc_date:
            continue
        if "UFC" in fight.event_name.upper():
            continue
        pre_ufc.append(fight)
    return pre_ufc


def compute_pre_ufc_stats(pre_ufc_fights: list[SherdogFight]) -> PreUFCRecord:
    """Compute pre-UFC career summary stats.

    Handles edge cases: zero fights returns zeros, zero wins returns 0.0 for
    finish rates.

    Args:
        pre_ufc_fights: Filtered list of pre-UFC fights.

    Returns:
        PreUFCRecord with computed stats.
    """
    total = len(pre_ufc_fights)
    if total == 0:
        return PreUFCRecord(
            total_wins=0,
            total_losses=0,
            total_draws=0,
            total_fights=0,
            win_pct=0.0,
            ko_finish_rate=0.0,
            sub_finish_rate=0.0,
            decision_rate=0.0,
            career_years=0.0,
            fights=[],
        )

    wins = [f for f in pre_ufc_fights if f.result == "win"]
    losses = [f for f in pre_ufc_fights if f.result == "loss"]
    draws = [f for f in pre_ufc_fights if f.result == "draw"]
    total_wins = len(wins)
    total_losses = len(losses)
    total_draws = len(draws)
    win_pct = total_wins / total if total > 0 else 0.0

    # Finish rates (based on wins only)
    if total_wins > 0:
        ko_wins = sum(1 for f in wins if f.method == "KO/TKO")
        sub_wins = sum(1 for f in wins if f.method == "Submission")
        dec_wins = sum(1 for f in wins if f.method == "Decision")
        ko_finish_rate = ko_wins / total_wins
        sub_finish_rate = sub_wins / total_wins
        decision_rate = dec_wins / total_wins
    else:
        ko_finish_rate = 0.0
        sub_finish_rate = 0.0
        decision_rate = 0.0

    # Career years
    dated_fights = [f for f in pre_ufc_fights if f.event_date is not None]
    if len(dated_fights) >= 2:
        dates = sorted(f.event_date for f in dated_fights if f.event_date is not None)
        career_days = (dates[-1] - dates[0]).days
        career_years = career_days / 365.25
    else:
        career_years = 0.0

    return PreUFCRecord(
        total_wins=total_wins,
        total_losses=total_losses,
        total_draws=total_draws,
        total_fights=total,
        win_pct=win_pct,
        ko_finish_rate=ko_finish_rate,
        sub_finish_rate=sub_finish_rate,
        decision_rate=decision_rate,
        career_years=career_years,
        fights=pre_ufc_fights,
    )


def match_fighter_name(
    db_name: str,
    candidates: list[tuple[str, str]],
    threshold: int = 85,
) -> tuple[str, str] | None:
    """Find best fuzzy match for a fighter name from search candidates.

    Uses rapidfuzz.fuzz.ratio for scoring. Also tries reversed name order
    ("Last, First" vs "First Last") to handle format differences.

    Args:
        db_name: Fighter name from our database.
        candidates: List of (name, url) tuples from Sherdog search.
        threshold: Minimum similarity score (0-100). Default 85 per D-06.

    Returns:
        The best (name, url) tuple if score >= threshold, else None.
    """
    if not candidates:
        return None

    best_score = 0.0
    best_match: tuple[str, str] | None = None

    for cand_name, cand_url in candidates:
        # Direct comparison
        score = fuzz.ratio(db_name.lower(), cand_name.lower())

        # Try reversed name comparison
        # "Last, First" -> "First Last"
        if "," in cand_name:
            parts = [p.strip() for p in cand_name.split(",", 1)]
            reversed_name = f"{parts[1]} {parts[0]}"
            reversed_score = fuzz.ratio(db_name.lower(), reversed_name.lower())
            score = max(score, reversed_score)

        # Also try reversing the db_name for "First Last" vs "Last, First"
        if "," in db_name:
            parts = [p.strip() for p in db_name.split(",", 1)]
            reversed_db = f"{parts[1]} {parts[0]}"
            reversed_score = fuzz.ratio(reversed_db.lower(), cand_name.lower())
            score = max(score, reversed_score)

        if score > best_score:
            best_score = score
            best_match = (cand_name, cand_url)

    if best_score >= threshold and best_match is not None:
        return best_match
    return None


def _safe_fetch_sherdog(
    client: object,
    url: str,
) -> tuple[str, str | None, Exception | None]:
    """Fetch ``url`` via ``client.get``, returning a sentinel tuple.

    Mirrors ``_safe_fetch`` in scraper/ingest.py: lets ``client.map(...)``
    batches survive individual URL failures instead of aborting on the
    first exception. Returns ``(url, html_or_None, exc_or_None)``.

    Kept local (not imported from ingest.py) so sherdog.py stays decoupled
    from the UFCStats pipeline.
    """
    try:
        return (url, client.get(url), None)  # type: ignore[attr-defined]
    except (RuntimeError, ValueError) as exc:
        return (url, None, exc)


class SherdogScraper:
    """Orchestrator for scraping pre-UFC records from Sherdog.com.

    Uses ScraperClient for HTTP (D-01) with rate limiting (D-04).
    Validates all data through Pydantic models (T-13-01).

    Args:
        client: ScraperClient instance for making HTTP requests.
        session: SQLAlchemy session for DB operations.
        workers: Number of concurrent worker threads for HTTP fetches
            (default 1 — preserves legacy byte-identical serial behavior).
            When >1, search and profile URL batches are dispatched via
            ``client.map`` with per-URL error isolation via
            :func:`_safe_fetch_sherdog`. DB writes remain serial.
        commit_batch_size: Number of successfully-processed fighters
            between intermediate ``session.commit()`` calls. Lower values
            give more granular resume after Ctrl-C at the cost of more
            commit overhead. Default 50.

    Raises:
        ValueError: If ``workers < 1`` or ``commit_batch_size < 1``.
    """

    def __init__(
        self,
        client: object,
        session: object,
        workers: int = 1,
        commit_batch_size: int = 50,
    ) -> None:
        if workers < 1:
            msg = f"workers must be >= 1, got {workers}"
            raise ValueError(msg)
        if commit_batch_size < 1:
            msg = f"commit_batch_size must be >= 1, got {commit_batch_size}"
            raise ValueError(msg)
        self._client = client
        self._session = session
        self._workers = workers
        self._commit_batch_size = commit_batch_size

    def scrape_all_fighters(
        self,
        progress_callback: object | None = None,
        dry_run: bool = False,
        since_year: int | None = None,
    ) -> dict:
        """Scrape pre-UFC records for all fighters in the database.

        Pipeline:
        1. Load fighters from DB (optionally filtered by ``since_year``)
        2. For each fighter, find first UFC fight date
        3. Search Sherdog for the fighter name
        4. Match using fuzzy matching (D-05, D-06)
        5. If matched, fetch profile and compute pre-UFC stats
        6. Update Fighter.sherdog_url (D-07) and Fighter.pre_ufc_record (D-11)
        7. Log unmatched fighters at WARNING level (D-08)
        8. ``session.commit()`` every ``commit_batch_size`` successful
           writes, plus a final commit at end of run (resilience).

        Args:
            progress_callback: Optional callable(current, total) for progress.
            dry_run: If True, search and match only (don't fetch profiles).
            since_year: If provided, only process fighters with at least one
                UFC fight on or after ``date(since_year, 1, 1)``. Default
                None = all fighters (legacy behavior).

        Returns:
            Summary dict with matched, unmatched, skipped, errors counts.
        """
        from sqlalchemy import func as sa_func
        from sqlalchemy import or_

        from ufc_prediction.models.event import Event
        from ufc_prediction.models.fight import Fight
        from ufc_prediction.models.fighter import Fighter

        # Load fighters, optionally filtered by since_year.
        fighters_query = self._session.query(Fighter)
        if since_year is not None:
            cutoff = date(since_year, 1, 1)
            # Fighters with at least one UFC fight on or after cutoff.
            # Use a correlated EXISTS subquery to avoid DISTINCT/join duplicates.
            exists_clause = (
                self._session.query(Fight.id)
                .join(Event, Fight.event_id == Event.id)
                .filter(
                    or_(
                        Fight.fighter_a_id == Fighter.id,
                        Fight.fighter_b_id == Fighter.id,
                    ),
                    Event.date >= cutoff,
                )
                .exists()
            )
            fighters_query = fighters_query.filter(exists_clause)

        fighters = fighters_query.all()
        total = len(fighters)
        matched = 0
        unmatched = 0
        skipped = 0
        errors = 0
        unmatched_names: list[str] = []
        processed_since_commit = 0

        def _maybe_commit() -> None:
            """Commit if the batch threshold has been reached."""
            nonlocal processed_since_commit
            if processed_since_commit >= self._commit_batch_size:
                self._session.commit()
                processed_since_commit = 0

        def _first_ufc_fight_date(fighter_obj: object) -> object:
            """Return the first UFC fight date for a fighter (None if any)."""
            return (
                self._session.query(sa_func.min(Event.date))
                .join(Fight, Fight.event_id == Event.id)
                .filter(
                    or_(
                        Fight.fighter_a_id == fighter_obj.id,  # type: ignore[attr-defined]
                        Fight.fighter_b_id == fighter_obj.id,  # type: ignore[attr-defined]
                    )
                )
                .scalar()
            )

        if self._workers == 1:
            # ── Legacy serial path (byte-identical to pre-rebuild loop,
            # except for intermediate commit bookkeeping) ──────────────────
            for idx, fighter in enumerate(fighters):
                if progress_callback:
                    progress_callback(idx, total)

                # Skip fighters already processed
                if fighter.sherdog_url is not None:
                    skipped += 1
                    continue

                try:
                    # Find first UFC fight date for this fighter
                    first_ufc_fight = _first_ufc_fight_date(fighter)

                    if first_ufc_fight is None:
                        logger.debug("No UFC fights found for %s, skipping", fighter.name)
                        skipped += 1
                        continue

                    # Search Sherdog for the fighter
                    search_url = _SHERDOG_SEARCH_URL.format(name=fighter.name.replace(" ", "+"))
                    search_html = self._client.get(search_url)  # type: ignore[attr-defined]
                    candidates = parse_sherdog_search_results(search_html)

                    if not candidates:
                        logger.warning("No Sherdog search results for: %s", fighter.name)
                        unmatched += 1
                        unmatched_names.append(fighter.name)
                        continue

                    # Match fighter name
                    match = match_fighter_name(fighter.name, candidates)
                    if match is None:
                        logger.warning(
                            "No Sherdog match for: %s (candidates: %s)",
                            fighter.name,
                            [c[0] for c in candidates[:3]],
                        )
                        unmatched += 1
                        unmatched_names.append(fighter.name)
                        continue

                    sherdog_name, sherdog_url = match

                    if dry_run:
                        logger.info(
                            "DRY RUN match: %s -> %s (%s)",
                            fighter.name,
                            sherdog_name,
                            sherdog_url,
                        )
                        matched += 1
                        fighter.sherdog_url = sherdog_url
                        self._session.flush()
                        processed_since_commit += 1
                        _maybe_commit()
                        continue

                    # Fetch and parse Sherdog profile
                    profile_html = self._client.get(sherdog_url)  # type: ignore[attr-defined]
                    profile = parse_sherdog_fighter_page(profile_html, sherdog_url)

                    # Filter pre-UFC fights
                    pre_ufc_fights = filter_pre_ufc_fights(profile.fights, first_ufc_fight)

                    # Compute stats
                    stats = compute_pre_ufc_stats(pre_ufc_fights)

                    # Update fighter record
                    fighter.sherdog_url = sherdog_url
                    fighter.pre_ufc_record = stats.model_dump(mode="json")
                    self._session.flush()

                    matched += 1
                    processed_since_commit += 1
                    _maybe_commit()
                    logger.info(
                        "Matched %s -> %s (%d pre-UFC fights)",
                        fighter.name,
                        sherdog_name,
                        stats.total_fights,
                    )

                except Exception:
                    logger.exception("Error processing fighter: %s", fighter.name)
                    errors += 1
        else:
            # ── Parallel batched path (workers > 1) ────────────────────────
            chunk_size = self._workers
            idx_counter = 0
            for chunk_start in range(0, total, chunk_size):
                chunk = fighters[chunk_start : chunk_start + chunk_size]

                # Filter out skips up-front and pre-compute first_ufc dates.
                work_items: list[tuple[object, object]] = []
                for fighter in chunk:
                    if progress_callback:
                        progress_callback(idx_counter, total)
                    idx_counter += 1
                    if fighter.sherdog_url is not None:
                        skipped += 1
                        continue
                    try:
                        first_ufc = _first_ufc_fight_date(fighter)
                    except Exception:
                        logger.exception("Error looking up first UFC fight for %s", fighter.name)
                        errors += 1
                        continue
                    if first_ufc is None:
                        logger.debug("No UFC fights found for %s, skipping", fighter.name)
                        skipped += 1
                        continue
                    work_items.append((fighter, first_ufc))

                if not work_items:
                    continue

                # Stage A: parallel search fetch
                search_urls = [
                    _SHERDOG_SEARCH_URL.format(name=f.name.replace(" ", "+"))  # type: ignore[attr-defined]
                    for f, _ in work_items
                ]
                search_results = self._client.map(  # type: ignore[attr-defined]
                    functools.partial(_safe_fetch_sherdog, self._client),
                    search_urls,
                )

                # Stage B: parse + match (CPU-only, serial)
                matches_to_fetch: list[tuple[object, object, str, str]] = []
                for (fighter, first_ufc), (_u, html, err) in zip(
                    work_items,
                    search_results,
                    strict=True,
                ):
                    if err is not None or html is None:
                        logger.warning(
                            "Failed to fetch Sherdog search for %s: %s",
                            fighter.name,
                            err,  # type: ignore[attr-defined]
                        )
                        errors += 1
                        continue
                    candidates = parse_sherdog_search_results(html)
                    if not candidates:
                        logger.warning(
                            "No Sherdog search results for: %s",
                            fighter.name,  # type: ignore[attr-defined]
                        )
                        unmatched += 1
                        unmatched_names.append(fighter.name)  # type: ignore[attr-defined]
                        continue
                    match = match_fighter_name(fighter.name, candidates)  # type: ignore[attr-defined]
                    if match is None:
                        logger.warning(
                            "No Sherdog match for: %s (candidates: %s)",
                            fighter.name,  # type: ignore[attr-defined]
                            [c[0] for c in candidates[:3]],
                        )
                        unmatched += 1
                        unmatched_names.append(fighter.name)  # type: ignore[attr-defined]
                        continue
                    sherdog_name, sherdog_url = match
                    matches_to_fetch.append((fighter, first_ufc, sherdog_url, sherdog_name))

                if not matches_to_fetch:
                    continue

                if dry_run:
                    for fighter, _first, sherdog_url, sherdog_name in matches_to_fetch:
                        logger.info(
                            "DRY RUN match: %s -> %s (%s)",
                            fighter.name,
                            sherdog_name,
                            sherdog_url,  # type: ignore[attr-defined]
                        )
                        fighter.sherdog_url = sherdog_url  # type: ignore[attr-defined]
                        self._session.flush()
                        matched += 1
                        processed_since_commit += 1
                    _maybe_commit()
                    continue

                # Stage C: parallel profile fetch
                profile_urls = [m[2] for m in matches_to_fetch]
                profile_results = self._client.map(  # type: ignore[attr-defined]
                    functools.partial(_safe_fetch_sherdog, self._client),
                    profile_urls,
                )

                # Stage D: parse + write (serial, includes DB I/O)
                for (fighter, first_ufc, sherdog_url, sherdog_name), (
                    _u,
                    html,
                    err,
                ) in zip(matches_to_fetch, profile_results, strict=True):
                    if err is not None or html is None:
                        logger.warning(
                            "Failed to fetch Sherdog profile for %s: %s",
                            fighter.name,
                            err,  # type: ignore[attr-defined]
                        )
                        errors += 1
                        continue
                    try:
                        profile = parse_sherdog_fighter_page(html, sherdog_url)
                        pre_ufc_fights = filter_pre_ufc_fights(profile.fights, first_ufc)
                        stats = compute_pre_ufc_stats(pre_ufc_fights)
                        fighter.sherdog_url = sherdog_url  # type: ignore[attr-defined]
                        fighter.pre_ufc_record = stats.model_dump(mode="json")  # type: ignore[attr-defined]
                        self._session.flush()
                        matched += 1
                        processed_since_commit += 1
                        logger.info(
                            "Matched %s -> %s (%d pre-UFC fights)",
                            fighter.name,
                            sherdog_name,
                            stats.total_fights,  # type: ignore[attr-defined]
                        )
                    except Exception:
                        logger.exception(
                            "Error processing fighter: %s",
                            fighter.name,  # type: ignore[attr-defined]
                        )
                        errors += 1

                _maybe_commit()

        # Final commit flushes any remaining work from the last partial batch.
        self._session.commit()

        if progress_callback:
            progress_callback(total, total)

        return {
            "matched": matched,
            "unmatched": unmatched,
            "skipped": skipped,
            "errors": errors,
            "unmatched_names": unmatched_names,
        }
