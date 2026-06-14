"""BFO probe-outcome classifier (extracted from Phase 20 audit script).

Single-source-of-truth for the 10-status content classifier originally
landed inline in ``scripts/audit_bfo_reachability.py`` (Phase 20 BFO-V22-00).
Phase 21 BFO Coverage Backfill imports the same classifier so the audit
script + backfill driver share one decision tree.

Status taxonomy (per-row probe outcomes):

  reachable           — HTTP 200 + page has real event content
  homepage_fallback   — HTTP 200 + <title> matches BFO_HOMEPAGE_TITLE
                        (invalid URL ID; BFO redirected to homepage)
  not_found           — HTTP 404
  http_other          — non-200, non-404 HTTP status (4xx/5xx)
  timeout             — read timeout / connection timeout
  error               — transport_fail / RuntimeError (retries exhausted)
  captcha             — HTTP 200 + Cloudflare challenge body (id="hfmr8")
  event_not_found     — CSV sentinel: operator marked URL as unfindable
  pre_archive_skip    — NEW (Phase 21): Event.date < BFO_ARCHIVE_START_DATE
                        (operator-locked 2007-06-01 per CONTEXT.md D-03)
  disambiguation_fail — NEW (Phase 21): /search returned a URL whose slug
                        does NOT start with "ufc-" (e.g. Bellator 272
                        collision on "272")

REACHABLE is the SOLE numerator status for reachability rates. All others
count toward the denominator only (Pitfall #5 denominator semantics).

This module deliberately does NO HTTP work — pass it parsed HTML + an
exception (or None) and it returns a status string. Phase 20 audit script
+ Phase 21 backfill driver own the HTTP + timing layers.
"""
from __future__ import annotations

import httpx
from bs4 import BeautifulSoup

# ── Status constants (string-equality; persisted into JSON artifacts) ────

REACHABLE: str = "reachable"
HOMEPAGE_FALLBACK: str = "homepage_fallback"
NOT_FOUND: str = "not_found"
HTTP_OTHER: str = "http_other"
TIMEOUT: str = "timeout"
ERROR: str = "error"
CAPTCHA: str = "captcha"
EVENT_NOT_FOUND: str = "event_not_found"
PRE_ARCHIVE_SKIP: str = "pre_archive_skip"            # NEW Phase 21
DISAMBIGUATION_FAIL: str = "disambiguation_fail"      # NEW Phase 21

# REACHABLE is the SOLE numerator status; everything else is denominator-only.
ALL_STATUSES: frozenset[str] = frozenset({
    REACHABLE, HOMEPAGE_FALLBACK, NOT_FOUND, HTTP_OTHER,
    TIMEOUT, ERROR, CAPTCHA, EVENT_NOT_FOUND,
    PRE_ARCHIVE_SKIP, DISAMBIGUATION_FAIL,
})
NON_REACHABLE_STATUSES: frozenset[str] = frozenset({
    HOMEPAGE_FALLBACK, NOT_FOUND, HTTP_OTHER, TIMEOUT,
    ERROR, CAPTCHA, EVENT_NOT_FOUND,
    PRE_ARCHIVE_SKIP, DISAMBIGUATION_FAIL,
})

# Sentinel for operator-marked unfindable rows (Phase 20 CSV-curation pattern).
EVENT_NOT_FOUND_SENTINEL: str = "EVENT_NOT_FOUND"

# Sentinel for events pre-dating the BFO archive (Phase 21 CONTEXT.md D-03).
PRE_ARCHIVE_SKIP_SENTINEL: str = "PRE_ARCHIVE_SKIP"

# The literal <title> string BFO serves for its generic homepage.
# Phase 20 Task 1 finding: BFO returns HTTP 200 with this body for
# invalid event slugs (rather than 404). BeautifulSoup decodes &amp;
# automatically when .get_text() is called.
BFO_HOMEPAGE_TITLE: str = "UFC & MMA Odds & Betting Lines | Best Fight Odds"


def is_real_event_title(title: str | None) -> bool:
    """Return True iff ``title`` is a real BFO event page title.

    Discriminator: any title that is NOT the generic homepage title is
    treated as a real event page. None or empty / whitespace-only titles
    return False (defensive — non-event-page responses with no title
    should not be treated as reachable).
    """
    if title is None:
        return False
    stripped = title.strip()
    if not stripped:
        return False
    return stripped != BFO_HOMEPAGE_TITLE


def extract_title(html: str | None) -> str | None:
    """Parse HTML and return the <title> text (stripped) or None.

    Pure helper; both ``classify`` and the audit/backfill drivers can use
    it directly when they need the title alongside the classification.
    """
    if not html:
        return None
    soup = BeautifulSoup(html, "lxml")
    title_el = soup.find("title")
    if title_el is None:
        return None
    return title_el.get_text(strip=True)


def classify(
    *,
    csv_sentinel: str | None = None,
    http_exception: Exception | None = None,
    html: str | None = None,
    captcha_detected: bool = False,
    slug: str | None = None,
) -> tuple[str, str | None]:
    """Classify a probe outcome into one of the 10 statuses.

    Mutually-exclusive decision order (first match wins):

    1. ``csv_sentinel`` precedence — operator-flagged rows bypass HTTP entirely:
       - ``EVENT_NOT_FOUND_SENTINEL`` → returns ``(EVENT_NOT_FOUND, None)``
       - ``PRE_ARCHIVE_SKIP_SENTINEL`` → returns ``(PRE_ARCHIVE_SKIP, None)``

    2. ``slug`` disambiguation — Phase 21 D-01a: a /search result whose
       slug does not start with ``ufc-`` (and isn't a sentinel) is a
       disambiguation failure (e.g. Bellator 272 returned for UFC 272).
       Returns ``(DISAMBIGUATION_FAIL, None)``.

       Important: pass ``slug=None`` from the audit script (Phase 20)
       since Phase 20 did NOT apply this guard at probe time. Phase 21
       backfill driver passes the slug and gets the new status.

    3. ``http_exception`` — wraps HTTP failures into one of three statuses:
       - ``httpx.HTTPStatusError`` with 404 → ``NOT_FOUND``
       - ``httpx.HTTPStatusError`` other → ``HTTP_OTHER``
       - ``httpx.TimeoutException`` → ``TIMEOUT``
       - any other exception (e.g. RuntimeError) → ``ERROR``

    4. ``captcha_detected`` → ``CAPTCHA``. (Caller is responsible for
       calling ``_check_captcha(soup)`` from ``bfo_scraper.py`` and
       passing the result here.)

    5. ``html``-based classification:
       - ``html`` is None or empty → ``ERROR`` (defensive; should never
         happen if neither exception nor captcha was raised)
       - <title> matches ``BFO_HOMEPAGE_TITLE`` → ``HOMEPAGE_FALLBACK``
       - any other <title> → ``REACHABLE`` (returns the title text in the
         tuple's second element)

    Returns:
        A tuple ``(status, title_extracted_or_None)``. ``title_extracted``
        is only non-None on the ``HOMEPAGE_FALLBACK`` and ``REACHABLE``
        paths; all other statuses return ``None`` for the title.
    """
    # 1. CSV-sentinel precedence (no HTTP work).
    if csv_sentinel == EVENT_NOT_FOUND_SENTINEL:
        return EVENT_NOT_FOUND, None
    if csv_sentinel == PRE_ARCHIVE_SKIP_SENTINEL:
        return PRE_ARCHIVE_SKIP, None

    # 2. Slug-disambiguation guard (Phase 21 D-01a).
    if slug is not None and slug and not slug.startswith("ufc-"):
        return DISAMBIGUATION_FAIL, None

    # 3. HTTP exceptions.
    if http_exception is not None:
        if isinstance(http_exception, httpx.HTTPStatusError):
            status_code = (
                http_exception.response.status_code
                if http_exception.response is not None
                else None
            )
            return (NOT_FOUND if status_code == 404 else HTTP_OTHER), None
        if isinstance(http_exception, httpx.TimeoutException):
            return TIMEOUT, None
        return ERROR, None

    # 4. Captcha detected upstream (caller already ran _check_captcha).
    if captcha_detected:
        return CAPTCHA, None

    # 5. HTML-based discrimination.
    title = extract_title(html)
    if title is None:
        return ERROR, None
    if title == BFO_HOMEPAGE_TITLE:
        return HOMEPAGE_FALLBACK, title
    return REACHABLE, title
