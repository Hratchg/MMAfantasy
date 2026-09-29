"""Shared anti-bot / Cloudflare challenge detection.

Factored out of ``scripts/ingest_pre_ufc_records_v25.py`` (Sherdog ingest) so
the same block-detection logic can be reused by the UFCStats browser fetcher
(``scraper/browser_fetch.py``). The signatures and status codes are the
verbatim Cloudflare "challenge" / rate-limit gates observed from both Sherdog
and UFCStats.

Detection is intentionally conservative: a fetcher should HALT honestly on a
positive detection rather than fabricate data (project decision D-04 / the
Sherdog anti-bot HALT policy, now extended to UFCStats).
"""

from __future__ import annotations

# Cloudflare / anti-bot HTML signatures (case-sensitive substring match
# — Cloudflare emits these verbatim). Shared across Sherdog + UFCStats.
_ANTIBOT_HTML_SIGNATURES: tuple[str, ...] = (
    "Just a moment...",
    "cf-browser-verification",
    "Cloudflare Ray ID",
    "Attention Required! | Cloudflare",
    # UFCStats' self-hosted JS proof-of-work page (seen 2026-09-29): HTTP 200,
    # "<p>Checking your browser...</p>", a SHA-256 nonce search, then
    # ``POST /__c`` and a reload. It carries none of the Cloudflare markers.
    "Checking your browser",
)

# HTTP status codes that indicate an anti-bot / rate-limit gate.
_ANTIBOT_STATUS_CODES: frozenset[int] = frozenset({403, 429, 503})


def detect_antibot(html: str, status_code: int) -> bool:
    """Detect an anti-bot challenge (Cloudflare / rate-limit gate).

    Returns True on:
    - HTTP status in {403, 429, 503}
    - HTML containing any known Cloudflare challenge signature

    Args:
        html: The response body / rendered page content.
        status_code: The HTTP status code (use 200 when a headless browser
            already resolved the navigation and only the content matters).

    Returns:
        True if the response looks like an anti-bot challenge, else False.
    """
    if status_code in _ANTIBOT_STATUS_CODES:
        return True
    if not html:
        return False
    return any(sig in html for sig in _ANTIBOT_HTML_SIGNATURES)


class ChallengePageError(RuntimeError):
    """A fetch returned an anti-bot challenge page instead of the real page.

    A ``RuntimeError`` on purpose: the ingest pipeline's per-URL isolation
    (``ingest._safe_fetch``, ``_ensure_fighter``) treats it as a failed FETCH —
    transient, so the event is skipped and retried — never as a page to parse.
    (Contrast ``browser_fetch.AntiBotChallengeError``, which is deliberately
    NOT a ``RuntimeError``: the browser backend raises it only after its own
    retries are exhausted, and it must abort the whole run.)
    """


def raise_if_challenge(html: str, url: str) -> None:
    """Raise :class:`ChallengePageError` if ``html`` is an anti-bot challenge.

    For bodies that already came back as HTTP 200 (a client raises on the
    blocking status codes itself), so only the HTML signatures are checked.
    """
    if detect_antibot(html, 200):
        msg = f"anti-bot challenge page returned for {url}"
        raise ChallengePageError(msg)
