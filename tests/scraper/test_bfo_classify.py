"""Tests for ufc_prediction.scraper.bfo_classify (Phase 21 Plan 21-01 Task 1).

Migrates the classifier tests from Phase 20's
``tests/scripts/test_audit_bfo_reachability.py`` and adds 2 new tests
for the Phase 21-only statuses (PRE_ARCHIVE_SKIP, DISAMBIGUATION_FAIL).
"""

from __future__ import annotations

import httpx

from ufc_prediction.scraper.bfo_classify import (
    BFO_HOMEPAGE_TITLE,
    CAPTCHA,
    DISAMBIGUATION_FAIL,
    ERROR,
    EVENT_NOT_FOUND,
    EVENT_NOT_FOUND_SENTINEL,
    HOMEPAGE_FALLBACK,
    HTTP_OTHER,
    NOT_FOUND,
    PRE_ARCHIVE_SKIP,
    PRE_ARCHIVE_SKIP_SENTINEL,
    REACHABLE,
    TIMEOUT,
    classify,
    extract_title,
    is_real_event_title,
)

# ── Constants ─────────────────────────────────────────────────────────────


def test_all_10_status_constants_present():
    """Every documented status string is exported and distinct."""
    statuses = {
        REACHABLE,
        HOMEPAGE_FALLBACK,
        NOT_FOUND,
        HTTP_OTHER,
        TIMEOUT,
        ERROR,
        CAPTCHA,
        EVENT_NOT_FOUND,
        PRE_ARCHIVE_SKIP,
        DISAMBIGUATION_FAIL,
    }
    assert len(statuses) == 10, "10 distinct statuses required"


def test_bfo_homepage_title_is_decoded_entity_form():
    """BFO_HOMEPAGE_TITLE uses ``&`` (decoded), not ``&amp;`` (raw)."""
    assert "&amp;" not in BFO_HOMEPAGE_TITLE
    assert "&" in BFO_HOMEPAGE_TITLE


# ── extract_title + is_real_event_title helpers ───────────────────────────


def test_extract_title_returns_stripped_text():
    html = "<html><head><title>  UFC 270: Ngannou vs Gane  </title></head><body></body></html>"
    assert extract_title(html) == "UFC 270: Ngannou vs Gane"


def test_extract_title_on_no_title_tag_returns_none():
    html = "<html><body>no title here</body></html>"
    assert extract_title(html) is None


def test_extract_title_on_empty_html_returns_none():
    assert extract_title(None) is None
    assert extract_title("") is None


def test_is_real_event_title_rejects_homepage_title():
    assert is_real_event_title(BFO_HOMEPAGE_TITLE) is False


def test_is_real_event_title_accepts_event_specific_title():
    assert is_real_event_title("UFC 270: Ngannou vs Gane") is True


def test_is_real_event_title_rejects_none_and_empty():
    assert is_real_event_title(None) is False
    assert is_real_event_title("") is False
    assert is_real_event_title("   ") is False


# ── classify() — Phase 20 status migration ────────────────────────────────


def test_classify_reachable_on_real_event_title():
    html = "<html><head><title>UFC 270: Ngannou vs Gane</title></head></html>"
    status, title = classify(html=html)
    assert status == REACHABLE
    assert title == "UFC 270: Ngannou vs Gane"


def test_classify_homepage_fallback_on_generic_title():
    # BFO returns HTTP 200 + this exact title for invalid event slugs.
    html = f"<html><head><title>{BFO_HOMEPAGE_TITLE}</title></head></html>"
    status, title = classify(html=html)
    assert status == HOMEPAGE_FALLBACK
    assert title == BFO_HOMEPAGE_TITLE


def test_classify_captcha_when_caller_flags_detected():
    # Caller is responsible for running _check_captcha() upstream.
    html = "<html><head><title>doesnt matter</title></head></html>"
    status, title = classify(html=html, captcha_detected=True)
    assert status == CAPTCHA
    assert title is None


def test_classify_event_not_found_sentinel_bypasses_html():
    # CSV sentinel takes precedence; HTML is ignored.
    status, title = classify(
        csv_sentinel=EVENT_NOT_FOUND_SENTINEL,
        html="<title>shouldnt be inspected</title>",
    )
    assert status == EVENT_NOT_FOUND
    assert title is None


def test_classify_http_404_returns_not_found():
    resp = httpx.Response(404, request=httpx.Request("GET", "https://x/y"))
    exc = httpx.HTTPStatusError("404", request=resp.request, response=resp)
    status, title = classify(http_exception=exc)
    assert status == NOT_FOUND


def test_classify_http_500_returns_http_other():
    resp = httpx.Response(500, request=httpx.Request("GET", "https://x/y"))
    exc = httpx.HTTPStatusError("500", request=resp.request, response=resp)
    status, title = classify(http_exception=exc)
    assert status == HTTP_OTHER


def test_classify_timeout_exception_returns_timeout():
    exc = httpx.ReadTimeout("read timed out")
    status, title = classify(http_exception=exc)
    assert status == TIMEOUT


def test_classify_runtime_error_returns_error():
    exc = RuntimeError("retries exhausted")
    status, title = classify(http_exception=exc)
    assert status == ERROR


def test_classify_none_html_no_exception_returns_error():
    # Defensive: caller passed neither exception nor captcha but no html.
    status, title = classify(html=None)
    assert status == ERROR


# ── classify() — Phase 21 NEW statuses ────────────────────────────────────


def test_classify_pre_archive_skip_sentinel_bypasses_http():
    """Phase 21 CONTEXT.md D-03: Event.date < 2007-06-01 → PRE_ARCHIVE_SKIP."""
    status, title = classify(
        csv_sentinel=PRE_ARCHIVE_SKIP_SENTINEL,
        html="<title>ignored</title>",
    )
    assert status == PRE_ARCHIVE_SKIP
    assert title is None


def test_classify_disambiguation_fail_when_slug_not_ufc_prefixed():
    """Phase 21 D-01a: /search returned a Bellator/Strikeforce slug."""
    # Slug from Phase 20 UFC 272 disambiguation failure.
    status, title = classify(slug="bellator-272-2377", html="<title>Bellator 272</title>")
    assert status == DISAMBIGUATION_FAIL
    assert title is None


def test_classify_disambiguation_pass_on_ufc_prefixed_slug():
    """UFC-prefixed slug + real HTML → REACHABLE."""
    html = "<html><head><title>UFC 270: Ngannou vs Gane</title></head></html>"
    status, title = classify(slug="ufc-270-2322", html=html)
    assert status == REACHABLE
    assert title == "UFC 270: Ngannou vs Gane"


# ── Precedence: CSV sentinel > slug > http_exception > captcha > html ─────


def test_classify_csv_sentinel_takes_precedence_over_slug_guard():
    """EVENT_NOT_FOUND_SENTINEL wins even if slug looks wrong."""
    status, _ = classify(
        csv_sentinel=EVENT_NOT_FOUND_SENTINEL,
        slug="bellator-272-2377",
    )
    assert status == EVENT_NOT_FOUND


def test_classify_slug_guard_takes_precedence_over_html():
    """DISAMBIGUATION_FAIL wins over a real-looking title."""
    status, _ = classify(
        slug="bellator-272-2377",
        html="<title>This looks like a real event</title>",
    )
    assert status == DISAMBIGUATION_FAIL


def test_classify_http_exception_takes_precedence_over_html():
    """If both exception and html are passed, exception wins."""
    exc = httpx.ReadTimeout("oops")
    status, _ = classify(
        http_exception=exc,
        html="<title>real event</title>",
    )
    assert status == TIMEOUT
