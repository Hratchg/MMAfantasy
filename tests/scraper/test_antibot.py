"""Unit tests for the shared anti-bot detection helper.

Covers ``detect_antibot`` against a captured Cloudflare challenge fixture and
a real UFCStats events-table fixture (which must NOT trip detection).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ufc_prediction.scraper.antibot import (
    _ANTIBOT_HTML_SIGNATURES,
    _ANTIBOT_STATUS_CODES,
    detect_antibot,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def test_challenge_fixture_detected() -> None:
    """The captured Cloudflare 'Just a moment...' page is flagged."""
    html = _load("ufcstats_antibot_challenge.html")
    assert detect_antibot(html, 200) is True


def test_real_events_table_not_detected() -> None:
    """A genuine UFCStats events listing is NOT flagged as a challenge."""
    html = _load("event_list_snippet.html")
    assert "b-statistics__table-events" in html
    assert detect_antibot(html, 200) is False


@pytest.mark.parametrize("status", sorted(_ANTIBOT_STATUS_CODES))
def test_blocked_status_codes(status: int) -> None:
    assert detect_antibot("<html>ok</html>", status) is True


@pytest.mark.parametrize("status", [200, 404, 301])
def test_ok_status_codes(status: int) -> None:
    assert detect_antibot("<html>ok</html>", status) is False


@pytest.mark.parametrize("signature", _ANTIBOT_HTML_SIGNATURES)
def test_each_signature_detected(signature: str) -> None:
    assert detect_antibot(f"<html>{signature}</html>", 200) is True


def test_empty_html_status_200_not_detected() -> None:
    assert detect_antibot("", 200) is False
