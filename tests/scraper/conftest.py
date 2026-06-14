"""Shared fixtures for scraper parser tests.

Provides HTML fixture loading helpers for all page types.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def fixture_path(filename: str) -> Path:
    """Return the path to a fixture file."""
    return Path(__file__).parent / "fixtures" / filename


@pytest.fixture
def event_list_html() -> str:
    """Load the event listing page HTML fixture."""
    return fixture_path("event_list_snippet.html").read_text(encoding="utf-8")


@pytest.fixture
def event_detail_html() -> str:
    """Load the event detail page HTML fixture."""
    return fixture_path("event_detail.html").read_text(encoding="utf-8")


@pytest.fixture
def fight_detail_1round_html() -> str:
    """Load the 1-round KO/TKO fight detail HTML fixture."""
    return fixture_path("fight_detail_1round.html").read_text(encoding="utf-8")


@pytest.fixture
def fight_detail_3round_html() -> str:
    """Load the 3-round decision fight detail HTML fixture."""
    return fixture_path("fight_detail_3round.html").read_text(encoding="utf-8")


@pytest.fixture
def fight_detail_draw_html() -> str:
    """Load the draw fight detail HTML fixture."""
    return fixture_path("fight_detail_draw.html").read_text(encoding="utf-8")


@pytest.fixture
def fighter_profile_html() -> str:
    """Load the complete fighter profile HTML fixture."""
    return fixture_path("fighter_profile.html").read_text(encoding="utf-8")


@pytest.fixture
def fighter_profile_missing_html() -> str:
    """Load the fighter profile with missing fields HTML fixture."""
    return fixture_path("fighter_profile_missing.html").read_text(encoding="utf-8")
