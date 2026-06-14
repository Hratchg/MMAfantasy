"""Shared fixtures for tests/scripts/ — Wave-0 RED suite for Phase 20 audit scripts (CAMP + BFO)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

FIXTURES_DIR: Path = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def mock_scraper_client() -> MagicMock:
    """Return a MagicMock standing in for ScraperClient.

    Callers override `.get.return_value` (success path) or `.get.side_effect`
    (HTTPStatusError / RuntimeError failure paths). Mirrors the pattern
    established in `tests/scraper/test_bfo_scraper.py` — patch at the
    audit-script's local import site, not `ufc_prediction.scraper.client`.
    """
    return MagicMock()


@pytest.fixture
def tmp_audit_out(tmp_path: Path) -> Path:
    """Per-test temp directory for audit JSON / MD outputs."""
    out = tmp_path / "audit_out"
    out.mkdir()
    return out
