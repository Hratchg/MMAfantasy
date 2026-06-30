"""Phase 23 Q6 coverage smoke test.

Reads the Q6 fact-finding JSON written by Plan 23-01 Task 1 and verifies it
is well-formed. Does NOT impose a coverage floor (CONTEXT D-02 says NaN /
global-rate fallback handles low coverage gracefully).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

Q6_PATH = Path(".planning/phases/23-feature-engineering-append-only/23-Q6-COVERAGE.json")


def test_phase22_referee_venue_coverage_smoke() -> None:
    """Verify Q6 JSON exists, parses, and has expected keys.

    DB-unreachable case (queried_at is None) → skip with explanatory message.
    Otherwise asserts schema + bounds on the coverage fractions.
    """
    if not Q6_PATH.exists():
        pytest.skip(
            f"Q6 coverage JSON not yet written at {Q6_PATH} — Plan 23-01 Task 1 Step B has not run."
        )

    data = json.loads(Q6_PATH.read_text(encoding="utf-8"))

    if data.get("queried_at") is None:
        pytest.skip(
            "Q6 coverage JSON has queried_at=None — operator must populate "
            "(DB was unreachable when the fact-finding query ran)."
        )

    assert "total_events" in data
    assert "with_referee" in data
    assert "with_venue" in data
    assert "pct_with_referee" in data
    assert "pct_with_venue" in data

    assert data["total_events"] > 0
    assert data["with_referee"] >= 0
    assert data["with_venue"] >= 0
    assert 0.0 <= data["pct_with_referee"] <= 1.0
    assert 0.0 <= data["pct_with_venue"] <= 1.0
