"""Phase 21 Plan 21-01 Task 2 — Couture-Liddell rematch regression guard.

These tests pin the existing ``BFOOddsIngester._resolve_fight`` closest-date
matching behavior (the 999.4 / ``260501-u9u`` fix) against the canonical
pre-2010 rematch class. If anyone modifies ``_resolve_fight`` in a way
that breaks rematch disambiguation, these tests fail BEFORE the bad code
can be committed.

DO NOT add a "schema-version dispatch" branch to ``_resolve_fight`` —
the existing closest-date logic is the regression guard. CONTEXT.md
REVISION (post-21-RESEARCH.md) made D-04 explicit on this: the fixture
exercises the EXISTING logic; ``_resolve_fight`` stays byte-identical.

Couture-Liddell I/II/III are pre-2007 events (pre-archive in BFO per
Phase 20 finding) so they will never actually flow through Phase 21's
live backfill. The CSV fixtures are kept here as regression test data
ONLY — they represent the URL-format class on which the original 999.4
rematch-overwrite bug hid (fixed in commit ``ce35a90``).
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

from ufc_prediction.scraper.bfo_ingest import BFOOddsIngester
from ufc_prediction.scraper.bfo_models import BFOOddsRow

FIXTURES = Path(__file__).parent.parent / "fixtures"


# Constants matching the synthetic fixture data.
COUTURE_DB_ID = 101
LIDDELL_DB_ID = 102


# Three historical Couture-Liddell fights with their DB Fight.id values
# (synthetic — these IDs only need to be unique within each test).
COUTURE_LIDDELL_I_FIGHT_ID = 1001
COUTURE_LIDDELL_I_EVENT_DATE = date(2003, 6, 6)  # UFC 43
COUTURE_LIDDELL_II_FIGHT_ID = 1002
COUTURE_LIDDELL_II_EVENT_DATE = date(2005, 4, 16)  # UFC 52
COUTURE_LIDDELL_III_FIGHT_ID = 1003
COUTURE_LIDDELL_III_EVENT_DATE = date(2006, 2, 4)  # UFC 57


from collections import namedtuple

# Use a namedtuple instead of MagicMock so .id and .date are real values
# (MagicMock(id=...) sets the attribute but `min(...).id` returns a mock,
# which breaks tuple-unpacking in _resolve_fight at line 318).
_Candidate = namedtuple("_Candidate", ["id", "date"])


def _build_ingester_with_three_rematches() -> BFOOddsIngester:
    """Mock a Session whose ``_resolve_fight`` query returns all 3 fights."""
    candidates = [
        _Candidate(id=COUTURE_LIDDELL_I_FIGHT_ID, date=COUTURE_LIDDELL_I_EVENT_DATE),
        _Candidate(id=COUTURE_LIDDELL_II_FIGHT_ID, date=COUTURE_LIDDELL_II_EVENT_DATE),
        _Candidate(id=COUTURE_LIDDELL_III_FIGHT_ID, date=COUTURE_LIDDELL_III_EVENT_DATE),
    ]
    # MagicMock the chained session.execute().all() pattern used in _resolve_fight.
    session = MagicMock()
    exec_result = MagicMock()
    exec_result.all.return_value = candidates
    session.execute.return_value = exec_result

    return BFOOddsIngester(
        session=session,
        data_folder=Path("/nonexistent"),  # never actually read
        date_window_days=14,
    )


# ── Couture-Liddell I (UFC 43, 2003-06-06) ────────────────────────────────


def test_resolve_fight_i_picks_2003_event_date():
    """BFO row for Couture-Liddell I resolves to UFC 43 (not II or III)."""
    fixture = FIXTURES / "bfo_resolve_couture_liddell_i.csv"
    with fixture.open() as fh:
        rows = [BFOOddsRow(**r) for r in csv.DictReader(fh)]
    assert rows[0].event_date == COUTURE_LIDDELL_I_EVENT_DATE

    ingester = _build_ingester_with_three_rematches()
    resolved = ingester._resolve_fight(COUTURE_DB_ID, LIDDELL_DB_ID, rows[0].event_date)
    assert resolved == COUTURE_LIDDELL_I_FIGHT_ID, (
        "Closest-date matching must pick I, not the most-recent fight"
    )


# ── Couture-Liddell II (UFC 52, 2005-04-16) ───────────────────────────────


def test_resolve_fight_ii_picks_2005_event_date():
    fixture = FIXTURES / "bfo_resolve_couture_liddell_ii.csv"
    with fixture.open() as fh:
        rows = [BFOOddsRow(**r) for r in csv.DictReader(fh)]
    assert rows[0].event_date == COUTURE_LIDDELL_II_EVENT_DATE

    ingester = _build_ingester_with_three_rematches()
    resolved = ingester._resolve_fight(COUTURE_DB_ID, LIDDELL_DB_ID, rows[0].event_date)
    assert resolved == COUTURE_LIDDELL_II_FIGHT_ID, "Closest-date matching must pick II"


# ── Couture-Liddell III (UFC 57, 2006-02-04) ──────────────────────────────


def test_resolve_fight_iii_picks_2006_event_date():
    fixture = FIXTURES / "bfo_resolve_couture_liddell_iii.csv"
    with fixture.open() as fh:
        rows = [BFOOddsRow(**r) for r in csv.DictReader(fh)]
    assert rows[0].event_date == COUTURE_LIDDELL_III_EVENT_DATE

    ingester = _build_ingester_with_three_rematches()
    resolved = ingester._resolve_fight(COUTURE_DB_ID, LIDDELL_DB_ID, rows[0].event_date)
    assert resolved == COUTURE_LIDDELL_III_FIGHT_ID, "Closest-date matching must pick III"


# ── Outside date-window: returns None (refuse to mis-attach) ──────────────


def test_resolve_fight_outside_window_returns_none():
    """An event_date far from all 3 candidates returns None (skip > mis-attach)."""
    ingester = _build_ingester_with_three_rematches()
    # 2010-01-01 is 4+ years past UFC 57; far outside the 14-day window.
    resolved = ingester._resolve_fight(COUTURE_DB_ID, LIDDELL_DB_ID, date(2010, 1, 1))
    assert resolved is None, (
        "Must return None when no candidate is within date_window_days; "
        "refusing to mis-attach is the 999.4 regression guard."
    )


# ── No candidates: returns None ───────────────────────────────────────────


def test_resolve_fight_no_candidates_returns_none():
    """Mock returns 0 candidates → _resolve_fight returns None."""
    session = MagicMock()
    exec_result = MagicMock()
    exec_result.all.return_value = []
    session.execute.return_value = exec_result
    ingester = BFOOddsIngester(
        session=session, data_folder=Path("/nonexistent"), date_window_days=14
    )
    assert (
        ingester._resolve_fight(COUTURE_DB_ID, LIDDELL_DB_ID, COUTURE_LIDDELL_I_EVENT_DATE) is None
    )


# ── Byte-identity guard: _resolve_fight signature unchanged ────────────────


def test_resolve_fight_signature_unchanged():
    """If anyone re-introduces an explicit schema_version arg, this fails."""
    import inspect

    sig = inspect.signature(BFOOddsIngester._resolve_fight)
    expected_params = ("self", "db_fighter_a", "db_fighter_b", "event_date")
    assert tuple(sig.parameters.keys()) == expected_params, (
        f"_resolve_fight signature drifted; got {tuple(sig.parameters.keys())}. "
        "CONTEXT.md D-04 REVISION: do NOT add schema_version dispatch — "
        "the existing closest-date matching IS the regression guard."
    )
