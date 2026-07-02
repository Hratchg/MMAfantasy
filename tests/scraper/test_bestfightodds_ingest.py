"""Unit tests for Pydantic validation of ufcscraper CSV rows.

Covers:
  - BFOOddsRow construction + boundary validation (T-13-01 type guard)
  - BFOOddsRow.event_date parsing from composite fight_id (999.4)
  - BFOFighterName construction + empty-name rejection (T-13-02)
  - End-to-end fixture CSVs parse cleanly through both models

Pure-logic: reads local CSV fixtures, no network, no DB.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from ufc_prediction.scraper.bfo_models import BFOFighterName, BFOOddsRow

FIXTURES = Path(__file__).parent / "fixtures"
ODDS_FIXTURE = FIXTURES / "bestfightodds_sample.csv"
NAMES_FIXTURE = FIXTURES / "bestfightodds_fighters_names_sample.csv"


# ── Direct construction tests ────────────────────────────────────────────────


def test_bfo_odds_row_accepts_valid_row() -> None:
    """Canonical valid row constructs and preserves field values."""
    row = BFOOddsRow(
        fight_id="abc123",
        fighter_id="fx1",
        opening=-200,
        closing_range_min=-205,
        closing_range_max=-195,
    )
    assert row.opening == -200
    assert row.closing_range_min == -205
    assert row.closing_range_max == -195


def test_bfo_odds_row_accepts_null_odds() -> None:
    """Missing odds fields are valid per D-07 / D-08 (XGBoost handles NaN)."""
    row = BFOOddsRow(fight_id="x", fighter_id="y")
    assert row.opening is None
    assert row.closing_range_min is None
    assert row.closing_range_max is None


def test_upsert_row_bad_moneyline_does_not_abort_batch() -> None:
    """review #12 (Major): an out-of-domain moneyline (|ml|<100) must skip that
    row's odds and still upsert (NULL prob) — NOT raise and kill the whole batch.
    """
    from unittest.mock import MagicMock

    from ufc_prediction.scraper.bfo_ingest import BFOOddsIngester

    ingester = BFOOddsIngester(session=MagicMock(), data_folder=Path("."))
    # this_row has an invalid closing range (-50 is not a valid American ML);
    # opening is valid. The devig helper will raise ValueError internally.
    this_row = BFOOddsRow(
        fight_id="f", fighter_id="a", opening=-200, closing_range_min=-50, closing_range_max=-40
    )
    other_row = BFOOddsRow(
        fight_id="f", fighter_id="b", opening=170, closing_range_min=170, closing_range_max=180
    )
    # Must not raise despite the bad closing moneyline.
    ingester._upsert_row(1, 1, this_row, other_row)
    # The row is still upserted, keeping the batch alive...
    assert ingester._session.execute.called
    # ...and the bad closing prob is NULL while the valid opening prob is kept
    # (a regression that swallowed the error but wrote garbage would fail here).
    from sqlalchemy.dialects import postgresql

    stmt = ingester._session.execute.call_args.args[0]
    params = stmt.compile(dialect=postgresql.dialect()).params
    assert params["closing_implied_prob"] is None
    assert params["opening_implied_prob"] is not None


def test_bfo_odds_row_rejects_non_int_opening() -> None:
    """Non-numeric string on opening ML raises ValidationError (T-13-01)."""
    with pytest.raises(ValidationError):
        BFOOddsRow(fight_id="x", fighter_id="y", opening="strong")


def test_bfo_odds_row_rejects_float_opening() -> None:
    """Floats are rejected — moneylines are integers (T-13-01)."""
    with pytest.raises(ValidationError):
        BFOOddsRow(fight_id="x", fighter_id="y", opening=1.5)


def test_bfo_odds_row_extracts_event_date_from_composite_fight_id() -> None:
    """BFOOddsRow.event_date parses YYYY-MM-DD prefix from composite key (999.4).

    The real BFO CSV uses ``YYYY-MM-DD|<bfo_a>|<bfo_b>`` for ``fight_id``
    (see data/bfo/BestFightOdds_odds.csv). The ingester must use this date
    to disambiguate rematches — every BFO row for a rematch pair otherwise
    resolves to the most recent fight via ORDER BY date DESC + .first(),
    silently overwriting earlier bouts' odds via the pk_fight_odds upsert.

    Three contracts:
      1. Happy path: parses to a typed ``date``.
      2. Malformed prefix (no ISO date): raises ``ValueError`` — explicit
         rejection > silent ``None``, because silent fallback would
         re-introduce the H1 fall-back-to-most-recent bug.
      3. Missing pipe (no composite key shape at all): raises ``ValueError``
         for the same reason.
    """
    # 1. Happy path — real-CSV format (e.g. 2021-09-17|3227|8697)
    row = BFOOddsRow(fight_id="2017-07-29|3227|8697", fighter_id="3227")
    assert row.event_date == date(2017, 7, 29)

    # 2. Malformed date prefix (has the pipe shape, but the prefix isn't ISO)
    with pytest.raises(ValueError):
        _ = BFOOddsRow(fight_id="not-a-date|3227|8697", fighter_id="3227").event_date

    # 3. No pipe at all — old fixture-style opaque key
    with pytest.raises(ValueError):
        _ = BFOOddsRow(fight_id="onlydate", fighter_id="x").event_date


def test_bfo_fighter_name_accepts_valid_row() -> None:
    """Canonical fighter_names row constructs."""
    row = BFOFighterName(
        fighter_id="fx1",
        database="ufcstats",
        name="Jon Jones",
        database_id="abc",
    )
    assert row.fighter_id == "fx1"
    assert row.database == "ufcstats"
    assert row.name == "Jon Jones"
    assert row.database_id == "abc"


def test_bfo_fighter_name_allows_missing_database_id() -> None:
    """database_id is optional (ufcscraper may emit blanks)."""
    row = BFOFighterName(fighter_id="fx1", database="ufcstats", name="Jon Jones")
    assert row.database_id is None


def test_bfo_fighter_name_rejects_empty_name() -> None:
    """Empty name is an invalid row (T-13-02)."""
    with pytest.raises(ValidationError):
        BFOFighterName(fighter_id="fx1", database="ufcstats", name="")


# ── CSV fixture integration tests ────────────────────────────────────────────


_INT_FIELDS = {"opening", "closing_range_min", "closing_range_max"}


def _coerce_row(row: dict[str, str]) -> dict[str, object]:
    """Map DictReader row strings to values the Pydantic model expects.

    Pydantic v2 coerces numeric strings to ints when strict=False, but
    empty strings need explicit ``None`` mapping before validation.
    """
    out: dict[str, object] = {}
    for k, v in row.items():
        if k in _INT_FIELDS:
            out[k] = None if v == "" else v
        else:
            out[k] = None if v == "" else v
    return out


def test_csv_fixture_parses_via_pydantic() -> None:
    """All rows of bestfightodds_sample.csv validate; fixture has >= 5 rows."""
    assert ODDS_FIXTURE.exists(), f"fixture missing: {ODDS_FIXTURE}"
    rows: list[BFOOddsRow] = []
    with ODDS_FIXTURE.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            rows.append(BFOOddsRow(**_coerce_row(raw)))
    assert len(rows) >= 5, f"expected >= 5 fixture rows, got {len(rows)}"


def test_fighters_names_fixture_parses_via_pydantic() -> None:
    """All rows of bestfightodds_fighters_names_sample.csv validate; >= 5 rows."""
    assert NAMES_FIXTURE.exists(), f"fixture missing: {NAMES_FIXTURE}"
    rows: list[BFOFighterName] = []
    with NAMES_FIXTURE.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            rows.append(BFOFighterName(**_coerce_row(raw)))
    assert len(rows) >= 5, f"expected >= 5 fixture rows, got {len(rows)}"
