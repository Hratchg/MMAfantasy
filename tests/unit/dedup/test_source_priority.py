"""HOUSE-04: parity tests for unified prefer_canonical helper.

Plan 16-01 Task 4. Two legacy call sites (predictor.py and fighter_queries.py)
unified behind ufc_prediction.dedup.source_priority. Tests below confirm:

  - SOURCE_PRIORITY literal contents match the legacy dict literals byte-for-byte
    (Pitfall #11 countermeasure — silent priority reorder is rejected by CI)
  - The new prefer_canonical reproduces both legacy tiebreak behaviours
    (predictor: lowest fighter_id wins; fighter_queries: highest Elo wins)
  - Empty-input contract (raises ValueError) is preserved
  - Unknown-source contract (priority 99 — sorts last) is preserved
  - Substring-edge fixtures from quick tasks 260501-uyd and 260501-u9u remain
    green via the new module (regression safety net)
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

source_priority = pytest.importorskip("ufc_prediction.dedup.source_priority")
from ufc_prediction.dedup.source_priority import (  # noqa: E402
    SOURCE_PRIORITY,
    prefer_canonical,
)


# ── Test doubles ─────────────────────────────────────────────────────────


@dataclass
class _FighterRow:
    """Stand-in for the predictor.py / fighter_queries.py row shape.

    Holds the only fields the dedup helper consults: id, source, and
    elo_after_shrinkage (as a float for the fighter_queries tiebreak path).
    """

    id: int
    name: str
    source: str
    elo_after_shrinkage: float = 1500.0


# ── 1. Dict literal parity (Pitfall #11) ────────────────────────────────


def test_source_priority_literal_matches_legacy_values():
    """HOUSE-04 invariant: SOURCE_PRIORITY must equal the legacy dict literals byte-for-byte.

    Legacy literals were:
      predictor._SOURCE_PRIORITY = {ufcstats:0, sherdog:1, kaggle-rajeevw:2, kaggle-mdabbert:3}
      fighter_queries._RANKINGS_SOURCE_PRIORITY = {ufcstats:0, sherdog:1, kaggle-rajeevw:2, kaggle-mdabbert:3}

    Adding a new ingest source means updating SOURCE_PRIORITY here AND adding a
    parity row to this test. Any silent reorder fails CI before it can ship.
    """
    assert SOURCE_PRIORITY == {
        "ufcstats": 0,
        "sherdog": 1,
        "kaggle-rajeevw": 2,
        "kaggle-mdabbert": 3,
    }
    # Spot-checks for "did someone reorder by accident" failure modes.
    assert SOURCE_PRIORITY["ufcstats"] == 0
    assert SOURCE_PRIORITY["sherdog"] == 1
    assert SOURCE_PRIORITY["kaggle-rajeevw"] < SOURCE_PRIORITY["kaggle-mdabbert"]


# ── 2. Predictor parity (lowest fighter.id wins on tie) ─────────────────


@pytest.mark.parametrize(
    "rows,expected_source",
    [
        # Mixed sources — ufcstats wins.
        (
            [
                _FighterRow(id=2, name="Joe", source="kaggle-rajeevw"),
                _FighterRow(id=1, name="Joe", source="ufcstats"),
            ],
            "ufcstats",
        ),
        # kaggle-only — must still resolve (D-03 graceful path).
        (
            [_FighterRow(id=1, name="Joe", source="kaggle-rajeevw")],
            "kaggle-rajeevw",
        ),
        # Unknown source — sorts last (priority 99).
        (
            [
                _FighterRow(id=1, name="Joe", source="ufcstats"),
                _FighterRow(id=2, name="Joe", source="some-future-source"),
            ],
            "ufcstats",
        ),
        # Same-source tiebreak via id (lowest wins).
        (
            [
                _FighterRow(id=42, name="Joe", source="ufcstats"),
                _FighterRow(id=7, name="Joe", source="ufcstats"),
            ],
            "ufcstats",  # both ufcstats; the id=7 row should win
        ),
    ],
)
def test_predictor_parity_lowest_id_wins(rows, expected_source):
    """Predictor.py call site uses tiebreak_key=lambda f: f.id."""
    canonical = prefer_canonical(rows, source_key=lambda f: f.source, tiebreak_key=lambda f: f.id)
    assert canonical.source == expected_source


def test_predictor_parity_same_source_picks_lowest_id():
    """Same-source row pair: id=7 must beat id=42 to match legacy tiebreak."""
    rows = [
        _FighterRow(id=42, name="Joe", source="ufcstats"),
        _FighterRow(id=7, name="Joe", source="ufcstats"),
    ]
    canonical = prefer_canonical(rows, source_key=lambda f: f.source, tiebreak_key=lambda f: f.id)
    assert canonical.id == 7


# ── 3. Fighter-queries parity (highest Elo wins on tie) ─────────────────


def test_fighter_queries_parity_highest_elo_wins_within_same_source():
    """fighter_queries.get_division_rankings ties on -float(r.elo_after_shrinkage).

    With two ufcstats rows at the same source-priority bucket, the row with
    the higher elo_after_shrinkage must win.
    """
    rows = [
        _FighterRow(id=10, name="Top", source="ufcstats", elo_after_shrinkage=1820.5),
        _FighterRow(id=11, name="Top", source="ufcstats", elo_after_shrinkage=1750.0),
    ]
    canonical = prefer_canonical(
        rows,
        source_key=lambda r: r.source,
        tiebreak_key=lambda r: -float(r.elo_after_shrinkage),
    )
    assert canonical.elo_after_shrinkage == 1820.5


def test_fighter_queries_parity_source_beats_elo():
    """Source priority is dominant — even a higher-Elo kaggle row loses to ufcstats."""
    rows = [
        _FighterRow(id=10, name="Top", source="kaggle-rajeevw", elo_after_shrinkage=1900.0),
        _FighterRow(id=11, name="Top", source="ufcstats", elo_after_shrinkage=1500.0),
    ]
    canonical = prefer_canonical(
        rows,
        source_key=lambda r: r.source,
        tiebreak_key=lambda r: -float(r.elo_after_shrinkage),
    )
    assert canonical.source == "ufcstats"


# ── 4. Empty-input + edge contracts ─────────────────────────────────────


def test_empty_rows_raises_value_error():
    """prefer_canonical([]) must raise — never return None or arbitrary."""
    with pytest.raises(ValueError, match="at least one row"):
        prefer_canonical([], source_key=lambda r: r.source)


def test_unknown_source_sorts_last():
    """An unrecognised source string falls through to priority 99."""
    rows = [
        _FighterRow(id=2, name="Joe", source="random_blog"),
        _FighterRow(id=1, name="Joe", source="ufcstats"),
    ]
    canonical = prefer_canonical(rows, source_key=lambda r: r.source)
    assert canonical.source == "ufcstats"


def test_unknown_only_source_still_resolves():
    """If the only row has an unknown source, it still wins (graceful)."""
    rows = [_FighterRow(id=1, name="Joe", source="random_blog")]
    canonical = prefer_canonical(rows, source_key=lambda r: r.source)
    assert canonical.source == "random_blog"


# ── 5. Substring-edge regression (260501-uyd / 260501-u9u) ──────────────


def test_substring_regression_cormier_dedups_to_ufcstats():
    """Quick task 260501-uyd: 'Cormier' substring matches Daniel Cormier across two
    ingest sources; canonical must be the ufcstats row.
    """
    rows = [
        _FighterRow(id=1, name="Daniel Cormier", source="ufcstats"),
        _FighterRow(id=2, name="Daniel Cormier", source="kaggle-rajeevw"),
    ]
    canonical = prefer_canonical(rows, source_key=lambda f: f.source, tiebreak_key=lambda f: f.id)
    assert canonical.source == "ufcstats"
    assert canonical.name == "Daniel Cormier"


def test_substring_regression_topuria_two_real_people_each_dedups():
    """Quick task 260501-uyd: 'Topuria' matches Ilia + Aleksandre. After per-name
    grouping each pair dedups to its ufcstats canonical row; the upstream caller
    surfaces both canonicals and raises a Multiple-fighters-match error.
    Here we just verify that EACH per-name group dedups correctly.
    """
    ilia_pair = [
        _FighterRow(id=1, name="Ilia Topuria", source="ufcstats"),
        _FighterRow(id=2, name="Ilia Topuria", source="kaggle-rajeevw"),
    ]
    alex_pair = [
        _FighterRow(id=3, name="Aleksandre Topuria", source="ufcstats"),
        _FighterRow(id=4, name="Aleksandre Topuria", source="kaggle-mdabbert"),
    ]
    ilia_canonical = prefer_canonical(
        ilia_pair, source_key=lambda f: f.source, tiebreak_key=lambda f: f.id
    )
    alex_canonical = prefer_canonical(
        alex_pair, source_key=lambda f: f.source, tiebreak_key=lambda f: f.id
    )
    assert ilia_canonical.source == "ufcstats"
    assert alex_canonical.source == "ufcstats"
    assert ilia_canonical.name != alex_canonical.name  # genuinely different people


def test_substring_regression_bfo_2026_id_resolution():
    """Quick task 260501-u9u: rematch dedup must still pick the ufcstats row even
    when both candidates share Elo. Sanity check that rematch handling does not
    interact with HOUSE-04 unification.
    """
    rows = [
        _FighterRow(id=1, name="Conor McGregor", source="ufcstats", elo_after_shrinkage=1700.0),
        _FighterRow(
            id=2, name="Conor McGregor", source="kaggle-rajeevw", elo_after_shrinkage=1700.0
        ),
    ]
    canonical = prefer_canonical(
        rows,
        source_key=lambda r: r.source,
        tiebreak_key=lambda r: -float(r.elo_after_shrinkage),
    )
    assert canonical.source == "ufcstats"
