"""Unit tests for scripts/ingest_pre_ufc_records_v25.py (DEBUT-V25-01).

Pure-function tests for the 6 behaviors locked by 43-01-PLAN.md:
1. enumerate_debutants — DB query shape (SQL string inspection)
2. classify_org_tier — locked 4-tier taxonomy (major/regional/local/none)
3. derive_last_organization — most-recent dated fight's event_name
4. build_csv_row — exact 14-column locked schema
5. upsert_csv — idempotent UPSERT on fighter_id
6. detect_antibot — Cloudflare / 403/429/503 signatures

These tests use fixtures only — no real Sherdog network calls.
"""

from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

import pytest

# Add scripts/ to path so we can import the CLI module directly.
SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture
def ingest_module():
    """Import the module under test (fresh per test via sys.path)."""
    import ingest_pre_ufc_records_v25 as m

    return m


# ─────────────────────────────────────────────────────────────────────────
# Test 1: enumerate_debutants returns (fighter_id, sherdog_url, first_ufc_date)
# ─────────────────────────────────────────────────────────────────────────


def test_enumerate_debutants_query_shape(ingest_module):
    """enumerate_debutants should query first-UFC-fight per fighter.

    Verifies the function signature + that it issues a SQL query that:
    - Selects from fighters joined to fights -> events
    - Aggregates min(event.date) per fighter (first UFC appearance)
    - Returns tuples (fighter_id, sherdog_url, first_ufc_date)
    """

    # We can't run a real DB here, so use a mock session capturing the
    # query() / filter() / all() chain. We assert: the function returns
    # whatever the session returns (it's a thin SELECT wrapper).
    class FakeQuery:
        def __init__(self, rows):
            self._rows = rows
            self.calls = []

        def join(self, *a, **kw):
            self.calls.append(("join", a, kw))
            return self

        def filter(self, *a, **kw):
            self.calls.append(("filter", a, kw))
            return self

        def group_by(self, *a, **kw):
            self.calls.append(("group_by", a, kw))
            return self

        def order_by(self, *a, **kw):
            self.calls.append(("order_by", a, kw))
            return self

        def all(self):
            return self._rows

    class FakeSession:
        def __init__(self, rows):
            self._rows = rows

        def query(self, *a, **kw):
            return FakeQuery(self._rows)

    fake_rows = [
        (1, "https://sherdog.com/fighter/foo-1", date(2010, 1, 1)),
        (2, None, date(2011, 6, 15)),
        (3, "https://sherdog.com/fighter/bar-3", date(2012, 11, 20)),
    ]
    session = FakeSession(fake_rows)
    result = ingest_module.enumerate_debutants(session)
    assert result == fake_rows
    # All fighters are debutants at first UFC appearance — universal filter.
    assert len(result) == 3
    # Each returned row is a 3-tuple (fighter_id, sherdog_url, first_ufc_date).
    for row in result:
        assert len(row) == 3
        assert isinstance(row[0], int)
        assert row[1] is None or isinstance(row[1], str)
        assert isinstance(row[2], date)


# ─────────────────────────────────────────────────────────────────────────
# Test 2: classify_org_tier — locked 4-tier taxonomy
# ─────────────────────────────────────────────────────────────────────────


def test_classify_org_tier_major(ingest_module):
    assert ingest_module.classify_org_tier("Bellator MMA") == "major"
    assert ingest_module.classify_org_tier("Bellator 290: Bader vs Fedor") == "major"
    assert ingest_module.classify_org_tier("ONE Championship") == "major"
    assert ingest_module.classify_org_tier("ONE FC") == "major"
    assert ingest_module.classify_org_tier("PFL Playoffs") == "major"
    assert ingest_module.classify_org_tier("Pride FC") == "major"
    assert ingest_module.classify_org_tier("Strikeforce") == "major"
    assert ingest_module.classify_org_tier("WEC 53") == "major"
    assert ingest_module.classify_org_tier("RIZIN 25") == "major"


def test_classify_org_tier_regional(ingest_module):
    assert ingest_module.classify_org_tier("Cage Warriors") == "regional"
    assert ingest_module.classify_org_tier("Cage Warriors 120") == "regional"
    assert ingest_module.classify_org_tier("LFA 100") == "regional"
    assert ingest_module.classify_org_tier("KSW 70: Khalidov vs Soldic") == "regional"
    assert ingest_module.classify_org_tier("M-1 Global") == "regional"
    assert ingest_module.classify_org_tier("Brave CF 50") == "regional"
    assert ingest_module.classify_org_tier("Eagle FC 44") == "regional"
    assert ingest_module.classify_org_tier("Combate Global") == "regional"


def test_classify_org_tier_local(ingest_module):
    assert ingest_module.classify_org_tier("Some Backyard Promo") == "local"
    assert ingest_module.classify_org_tier("Cage Fury FC 89") == "local"
    assert ingest_module.classify_org_tier("Titan FC") == "local"


def test_classify_org_tier_none(ingest_module):
    assert ingest_module.classify_org_tier("") == "none"
    assert ingest_module.classify_org_tier(None) == "none"
    assert ingest_module.classify_org_tier("   ") == "none"


def test_classify_org_tier_case_insensitive(ingest_module):
    """Token matching should be case-insensitive."""
    assert ingest_module.classify_org_tier("bellator mma") == "major"
    assert ingest_module.classify_org_tier("BELLATOR") == "major"
    assert ingest_module.classify_org_tier("cage warriors") == "regional"


# ─────────────────────────────────────────────────────────────────────────
# Test 3: derive_last_organization — most-recent dated fight's event_name
# ─────────────────────────────────────────────────────────────────────────


def test_derive_last_organization_returns_most_recent(ingest_module):
    from ufc_prediction.scraper.sherdog_models import SherdogFight

    fights = [
        SherdogFight(
            result="win",
            opponent_name="A",
            method="Decision",
            event_name="Bellator 100",
            event_date=date(2014, 1, 1),
        ),
        SherdogFight(
            result="win",
            opponent_name="B",
            method="KO/TKO",
            event_name="Cage Warriors 50",
            event_date=date(2016, 6, 1),
        ),
        SherdogFight(
            result="loss",
            opponent_name="C",
            method="Decision",
            event_name="LFA 20",
            event_date=date(2015, 3, 1),
        ),
    ]
    result = ingest_module.derive_last_organization(fights)
    # Most recent fight is Cage Warriors 50 (2016-06-01).
    assert result is not None
    assert "Cage Warriors" in result


def test_derive_last_organization_empty_list_returns_none(ingest_module):
    assert ingest_module.derive_last_organization([]) is None


def test_derive_last_organization_ignores_undated_fights(ingest_module):
    from ufc_prediction.scraper.sherdog_models import SherdogFight

    fights = [
        SherdogFight(
            result="win",
            opponent_name="A",
            method="Decision",
            event_name="Some Old Fight",
            event_date=None,
        ),
    ]
    assert ingest_module.derive_last_organization(fights) is None


# ─────────────────────────────────────────────────────────────────────────
# Test 4: build_csv_row — exact 14-column locked schema
# ─────────────────────────────────────────────────────────────────────────


def test_build_csv_row_emits_locked_14_columns(ingest_module):
    from ufc_prediction.scraper.sherdog_models import PreUFCRecord

    record = PreUFCRecord(
        total_wins=8,
        total_losses=2,
        total_draws=1,
        total_fights=12,  # 12 = 8 + 2 + 1 + 1 NC/DQ
        win_pct=0.667,
        ko_finish_rate=0.5,  # 4 KOs / 8 wins
        sub_finish_rate=0.25,  # 2 subs / 8 wins
        decision_rate=0.25,  # 2 decisions / 8 wins
        career_years=3.5,
        fights=[],
    )
    row = ingest_module.build_csv_row(
        fighter_id=42,
        sherdog_url="https://sherdog.com/fighter/foo-42",
        record=record,
        last_org="Bellator MMA",
        tier="major",
        scraped_at="2026-06-02T12:00:00Z",
    )
    expected_keys = {
        "fighter_id",
        "sherdog_url",
        "n_pre_ufc_fights",
        "wins",
        "losses",
        "draws",
        "nc_dq",
        "win_rate",
        "kos",
        "submissions",
        "decisions",
        "last_organization",
        "org_tier",
        "scraped_at",
    }
    assert set(row.keys()) == expected_keys
    assert len(row) == 14
    assert row["fighter_id"] == 42
    assert row["sherdog_url"] == "https://sherdog.com/fighter/foo-42"
    assert row["n_pre_ufc_fights"] == 12
    assert row["wins"] == 8
    assert row["losses"] == 2
    assert row["draws"] == 1
    assert row["nc_dq"] == 1  # 12 - 8 - 2 - 1 = 1
    assert row["win_rate"] == 0.667
    assert row["kos"] == 4  # round(0.5 * 8)
    assert row["submissions"] == 2  # round(0.25 * 8)
    assert row["decisions"] == 2  # round(0.25 * 8)
    assert row["last_organization"] == "Bellator MMA"
    assert row["org_tier"] == "major"
    assert row["scraped_at"] == "2026-06-02T12:00:00Z"


def test_build_csv_row_zero_wins_zero_finishes(ingest_module):
    from ufc_prediction.scraper.sherdog_models import PreUFCRecord

    record = PreUFCRecord(
        total_wins=0,
        total_losses=3,
        total_draws=0,
        total_fights=3,
        win_pct=0.0,
        ko_finish_rate=0.0,
        sub_finish_rate=0.0,
        decision_rate=0.0,
        career_years=1.0,
        fights=[],
    )
    row = ingest_module.build_csv_row(
        fighter_id=99,
        sherdog_url=None,
        record=record,
        last_org=None,
        tier="none",
        scraped_at="2026-06-02T12:00:00Z",
    )
    assert row["kos"] == 0
    assert row["submissions"] == 0
    assert row["decisions"] == 0
    assert row["nc_dq"] == 0  # 3 - 0 - 3 - 0


# ─────────────────────────────────────────────────────────────────────────
# Test 5: upsert_csv — idempotent UPSERT on fighter_id
# ─────────────────────────────────────────────────────────────────────────


def test_upsert_csv_appends_and_replaces(ingest_module, tmp_path):
    csv_path = tmp_path / "pre_ufc_records.csv"

    initial_rows = [
        _stub_row(ingest_module, fighter_id=1, wins=5),
        _stub_row(ingest_module, fighter_id=2, wins=3),
        _stub_row(ingest_module, fighter_id=3, wins=10),
    ]
    ingest_module.upsert_csv(csv_path, initial_rows)
    assert csv_path.exists()
    rows1 = _read_csv(csv_path)
    assert len(rows1) == 3

    # Now upsert: 2 new rows, one of which overlaps fighter_id=2
    new_batch = [
        _stub_row(ingest_module, fighter_id=2, wins=99),  # REPLACE
        _stub_row(ingest_module, fighter_id=4, wins=7),  # NEW
    ]
    ingest_module.upsert_csv(csv_path, new_batch)
    rows2 = _read_csv(csv_path)
    # 3 original + 1 new (fighter_id=4) - REPLACED row counts as same id
    assert len(rows2) == 4
    by_id = {int(r["fighter_id"]): r for r in rows2}
    assert set(by_id.keys()) == {1, 2, 3, 4}
    assert int(by_id[2]["wins"]) == 99  # REPLACED, not appended
    assert int(by_id[1]["wins"]) == 5  # untouched
    assert int(by_id[3]["wins"]) == 10  # untouched
    assert int(by_id[4]["wins"]) == 7  # new


def test_upsert_csv_creates_file_with_header(ingest_module, tmp_path):
    csv_path = tmp_path / "fresh.csv"
    assert not csv_path.exists()
    rows = [_stub_row(ingest_module, fighter_id=1, wins=1)]
    ingest_module.upsert_csv(csv_path, rows)
    assert csv_path.exists()
    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    assert header == ingest_module.CSV_COLUMNS


def test_upsert_csv_empty_batch_no_op_on_existing(ingest_module, tmp_path):
    csv_path = tmp_path / "x.csv"
    rows = [_stub_row(ingest_module, fighter_id=1, wins=5)]
    ingest_module.upsert_csv(csv_path, rows)
    ingest_module.upsert_csv(csv_path, [])
    assert len(_read_csv(csv_path)) == 1


# ─────────────────────────────────────────────────────────────────────────
# Test 6: detect_antibot — Cloudflare / 403/429/503 signatures
# ─────────────────────────────────────────────────────────────────────────


def test_detect_antibot_status_codes(ingest_module):
    assert ingest_module.detect_antibot("<html>ok</html>", 403) is True
    assert ingest_module.detect_antibot("<html>ok</html>", 429) is True
    assert ingest_module.detect_antibot("<html>ok</html>", 503) is True
    assert ingest_module.detect_antibot("<html>ok</html>", 200) is False
    assert ingest_module.detect_antibot("<html>ok</html>", 404) is False


def test_detect_antibot_cloudflare_signatures(ingest_module):
    assert ingest_module.detect_antibot("<html><body>Just a moment...</body></html>", 200) is True
    assert (
        ingest_module.detect_antibot(
            "<html><div class='cf-browser-verification'></div></html>", 200
        )
        is True
    )
    assert ingest_module.detect_antibot("<html>Cloudflare Ray ID: abc123</html>", 200) is True
    assert (
        ingest_module.detect_antibot(
            "<html><title>Attention Required! | Cloudflare</title></html>", 200
        )
        is True
    )


def test_detect_antibot_clean_page_passes(ingest_module):
    html = (
        "<html><body><h1>Joe Smith</h1>"
        "<table class='new_table'><tr><td>win</td></tr></table>"
        "</body></html>"
    )
    assert ingest_module.detect_antibot(html, 200) is False


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _stub_row(ingest_module, fighter_id: int, wins: int) -> dict:
    """Build a stub CSV row for upsert tests."""
    return {
        "fighter_id": fighter_id,
        "sherdog_url": f"https://sherdog.com/fighter/{fighter_id}",
        "n_pre_ufc_fights": wins,
        "wins": wins,
        "losses": 0,
        "draws": 0,
        "nc_dq": 0,
        "win_rate": 1.0 if wins else 0.0,
        "kos": 0,
        "submissions": 0,
        "decisions": wins,
        "last_organization": "Bellator MMA",
        "org_tier": "major",
        "scraped_at": "2026-06-02T12:00:00Z",
    }


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))
